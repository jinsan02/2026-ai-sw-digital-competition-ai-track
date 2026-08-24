"""Build and smoke-test submit.zip from a model directory.

Usage:
    .venv/bin/python package_submission.py --out baseline_0702.zip          # repackage ./model (baseline stack)
    .venv/bin/python package_submission.py \
        --hf-dir experiments/incoming/models/NAME --no-sparse --out m3_len448_s42.zip

Zips land in submissions/ (gitignored). Naming rule: filename ≤ 30 chars, no
"submit" prefix — e.g. m3_len448_s42.zip.

Assembles the submission contract (script.py, requirements.txt, model/) into a
zip with exactly those three root entries, then smoke-tests a clean extraction:
data/ -> symlink to open/data, TRANSFORMERS_OFFLINE=1 python script.py, and
validates submission.csv (columns id,action; ID order == sample_submission.csv;
labels within the model's classes). Reports zip size and inference wall time.

--hf-dir needs hf_model/ + hf_meta.json (what --save-val-model / --final-model
write). Sparse SVC files ride along from --sparse-dir unless --no-sparse; the
sparse blend weight is tuned against a specific transformer's logit scale, so
pair them only if they were tuned together (new encoders: --no-sparse).
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
ZIP_LIMIT_MB = 1024
VALID_ROOT = {"script.py", "requirements.txt", "model"}


def fail(msg):
    sys.exit(f"package_submission: {msg}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_hf_configs(model_dir):
    """Backport newer HF config keys to the 4.51 eval/runtime format.

    Transformers 5.x serializes Llama RoPE as rope_parameters; 4.51 ignores
    that key and falls back to rope_theta=10000 unless the legacy top-level
    key is present.
    """
    for config_path in sorted(model_dir.glob("hf_model*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        changed = False
        rope_params = config.get("rope_parameters") or {}
        if "rope_theta" in rope_params and "rope_theta" not in config:
            config["rope_theta"] = rope_params["rope_theta"]
            changed = True
        if "rope_theta" in config and "rope_scaling" not in config:
            config["rope_scaling"] = None
            changed = True
        if config.get("dtype") and "torch_dtype" not in config:
            config["torch_dtype"] = config["dtype"]
            changed = True
        if changed:
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"normalized HF config for 4.51 runtime: {config_path.relative_to(model_dir)}")


def stage(hf_dir, sparse_dir, staging, leak_lookup=False, test_graph_backfill=False,
          test_graph_aligned_only=False, requirements=None):
    hf_dir = Path(hf_dir)
    if not (hf_dir / "hf_model").is_dir() or not (hf_dir / "hf_meta.json").is_file():
        fail(f"{hf_dir} must contain hf_model/ and hf_meta.json")
    source_meta = json.loads((hf_dir / "hf_meta.json").read_text(encoding="utf-8"))
    specialist_enabled = bool((source_meta.get("weak4_specialist") or {}).get("enabled", False))
    if specialist_enabled:
        expected_script = (source_meta.get("weak4_provenance") or {}).get("script_sha256")
        actual_script = sha256_file(REPO / "script.py")
        if expected_script != actual_script:
            fail(
                "weak4 pack was built for a different script.py: "
                f"pack={expected_script!r} current={actual_script}"
            )
    if specialist_enabled and sparse_dir is not None:
        fail("weak4_specialist packages require --no-sparse")
    if specialist_enabled and (leak_lookup or test_graph_backfill or test_graph_aligned_only):
        fail("weak4_specialist packages forbid leak lookup and test graph backfill")
    shutil.copy2(REPO / "script.py", staging / "script.py")
    shutil.copy2(requirements or (REPO / "requirements.txt"), staging / "requirements.txt")
    model_dir = staging / "model"
    model_dir.mkdir()
    for enc_dir in sorted(hf_dir.glob("hf_model*")):
        if enc_dir.is_dir():
            shutil.copytree(enc_dir, model_dir / enc_dir.name)
    for lora_dir in sorted(hf_dir.glob("lora_*")):
        if lora_dir.is_dir():
            shutil.copytree(lora_dir, model_dir / lora_dir.name)
    normalize_hf_configs(model_dir)
    shutil.copy2(hf_dir / "hf_meta.json", model_dir / "hf_meta.json")
    if test_graph_backfill or test_graph_aligned_only:
        enable_test_graph_backfill(model_dir, aligned_only=test_graph_aligned_only)
    if sparse_dir is not None:
        sparse_dir = Path(sparse_dir)
        for name in ("sparse_svc.pkl", "sparse_meta.json"):
            if not (sparse_dir / name).is_file():
                fail(f"{sparse_dir}/{name} missing; use --no-sparse for an encoder-only package")
            shutil.copy2(sparse_dir / name, model_dir / name)
        if sparse_dir.resolve() != hf_dir.resolve():
            print(f"WARNING: sparse files from {sparse_dir} but transformer from {hf_dir} -- "
                  "the blend weight is only valid if they were tuned together")
    if leak_lookup:
        stage_leak_lookup(model_dir)
    meta = json.loads((model_dir / "hf_meta.json").read_text(encoding="utf-8"))
    print(f"staged: base={meta.get('base_model')} max_length={meta.get('max_length')} "
          f"final_refit={meta.get('final_refit')} fp16={meta.get('saved_fp16')} "
          f"sparse={'yes' if sparse_dir is not None else 'no'} "
          f"leak_lookup={'yes' if leak_lookup else 'no'} "
          f"test_graph={'yes' if (test_graph_backfill or test_graph_aligned_only) else 'no'} "
          f"rule_boosts={len(meta.get('rule_boosts') or [])}")
    return meta


def stage_leak_lookup(model_dir):
    """Build the leak lookup fresh from open/data so it can never go stale."""
    import gzip

    from build_leak_lookup import build_lookup
    from script import LEAK_LOOKUP_FILENAME, load_jsonl

    samples = load_jsonl(str(REPO / "open/data/train.jsonl"))
    with (REPO / "open/data/train_labels.csv").open(newline="", encoding="utf-8") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    payload = build_lookup(samples, labels)
    with gzip.open(model_dir / LEAK_LOOKUP_FILENAME, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"staged leak lookup: by_prompt={len(payload['by_prompt'])} "
          f"by_prompt_last={len(payload['by_prompt_last'])} entries")


def enable_test_graph_backfill(model_dir, aligned_only=False):
    meta_path = model_dir / "hf_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["test_batch_graph_backfill"] = {
        "enabled": True,
        "positional": not aligned_only,
        "aligned": True,
        "notes": "Same-test-batch history graph backfill only; no train-derived prompt lookup.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mode = "aligned-only" if aligned_only else "positional+aligned"
    print(f"enabled test-batch graph backfill in hf_meta.json ({mode})")


def build_zip(staging, out_path):
    tmp_zip = out_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(staging.rglob("*")):
            zf.write(path, path.relative_to(staging))
    tmp_zip.replace(out_path)
    with zipfile.ZipFile(out_path) as zf:
        roots = {name.split("/")[0] for name in zf.namelist()}
    if roots != VALID_ROOT:
        fail(f"zip root entries {sorted(roots)} != {sorted(VALID_ROOT)}")
    size_mb = out_path.stat().st_size / 1e6
    print(f"built {out_path} ({size_mb:.0f} MB, root entries OK)")
    if size_mb > ZIP_LIMIT_MB:
        fail(f"zip exceeds the {ZIP_LIMIT_MB} MB submission limit")
    return size_mb


def smoke(out_path, meta, python_bin, force_cpu):
    with tempfile.TemporaryDirectory(prefix="aadp_smoke_") as td:
        workdir = Path(td)
        with zipfile.ZipFile(out_path) as zf:
            zf.extractall(workdir)
        (workdir / "data").symlink_to(REPO / "open/data")
        env = dict(os.environ, TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        if force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        start = time.perf_counter()
        run = subprocess.run([python_bin, "script.py"], cwd=workdir, env=env,
                             capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        print(run.stdout.strip())
        if run.returncode != 0:
            print(run.stderr, file=sys.stderr)
            fail(f"smoke run failed (rc={run.returncode})")

        sub_path = workdir / "output/submission.csv"
        if not sub_path.is_file():
            fail("smoke run produced no output/submission.csv")
        with sub_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        if header != ["id", "action"]:
            fail(f"submission columns {header} != ['id', 'action']")
        with (REPO / "open/data/sample_submission.csv").open(newline="", encoding="utf-8") as f:
            sample_ids = [row["id"] for row in csv.DictReader(f)]
        if [row[0] for row in rows] != sample_ids:
            fail("submission ID order does not match sample_submission.csv")
        bad = sorted({row[1] for row in rows} - set(meta["classes"]))
        if bad:
            fail(f"invalid labels in submission: {bad}")
        n = len(rows)
        print(f"smoke OK: {n} rows, columns/ID-order/labels valid, "
              f"wall {elapsed:.1f}s ({elapsed / max(1, n):.2f}s/row incl. model load; "
              f"local stub test set -- server timing needs the real 10-min budget check)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-dir", default="model",
                        help="dir with hf_model/ + hf_meta.json (default: model)")
    parser.add_argument("--sparse-dir", default="model",
                        help="dir with sparse_svc.pkl + sparse_meta.json (default: model)")
    parser.add_argument("--no-sparse", action="store_true", help="encoder-only package")
    parser.add_argument("--leak-lookup", action="store_true",
                        help="include model/leak_lookup.json.gz, which re-enables ALL leak-override "
                             "tiers in script.py (07-04 probe: Public 0.710 vs 0.743 — off by default)")
    parser.add_argument("--test-graph-backfill", action="store_true",
                        help="enable same-test-batch history graph backfill only; does not package "
                             "train-derived leak_lookup.json.gz")
    parser.add_argument("--test-graph-aligned-only", action="store_true",
                        help="same as --test-graph-backfill but disables id/step positional matching")
    parser.add_argument("--out", default=None,
                        help="zip filename, ≤30 chars (default: <hf-dir name>.zip); "
                             "relative paths land in submissions/")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="smoke with CUDA_VISIBLE_DEVICES=''")
    parser.add_argument("--python", default=str(REPO / ".venv/bin/python"))
    parser.add_argument("--requirements", default=str(REPO / "requirements.txt"),
                        help="requirements.txt variant to ship (e.g. transformers 4.51 for Qwen3 packs)")
    args = parser.parse_args()
    if args.leak_lookup and (args.test_graph_backfill or args.test_graph_aligned_only):
        fail("--leak-lookup cannot be combined with test graph backfill; probe them separately")

    out_path = Path(args.out or Path(args.hf_dir).resolve().name + ".zip")
    if not out_path.is_absolute():
        out_path = REPO / "submissions" / out_path
    if len(out_path.name) > 30:
        fail(f"zip filename '{out_path.name}' is over the 30-char naming rule; pass a shorter --out")
    if out_path.name.lower().startswith("submit"):
        fail(f"zip filename '{out_path.name}' must not start with 'submit'")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aadp_stage_") as td:
        staging = Path(td)
        meta = stage(args.hf_dir, None if args.no_sparse else args.sparse_dir, staging,
                     leak_lookup=args.leak_lookup,
                     test_graph_backfill=args.test_graph_backfill,
                     test_graph_aligned_only=args.test_graph_aligned_only,
                     requirements=args.requirements)
        build_zip(staging, out_path)
    if args.skip_smoke:
        print("smoke skipped (--skip-smoke)")
        return
    smoke(out_path, meta, args.python, args.cpu)


if __name__ == "__main__":
    main()
