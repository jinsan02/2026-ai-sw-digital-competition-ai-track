"""M8 max-optimized pack probe: py3.11 venv, compiled bucketed inference, shippable cache.

Purpose: the user wants one Public probe of Qwen3.5-0.8B despite the RED timing wall.
This builds and measures the maximum optimization stack on a Colab T4 with the exact
server runtime (python 3.11 + torch 2.7.1+cu128 + transformers 5.13, no fla):

  torch.compile(mode="reduce-overhead") + bucket padding + batch 128
  + shorter max_length (config A: len336 buckets {192,256,336}; B: len400 {256,400})
  + inductor/triton caches built under py3.11 for shipping inside the zip.

Gate math vs M7 anchor 530s: projected = ratio*530 + warm_startup. The real go/no-go
is the end-to-end rehearsal of the actual pack afterwards, not this projection.

Stages:
  controller (system py): deadsnakes py3.11 venv -> torch stack -> venv stages
  measure (venv py3.11): denominator + per-config eager correctness base +
      compiled warm/correctness/timing + cache artifacts saved
  warm (venv py3.11, fresh proc): same cache dirs -> warm startup + timing
      (= server startup rehearsal for the shipped-cache scenario)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from colab.m8_qwen35_t4_replica_probe import (  # noqa: E402
    model_timing,
    run,
    selected_samples,
    stack_info,
    tokenize_features,
    utc_now,
)
from colab.m8_qwen35_compile_probe import (  # noqa: E402
    load_qwen35,
    make_batches,
    run_batches,
    warm_shapes,
)

EXPERIMENT_ID = "m8_qwen35_maxpack"
QWEN3 = "Qwen/Qwen3-0.6B"
SERVER_ANCHOR_SEC = 530.0
GATE_SEC = 510.0
HARD_SEC = 600.0
VENV_PY = "/content/venv311/bin/python"
INDUCTOR_CACHE = "/content/venv311_inductor_cache"
TRITON_CACHE = "/content/venv311_triton_cache"

CONFIGS = {
    "A": {"max_length": 336, "buckets": [192, 256, 336], "batch_size": 128},
    "B": {"max_length": 400, "buckets": [256, 400], "batch_size": 128},
}


def art_path(suffix=""):
    name = EXPERIMENT_ID + (f"_{suffix}" if suffix else "")
    return Path("experiments/artifacts") / f"{name}.json"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def vrun(args, check=True, timeout=None):
    return run([VENV_PY, *args], check=check, timeout=timeout)


def mirror_to_drive(*paths):
    """Copy artifacts straight to the Drive exchange as they are produced —
    a reclaimed VM loses /content, and collect only fires on clean run exit."""
    import shutil

    base = Path("/content/drive/MyDrive") / os.environ.get("AADP_EXCHANGE_DIR", "AADP_exchange")
    out = base / "runs" / "maxpack_mirror"
    try:
        out.mkdir(parents=True, exist_ok=True)
        for p in paths:
            p = Path(p)
            if p.exists():
                shutil.copy2(p, out / p.name)
                print(f"mirrored {p.name} -> Drive", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"drive mirror skipped: {exc!r}", flush=True)


def save_caches(tag):
    info = {}
    try:
        import torch

        result = torch.compiler.save_cache_artifacts()
        if result is not None:
            blob = result[0]
            out = Path("experiments/artifacts") / f"{tag}_megacache.bin"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(blob)
            info["megacache_bytes"] = len(blob)
    except Exception as exc:  # noqa: BLE001
        info["megacache_error"] = repr(exc)
    try:
        out = Path("experiments/artifacts") / f"{tag}_cachedirs.tar.gz"
        with tarfile.open(out, "w:gz") as tf:
            for d in (INDUCTOR_CACHE, TRITON_CACHE):
                if Path(d).exists():
                    tf.add(d, arcname=Path(d).name)
        info["cachedirs_tar_bytes"] = out.stat().st_size
    except Exception as exc:  # noqa: BLE001
        info["cachedirs_error"] = repr(exc)
    return info


def eager_logits_at(max_length, rows):
    model, tokenizer, device, _ = load_qwen35()
    _, texts, _ = selected_samples(rows)
    features, lengths, _ = tokenize_features(tokenizer, texts, max_length)
    batches = make_batches(features, lengths, 64, None)
    logits, _, _ = run_batches(model, tokenizer, features, batches, 64, None, device, len(texts))
    import gc

    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def measure_stage(args):
    import gc

    import torch

    torch._dynamo.config.cache_size_limit = 64
    started = time.perf_counter()
    payload = {"stage": "measure", "stack": stack_info(), "created_utc": utc_now(), "configs": {}}
    step = "denominator"
    try:
        denom = model_timing(QWEN3, 416, 64, "sorted", args.timing_rows)
        payload["qwen3_denominator"] = denom
        denom_sec = denom["tokenize_plus_infer_sec"]
        gc.collect()
        torch.cuda.empty_cache()

        eager_base = {}
        for key in args.configs.split(","):
            cfg = CONFIGS[key]
            step = f"eager_logits_{key}"
            eager_base[key] = eager_logits_at(cfg["max_length"], args.correctness_rows)

        step = "load_compiled"
        model, tokenizer, device, load_warnings = load_qwen35()
        compiled = torch.compile(model, mode="reduce-overhead")
        payload["load_warnings"] = load_warnings

        for key in args.configs.split(","):
            cfg = CONFIGS[key]
            step = f"config_{key}"
            entry = dict(cfg)
            # warm on the TIMING batch set so every bucket shape compiles before
            # the timing loop (the correctness subset may not touch all buckets)
            _, t_texts, _ = selected_samples(args.timing_rows)
            t_features, t_lengths, tokenize_sec = tokenize_features(tokenizer, t_texts, cfg["max_length"])
            t_batches = make_batches(t_features, t_lengths, cfg["batch_size"], cfg["buckets"])
            t0 = time.perf_counter()
            entry["warm_shape_times"] = warm_shapes(
                compiled, tokenizer, t_features, t_batches, cfg["batch_size"], cfg["buckets"], device
            )
            entry["compile_wall_sec"] = time.perf_counter() - t0
            # cache is the most valuable artifact: persist+mirror right after
            # compile, before timing, so a reclaimed VM cannot take it away
            payload["cache_artifacts"] = save_caches(EXPERIMENT_ID)
            payload["configs"][key] = entry
            write_json(art_path("measure"), payload)
            mirror_to_drive(
                art_path("measure"),
                Path("experiments/artifacts") / f"{EXPERIMENT_ID}_megacache.bin",
                Path("experiments/artifacts") / f"{EXPERIMENT_ID}_cachedirs.tar.gz",
            )

            _, c_texts, _ = selected_samples(args.correctness_rows)
            c_features, c_lengths, _ = tokenize_features(tokenizer, c_texts, cfg["max_length"])
            c_batches = make_batches(c_features, c_lengths, cfg["batch_size"], cfg["buckets"])
            logits, _, _ = run_batches(
                compiled, tokenizer, c_features, c_batches, cfg["batch_size"], cfg["buckets"], device, len(c_texts)
            )
            base = eager_base[key].float()
            diff = (logits.float() - base).abs()
            entry["correctness"] = {
                "argmax_agreement": (logits.argmax(1) == base.argmax(1)).float().mean().item(),
                "max_abs_diff": float(diff.max().item()),
            }

            _, infer_sec, batch_times = run_batches(
                compiled, tokenizer, t_features, t_batches, cfg["batch_size"], cfg["buckets"], device, len(t_texts)
            )
            total = tokenize_sec + infer_sec
            entry["timing"] = {
                "tokenize_sec": tokenize_sec,
                "infer_sec": infer_sec,
                "tokenize_plus_infer_sec": total,
                "batch_count": len(batch_times),
                "cuda_max_mb": torch.cuda.max_memory_allocated() / 1e6,
            }
            torch.cuda.reset_peak_memory_stats()
            entry["ratio_vs_qwen3"] = total / denom_sec
            entry["projected_infer_only_sec"] = total / denom_sec * SERVER_ANCHOR_SEC
            payload["configs"][key] = entry
            print(f"config {key}: ratio={entry['ratio_vs_qwen3']:.3f} "
                  f"projected_infer={entry['projected_infer_only_sec']:.0f}s "
                  f"compile={entry['compile_wall_sec']:.0f}s", flush=True)
            write_json(art_path("measure"), payload)
            mirror_to_drive(art_path("measure"))

        step = "save_caches"
        payload["cache_artifacts"] = save_caches(EXPERIMENT_ID)
        payload["runtime_sec"] = time.perf_counter() - started
        write_json(art_path("measure"), payload)
        mirror_to_drive(
            art_path("measure"),
            Path("experiments/artifacts") / f"{EXPERIMENT_ID}_megacache.bin",
            Path("experiments/artifacts") / f"{EXPERIMENT_ID}_cachedirs.tar.gz",
        )
        return 0
    except Exception:  # noqa: BLE001
        payload["verdict"] = "RED_measure_error"
        payload["failed_step"] = step
        payload["traceback"] = traceback.format_exc()[-4000:]
        payload["runtime_sec"] = time.perf_counter() - started
        write_json(art_path("measure"), payload)
        return 43


def warm_stage(args):
    import torch

    torch._dynamo.config.cache_size_limit = 64
    started = time.perf_counter()
    cfg = CONFIGS[args.config]
    payload = {"stage": "warm", "config": args.config, "stack": stack_info(), "created_utc": utc_now()}
    step = "load"
    try:
        mega = Path("experiments/artifacts") / f"{EXPERIMENT_ID}_megacache.bin"
        if mega.exists():
            try:
                torch.compiler.load_cache_artifacts(mega.read_bytes())
                payload["megacache_loaded"] = True
            except Exception as exc:  # noqa: BLE001
                payload["megacache_loaded"] = repr(exc)

        model, tokenizer, device, _ = load_qwen35()
        compiled = torch.compile(model, mode="reduce-overhead")
        payload["model_load_plus_compileobj_sec"] = time.perf_counter() - started

        step = "warm_shapes"
        _, texts, _ = selected_samples(args.timing_rows)
        features, lengths, tokenize_sec = tokenize_features(tokenizer, texts, cfg["max_length"])
        t_batches = make_batches(features, lengths, cfg["batch_size"], cfg["buckets"])
        t0 = time.perf_counter()
        payload["warm_shape_times"] = warm_shapes(
            compiled, tokenizer, features, t_batches, cfg["batch_size"], cfg["buckets"], device
        )
        payload["warm_compile_wall_sec"] = time.perf_counter() - t0
        payload["startup_total_sec"] = time.perf_counter() - started

        step = "timing"
        _, infer_sec, _ = run_batches(
            compiled, tokenizer, features, t_batches, cfg["batch_size"], cfg["buckets"], device, len(texts)
        )
        payload["timing"] = {"tokenize_sec": tokenize_sec, "infer_sec": infer_sec,
                             "tokenize_plus_infer_sec": tokenize_sec + infer_sec}
        payload["runtime_sec"] = time.perf_counter() - started
        write_json(art_path("warm"), payload)
        mirror_to_drive(art_path("warm"))
        return 0
    except Exception:  # noqa: BLE001
        payload["verdict"] = "RED_warm_error"
        payload["failed_step"] = step
        payload["traceback"] = traceback.format_exc()[-4000:]
        payload["runtime_sec"] = time.perf_counter() - started
        write_json(art_path("warm"), payload)
        return 43


def controller_stage(args):
    started = time.perf_counter()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = INDUCTOR_CACHE
    os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logs = []
    py311_ok = True
    if not Path(VENV_PY).exists():
        logs.append(run(["add-apt-repository", "-y", "ppa:deadsnakes/ppa"], check=False, timeout=300))
        # python3.11-dev is required: triton builds its cuda_utils module against
        # the python headers (the server image ships python3.11-dev per rule.md)
        logs.append(run(["apt-get", "install", "-y", "-q", "python3.11", "python3.11-venv", "python3.11-dev"], check=False, timeout=600))
        mk = run(["python3.11", "-m", "venv", "/content/venv311"], check=False, timeout=300)
        logs.append(mk)
        py311_ok = mk["returncode"] == 0 and Path(VENV_PY).exists()
        if not py311_ok:
            # fallback: system-python venv (cache py-version risk recorded in artifact)
            logs.append(run([sys.executable, "-m", "venv", "/content/venv311"], timeout=300))
        logs.append(vrun(["-m", "pip", "install", "-q", "--upgrade", "pip"], timeout=600))
        logs.append(vrun(["-m", "pip", "install", "-q", "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128"], timeout=1800))
        logs.append(vrun(["-m", "pip", "install", "-q", "transformers>=5.13,<5.14", "safetensors==0.8.0"], timeout=900))

    common = ["--experiment-id", EXPERIMENT_ID, "--correctness-rows", str(args.correctness_rows), "--timing-rows", str(args.timing_rows)]
    measure_rc = vrun(
        ["colab/m8_qwen35_maxpack.py", "measure", "--configs", args.configs, *common],
        check=False, timeout=args.stage_timeout,
    )
    logs.append(measure_rc)

    measure = json.loads(art_path("measure").read_text(encoding="utf-8")) if art_path("measure").exists() else {}
    best_key, best_proj = None, None
    for key, entry in (measure.get("configs") or {}).items():
        proj = entry.get("projected_infer_only_sec")
        agree = (entry.get("correctness") or {}).get("argmax_agreement", 0)
        if isinstance(proj, (int, float)) and agree >= 0.995 and (best_proj is None or proj < best_proj):
            best_key, best_proj = key, proj

    warm = {}
    if best_key:
        warm_rc = vrun(
            ["colab/m8_qwen35_maxpack.py", "warm", "--config", best_key, *common],
            check=False, timeout=args.stage_timeout,
        )
        logs.append(warm_rc)
        warm = json.loads(art_path("warm").read_text(encoding="utf-8")) if art_path("warm").exists() else {}

    projected_total = None
    if best_proj is not None and isinstance(warm.get("startup_total_sec"), (int, float)):
        projected_total = best_proj + warm["warm_compile_wall_sec"]
    payload = {
        "what": "max-optimization pack probe (py3.11 venv, compiled buckets, shippable cache)",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": utc_now(),
        "py311_venv": py311_ok,
        "best_config": best_key,
        "best_projected_infer_only_sec": best_proj,
        "warm_compile_wall_sec": warm.get("warm_compile_wall_sec"),
        "warm_startup_total_sec": warm.get("startup_total_sec"),
        "projected_total_sec": projected_total,
        "verdict": (
            "GREEN" if projected_total is not None and projected_total <= GATE_SEC
            else "YELLOW" if projected_total is not None and projected_total <= HARD_SEC
            else "RED_timing" if projected_total is not None
            else "RED_no_valid_config"
        ),
        "stage_rcs": [{k: item.get(k) for k in ("returncode", "duration_sec")} for item in logs if isinstance(item, dict) and "returncode" in item],
        "runtime_sec": time.perf_counter() - started,
    }
    write_json(art_path(), payload)
    print(json.dumps(payload, indent=1, default=str)[:2000], flush=True)


def main():
    global EXPERIMENT_ID
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("controller", "measure", "warm"):
        p = sub.add_parser(name)
        p.add_argument("--experiment-id", default=EXPERIMENT_ID)
        p.add_argument("--correctness-rows", type=int, default=256)
        p.add_argument("--timing-rows", type=int, default=4096)
        if name == "controller":
            p.add_argument("--configs", default="A,B")
            p.add_argument("--stage-timeout", type=int, default=7200)
        if name == "measure":
            p.add_argument("--configs", default="A,B")
        if name == "warm":
            p.add_argument("--config", required=True, choices=list(CONFIGS))
    args = parser.parse_args()
    EXPERIMENT_ID = args.experiment_id

    if args.command == "controller":
        controller_stage(args)
    elif args.command == "measure":
        raise SystemExit(measure_stage(args))
    elif args.command == "warm":
        raise SystemExit(warm_stage(args))


if __name__ == "__main__":
    main()
