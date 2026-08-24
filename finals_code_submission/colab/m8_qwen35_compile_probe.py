"""M8 Qwen3.5 torch.compile fallback-acceleration probe (T4, server-replica stack).

Context: the fla fast path is dead on T4/SM75 (Triton PassManager::run failed for
every fla-core candidate on the server-matched stack). This probe measures whether
torch.compile on the *PyTorch fallback* path recovers enough speed to fit the
server budget. No fla / causal-conv1d anywhere — the fallback path IS the target.

Gate math (vs M7 anchor: Qwen3-0.6B len416 sorted b64 = 530s server):
  projected_with_compile   = ratio * 530 + compile_sec      (cold compile on server)
  projected_with_warm_cache = ratio * 530 + warm_startup_sec (shipped inductor cache)
  GREEN <= 510s, YELLOW <= 600s, else RED.

Stages (controller runs each in a fresh subprocess to isolate dynamo state):
  controller -> install server stack -> baseline -> variant xN -> warm rerun of best.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from colab.m8_qwen35_t4_replica_probe import (  # noqa: E402
    capture_model_load,
    head_payload,
    load_head_payload,
    model_timing,
    py_run,
    run,
    selected_samples,
    set_pad_token,
    stack_info,
    tokenize_features,
    utc_now,
)

EXPERIMENT_ID = "m8_qwen35_compile_probe"
QWEN35 = "igorktech/Qwen3.5-0.8B-Base-LM"
QWEN3 = "Qwen/Qwen3-0.6B"
NUM_LABELS = 14
SERVER_ANCHOR_SEC = 530.0
GATE_SEC = 510.0
HARD_SEC = 600.0
RESULTS_PATH = Path("experiments/results.csv")

INDUCTOR_CACHE = "/content/aadp_inductor_cache"
TRITON_CACHE = "/content/aadp_triton_cache"


def art_path(suffix=""):
    name = EXPERIMENT_ID + (f"_{suffix}" if suffix else "")
    return Path("experiments/artifacts") / f"{name}.json"


def baseline_pt_path():
    return Path("experiments/artifacts") / f"{EXPERIMENT_ID}_baseline.pt"


def configure(experiment_id):
    global EXPERIMENT_ID
    EXPERIMENT_ID = experiment_id


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def append_result_row(payload):
    try:
        from train import append_results_csv
    except Exception:
        return
    append_results_csv(
        RESULTS_PATH,
        {
            "experiment_id": EXPERIMENT_ID,
            "model_family": "t4_compile_probe",
            "base_model": QWEN35,
            "features": "current_v1 serialized text, random seq-cls heads, torch.compile fallback",
            "serializer_name": "current_v1",
            "split_type": "timing_probe",
            "seed": "42",
            "max_length": "400",
            "batch_size": "64",
            "artifact_path": str(art_path()),
            "inference_time_sec": str(payload.get("best_projected_server_sec", "")),
            "runtime_sec": f"{payload.get('runtime_sec', 0.0):.3f}",
            "train_command": "colab/m8_qwen35_compile_probe.py",
            "notes": payload.get("notes", ""),
            "decision": payload.get("verdict", ""),
        },
    )


# ---------------------------------------------------------------- inference


def load_qwen35(head=None):
    import torch
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(QWEN35)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(1234)
    model, load_warnings = capture_model_load(QWEN35, torch.float16)
    set_pad_token(model, tokenizer)
    if head is not None:
        load_head_payload(model, head)
    model.to(device).eval().half()
    return model, tokenizer, device, load_warnings


def pick_bucket(buckets, need):
    for b in buckets:
        if b >= need:
            return b
    return buckets[-1]


def make_batches(features, lengths, batch_size, buckets):
    """Length-sorted batches; each tagged with its padded bucket length."""
    order = sorted(range(len(features)), key=lambda i: lengths[i])
    batches = []
    for offset in range(0, len(order), batch_size):
        chunk = order[offset : offset + batch_size]
        need = max(lengths[i] for i in chunk)
        bucket = pick_bucket(buckets, need) if buckets else need
        batches.append({"idx": chunk, "bucket": bucket})
    return batches


def pad_batch(tokenizer, features, batch, batch_size, buckets, device):
    feats = [features[i] for i in batch["idx"]]
    fill = 0
    if buckets and len(feats) < batch_size:
        fill = batch_size - len(feats)
        feats = feats + [feats[-1]] * fill
    if buckets:
        enc = tokenizer.pad(feats, padding="max_length", max_length=batch["bucket"], return_tensors="pt")
    else:
        enc = tokenizer.pad(feats, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}, fill


def run_batches(model, tokenizer, features, batches, batch_size, buckets, device, n_rows):
    import torch

    outputs = [None] * n_rows
    batch_times = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            enc, fill = pad_batch(tokenizer, features, batch, batch_size, buckets, device)
            b0 = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**enc).logits.float().cpu()
            torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - b0)
            if fill:
                logits = logits[: len(batch["idx"])]
            for row, sample_idx in enumerate(batch["idx"]):
                outputs[sample_idx] = logits[row]
    infer_sec = time.perf_counter() - start
    return torch.stack(outputs, dim=0), infer_sec, batch_times


def warm_shapes(model, tokenizer, features, batches, batch_size, buckets, device):
    """Run one batch per distinct shape once; wall time ~= compile cost."""
    import torch

    if buckets:
        picks, seen = [], set()
        for batch in batches:
            if batch["bucket"] not in seen:
                seen.add(batch["bucket"])
                picks.append(batch)
    else:
        # dynamic: shortest + longest batch covers the shape/chunk-count range
        picks = [batches[0], batches[-1]] if len(batches) > 1 else batches[:1]
    times = {}
    with torch.no_grad():
        for i, batch in enumerate(picks):
            enc, _ = pad_batch(tokenizer, features, batch, batch_size, buckets, device)
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                model(**enc)
            torch.cuda.synchronize()
            times[str(batch["bucket"]) if buckets else f"warm{i}"] = time.perf_counter() - t0
    return times


# ------------------------------------------------------------------ stages


def baseline_stage(args):
    import torch

    started = time.perf_counter()
    step = "start"
    try:
        step = "qwen3_denominator"
        denom = model_timing(QWEN3, 416, 64, "sorted", args.timing_rows)
        print(f"denominator done, cuda_max_mb={torch.cuda.max_memory_allocated() / 1e6:.0f}", flush=True)
        torch.cuda.reset_peak_memory_stats()

        step = "qwen35_eager_timing"
        eager = model_timing(QWEN35, 400, 64, "sorted", args.timing_rows)
        print(f"eager timing done, cuda_max_mb={torch.cuda.max_memory_allocated() / 1e6:.0f}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        step = "qwen35_correctness_logits"
        model, tokenizer, device, load_warnings = load_qwen35()
        _, texts, ids = selected_samples(args.correctness_rows)
        features, lengths, _ = tokenize_features(tokenizer, texts, 400)
        batches = make_batches(features, lengths, 64, None)
        logits, _, _ = run_batches(model, tokenizer, features, batches, 64, None, device, len(texts))
        payload = {
            "logits": logits.cpu(),
            "head": head_payload(model),
            "ids": ids,
            "qwen3_denominator": denom,
            "qwen35_eager": eager,
            "stack": stack_info(),
            "load_warnings": load_warnings,
            "runtime_sec": time.perf_counter() - started,
        }
        step = "save"
        baseline_pt_path().parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, baseline_pt_path())
        write_json(
            art_path("baseline"),
            {
                "qwen3_denominator": denom,
                "qwen35_eager": eager,
                "eager_ratio": eager["tokenize_plus_infer_sec"] / denom["tokenize_plus_infer_sec"],
                "stack": stack_info(),
                "created_utc": utc_now(),
            },
        )
        print(f"saved baseline payload: {baseline_pt_path()}", flush=True)
        return 0
    except Exception:  # noqa: BLE001 - leave a diagnosable artifact, do not just die
        write_json(
            art_path("baseline"),
            {
                "verdict": "RED_baseline_error",
                "failed_step": step,
                "traceback": traceback.format_exc()[-4000:],
                "stack": stack_info(),
                "created_utc": utc_now(),
                "runtime_sec": time.perf_counter() - started,
            },
        )
        return 43


def variant_stage(args):
    import torch

    torch._dynamo.config.cache_size_limit = 64
    started = time.perf_counter()
    baseline = torch.load(baseline_pt_path(), map_location="cpu", weights_only=False)
    denom_sec = baseline["qwen3_denominator"]["tokenize_plus_infer_sec"]

    buckets = None
    if "buckets" in args.variant:
        buckets = sorted(int(b) for b in args.buckets.split(","))

    model, tokenizer, device, load_warnings = load_qwen35(head=baseline.get("head"))
    compile_kwargs = {}
    if args.variant == "cudagraph_buckets":
        compile_kwargs = {"mode": "reduce-overhead"}
    elif args.variant == "dynamic":
        compile_kwargs = {"dynamic": True}
    compiled = torch.compile(model, **compile_kwargs)

    payload = {
        "variant": args.variant,
        "buckets": buckets,
        "compile_kwargs": {k: str(v) for k, v in compile_kwargs.items()},
        "batch_size": args.batch_size,
        "stack": stack_info(),
        "load_warnings": load_warnings,
        "created_utc": utc_now(),
    }
    try:
        # -- compile/warm on the correctness set (measures cold-compile wall time)
        _, texts, _ = selected_samples(args.correctness_rows)
        features, lengths, _ = tokenize_features(tokenizer, texts, 400)
        c_batches = make_batches(features, lengths, args.batch_size, buckets)
        t0 = time.perf_counter()
        payload["warm_shape_times"] = warm_shapes(
            compiled, tokenizer, features, c_batches, args.batch_size, buckets, device
        )
        payload["compile_wall_sec"] = time.perf_counter() - t0

        # -- correctness vs eager fallback logits
        logits, _, _ = run_batches(
            compiled, tokenizer, features, c_batches, args.batch_size, buckets, device, len(texts)
        )
        base_logits = baseline["logits"].float()
        diff = (logits.float() - base_logits).abs()
        agreement = (logits.argmax(dim=1) == base_logits.argmax(dim=1)).float().mean().item()
        payload["correctness"] = {
            "rows": args.correctness_rows,
            "argmax_agreement": agreement,
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
        }
        if agreement < 0.995:
            payload["verdict"] = "RED_correctness"
            payload["runtime_sec"] = time.perf_counter() - started
            write_json(art_path(args.variant), payload)
            return 41

        # -- timing on the full probe workload (kernels now warm)
        _, texts, _ = selected_samples(args.timing_rows)
        features, lengths, tokenize_sec = tokenize_features(tokenizer, texts, 400)
        t_batches = make_batches(features, lengths, args.batch_size, buckets)
        _, infer_sec, batch_times = run_batches(
            compiled, tokenizer, features, t_batches, args.batch_size, buckets, device, len(texts)
        )
        total = tokenize_sec + infer_sec
        ratio = total / denom_sec
        payload["timing"] = {
            "tokenize_sec": tokenize_sec,
            "infer_sec": infer_sec,
            "tokenize_plus_infer_sec": total,
            "batch_count": len(batch_times),
            "batch_times_first8": batch_times[:8],
            "batch_times_slowest5": sorted(batch_times)[-5:],
        }
        payload["ratio_vs_qwen3"] = ratio
        payload["projected_infer_only_sec"] = ratio * SERVER_ANCHOR_SEC
        payload["projected_with_compile_sec"] = ratio * SERVER_ANCHOR_SEC + payload["compile_wall_sec"]
        p = payload["projected_with_compile_sec"]
        payload["verdict"] = "GREEN" if p <= GATE_SEC else ("YELLOW" if p <= HARD_SEC else "RED_timing")
    except Exception as exc:  # noqa: BLE001 - probe must report, not crash the lane
        payload["verdict"] = "RED_error"
        payload["error"] = repr(exc)[:2000]
        payload["runtime_sec"] = time.perf_counter() - started
        write_json(art_path(args.variant), payload)
        return 42
    payload["runtime_sec"] = time.perf_counter() - started
    write_json(art_path(args.variant), payload)
    return 0


def controller_stage(args):
    started = time.perf_counter()
    gpu = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=False)
    if "T4" not in gpu.get("output_tail", ""):
        write_json(
            art_path(),
            {"verdict": "WRONG_GPU", "gpu": gpu.get("output_tail", ""), "created_utc": utc_now()},
        )
        return

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = INDUCTOR_CACHE
    os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logs = []
    if not args.skip_install:
        logs.append(py_run(["-m", "pip", "install", "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128"], timeout=1800))
        logs.append(py_run(["-m", "pip", "uninstall", "-y", "torchvision", "torchaudio", "torchtext"], check=False, timeout=300))
        logs.append(py_run(["-m", "pip", "install", "transformers>=5.13,<5.14", "safetensors==0.8.0"], timeout=900))
        # the fallback path is the target: make sure the optional kernels are absent
        logs.append(py_run(["-m", "pip", "uninstall", "-y", "causal-conv1d", "causal_conv1d", "fla-core", "flash-linear-attention"], check=False, timeout=300))

    common = ["--experiment-id", EXPERIMENT_ID, "--correctness-rows", str(args.correctness_rows), "--timing-rows", str(args.timing_rows)]
    base_rc = py_run(["colab/m8_qwen35_compile_probe.py", "baseline", *common], check=False, timeout=3600)
    logs.append(base_rc)
    if base_rc["returncode"] != 0 or not baseline_pt_path().exists():
        baseline_art = {}
        if art_path("baseline").exists():
            baseline_art = json.loads(art_path("baseline").read_text(encoding="utf-8"))
        write_json(
            art_path(),
            {
                "what": "torch.compile fallback acceleration probe on server-replica stack (no fla)",
                "experiment_id": EXPERIMENT_ID,
                "created_utc": utc_now(),
                "verdict": "RED_baseline_failed",
                "baseline_artifact": baseline_art,
                "baseline_output_tail": base_rc.get("output_tail", ""),
                "install_logs_tail": [
                    {k: item.get(k) for k in ("returncode", "duration_sec")}
                    for item in logs
                    if isinstance(item, dict) and "returncode" in item
                ],
                "runtime_sec": time.perf_counter() - started,
            },
        )
        append_result_row({"verdict": "RED_baseline_failed", "runtime_sec": time.perf_counter() - started})
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = {}
    for variant in variants:
        rc = py_run(
            ["colab/m8_qwen35_compile_probe.py", "variant", "--variant", variant, "--buckets", args.buckets, "--batch-size", str(args.batch_size), *common],
            check=False,
            timeout=args.variant_timeout,
        )
        logs.append(rc)
        path = art_path(variant)
        if path.exists():
            results[variant] = json.loads(path.read_text(encoding="utf-8"))
        else:
            results[variant] = {"verdict": "RED_no_artifact", "returncode": rc["returncode"]}

    # warm rerun of the best-projected variant: simulates a shipped inductor cache
    scored = [
        (v, r["projected_with_compile_sec"])
        for v, r in results.items()
        if isinstance(r.get("projected_with_compile_sec"), (int, float))
    ]
    warm = None
    if scored:
        best_variant = min(scored, key=lambda item: item[1])[0]
        rc = py_run(
            ["colab/m8_qwen35_compile_probe.py", "variant", "--variant", best_variant, "--buckets", args.buckets, "--batch-size", str(args.batch_size), *common],
            check=False,
            timeout=args.variant_timeout,
        )
        logs.append(rc)
        path = art_path(best_variant)
        if path.exists():
            warm = json.loads(path.read_text(encoding="utf-8"))
            warm["note"] = "second run over the same TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR - compile_wall_sec here ~= server startup with a shipped cache"
            results[f"{best_variant}_warmcache"] = warm

    projections = {
        v: {
            "projected_with_compile_sec": r.get("projected_with_compile_sec"),
            "compile_wall_sec": r.get("compile_wall_sec"),
            "ratio_vs_qwen3": r.get("ratio_vs_qwen3"),
            "verdict": r.get("verdict"),
        }
        for v, r in results.items()
    }
    candidates = [
        r["projected_with_compile_sec"]
        for r in results.values()
        if isinstance(r.get("projected_with_compile_sec"), (int, float)) and not str(r.get("verdict", "")).startswith("RED")
    ]
    best = min(candidates) if candidates else None
    baseline_json = json.loads(art_path("baseline").read_text(encoding="utf-8")) if art_path("baseline").exists() else {}
    payload = {
        "what": "torch.compile fallback acceleration probe on server-replica stack (no fla)",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": utc_now(),
        "stack": stack_info(),
        "gpu": gpu.get("output_tail", "").strip(),
        "eager_baseline": baseline_json,
        "variants": projections,
        "best_projected_server_sec": best,
        "gate_sec": GATE_SEC,
        "hard_sec": HARD_SEC,
        "verdict": "GREEN" if best is not None and best <= GATE_SEC else ("YELLOW" if best is not None and best <= HARD_SEC else "RED_timing"),
        "install_logs_tail": [{k: item[k] for k in ("returncode", "duration_sec")} for item in logs if isinstance(item, dict) and "returncode" in item],
        "notes": "projected = ratio*530 + compile_wall; warmcache row = shipped-cache scenario",
        "runtime_sec": time.perf_counter() - started,
    }
    write_json(art_path(), payload)
    append_result_row(payload)
    print(json.dumps({k: payload[k] for k in ("variants", "best_projected_server_sec", "verdict")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_ctrl = sub.add_parser("controller")
    p_ctrl.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p_ctrl.add_argument("--correctness-rows", type=int, default=256)
    p_ctrl.add_argument("--timing-rows", type=int, default=4096)
    p_ctrl.add_argument("--variants", default="cudagraph_buckets,default_buckets,dynamic")
    p_ctrl.add_argument("--buckets", default="256,400")
    p_ctrl.add_argument("--batch-size", type=int, default=64)
    p_ctrl.add_argument("--variant-timeout", type=int, default=5400)
    p_ctrl.add_argument("--skip-install", action="store_true")

    p_base = sub.add_parser("baseline")
    p_base.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p_base.add_argument("--correctness-rows", type=int, default=256)
    p_base.add_argument("--timing-rows", type=int, default=4096)

    p_var = sub.add_parser("variant")
    p_var.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p_var.add_argument("--variant", required=True, choices=["cudagraph_buckets", "default_buckets", "dynamic"])
    p_var.add_argument("--buckets", default="256,400")
    p_var.add_argument("--batch-size", type=int, default=64)
    p_var.add_argument("--correctness-rows", type=int, default=256)
    p_var.add_argument("--timing-rows", type=int, default=4096)

    args = parser.parse_args()
    configure(args.experiment_id)
    if args.command == "controller":
        controller_stage(args)
    elif args.command == "baseline":
        baseline_stage(args)
    elif args.command == "variant":
        raise SystemExit(variant_stage(args))


if __name__ == "__main__":
    main()
