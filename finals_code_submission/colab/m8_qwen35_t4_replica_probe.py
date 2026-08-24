"""M8 Qwen3.5 T4 replica timing probe.

This is a Colab-only diagnostic runner for m8_fla_t4_reprobe_spec.md.
It installs the server-like torch stack in the runtime, captures fallback
Qwen3.5 logits without optional kernels, installs causal-conv1d + fla-core,
then checks fast-path activation, correctness, and T4 timing ratios.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

EXPERIMENT_ID = "m8_qwen35_t4_replica_probe"
QWEN35 = "igorktech/Qwen3.5-0.8B-Base-LM"
QWEN3 = "Qwen/Qwen3-0.6B"
NUM_LABELS = 14
DATA_PATH = Path("open/data/train.jsonl")
ARTIFACT_PATH = Path("experiments/artifacts/m8_qwen35_t4_replica_probe.json")
FALLBACK_PATH = Path("experiments/artifacts/m8_qwen35_t4_replica_fallback.pt")
RESULTS_PATH = Path("experiments/results.csv")


def configure_experiment(experiment_id):
    global EXPERIMENT_ID, ARTIFACT_PATH, FALLBACK_PATH
    EXPERIMENT_ID = experiment_id
    ARTIFACT_PATH = Path("experiments/artifacts") / f"{experiment_id}.json"
    FALLBACK_PATH = Path("experiments/artifacts") / f"{experiment_id}_fallback.pt"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run(cmd, check=True, timeout=None):
    start = time.perf_counter()
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    duration = time.perf_counter() - start
    tail = proc.stdout[-12000:]
    if tail:
        print(tail, flush=True)
    item = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_sec": round(duration, 3),
        "output_tail": tail[-4000:],
    }
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")
    return item


def py_run(args, check=True, timeout=None):
    return run([sys.executable, *args], check=check, timeout=timeout)


def module_version(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {"available": True, "version": getattr(module, "__version__", "unknown")}


def package_spec(name):
    return importlib.util.find_spec(name) is not None


def stack_info(extra=None):
    info = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception as exc:
        info["torch_error"] = repr(exc)
    for name in ("triton", "transformers", "fla", "causal_conv1d"):
        info[name] = module_version(name)
    info["find_spec"] = {
        "fla": package_spec("fla"),
        "causal_conv1d": package_spec("causal_conv1d"),
        "flash_linear_attention": package_spec("flash_linear_attention"),
    }
    if extra:
        info.update(extra)
    return info


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def selected_samples(count):
    from script import serialize_transformer_sample

    samples = read_jsonl(DATA_PATH)
    order = list(range(len(samples)))
    random.Random(42).shuffle(order)
    chosen = [samples[i] for i in order[:count]]
    texts = [serialize_transformer_sample(sample, "current_v1") for sample in chosen]
    ids = [sample.get("id") for sample in chosen]
    return chosen, texts, ids


def capture_model_load(model_id, dtype):
    import torch
    from transformers import AutoModelForSequenceClassification

    warn_buf = []
    stderr = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stderr(stderr):
        warnings.simplefilter("always")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=NUM_LABELS,
            id2label={i: str(i) for i in range(NUM_LABELS)},
            label2id={str(i): i for i in range(NUM_LABELS)},
            torch_dtype=dtype,
        )
        warn_buf.extend(str(item.message) for item in caught)
    text = stderr.getvalue()
    if text:
        warn_buf.append(text)
    return model, warn_buf


def set_pad_token(model, tokenizer):
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def head_payload(model):
    for name in ("score", "classifier", "classification_head"):
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "state_dict"):
            state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
            if state:
                return {"kind": "module", "name": name, "state": state}
    picked = {}
    for key, value in model.state_dict().items():
        if key.startswith(("score.", "classifier.", "classification_head.")):
            picked[key] = value.detach().cpu()
    return {"kind": "state_keys", "name": "", "state": picked}


def load_head_payload(model, payload):
    if not payload:
        return
    state = payload.get("state") or {}
    if payload.get("kind") == "module":
        module = getattr(model, payload.get("name", ""), None)
        if module is not None:
            module.load_state_dict(state, strict=True)
        return
    if state:
        current = model.state_dict()
        current.update(state)
        model.load_state_dict(current, strict=False)


def tokenize_features(tokenizer, texts, max_length):
    start = time.perf_counter()
    encoded = tokenizer(texts, padding=False, truncation=True, max_length=max_length)
    tokenize_sec = time.perf_counter() - start
    keys = list(encoded.keys())
    features = [{key: encoded[key][i] for key in keys} for i in range(len(texts))]
    lengths = [len(feature["input_ids"]) for feature in features]
    return features, lengths, tokenize_sec


def infer_logits(model, tokenizer, features, lengths, batch_size, mode, max_length, device):
    import torch

    model.eval()
    if mode == "sorted":
        order = sorted(range(len(features)), key=lambda i: lengths[i])
    else:
        order = list(range(len(features)))
    outputs = [None] * len(features)
    batch_times = []
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            chunk = order[offset : offset + batch_size]
            pad_kwargs = {"padding": True}
            if mode == "fixed":
                pad_kwargs = {"padding": "max_length", "max_length": max_length}
            batch = tokenizer.pad(
                [features[i] for i in chunk],
                **pad_kwargs,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            b_start = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.float16):
                logits = model(**batch).logits.float().cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - b_start)
            for row, sample_idx in enumerate(chunk):
                outputs[sample_idx] = logits[row]
    infer_sec = time.perf_counter() - start
    return torch.stack(outputs, dim=0), infer_sec, batch_times


def model_timing(model_id, max_length, batch_size, mode, sample_count):
    import torch
    from transformers import AutoTokenizer

    _, texts, ids = selected_samples(sample_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(1234)
    model, load_warnings = capture_model_load(model_id, torch.float16)
    set_pad_token(model, tokenizer)
    model.to(device).eval()
    if device.type == "cuda":
        model.half()
    features, lengths, tokenize_sec = tokenize_features(tokenizer, texts, max_length)
    _, infer_sec, batch_times = infer_logits(
        model,
        tokenizer,
        features,
        lengths,
        batch_size,
        mode,
        max_length,
        device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    steady = sum(batch_times[4:]) if len(batch_times) > 4 else sum(batch_times)
    return {
        "model_id": model_id,
        "max_length": max_length,
        "batch_size": batch_size,
        "mode": mode,
        "sample_count": sample_count,
        "tokenize_sec": tokenize_sec,
        "infer_sec": infer_sec,
        "tokenize_plus_infer_sec": tokenize_sec + infer_sec,
        "steady_infer_sec_excluding_first4": steady,
        "batch_count": len(batch_times),
        "batch_times_first8": batch_times[:8],
        "lengths": {
            "mean": sum(lengths) / max(1, len(lengths)),
            "max": max(lengths) if lengths else 0,
        },
        "sample_ids_first5": ids[:5],
        "load_warnings": load_warnings,
    }


def qwen35_flags():
    flags = {}
    for modname in (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3_5_text.modeling_qwen3_5_text",
    ):
        try:
            module = importlib.import_module(modname)
        except Exception as exc:
            flags[modname] = {"import_error": repr(exc)}
            continue
        values = {}
        for name in dir(module):
            lower = name.lower()
            if not any(token in lower for token in ("fla", "causal", "conv", "fast", "delta")):
                continue
            try:
                value = getattr(module, name)
            except Exception as exc:
                values[name] = f"error:{exc!r}"
                continue
            if isinstance(value, (bool, int, float, str, type(None))):
                values[name] = value
            else:
                values[name] = repr(value)[:160]
        flags[modname] = values
    return flags


def fastpath_active_from_warnings(warnings_texts):
    text = "\n".join(warnings_texts).lower()
    markers = (
        "fast path is not available",
        "falling back to",
        "fallback to",
    )
    return not any(marker in text for marker in markers)


def append_result_row(payload):
    try:
        from train import append_results_csv
    except Exception:
        append_results_csv = None
    row = {
        "experiment_id": EXPERIMENT_ID,
        "model_family": "t4_replica_timing_probe",
        "base_model": QWEN35,
        "features": "current_v1 serialized text, random seq-cls heads",
        "serializer_name": "current_v1",
        "split_type": "timing_probe",
        "seed": "42",
        "max_length": "400",
        "batch_size": "64",
        "artifact_path": str(ARTIFACT_PATH),
        "inference_time_sec": str(payload.get("best_projected_server_sec", "")),
        "runtime_sec": f"{payload.get('runtime_sec', 0.0):.3f}",
        "train_command": "colab/m8_qwen35_t4_replica_probe.py",
        "notes": payload.get("notes", ""),
        "decision": payload.get("verdict", ""),
    }
    if append_results_csv is not None:
        append_results_csv(RESULTS_PATH, row)
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    exists = RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def finalize(payload):
    payload.setdefault("experiment_id", EXPERIMENT_ID)
    payload.setdefault("created_utc", utc_now())
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_result_row(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def fallback_stage(args):
    import torch
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, texts, ids = selected_samples(args.correctness_rows)
    tokenizer = AutoTokenizer.from_pretrained(QWEN35)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(1234)
    model, load_warnings = capture_model_load(QWEN35, torch.float16)
    set_pad_token(model, tokenizer)
    model.to(device).eval()
    if device.type == "cuda":
        model.half()
    features, lengths, tokenize_sec = tokenize_features(tokenizer, texts, 400)
    logits, infer_sec, batch_times = infer_logits(
        model,
        tokenizer,
        features,
        lengths,
        64,
        "sorted",
        400,
        device,
    )
    payload = {
        "logits": logits.cpu(),
        "head": head_payload(model),
        "ids": ids,
        "stack": stack_info(),
        "load_warnings": load_warnings,
        "timing": {
            "tokenize_sec": tokenize_sec,
            "infer_sec": infer_sec,
            "batch_times_first8": batch_times[:8],
        },
    }
    FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, FALLBACK_PATH)
    print(f"saved fallback payload: {FALLBACK_PATH}", flush=True)


def fast_stage(args):
    import torch
    from transformers import AutoTokenizer

    started = time.perf_counter()
    fallback = torch.load(FALLBACK_PATH, map_location="cpu", weights_only=False)
    _, texts, ids = selected_samples(args.correctness_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(QWEN35)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(1234)
    model, load_warnings = capture_model_load(QWEN35, torch.float16)
    set_pad_token(model, tokenizer)
    load_head_payload(model, fallback.get("head"))
    model.to(device).eval()
    if device.type == "cuda":
        model.half()
    flags = qwen35_flags()
    fastpath_active = fastpath_active_from_warnings(load_warnings)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": utc_now(),
        "stack": stack_info({"fla_core_candidate": args.fla_candidate}),
        "fallback_stack": fallback.get("stack"),
        "fastpath_active": fastpath_active,
        "load_warnings": load_warnings,
        "qwen35_flags": flags,
        "sample_ids_first5": ids[:5],
        "notes": "M8 Qwen3.5 T4 replica probe per m8_fla_t4_reprobe_spec.md",
    }
    if not fastpath_active:
        payload["verdict"] = "RED_fastpath"
        payload["runtime_sec"] = time.perf_counter() - started
        finalize(payload)
        return 0

    features, lengths, _ = tokenize_features(tokenizer, texts, 400)
    fast_logits, _, _ = infer_logits(
        model,
        tokenizer,
        features,
        lengths,
        64,
        "sorted",
        400,
        device,
    )
    fallback_logits = fallback["logits"].float()
    diff = (fast_logits.float() - fallback_logits).abs()
    agreement = (
        fast_logits.argmax(dim=1).cpu() == fallback_logits.argmax(dim=1).cpu()
    ).float().mean().item()
    payload["correctness"] = {
        "rows": args.correctness_rows,
        "argmax_agreement": agreement,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "p95_abs_diff": float(torch.quantile(diff.flatten(), 0.95).item()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if agreement < 0.995:
        payload["verdict"] = "RED_correctness"
        payload["runtime_sec"] = time.perf_counter() - started
        finalize(payload)
        return 41

    timings = {}
    for model_id, max_length, prefix in (
        (QWEN3, 416, "qwen3"),
        (QWEN35, 400, "qwen35"),
    ):
        for mode in ("sorted", "fixed"):
            key = f"{prefix}_{mode}_b64_len{max_length}"
            print(f"timing {key}", flush=True)
            timings[key] = model_timing(model_id, max_length, 64, mode, args.timing_rows)
    payload["timings"] = timings
    ratios = {}
    for mode in ("sorted", "fixed"):
        q3 = timings[f"qwen3_{mode}_b64_len416"]["tokenize_plus_infer_sec"]
        q35 = timings[f"qwen35_{mode}_b64_len400"]["tokenize_plus_infer_sec"]
        ratios[mode] = {
            "qwen35_over_qwen3_tokenize_plus_infer": q35 / q3 if q3 else None,
            "projected_server_sec": (q35 / q3) * 530.0 if q3 else None,
        }
    payload["ratios"] = ratios
    projected_values = [
        value["projected_server_sec"]
        for value in ratios.values()
        if value.get("projected_server_sec") is not None
    ]
    best_projected = min(projected_values) if projected_values else None
    payload["best_projected_server_sec"] = best_projected

    if best_projected is not None and 510.0 < best_projected <= 600.0:
        grid = []
        for batch_size, max_length, mode in (
            (96, 400, "fixed"),
            (128, 400, "fixed"),
            (64, 336, "sorted"),
            (96, 336, "fixed"),
            (128, 336, "fixed"),
        ):
            item = model_timing(QWEN35, max_length, batch_size, mode, args.timing_rows)
            denom = timings["qwen3_sorted_b64_len416"]["tokenize_plus_infer_sec"]
            item["projected_vs_qwen3_sorted_b64_sec"] = (
                item["tokenize_plus_infer_sec"] / denom * 530.0 if denom else None
            )
            grid.append(item)
        payload["yellow_grid"] = grid

    if best_projected is not None and best_projected <= 510.0:
        payload["verdict"] = "GREEN"
    elif best_projected is not None and best_projected <= 600.0:
        payload["verdict"] = "YELLOW"
    else:
        payload["verdict"] = "RED_timing_failed"
    payload["runtime_sec"] = time.perf_counter() - started
    finalize(payload)
    return 0


def install_stack(args):
    logs = []
    logs.append(py_run(["-m", "pip", "install", "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128"], timeout=1200))
    # Colab images keep torchvision/torchaudio pinned to their original torch.
    # After downgrading torch to the server version, stale torchvision can break
    # transformers import via operator registration (torchvision::nms).
    logs.append(py_run(["-m", "pip", "uninstall", "-y", "torchvision", "torchaudio", "torchtext"], check=False, timeout=300))
    logs.append(py_run(["-m", "pip", "install", "transformers>=5.13,<5.14", "safetensors==0.8.0"], timeout=900))
    logs.append(py_run(["-m", "pip", "uninstall", "-y", "causal-conv1d", "causal_conv1d", "fla-core", "flash-linear-attention", "flash_linear_attention"], check=False, timeout=300))
    logs.append(py_run(
        [
            "colab/m8_qwen35_t4_replica_probe.py",
            "fallback",
            "--experiment-id",
            EXPERIMENT_ID,
            "--correctness-rows",
            str(args.correctness_rows),
        ],
        timeout=1800,
    ))
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel = (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/"
        f"causal_conv1d-1.6.2.post1+cu12torch2.7cxx11abiTRUE-{py_tag}-{py_tag}-linux_x86_64.whl"
    )
    logs.append(py_run(["-m", "pip", "install", wheel], timeout=600))

    fla_candidates = args.fla_candidates.split(",")
    last_error = None
    for candidate in fla_candidates:
        candidate = candidate.strip()
        if not candidate:
            spec = "fla-core"
            label = "latest"
        else:
            spec = f"fla-core=={candidate}"
            label = candidate
        try:
            # Do not let fla-core dependency resolution move torch/triton away
            # from the server-replica stack. A forced reinstall without
            # --no-deps can upgrade torch and invalidate the causal-conv1d wheel.
            logs.append(py_run(["-m", "pip", "install", "--force-reinstall", "--no-deps", spec], timeout=900))
            logs.append(py_run(["-c", "import fla, causal_conv1d; print('optional imports ok')"], timeout=120))
            rc = py_run(
                [
                    "colab/m8_qwen35_t4_replica_probe.py",
                    "fast",
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--correctness-rows",
                    str(args.correctness_rows),
                    "--timing-rows",
                    str(args.timing_rows),
                    "--fla-candidate",
                    label,
                ],
                check=False,
                timeout=args.fast_timeout,
            )
            logs.append(rc)
            if rc["returncode"] == 41 and not args.no_correctness_retry:
                print("correctness failed; trying next fla-core candidate", flush=True)
                continue
            if rc["returncode"] != 0:
                raise RuntimeError(f"fast stage failed rc={rc['returncode']}")
            return
        except Exception as exc:
            last_error = repr(exc)
            print(f"fla candidate failed: {label}: {last_error}", flush=True)
            continue

    finalize(
        {
            "experiment_id": EXPERIMENT_ID,
            "created_utc": utc_now(),
            "stack": stack_info(),
            "install_logs": logs[-8:],
            "verdict": "RED_setup_failed",
            "notes": f"no fla-core candidate completed: {last_error}",
            "runtime_sec": 0.0,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_controller = sub.add_parser("controller")
    p_controller.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p_controller.add_argument("--correctness-rows", type=int, default=256)
    p_controller.add_argument("--timing-rows", type=int, default=4096)
    p_controller.add_argument("--fast-timeout", type=int, default=7200)
    p_controller.add_argument("--fla-candidates", default=",0.5.1,0.5.0,0.4.0")
    p_controller.add_argument("--no-correctness-retry", action="store_true")
    p_fallback = sub.add_parser("fallback")
    p_fallback.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p_fallback.add_argument("--correctness-rows", type=int, default=256)
    p_fast = sub.add_parser("fast")
    p_fast.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p_fast.add_argument("--correctness-rows", type=int, default=256)
    p_fast.add_argument("--timing-rows", type=int, default=4096)
    p_fast.add_argument("--fla-candidate", default="")
    args = parser.parse_args()
    configure_experiment(args.experiment_id)

    if args.command == "controller":
        try:
            install_stack(args)
        except Exception as exc:
            finalize(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "created_utc": utc_now(),
                    "stack": stack_info(),
                    "verdict": "RED_setup_failed",
                    "notes": repr(exc),
                    "runtime_sec": 0.0,
                }
            )
    elif args.command == "fallback":
        fallback_stage(args)
    elif args.command == "fast":
        raise SystemExit(fast_stage(args))


if __name__ == "__main__":
    main()
