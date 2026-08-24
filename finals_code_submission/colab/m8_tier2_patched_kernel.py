"""M8 tier-2: monkeypatched DeltaNet fallback (solve_triangular) timing probe.

Tier-1 findings this probe acts on:
- The fallback's inner loop `for i in range(1, chunk_size)` (63 sequential
  launches per chunk-solve, per DeltaNet layer) is mathematically a batched
  unit-lower-triangular solve: T = solve_triangular(I - L, I, unitriangular).
  Verified equivalent locally (max diff 1.2e-7).
- Eager profile: copies 28.9%, elementwise ~40%, "Command Buffer Full" 79% CPU
  -> the path is launch/overhead-bound; killing the sequential loop attacks
  exactly that. It also shrinks the torch.compile graph (the loop unrolled to
  63 iterations x 18 layers, which is why cold compile took ~40 min/shape).

Stages: patch -> correctness vs stock -> eager timing + chunk grid ->
compiled (cudagraph, buckets {256,400}, b64) -> warm rerun (cache-ship).
Anchor correction from tier-1: M7 server 530s = load ~52s + infer ~478s;
projections here use the 478s infer anchor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from colab.m8_qwen35_t4_replica_probe import (  # noqa: E402
    model_timing,
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
from colab.m8_qwen35_maxpack import mirror_to_drive, write_json  # noqa: E402

QWEN3 = "Qwen/Qwen3-0.6B"
ART = Path("experiments/artifacts/m8_tier2_patched_kernel.json")
INFER_ANCHOR_SEC = 478.0  # tier-1: 530 total - 52 load
VENV_PY = "/content/venv311/bin/python"
INDUCTOR_CACHE = "/content/tier2_inductor_cache"
TRITON_CACHE = "/content/tier2_triton_cache"


def make_patched_rule(chunk_size_override=None):
    import torch
    import torch.nn.functional as F

    def patched(query, key, value, g, beta, chunk_size=64, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

        if chunk_size_override:
            chunk_size = chunk_size_override
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]
        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        # --- tier-2 patch: one batched triangular solve replaces the
        # chunk_size-1 step sequential substitution loop (verified equivalent)
        eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        attn = torch.linalg.solve_triangular(
            eye - attn, eye.expand(*attn.shape[:-2], chunk_size, chunk_size).contiguous(),
            upper=False, unitriangular=True,
        )
        # --- end patch (stock code's final `attn + eye` is folded into the solve)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    return patched


PATCH_MODULES = (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_5_text.modeling_qwen3_5_text",
    "transformers.models.qwen3_next.modeling_qwen3_next",
)


def apply_patch(fn):
    import importlib

    patched_into = []
    for modname in PATCH_MODULES:
        try:
            module = importlib.import_module(modname)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(module, "torch_chunk_gated_delta_rule"):
            module.torch_chunk_gated_delta_rule = fn
            patched_into.append(modname)
    return patched_into


def timed_sorted_run(model, tokenizer, device, rows, max_length, batch_size):
    _, texts, _ = selected_samples(rows)
    features, lengths, tokenize_sec = tokenize_features(tokenizer, texts, max_length)
    batches = make_batches(features, lengths, batch_size, None)
    logits, infer_sec, _ = run_batches(model, tokenizer, features, batches, batch_size, None, device, len(texts))
    return logits, tokenize_sec + infer_sec


def main():
    if sys.executable != VENV_PY and Path(VENV_PY).exists():
        os.execv(VENV_PY, [VENV_PY, "-u", str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--correctness-rows", type=int, default=256)
    parser.add_argument("--grid-rows", type=int, default=1024)
    parser.add_argument("--chunk-grid", default="32,64,128")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = INDUCTOR_CACHE
    os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import gc

    import torch

    torch._dynamo.config.cache_size_limit = 64
    payload = {"what": "tier-2 patched DeltaNet fallback probe", "created_utc": utc_now(),
               "stack": stack_info(), "infer_anchor_sec": INFER_ANCHOR_SEC}

    def checkpoint(step):
        write_json(ART, payload)
        mirror_to_drive(ART)
        print(f"checkpoint: {step}", flush=True)

    try:
        # denominator (same session)
        denom = model_timing(QWEN3, 416, 64, "sorted", args.rows)
        payload["qwen3_denominator"] = {k: denom[k] for k in ("tokenize_plus_infer_sec", "infer_sec")}
        denom_sec = denom["tokenize_plus_infer_sec"]
        gc.collect(); torch.cuda.empty_cache()
        checkpoint("denominator")

        # stock fallback: correctness reference + eager baseline timing
        model, tokenizer, device, _ = load_qwen35()
        stock_logits, _ = timed_sorted_run(model, tokenizer, device, args.correctness_rows, 400, 64)
        _, stock_time = timed_sorted_run(model, tokenizer, device, args.rows, 400, 64)
        payload["stock_eager"] = {"tokenize_plus_infer_sec": stock_time, "ratio": stock_time / denom_sec}
        checkpoint("stock")

        # patched: correctness gate
        patched_into = apply_patch(make_patched_rule())
        payload["patched_modules"] = patched_into
        if not patched_into:
            payload["verdict"] = "RED_patch_not_applied"
            checkpoint("no-patch")
            return
        pat_logits, _ = timed_sorted_run(model, tokenizer, device, args.correctness_rows, 400, 64)
        diff = (pat_logits.float() - stock_logits.float()).abs()
        agree = (pat_logits.argmax(1) == stock_logits.argmax(1)).float().mean().item()
        payload["correctness"] = {"argmax_agreement": agree, "max_abs_diff": float(diff.max()),
                                  "mean_abs_diff": float(diff.mean())}
        checkpoint("correctness")
        if agree < 0.995:
            payload["verdict"] = "RED_correctness"
            checkpoint("verdict")
            return

        # patched eager timing + chunk grid
        _, pat_time = timed_sorted_run(model, tokenizer, device, args.rows, 400, 64)
        payload["patched_eager"] = {"tokenize_plus_infer_sec": pat_time, "ratio": pat_time / denom_sec,
                                    "projected_infer_sec": pat_time / denom_sec * INFER_ANCHOR_SEC}
        checkpoint("patched-eager")

        grid = {}
        for cs in (int(c) for c in args.chunk_grid.split(",")):
            apply_patch(make_patched_rule(chunk_size_override=cs))
            _, t = timed_sorted_run(model, tokenizer, device, args.grid_rows, 400, 64)
            grid[str(cs)] = t
            print(f"chunk {cs}: {t:.2f}s @{args.grid_rows} rows", flush=True)
        payload["chunk_grid_rows"] = args.grid_rows
        payload["chunk_grid"] = grid
        best_chunk = min(grid, key=grid.get)
        payload["best_chunk"] = int(best_chunk)
        checkpoint("chunk-grid")

        # full-rows timing at best chunk (if not 64, remeasure at full rows)
        apply_patch(make_patched_rule(chunk_size_override=int(best_chunk)))
        if best_chunk != "64":
            _, best_time = timed_sorted_run(model, tokenizer, device, args.rows, 400, 64)
        else:
            best_time = pat_time
        payload["patched_eager_best"] = {"chunk": int(best_chunk),
                                         "tokenize_plus_infer_sec": best_time,
                                         "ratio": best_time / denom_sec,
                                         "projected_infer_sec": best_time / denom_sec * INFER_ANCHOR_SEC}
        checkpoint("patched-best")

        if args.skip_compile:
            payload["verdict"] = "EAGER_ONLY"
            checkpoint("verdict")
            return

        # compiled on patched kernel (fresh cache dirs -> honest cold compile)
        del model
        gc.collect(); torch.cuda.empty_cache()
        model, tokenizer, device, _ = load_qwen35()
        compiled = torch.compile(model, mode="reduce-overhead")
        buckets = [256, 400]
        _, texts, _ = selected_samples(args.rows)
        features, lengths, tokenize_sec = tokenize_features(tokenizer, texts, 400)
        t_batches = make_batches(features, lengths, 64, buckets)
        t0 = time.perf_counter()
        payload["compile_warm_shape_times"] = warm_shapes(compiled, tokenizer, features, t_batches, 64, buckets, device)
        payload["cold_compile_wall_sec"] = time.perf_counter() - t0
        checkpoint("compiled-warmup")

        c_logits, infer_sec, _ = run_batches(compiled, tokenizer, features, t_batches, 64, buckets, device, len(texts))
        total = tokenize_sec + infer_sec
        ratio = total / denom_sec
        proj_infer = ratio * INFER_ANCHOR_SEC
        payload["compiled_patched"] = {"tokenize_plus_infer_sec": total, "ratio": ratio,
                                       "projected_infer_sec": proj_infer}
        # compiled correctness vs stock (subset: first correctness_rows entries)
        sub = c_logits[: args.correctness_rows]
        agree_c = (sub.argmax(1) == stock_logits.argmax(1)).float().mean().item()
        payload["compiled_correctness_agree"] = agree_c
        # projection: load(est 70-90) + warm compile(shipped cache -> measured next run) + infer
        payload["projection_note"] = (
            "server_total ~= M8_load(70-90 est) + warm_compile(cache-ship; cold measured here) "
            f"+ {proj_infer:.0f}"
        )
        payload["verdict"] = ("GREEN_candidate" if proj_infer <= 350
                              else "YELLOW_candidate" if proj_infer <= 470
                              else "RED_still_over")
        checkpoint("verdict")
    except Exception:  # noqa: BLE001
        payload["error"] = traceback.format_exc()[-3500:]
        payload.setdefault("verdict", "RED_error")
        checkpoint("error")


if __name__ == "__main__":
    main()
