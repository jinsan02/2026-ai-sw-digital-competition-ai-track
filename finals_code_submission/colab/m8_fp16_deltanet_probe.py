"""M8 volume lever: fp16 DeltaNet fallback numerics + timing probe (T4 replica).

Tier-1 profiled the eager fallback as memory-volume bound: fp32-conversion
copies 28.9% + elementwise ~40% (fp32 bytes), GEMM only 20% (no tensor cores
in fp32 on T4). The untested gamble: run the fallback math in fp16 —
conversions vanish, elementwise bytes halve, GEMMs hit tensor cores.
The risk is numeric drift in the chunked recurrence (the stock upcast exists
for a reason); the probe's whole point is the correctness gate.

Two variants, both keeping the numerically fragile parts in fp32
(gate cumsum/exp on the small [B,H,L] g tensor; the chunk triangular inverse —
cublas trsm has no fp16 path anyway):
  fp16_state   — recurrent state in fp16 (max volume win)
  fp32_state   — recurrent state kept fp32, per-chunk casts (safer)

Stages: venv setup (if missing) -> denominator (Qwen3-0.6B) -> stock eager
(reference logits + timing) -> per-variant correctness gate (256 rows,
call-counter canary proves the patch executed) -> grid timing (1024 rows) ->
best variant full timing + 4096-row agreement -> verdict.
Anchor: M7 server infer ~478s; stock eager ratio was 2.305 (tier-2).
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
    run,
    stack_info,
    utc_now,
)
from colab.m8_qwen35_compile_probe import load_qwen35  # noqa: E402
from colab.m8_qwen35_maxpack import mirror_to_drive, write_json  # noqa: E402
from colab.m8_tier2_patched_kernel import make_patched_rule, timed_sorted_run  # noqa: E402

ART = Path("experiments/artifacts/m8_fp16_deltanet_probe.json")
INFER_ANCHOR_SEC = 478.0
STOCK_RATIO_PRIOR = 2.305  # tier-2 measured, same rows/recipe
VENV_PY = "/content/venv311/bin/python"
TRAINED_DIR = "/content/drive/MyDrive/AADP_exchange/models/m8_qwen35_refit/hf_model"


def ensure_venv():
    if Path(VENV_PY).exists():
        return True
    run(["add-apt-repository", "-y", "ppa:deadsnakes/ppa"], check=False, timeout=300)
    run(["apt-get", "install", "-y", "-q", "python3.11", "python3.11-venv", "python3.11-dev"],
        check=False, timeout=600)
    mk = run(["python3.11", "-m", "venv", "/content/venv311"], check=False, timeout=300)
    if mk["returncode"] != 0 or not Path(VENV_PY).exists():
        run([sys.executable, "-m", "venv", "/content/venv311"], check=False, timeout=300)
    run([VENV_PY, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=False, timeout=600)
    run([VENV_PY, "-m", "pip", "install", "-q", "torch==2.7.1",
         "--index-url", "https://download.pytorch.org/whl/cu128"], check=False, timeout=1800)
    run([VENV_PY, "-m", "pip", "install", "-q", "transformers>=5.13,<5.14",
         "safetensors==0.8.0"], check=False, timeout=900)
    return Path(VENV_PY).exists()


def make_fp16_rule(state_in_fp32=False):
    import torch
    import torch.nn.functional as F

    calls = {"n": 0}

    def fp16_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

        calls["n"] += 1
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        # fp16 lane: the big [B,H,L,D] tensors stay in the input dtype; only the
        # small [B,H,L] gate path runs fp32 (cumsum/exp precision)
        query, key, value, beta = [
            x.transpose(1, 2).contiguous() for x in (query, key, value, beta)
        ]
        g = g.transpose(1, 2).contiguous().to(torch.float32)
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

        g = g.cumsum(dim=-1)  # fp32
        decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril().to(initial_dtype)
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        # chunk triangular inverse stays fp32: cublas trsm has no fp16 path and
        # the 63-step substitution chain is the numerically fragile part
        eye = torch.eye(chunk_size, dtype=torch.float32, device=attn.device)
        attn = torch.linalg.solve_triangular(
            eye - attn.float(),
            eye.expand(*attn.shape[:-2], chunk_size, chunk_size).contiguous(),
            upper=False, unitriangular=True,
        ).to(initial_dtype)
        value = attn @ v_beta
        g_exp = g.exp().to(initial_dtype)  # exp'd in fp32, used in fp16
        k_cumdecay = attn @ (k_beta * g_exp.unsqueeze(-1))
        state_dtype = torch.float32 if state_in_fp32 else initial_dtype
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                        dtype=state_dtype, device=value.device)
            if initial_state is None
            else initial_state.to(device=value.device, dtype=state_dtype)
        )
        core_attn_out = torch.zeros_like(value)
        g_last = g[:, :, :, -1].exp()  # fp32 [B,H,C]
        g_tail = (g[:, :, :, -1, None] - g).exp().to(initial_dtype)  # [B,H,C,chunk]

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            state_h = last_recurrent_state.to(initial_dtype) if state_in_fp32 else last_recurrent_state
            v_prime = k_cumdecay[:, :, i] @ state_h
            v_new = v_i - v_prime
            attn_inter = (q_i * g_exp[:, :, i, :, None]) @ state_h
            core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
            delta = (k_i * g_tail[:, :, i, :, None]).transpose(-1, -2) @ v_new
            if state_in_fp32:
                last_recurrent_state = (
                    last_recurrent_state * g_last[:, :, i, None, None] + delta.float()
                )
            else:
                last_recurrent_state = (
                    last_recurrent_state * g_last[:, :, i, None, None].to(initial_dtype) + delta
                )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    fp16_rule.calls = calls
    return fp16_rule


def rebind_rule(model, fn):
    """The modeling code binds the fallback to the layer INSTANCE at __init__
    (`self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_...`), so
    module-global monkeypatching after load never fires — the tier-2 probe's
    'patch has zero effect' result was this, not physics. Rebind in place."""
    n = 0
    for mod in model.modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            mod.chunk_gated_delta_rule = fn
            n += 1
    return n


def load_m8_trained():
    """Trained refit weights (real decision margins) if the Drive dir is there;
    hub base + random head otherwise (tier-2 comparability fallback)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if Path(TRAINED_DIR).exists():
        tokenizer = AutoTokenizer.from_pretrained(TRAINED_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            TRAINED_DIR, torch_dtype=torch.float16
        )
        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.to("cuda").eval()
        return model, tokenizer, "cuda", "m8_qwen35_refit (trained, Drive)"
    model, tokenizer, device, _ = load_qwen35()
    return model, tokenizer, device, "hub base + random head (Drive dir missing)"


def main():
    if sys.executable != VENV_PY:
        ensure_venv()
        if Path(VENV_PY).exists():
            os.execv(VENV_PY, [VENV_PY, "-u", str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--correctness-rows", type=int, default=256)
    parser.add_argument("--grid-rows", type=int, default=1024)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import gc

    import torch

    payload = {"what": "fp16 DeltaNet fallback numerics + timing probe",
               "created_utc": utc_now(), "stack": stack_info(),
               "infer_anchor_sec": INFER_ANCHOR_SEC,
               "stock_ratio_prior": STOCK_RATIO_PRIOR}

    def checkpoint(step):
        write_json(ART, payload)
        mirror_to_drive(ART)
        print(f"checkpoint: {step}", flush=True)

    try:
        denom = model_timing("Qwen/Qwen3-0.6B", 416, 64, "sorted", args.rows)
        payload["qwen3_denominator"] = {k: denom[k] for k in ("tokenize_plus_infer_sec", "infer_sec")}
        denom_sec = denom["tokenize_plus_infer_sec"]
        gc.collect(); torch.cuda.empty_cache()
        checkpoint("denominator")

        model, tokenizer, device, weights_note = load_m8_trained()
        payload["weights"] = weights_note
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule as stock_rule,
        )
        stock_ref, _ = timed_sorted_run(model, tokenizer, device, args.correctness_rows, 400, 64)
        stock_logits, stock_time = timed_sorted_run(model, tokenizer, device, args.rows, 400, 64)
        _, stock_grid_time = timed_sorted_run(model, tokenizer, device, args.grid_rows, 400, 64)
        payload["stock_eager"] = {"tokenize_plus_infer_sec": stock_time,
                                  "grid_time_sec": stock_grid_time,
                                  "ratio": stock_time / denom_sec}
        checkpoint("stock")

        # variants: the two fp16 lanes + a proper retest of the tier-2
        # solve_triangular fp32 patch (its earlier 'zero effect' verdict was
        # measured with the patch never actually executing)
        def solve32_rule():
            fn = make_patched_rule()
            calls = {"n": 0}
            def wrapped(*a, **kw):
                calls["n"] += 1
                return fn(*a, **kw)
            wrapped.calls = calls
            return wrapped

        variants = {
            "fp16_state": lambda: make_fp16_rule(state_in_fp32=False),
            "fp32_state": lambda: make_fp16_rule(state_in_fp32=True),
            "solve32_retest": solve32_rule,
        }
        results = {}
        for name, factory in variants.items():
            rule = factory()
            rebound = rebind_rule(model, rule)
            entry = {"rebound_layers": rebound}
            var_ref, _ = timed_sorted_run(model, tokenizer, device, args.correctness_rows, 400, 64)
            diff = (var_ref.float() - stock_ref.float()).abs()
            agree = (var_ref.argmax(1) == stock_ref.argmax(1)).float().mean().item()
            entry.update({"argmax_agreement_256": agree,
                          "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                          "kernel_calls": rule.calls["n"]})
            if rule.calls["n"] == 0:
                entry["error"] = "canary: patched kernel never called"
            elif agree >= 0.995:
                _, t = timed_sorted_run(model, tokenizer, device, args.grid_rows, 400, 64)
                entry["grid_time_sec"] = t
                entry["grid_speedup_vs_stock"] = stock_grid_time / t
            results[name] = entry
            rebind_rule(model, stock_rule)  # restore before next variant
            payload["variants"] = results
            checkpoint(f"variant-{name}")

        timed = {k: v for k, v in results.items() if "grid_time_sec" in v}
        if not timed:
            payload["verdict"] = "RED_numerics"
            checkpoint("verdict")
            return
        best = min(timed, key=lambda k: timed[k]["grid_time_sec"])
        payload["best_variant"] = best
        rebind_rule(model, variants[best]())
        full_logits, full_time = timed_sorted_run(model, tokenizer, device, args.rows, 400, 64)
        agree_full = (full_logits.argmax(1) == stock_logits.argmax(1)).float().mean().item()
        ratio = full_time / denom_sec
        proj = ratio * INFER_ANCHOR_SEC
        payload["fp16_best_full"] = {
            "tokenize_plus_infer_sec": full_time, "ratio": ratio,
            "argmax_agreement_4096": agree_full,
            "projected_infer_sec": proj,
            "projected_infer_sec_v5": proj * 0.79,
            "speedup_vs_stock": stock_time / full_time,
        }
        payload["verdict"] = (
            "RED_numerics" if agree_full < 0.995
            else "GREEN_big_win" if ratio <= 1.7
            else "YELLOW_gain" if ratio <= 2.1
            else "RED_no_gain"
        )
        checkpoint("verdict")
    except Exception:  # noqa: BLE001
        payload["error"] = traceback.format_exc()[-3500:]
        payload.setdefault("verdict", "RED_error")
        checkpoint("error")


if __name__ == "__main__":
    main()
