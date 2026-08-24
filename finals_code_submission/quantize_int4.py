"""Storage-only int4 codec for HF safetensors checkpoints (aggressive package-size relief).

Extends the team's int8-rowwise-v1 codec (quantize_checkpoint.py): float weights
(ndim >= 2) are flattened to 2D, split into groups of 128 along the column axis,
and quantized to symmetric int4 ([-7, 7]) with one fp16 scale per (row, group).
Two 4-bit values pack into one uint8 byte. 1D floats and non-float tensors pass
through unchanged. The loader reconstructs an fp16/fp32 state_dict, so GPU
inference stays plain fp16 -- compression codec only, not an inference mode.

Usage:
    python quantize_int4.py quantize --input model/hf_model/model.safetensors --output OUT.safetensors
    python quantize_int4.py verify --model-dir model --quantized OUT.safetensors \
        [--data-dir open/data] [--samples 512] [--device cuda]
"""

import argparse
import fnmatch
import gc
import json
import os
import time

import torch
from safetensors.torch import load_file, save_file

FORMAT_VERSION = "int4-group128-v1"
GROUP_SIZE = 128
SCALE_SUFFIX = ".__scale__"
BASE_ROW_IDX_SUFFIX = ".__base_row_idx__"
INT8_ROW_IDX_SUFFIX = ".__int8_row_idx__"
INT8_ROWS_SUFFIX = ".__int8_rows__"
INT8_ROW_SCALE_SUFFIX = ".__int8_row_scale__"
ROW_AUX_SUFFIXES = (
    BASE_ROW_IDX_SUFFIX,
    INT8_ROW_IDX_SUFFIX,
    INT8_ROWS_SUFFIX,
    INT8_ROW_SCALE_SUFFIX,
)


def quantize_state_dict(
    state,
    group_size=GROUP_SIZE,
    keep_fp16=(),
    keep_int8=(),
    group_overrides=(),
    int8_rows=(),
):
    """fp state_dict -> packed symmetric int4 tensors and codec metadata."""
    if group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size}")
    packed, quantized, dtypes, shapes, kept_fp16 = {}, [], {}, {}, []
    tensor_group_sizes = {}
    split_rowwise_int8 = []
    rowwise_int8 = []
    for name, tensor in state.items():
        dtypes[name] = str(tensor.dtype).replace("torch.", "")
        preserve = any(fnmatch.fnmatchcase(name, pattern) for pattern in keep_fp16)
        preserve_int8 = any(fnmatch.fnmatchcase(name, pattern) for pattern in keep_int8)
        if tensor.is_floating_point() and tensor.ndim >= 2 and preserve_int8 and not preserve:
            shapes[name] = list(tensor.shape)
            w = tensor.float().reshape(tensor.shape[0], -1)
            scale = w.abs().amax(dim=1) / 127.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            packed[name] = torch.clamp(
                (w / scale.unsqueeze(1)).round(), -127, 127
            ).to(torch.int8).reshape(tensor.shape).contiguous()
            packed[name + SCALE_SUFFIX] = scale.to(torch.float16)
            rowwise_int8.append(name)
        elif tensor.is_floating_point() and tensor.ndim >= 2 and not preserve:
            tensor_group_size = group_size
            for pattern, override_size in group_overrides:
                if fnmatch.fnmatchcase(name, pattern):
                    tensor_group_size = override_size
            if tensor_group_size <= 0:
                raise ValueError(
                    f"group size for {name} must be positive, got {tensor_group_size}"
                )
            shapes[name] = list(tensor.shape)
            full_w = tensor.float().reshape(tensor.shape[0], -1)
            split_idx = None
            for pattern, row_indices in int8_rows:
                if fnmatch.fnmatchcase(name, pattern):
                    split_idx = torch.as_tensor(row_indices, dtype=torch.long)
            if split_idx is not None:
                if tensor.ndim != 2:
                    raise ValueError(f"--int8-rows requires a 2D tensor, got {name} {tensor.shape}")
                split_idx = torch.unique(split_idx, sorted=True)
                if split_idx.numel() == 0 or split_idx.min() < 0 or split_idx.max() >= full_w.shape[0]:
                    raise ValueError(f"invalid --int8-rows indices for {name}")
                base_mask = torch.ones(full_w.shape[0], dtype=torch.bool)
                base_mask[split_idx] = False
                base_idx = base_mask.nonzero(as_tuple=False).flatten()
                w = full_w[base_idx]
            else:
                base_idx = None
                w = full_w
            rows, cols = w.shape
            pad = (-cols) % tensor_group_size
            if pad:
                w = torch.nn.functional.pad(w, (0, pad))
            g = w.reshape(rows, -1, tensor_group_size)
            scale = g.abs().amax(dim=2) / 7.0                        # (rows, n_groups)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.clamp((g / scale.unsqueeze(2)).round(), -7, 7).to(torch.int8)
            nib = (q + 8).to(torch.uint8).reshape(rows, -1)          # [1, 15]
            if nib.shape[1] % 2:
                # One neutral nibble makes odd group sizes packable. The loader
                # removes this row-tail padding before regrouping values.
                nib = torch.nn.functional.pad(nib, (0, 1), value=8)
            packed[name] = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()  # 2 values / byte
            packed[name + SCALE_SUFFIX] = scale.to(torch.float16)
            quantized.append(name)
            if split_idx is not None:
                int8_w = full_w[split_idx]
                int8_scale = int8_w.abs().amax(dim=1) / 127.0
                int8_scale = torch.where(
                    int8_scale == 0, torch.ones_like(int8_scale), int8_scale
                )
                int8_q = torch.clamp(
                    (int8_w / int8_scale.unsqueeze(1)).round(), -127, 127
                ).to(torch.int8)
                packed[name + BASE_ROW_IDX_SUFFIX] = base_idx.to(torch.int32)
                packed[name + INT8_ROW_IDX_SUFFIX] = split_idx.to(torch.int32)
                packed[name + INT8_ROWS_SUFFIX] = int8_q.contiguous()
                packed[name + INT8_ROW_SCALE_SUFFIX] = int8_scale.to(torch.float16)
                split_rowwise_int8.append(name)
            if tensor_group_size != group_size:
                tensor_group_sizes[name] = tensor_group_size
        elif tensor.is_floating_point():
            packed[name] = tensor.to(torch.float16)
            if preserve:
                kept_fp16.append(name)
        else:
            packed[name] = tensor
    meta = {
        "format": (
            "int4-mixed-v1"
            if tensor_group_sizes or split_rowwise_int8 or rowwise_int8
            else f"int4-group{group_size}-v1"
        ),
        "group_size": group_size,
        "tensor_group_sizes": tensor_group_sizes,
        "quantized": quantized,
        "dtypes": dtypes,
        "shapes": shapes,
        "kept_fp16": kept_fp16,
        "split_rowwise_int8": split_rowwise_int8,
        "rowwise_int8": rowwise_int8,
    }
    return packed, meta


def load_int4_state_dict(path, dtype=torch.float16):
    """Quantized safetensors file -> reconstructed fp state_dict (the script.py loader)."""
    packed = load_file(path)
    with open(path + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    expected_formats = {f"int4-group{meta['group_size']}-v1", "int4-mixed-v1"}
    assert meta["format"] in expected_formats, f"unknown codec format {meta['format']}"
    quantized = set(meta["quantized"])
    tensor_group_sizes = meta.get("tensor_group_sizes") or {}
    split_rowwise_int8 = set(meta.get("split_rowwise_int8") or [])
    rowwise_int8 = set(meta.get("rowwise_int8") or [])
    state = {}
    for name, tensor in packed.items():
        if name.endswith(SCALE_SUFFIX) or name.endswith(ROW_AUX_SUFFIXES):
            continue
        if name in rowwise_int8:
            scale = packed[name + SCALE_SUFFIX].float()
            shaped = scale.view(-1, *([1] * (tensor.ndim - 1)))
            state[name] = (tensor.float() * shaped).to(dtype)
        elif name in quantized:
            shape = meta["shapes"][name]
            rows = tensor.shape[0]
            lo = (tensor & 0x0F).to(torch.int8) - 8
            hi = (tensor >> 4).to(torch.int8) - 8
            q = torch.stack((lo, hi), dim=2).reshape(rows, -1)       # interleave back
            scale = packed[name + SCALE_SUFFIX].float()
            cols = 1
            for d in shape[1:]:
                cols *= d
            tensor_group_size = int(tensor_group_sizes.get(name, meta["group_size"]))
            padded_cols = ((cols + tensor_group_size - 1) // tensor_group_size) * tensor_group_size
            q = q[:, :padded_cols]
            w = (q.float().reshape(rows, -1, tensor_group_size) * scale.unsqueeze(2)).reshape(rows, -1)
            base_w = w[:, :cols]
            if name in split_rowwise_int8:
                restored = torch.empty(shape, dtype=dtype)
                base_idx = packed[name + BASE_ROW_IDX_SUFFIX].long()
                int8_idx = packed[name + INT8_ROW_IDX_SUFFIX].long()
                restored[base_idx] = base_w.to(dtype)
                int8_scale = packed[name + INT8_ROW_SCALE_SUFFIX].float().unsqueeze(1)
                int8_w = packed[name + INT8_ROWS_SUFFIX].float() * int8_scale
                restored[int8_idx] = int8_w.to(dtype)
                state[name] = restored
            else:
                state[name] = base_w.reshape(shape).to(dtype)
        elif tensor.is_floating_point():
            state[name] = tensor.to(dtype)
        else:
            state[name] = tensor
    return state


def cmd_quantize(args):
    state = load_file(args.input)
    group_overrides = []
    for value in args.group_override:
        if "=" not in value:
            raise ValueError(f"--group-override must be PATTERN=SIZE, got {value!r}")
        pattern, size_text = value.rsplit("=", 1)
        group_overrides.append((pattern, int(size_text)))
    int8_rows = []
    for value in args.int8_rows:
        if "=" not in value:
            raise ValueError(f"--int8-rows must be PATTERN=JSON_PATH, got {value!r}")
        pattern, json_path = value.rsplit("=", 1)
        with open(json_path, encoding="utf-8") as f:
            row_payload = json.load(f)
        if isinstance(row_payload, dict):
            row_payload = row_payload.get("token_ids", row_payload.get("row_indices"))
        if not isinstance(row_payload, list):
            raise ValueError(f"{json_path} must contain a list or token_ids/row_indices list")
        int8_rows.append((pattern, row_payload))
    packed, meta = quantize_state_dict(
        state,
        group_size=args.group_size,
        keep_fp16=args.keep_fp16,
        keep_int8=args.keep_int8,
        group_overrides=group_overrides,
        int8_rows=int8_rows,
    )
    save_file(packed, args.output)
    with open(args.output + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    in_mb = os.path.getsize(args.input) / 1e6
    out_mb = os.path.getsize(args.output) / 1e6
    print(f"quantized {len(meta['quantized'])}/{len(state)} tensors")
    print(f"size: {in_mb:.1f} MB -> {out_mb:.1f} MB ({out_mb / in_mb:.2%})")


def cmd_roundtrip(args):
    """Synthetic sanity check: random tensors -> quantize -> dequantize -> error report."""
    torch.manual_seed(0)
    state = {
        "emb.weight": torch.randn(1000, 256) * 0.02,
        "layer.weight": torch.randn(512, 300),  # cols not a multiple of 128 (pad path)
        "odd_group.weight": torch.randn(7, 5),  # odd group-size row-tail nibble path
        "norm.weight": torch.randn(512),        # 1D passthrough
        "ids": torch.arange(10),                # non-float passthrough
    }
    packed, meta = quantize_state_dict(state, group_size=args.group_size)
    tmp = args.output
    save_file(packed, tmp)
    with open(tmp + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    restored = load_int4_state_dict(tmp, dtype=torch.float32)
    for name, tensor in state.items():
        r = restored[name]
        assert r.shape == tensor.shape, f"{name}: shape {r.shape} != {tensor.shape}"
        if tensor.is_floating_point():
            err = (r.float() - tensor.float()).abs()
            rel = err.sum().item() / tensor.float().abs().sum().item()
            print(f"{name}: max_abs={err.max().item():.6f} mean_rel={rel:.4%}")
        else:
            assert torch.equal(r, tensor), f"{name}: non-float mismatch"
            print(f"{name}: exact")
    os.remove(tmp)
    os.remove(tmp + ".meta.json")
    print("roundtrip OK")


def cmd_verify(args):
    import sys

    toolkit_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(toolkit_dir, "..", ".."))
    sys.path.insert(0, project_root)
    from script import (
        disable_decoder_cache,
        load_jsonl,
        model_logits_sorted,
        serialize_transformer_sample,
    )
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    hf_dir = os.path.join(args.model_dir, "hf_model")
    with open(os.path.join(args.model_dir, "hf_meta.json"), encoding="utf-8") as f:
        hf_meta = json.load(f)
    with open(args.quantized + ".meta.json", encoding="utf-8") as f:
        codec_meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples_path = os.path.join(args.data_dir, args.data_file)
    samples = load_jsonl(samples_path)[: args.samples]
    serializer_name = hf_meta.get("serializer_name", "current_v1")
    texts = [serialize_transformer_sample(s, serializer_name) for s in samples]
    max_length = int(hf_meta.get("max_length", 192))
    terminal_token = hf_meta.get("terminal_token", "")

    def infer(model):
        disable_decoder_cache(model)
        model.to(device).eval()
        started = time.perf_counter()
        logits = model_logits_sorted(
            model,
            tokenizer,
            texts,
            max_length,
            args.batch_size,
            device,
            terminal_token,
        )
        return logits, time.perf_counter() - started

    # The old verifier held both 1.5B models on the GPU at once. Run the fp16
    # reference first, release it completely, then reconstruct/load INT4.
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_dir, local_files_only=True, torch_dtype=dtype
    )
    original_logits, original_seconds = infer(model)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"original inference: {len(texts)} rows in {original_seconds:.2f}s", flush=True)

    restored = load_int4_state_dict(args.quantized, dtype=dtype)

    # Compare weights without loading the complete fp16 reference checkpoint a
    # second time. safe_open maps/reads one tensor at a time, keeping host RAM
    # bounded by the restored state plus the largest reference tensor.
    from safetensors import safe_open

    max_w_err = rel_num = rel_den = 0.0
    original_path = os.path.join(hf_dir, "model.safetensors")
    with safe_open(original_path, framework="pt", device="cpu") as original:
        original_names = set(original.keys())
        restored_names = set(restored)
        if original_names != restored_names:
            raise RuntimeError(
                f"int4 tensor mismatch: missing={sorted(original_names - restored_names)} "
                f"unexpected={sorted(restored_names - original_names)}"
            )
        for name in original.keys():
            reference = original.get_tensor(name).float()
            err = (restored[name].float() - reference).abs()
            max_w_err = max(max_w_err, err.max().item())
            rel_num += err.sum().item()
            rel_den += reference.abs().sum().item()
            del reference, err
    mean_relative_weight_error = rel_num / rel_den
    print(
        f"weight error: max_abs={max_w_err:.6f} "
        f"mean_rel={mean_relative_weight_error:.6%}",
        flush=True,
    )

    config = AutoConfig.from_pretrained(hf_dir, local_files_only=True)
    config.torch_dtype = dtype
    model = AutoModelForSequenceClassification.from_config(config, torch_dtype=dtype)
    if dtype == torch.float16:
        model.half()
    missing, unexpected = model.load_state_dict(restored, strict=False)
    missing = [name for name in missing if not name.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(
            f"int4 checkpoint mismatch: missing={missing} unexpected={list(unexpected)}"
        )
    del restored
    gc.collect()
    int4_logits, int4_seconds = infer(model)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"int4 inference: {len(texts)} rows in {int4_seconds:.2f}s", flush=True)

    logit_error = (original_logits - int4_logits).abs()
    original_probs = torch.softmax(original_logits, dim=1)
    int4_probs = torch.softmax(int4_logits, dim=1)
    total_variation = 0.5 * (original_probs - int4_probs).abs().sum(dim=1)
    bias = torch.tensor(
        hf_meta.get("class_bias", [0.0] * original_logits.shape[1]),
        dtype=torch.float32,
    )
    original_pred = (original_logits + bias).argmax(dim=1)
    int4_pred = (int4_logits + bias).argmax(dim=1)
    agree_mask = original_pred == int4_pred
    agree = int(agree_mask.sum().item())
    disagreements = [
        {
            "index": index,
            "id": str(samples[index].get("id", "")),
            "original_class_id": int(original_pred[index]),
            "int4_class_id": int(int4_pred[index]),
        }
        for index in (~agree_mask).nonzero(as_tuple=False).flatten().tolist()
    ]
    report = {
        "format": codec_meta["format"],
        "group_size": codec_meta["group_size"],
        "model_dir": args.model_dir,
        "original_path": original_path,
        "quantized_path": args.quantized,
        "original_bytes": os.path.getsize(original_path),
        "quantized_bytes": os.path.getsize(args.quantized),
        "size_ratio": os.path.getsize(args.quantized) / os.path.getsize(original_path),
        "data_path": samples_path,
        "samples": len(texts),
        "serializer_name": serializer_name,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "weight_error": {
            "max_abs": max_w_err,
            "mean_relative_l1": mean_relative_weight_error,
        },
        "logit_error": {
            "max_abs": float(logit_error.max().item()),
            "mean_abs": float(logit_error.mean().item()),
        },
        "probability_total_variation": {
            "mean": float(total_variation.mean().item()),
            "max": float(total_variation.max().item()),
        },
        "argmax": {
            "agree": agree,
            "total": len(texts),
            "agreement": agree / len(texts),
            "disagreements": disagreements,
        },
        "timing_seconds": {
            "original_inference": original_seconds,
            "int4_inference": int4_seconds,
        },
    }
    print(
        f"logit error: max_abs={report['logit_error']['max_abs']:.4f} "
        f"mean_abs={report['logit_error']['mean_abs']:.5f}"
    )
    print(
        f"probability TV: mean={report['probability_total_variation']['mean']:.6f} "
        f"max={report['probability_total_variation']['max']:.6f}"
    )
    print(f"argmax agreement: {agree}/{len(texts)} ({agree / len(texts):.4%})")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_q = sub.add_parser("quantize", help="safetensors fp checkpoint -> int4 codec file")
    p_q.add_argument("--input", required=True)
    p_q.add_argument("--output", required=True)
    p_q.add_argument("--group-size", type=int, default=GROUP_SIZE)
    p_q.add_argument("--keep-fp16", action="append", default=[],
                     help="fnmatch pattern for tensors to store as fp16 (repeatable)")
    p_q.add_argument("--keep-int8", action="append", default=[],
                     help="fnmatch pattern for tensors to store as rowwise int8 (repeatable)")
    p_q.add_argument("--group-override", action="append", default=[],
                     help="fnmatch PATTERN=SIZE override (repeatable; last match wins)")
    p_q.add_argument("--int8-rows", action="append", default=[],
                     help="PATTERN=JSON_PATH rows to split into rowwise int8 (repeatable)")
    p_r = sub.add_parser("roundtrip", help="synthetic random-tensor roundtrip sanity check")
    p_r.add_argument("--output", default="_int4_roundtrip_tmp.safetensors")
    p_r.add_argument("--group-size", type=int, default=GROUP_SIZE)
    p_v = sub.add_parser("verify", help="compare original vs dequantized logits on real samples")
    p_v.add_argument("--model-dir", default="model")
    p_v.add_argument("--quantized", required=True)
    p_v.add_argument("--data-dir", default="open/data")
    p_v.add_argument("--data-file", default="test.jsonl")
    p_v.add_argument("--samples", type=int, default=512)
    p_v.add_argument("--batch-size", type=int, default=8)
    p_v.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_v.add_argument("--output")
    args = parser.parse_args()
    if args.command == "quantize":
        cmd_quantize(args)
    elif args.command == "roundtrip":
        cmd_roundtrip(args)
    else:
        cmd_verify(args)


if __name__ == "__main__":
    main()
