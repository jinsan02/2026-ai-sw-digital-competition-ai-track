"""Storage-only int8 codec for HF safetensors checkpoints (package-size relief).

Quantizes float weights (ndim >= 2) to per-row symmetric int8 with fp32 scales;
1D floats (biases, LayerNorm) and non-float tensors pass through unchanged. The
loader reconstructs an fp16/fp32 state_dict, so GPU inference stays plain fp16 --
quantization is used purely as a compression codec, not an inference mode. No
dependencies beyond torch + safetensors (both already in requirements.txt).

Usage:
    python quantize_checkpoint.py quantize --input model/hf_model/model.safetensors --output OUT.safetensors
    python quantize_checkpoint.py verify --model-dir model --quantized OUT.safetensors \
        [--data-dir open/data] [--samples 512] [--device cuda]

`verify` compares original-fp16 vs dequantized-fp16 logits on real serialized
training samples (same serializer/max_length/batching as script.py) and reports
weight error, logit error, and argmax agreement.
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

FORMAT_VERSION = "int8-rowwise-v1"
SCALE_SUFFIX = ".__scale__"


def quantize_state_dict(state):
    """fp state_dict -> (packed tensors, meta). Per-row symmetric int8 for ndim>=2 floats."""
    packed, quantized, dtypes = {}, [], {}
    for name, tensor in state.items():
        dtypes[name] = str(tensor.dtype).replace("torch.", "")
        if tensor.is_floating_point() and tensor.ndim >= 2:
            w = tensor.float()
            amax = w.abs().amax(dim=tuple(range(1, w.ndim)))
            scale = amax / 127.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            shaped = scale.view(-1, *([1] * (w.ndim - 1)))
            packed[name] = torch.clamp((w / shaped).round(), -127, 127).to(torch.int8)
            packed[name + SCALE_SUFFIX] = scale
            quantized.append(name)
        elif tensor.is_floating_point():
            packed[name] = tensor.to(torch.float16)
        else:
            packed[name] = tensor
    meta = {"format": FORMAT_VERSION, "quantized": quantized, "dtypes": dtypes}
    return packed, meta


def load_int8_state_dict(path, dtype=torch.float16):
    """Quantized safetensors file -> reconstructed fp state_dict (the script.py loader)."""
    packed = load_file(path)
    with open(path + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["format"] == FORMAT_VERSION, f"unknown codec format {meta['format']}"
    quantized = set(meta["quantized"])
    state = {}
    for name, tensor in packed.items():
        if name.endswith(SCALE_SUFFIX):
            continue
        if name in quantized:
            scale = packed[name + SCALE_SUFFIX]
            shaped = scale.view(-1, *([1] * (tensor.ndim - 1)))
            state[name] = (tensor.float() * shaped).to(dtype)
        elif tensor.is_floating_point():
            state[name] = tensor.to(dtype)
        else:
            state[name] = tensor
    return state


def cmd_quantize(args):
    state = load_file(args.input)
    packed, meta = quantize_state_dict(state)
    save_file(packed, args.output)
    with open(args.output + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    in_mb = os.path.getsize(args.input) / 1e6
    out_mb = os.path.getsize(args.output) / 1e6
    print(f"quantized {len(meta['quantized'])}/{len(state)} tensors")
    print(f"size: {in_mb:.1f} MB -> {out_mb:.1f} MB ({out_mb / in_mb:.2%})")


def cmd_verify(args):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from script import (
        load_jsonl,
        serialize_transformer_sample,
        tokenize_texts_with_terminal,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    hf_dir = os.path.join(args.model_dir, "hf_model")
    with open(os.path.join(args.model_dir, "hf_meta.json"), encoding="utf-8") as f:
        hf_meta = json.load(f)

    original = load_file(os.path.join(hf_dir, "model.safetensors"))
    restored = load_int8_state_dict(args.quantized, dtype=torch.float32)
    max_w_err = rel_num = rel_den = 0.0
    for name, tensor in original.items():
        err = (restored[name].float() - tensor.float()).abs()
        max_w_err = max(max_w_err, err.max().item())
        rel_num += err.sum().item()
        rel_den += tensor.float().abs().sum().item()
    print(f"weight error: max_abs={max_w_err:.6f} mean_rel={rel_num / rel_den:.6%}")

    tokenizer = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    model_a = AutoModelForSequenceClassification.from_pretrained(hf_dir, local_files_only=True)
    model_b = AutoModelForSequenceClassification.from_pretrained(hf_dir, local_files_only=True)
    missing, unexpected = model_b.load_state_dict(
        {k: v for k, v in restored.items()}, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={list(missing)} unexpected={list(unexpected)}")
    for model in (model_a, model_b):
        model.to(device)
        model.to(dtype)
        model.eval()

    samples = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))[: args.samples]
    serializer_name = hf_meta.get("serializer_name", "current_v1")
    texts = [serialize_transformer_sample(s, serializer_name) for s in samples]
    max_length = int(hf_meta.get("max_length", 192))
    terminal_token = hf_meta.get("terminal_token", "")

    agree = 0
    max_logit_err = sum_logit_err = n_logits = 0.0
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start:start + args.batch_size]
            encoded_rows = tokenize_texts_with_terminal(
                tokenizer,
                batch_texts,
                max_length,
                terminal_token,
            )
            keys = list(encoded_rows.keys())
            encoded = tokenizer.pad(
                [
                    {key: encoded_rows[key][row] for key in keys}
                    for row in range(len(batch_texts))
                ],
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            la = model_a(**encoded).logits.float()
            lb = model_b(**encoded).logits.float()
            err = (la - lb).abs()
            max_logit_err = max(max_logit_err, err.max().item())
            sum_logit_err += err.sum().item()
            n_logits += err.numel()
            agree += (la.argmax(dim=1) == lb.argmax(dim=1)).sum().item()
    print(f"logit error: max_abs={max_logit_err:.4f} mean_abs={sum_logit_err / n_logits:.5f}")
    print(f"argmax agreement: {agree}/{len(texts)} ({agree / len(texts):.4%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_q = sub.add_parser("quantize", help="safetensors fp checkpoint -> int8 codec file")
    p_q.add_argument("--input", required=True)
    p_q.add_argument("--output", required=True)
    p_v = sub.add_parser("verify", help="compare original vs dequantized logits on real samples")
    p_v.add_argument("--model-dir", default="model")
    p_v.add_argument("--quantized", required=True)
    p_v.add_argument("--data-dir", default="open/data")
    p_v.add_argument("--samples", type=int, default=512)
    p_v.add_argument("--batch-size", type=int, default=32)
    p_v.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "quantize":
        cmd_quantize(args)
    else:
        cmd_verify(args)


if __name__ == "__main__":
    main()
