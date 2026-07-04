"""저장된 HF 모델(soup 포함)을 세션 split val에서 평가. (노진산)

사용: python eval_fixed.py --model-dir model_soup --data-dir /mnt/c/dacon/open/data --split-seed 42 --tune-bias
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from script import ALL_CLASSES, load_jsonl, serialize_transformer_sample
from train import CLASS_TO_ID, f1_metrics, load_labels, predict_with_bias, split_indices, tune_class_bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--serializer", default="current_v1")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tune-bias", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_dir = str(Path(args.model_dir) / "hf_model")
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_dir, local_files_only=True).to(device)
    if device.type == "cuda":
        model.half()
    model.eval()

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    labels_by_id = load_labels(Path(args.data_dir) / "train_labels.csv")
    y = [CLASS_TO_ID[labels_by_id[s["id"]]] for s in samples]
    _, val_idx = split_indices(samples, y, "session", args.split_seed)
    texts = [serialize_transformer_sample(samples[i], args.serializer) for i in val_idx]
    y_val = [y[i] for i in val_idx]
    print(f"val={len(val_idx)} model={args.model_dir}")

    parts = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            enc = tokenizer(texts[start:start + args.batch_size], padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            parts.append(model(**enc).logits.float().cpu())
    logits = torch.cat(parts, dim=0)
    raw = f1_metrics(y_val, torch.argmax(logits, dim=1).tolist())["macro_f1"]
    print(f"raw_macro_f1={raw:.6f}")
    if args.tune_bias:
        bias, tuned = tune_class_bias(logits, y_val, rounds=3)
        print(f"tuned_macro_f1={tuned:.6f}")
        print("bias:", {c: round(float(v), 3) for c, v in zip(ALL_CLASSES, bias.tolist()) if abs(v) > 1e-9})


if __name__ == "__main__":
    main()
