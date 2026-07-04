import argparse
import json
from pathlib import Path

import torch

from script import ALL_CLASSES, load_jsonl, rule_feature_flags, safe_text
from train import f1_metrics
from tune_oof_rule_boosts import torch_load


CLASS_TO_ID = {label: idx for idx, label in enumerate(ALL_CLASSES)}


def load_samples_by_id(data_dir, split_name):
    samples = load_jsonl(Path(data_dir) / f"{split_name}.jsonl")
    return {safe_text(sample.get("id")): sample for sample in samples}


def bias_from_payload(payload):
    bias = payload.get("class_bias", [0.0] * len(ALL_CLASSES))
    if isinstance(bias, dict):
        return torch.tensor([float(bias.get(label, 0.0)) for label in ALL_CLASSES], dtype=torch.float32)
    return torch.tensor([float(value) for value in bias], dtype=torch.float32)


def bias_from_rule_artifact(rule_artifact):
    bias = rule_artifact.get("class_bias", {})
    return torch.tensor([float(bias.get(label, 0.0)) for label in ALL_CLASSES], dtype=torch.float32)


def apply_rules(base_scores, ids, samples_by_id, rules):
    boosted = base_scores.clone()
    applied_counts = []
    for rule in rules:
        feature = rule["feature"]
        target_id = CLASS_TO_ID[rule["target"]]
        boost = float(rule["boost"])
        applied = 0
        for row_idx, sample_id in enumerate(ids):
            sample = samples_by_id.get(sample_id)
            if sample is None:
                raise ValueError(f"missing sample for id: {sample_id}")
            if feature in rule_feature_flags(sample, base_scores[row_idx], ALL_CLASSES):
                boosted[row_idx, target_id] += boost
                applied += 1
        applied_counts.append({**rule, "applied": applied})
    return boosted, applied_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-artifact", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--bias-source", choices=["payload", "rule", "none"], default="payload")
    args = parser.parse_args()

    rule_artifact = json.loads(Path(args.rule_artifact).read_text(encoding="utf-8"))
    payload = torch_load(args.logits)
    logits = payload["logits"].float()
    ids = payload["ids"]
    y_true = [int(value) for value in payload["y_true"]]
    if args.bias_source == "payload":
        bias = bias_from_payload(payload)
    elif args.bias_source == "rule":
        bias = bias_from_rule_artifact(rule_artifact)
    else:
        bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
    base_scores = logits + bias
    base_pred = base_scores.argmax(dim=1).tolist()
    base_metrics = f1_metrics(y_true, base_pred)

    samples_by_id = load_samples_by_id(args.data_dir, args.split_name)
    boosted_scores, applied_counts = apply_rules(base_scores, ids, samples_by_id, rule_artifact["rules"])
    boosted_pred = boosted_scores.argmax(dim=1).tolist()
    boosted_metrics = f1_metrics(y_true, boosted_pred)

    print(f"bias_source={args.bias_source}")
    print(f"base_macro_f1={base_metrics['macro_f1']:.6f}")
    print(f"boosted_macro_f1={boosted_metrics['macro_f1']:.6f}")
    print("base_weakest=" + ";".join(
        f"{name}:{score:.4f}"
        for name, score in sorted(base_metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:6]
    ))
    print("boosted_weakest=" + ";".join(
        f"{name}:{score:.4f}"
        for name, score in sorted(boosted_metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:6]
    ))
    print("applied_rules=" + json.dumps(applied_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
