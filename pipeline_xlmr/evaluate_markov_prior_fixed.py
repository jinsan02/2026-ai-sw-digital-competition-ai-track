import argparse
import json
from pathlib import Path

import torch

from script import ALL_CLASSES, load_jsonl, safe_text
from train import CLASS_TO_ID, f1_metrics, load_labels, session_id
from tune_markov_prior_oof import fit_prior_tables, predict_prior_scores
from tune_oof_rule_boosts import torch_load
from evaluate_rule_boosts import apply_rules


def load_train_data(data_dir):
    data_dir = Path(data_dir)
    samples = load_jsonl(data_dir / "train.jsonl")
    labels_by_id = load_labels(data_dir / "train_labels.csv")
    y = [CLASS_TO_ID[labels_by_id[sample["id"]]] for sample in samples]
    samples_by_id = {safe_text(sample.get("id")): sample for sample in samples}
    return samples, y, samples_by_id


def bias_from_payload(payload):
    bias = payload.get("class_bias", [0.0] * len(ALL_CLASSES))
    if isinstance(bias, dict):
        return torch.tensor([float(bias.get(label, 0.0)) for label in ALL_CLASSES], dtype=torch.float32)
    return torch.tensor([float(value) for value in bias], dtype=torch.float32)


def extra_bias_from_markov_artifact(path):
    if not path:
        return torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bias = payload.get("best_class_bias", {})
    return torch.tensor([float(bias.get(label, 0.0)) for label in ALL_CLASSES], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True)
    parser.add_argument("--rule-artifact", default="")
    parser.add_argument("--markov-artifact", default="")
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--smoothing", type=float, default=2.0)
    parser.add_argument("--feature-weight", type=float, default=0.35)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--use-markov-extra-bias", action="store_true")
    args = parser.parse_args()

    samples, y, samples_by_id = load_train_data(args.data_dir)
    payload = torch_load(args.logits)
    ids = [safe_text(sample_id) for sample_id in payload["ids"]]
    y_true = [int(value) for value in payload["y_true"]]
    val_sessions = {session_id(sample_id) for sample_id in ids}
    train_indices = [
        idx for idx, sample in enumerate(samples)
        if session_id(sample.get("id", "")) not in val_sessions
    ]

    global_logp, feature_deltas = fit_prior_tables(samples, y, train_indices, args.min_support, args.smoothing)
    prior_scores = predict_prior_scores(samples, ids, samples_by_id, global_logp, feature_deltas, args.feature_weight)

    scores = payload["logits"].float() + bias_from_payload(payload)
    base_metrics = f1_metrics(y_true, scores.argmax(dim=1).tolist())
    if args.rule_artifact:
        rule_payload = json.loads(Path(args.rule_artifact).read_text(encoding="utf-8"))
        scores, applied_rules = apply_rules(scores, ids, samples_by_id, rule_payload.get("rules", []))
    else:
        applied_rules = []
    rules_metrics = f1_metrics(y_true, scores.argmax(dim=1).tolist())
    combined = scores + args.prior_weight * prior_scores
    if args.use_markov_extra_bias:
        combined = combined + extra_bias_from_markov_artifact(args.markov_artifact)
    combined_metrics = f1_metrics(y_true, combined.argmax(dim=1).tolist())

    print(f"train_rows={len(train_indices)} val_rows={len(ids)} feature_count={len(feature_deltas)}")
    print(f"base_macro_f1={base_metrics['macro_f1']:.6f}")
    print(f"rules_macro_f1={rules_metrics['macro_f1']:.6f}")
    print(f"combined_macro_f1={combined_metrics['macro_f1']:.6f}")
    print("combined_weakest=" + ";".join(
        f"{name}:{score:.4f}"
        for name, score in sorted(combined_metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:6]
    ))
    if applied_rules:
        print("applied_rules=" + json.dumps(applied_rules, ensure_ascii=False))


if __name__ == "__main__":
    main()
