import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from script import ALL_CLASSES, load_jsonl, safe_text, serialize_transformer_sample_current
from train import CLASS_TO_ID, f1_metrics, load_labels, session_id
from tune_oof_rule_boosts import torch_load
from evaluate_rule_boosts import apply_rules


def load_train_data(data_dir):
    data_dir = Path(data_dir)
    samples = load_jsonl(data_dir / "train.jsonl")
    labels_by_id = load_labels(data_dir / "train_labels.csv")
    y = np.array([CLASS_TO_ID[labels_by_id[sample["id"]]] for sample in samples], dtype=np.int64)
    samples_by_id = {safe_text(sample.get("id")): sample for sample in samples}
    sample_index_by_id = {safe_text(sample.get("id")): idx for idx, sample in enumerate(samples)}
    return samples, y, samples_by_id, sample_index_by_id


def make_vectorizer(args):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=args.word_min_df,
        max_features=args.word_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=args.char_min_df,
        max_features=args.char_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)], n_jobs=1)


def bias_from_payload(payload):
    bias = payload.get("class_bias", [0.0] * len(ALL_CLASSES))
    if isinstance(bias, dict):
        return torch.tensor([float(bias.get(label, 0.0)) for label in ALL_CLASSES], dtype=torch.float32)
    return torch.tensor([float(value) for value in bias], dtype=torch.float32)


def normalize_scores(train_scores, val_scores):
    mean = train_scores.mean(axis=0, keepdims=True)
    std = train_scores.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (val_scores - mean) / std


def sparse_text(sample):
    return serialize_transformer_sample_current(sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True)
    parser.add_argument("--rule-artifact", default="")
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--word-features", type=int, default=180000)
    parser.add_argument("--char-features", type=int, default=220000)
    parser.add_argument("--word-min-df", type=int, default=2)
    parser.add_argument("--char-min-df", type=int, default=2)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--sparse-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples, y, samples_by_id, sample_index_by_id = load_train_data(args.data_dir)
    payload = torch_load(args.logits)
    ids = [safe_text(sample_id) for sample_id in payload["ids"]]
    y_true = [int(value) for value in payload["y_true"]]
    val_sessions = {session_id(sample_id) for sample_id in ids}
    train_idx = [
        idx for idx, sample in enumerate(samples)
        if session_id(sample.get("id", "")) not in val_sessions
    ]
    val_idx = [sample_index_by_id[sample_id] for sample_id in ids]

    vectorizer = make_vectorizer(args)
    x_train = vectorizer.fit_transform([sparse_text(samples[idx]) for idx in train_idx])
    x_val = vectorizer.transform([sparse_text(samples[idx]) for idx in val_idx])
    model = LinearSVC(
        C=args.c,
        class_weight="balanced" if args.class_weight == "balanced" else None,
        dual="auto",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    model.fit(x_train, y[train_idx])
    train_scores = model.decision_function(x_train)
    sparse_scores = model.decision_function(x_val)
    if args.normalize:
        sparse_scores = normalize_scores(train_scores, sparse_scores)
    sparse_scores = torch.tensor(sparse_scores, dtype=torch.float32)

    base_scores = payload["logits"].float() + bias_from_payload(payload)
    base_metrics = f1_metrics(y_true, base_scores.argmax(dim=1).tolist())
    if args.rule_artifact:
        rule_payload = json.loads(Path(args.rule_artifact).read_text(encoding="utf-8"))
        base_scores, applied_rules = apply_rules(base_scores, ids, samples_by_id, rule_payload.get("rules", []))
    else:
        applied_rules = []
    rules_metrics = f1_metrics(y_true, base_scores.argmax(dim=1).tolist())
    sparse_metrics = f1_metrics(y_true, sparse_scores.argmax(dim=1).tolist())
    combined = base_scores + args.sparse_weight * sparse_scores
    combined_metrics = f1_metrics(y_true, combined.argmax(dim=1).tolist())

    print(f"train_rows={len(train_idx)} val_rows={len(ids)} feature_count={x_train.shape[1]}")
    print(f"base_macro_f1={base_metrics['macro_f1']:.6f}")
    print(f"rules_macro_f1={rules_metrics['macro_f1']:.6f}")
    print(f"sparse_macro_f1={sparse_metrics['macro_f1']:.6f}")
    print(f"combined_macro_f1={combined_metrics['macro_f1']:.6f}")
    print("combined_weakest=" + ";".join(
        f"{name}:{score:.4f}"
        for name, score in sorted(combined_metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:6]
    ))
    if applied_rules:
        print("applied_rules=" + json.dumps(applied_rules, ensure_ascii=False))


if __name__ == "__main__":
    main()
