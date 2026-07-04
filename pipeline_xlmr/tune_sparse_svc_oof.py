import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from script import ALL_CLASSES, load_jsonl, safe_text, serialize_transformer_sample_current
from train import CLASS_TO_ID, append_results_csv, f1_metrics, load_labels, predict_with_bias, session_id, tune_class_bias
from tune_oof_rule_boosts import torch_load
from evaluate_rule_boosts import apply_rules


def load_samples_and_labels(data_dir):
    data_dir = Path(data_dir)
    samples = load_jsonl(data_dir / "train.jsonl")
    labels_by_id = load_labels(data_dir / "train_labels.csv")
    y = np.array([CLASS_TO_ID[labels_by_id[sample["id"]]] for sample in samples], dtype=np.int64)
    samples_by_id = {safe_text(sample.get("id")): sample for sample in samples}
    sample_index_by_id = {safe_text(sample.get("id")): idx for idx, sample in enumerate(samples)}
    return samples, y, samples_by_id, sample_index_by_id


SPARSE_TEXT_MODE = "current_v1"


def sparse_text(sample):
    # 텍스트 다양화 (팀 로컬 과제): current_v1 / state_v2 / compact_events_v1 / union
    from script import serialize_transformer_sample
    if SPARSE_TEXT_MODE == "union":
        return "\n".join(
            serialize_transformer_sample(sample, name)
            for name in ("current_v1", "state_v2", "compact_events_v1")
        )
    if SPARSE_TEXT_MODE == "current_v1":
        return serialize_transformer_sample_current(sample)
    return serialize_transformer_sample(sample, SPARSE_TEXT_MODE)


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


def load_oof_payloads(oof_artifact):
    artifact = json.loads(Path(oof_artifact).read_text(encoding="utf-8"))
    payloads = []
    seen = set()
    for path in artifact["fold_logits"]:
        payload = torch_load(path)
        overlap = seen & set(payload["ids"])
        if overlap:
            raise ValueError(f"duplicate OOF ids in {path}: {sorted(overlap)[:5]}")
        seen.update(payload["ids"])
        payloads.append((Path(path), payload))
    return artifact, payloads


def bias_from_artifact(artifact):
    class_bias = artifact.get("class_bias", {})
    return torch.tensor([float(class_bias.get(label, 0.0)) for label in ALL_CLASSES], dtype=torch.float32)


def load_transformer_scores(artifact, payloads, rule_artifact_path, samples_by_id):
    logits = torch.cat([payload["logits"].float() for _, payload in payloads], dim=0)
    ids = [safe_text(sample_id) for _, payload in payloads for sample_id in payload["ids"]]
    y_true = torch.tensor([int(label) for _, payload in payloads for label in payload["y_true"]], dtype=torch.long)
    scores = logits + bias_from_artifact(artifact)
    rules = []
    if rule_artifact_path:
        rule_payload = json.loads(Path(rule_artifact_path).read_text(encoding="utf-8"))
        rules = rule_payload.get("rules", [])
        scores, _ = apply_rules(scores, ids, samples_by_id, rules)
    return scores, ids, y_true, rules


def normalize_scores(train_scores, val_scores):
    mean = train_scores.mean(axis=0, keepdims=True)
    std = train_scores.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (val_scores - mean) / std


def fit_predict_fold(args, samples, y, payload, all_session_ids):
    val_ids = [safe_text(sample_id) for sample_id in payload["ids"]]
    val_sessions = {session_id(sample_id) for sample_id in val_ids}
    train_idx = [idx for idx, sess in enumerate(all_session_ids) if sess not in val_sessions]
    sample_index_by_id = {safe_text(sample.get("id")): idx for idx, sample in enumerate(samples)}
    val_idx = [sample_index_by_id[sample_id] for sample_id in val_ids]

    vectorizer = make_vectorizer(args)
    train_texts = [sparse_text(samples[idx]) for idx in train_idx]
    val_texts = [sparse_text(samples[idx]) for idx in val_idx]
    fit_start = time.perf_counter()
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    model = LinearSVC(
        C=args.c,
        class_weight="balanced" if args.class_weight == "balanced" else None,
        dual="auto",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    model.fit(x_train, y[train_idx])
    train_scores = model.decision_function(x_train)
    val_scores = model.decision_function(x_val)
    if val_scores.ndim == 1:
        raise ValueError("expected multiclass decision_function scores")
    if args.normalize:
        val_scores = normalize_scores(train_scores, val_scores)
    elapsed = time.perf_counter() - fit_start
    return {
        "scores": torch.tensor(val_scores, dtype=torch.float32),
        "ids": val_ids,
        "y_true": torch.tensor([int(label) for label in payload["y_true"]], dtype=torch.long),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "elapsed_sec": elapsed,
        "feature_count": int(x_train.shape[1]),
    }


def build_sparse_oof(args, payloads, samples, y):
    all_session_ids = [session_id(sample.get("id", "")) for sample in samples]
    fold_outputs = []
    for path, payload in payloads:
        print(f"fitting sparse fold={payload.get('fold_id')} path={path}", flush=True)
        output = fit_predict_fold(args, samples, y, payload, all_session_ids)
        fold_outputs.append(output)
        pred = output["scores"].argmax(dim=1).tolist()
        metrics = f1_metrics(output["y_true"].tolist(), pred)
        print(
            f"  fold={payload.get('fold_id')} macro={metrics['macro_f1']:.6f} "
            f"features={output['feature_count']} elapsed={output['elapsed_sec']:.1f}s",
            flush=True,
        )
    scores = torch.cat([fold["scores"] for fold in fold_outputs], dim=0)
    y_true = torch.cat([fold["y_true"] for fold in fold_outputs], dim=0)
    ids = [sample_id for fold in fold_outputs for sample_id in fold["ids"]]
    summaries = [
        {
            "train_rows": fold["train_rows"],
            "val_rows": fold["val_rows"],
            "elapsed_sec": fold["elapsed_sec"],
            "feature_count": fold["feature_count"],
        }
        for fold in fold_outputs
    ]
    return scores, y_true, ids, summaries


def score_metrics(y_true, scores):
    pred = scores.argmax(dim=1).tolist()
    return f1_metrics(y_true.tolist(), pred)


def tune_ensemble(base_scores, sparse_scores, y_true, weights, tune_bias):
    best = None
    for weight in weights:
        combined = base_scores + weight * sparse_scores
        raw_metrics = score_metrics(y_true, combined)
        bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
        metrics = raw_metrics
        if tune_bias:
            bias, _ = tune_class_bias(combined, y_true.tolist(), rounds=2)
            pred = predict_with_bias(combined, bias)
            metrics = f1_metrics(y_true.tolist(), pred)
        candidate = {
            "sparse_weight": weight,
            "macro_f1_raw": raw_metrics["macro_f1"],
            "macro_f1": metrics["macro_f1"],
            "class_bias": [float(value) for value in bias.tolist()],
            "metrics": metrics,
        }
        if best is None or candidate["macro_f1"] > best["macro_f1"]:
            best = candidate
        print(f"sparse_weight={weight:.3f} raw={raw_metrics['macro_f1']:.6f} metric={metrics['macro_f1']:.6f}", flush=True)
    return best


def summarize_weak(metrics, count=5):
    return ";".join(
        f"{name}:{score:.4f}"
        for name, score in sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:count]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-artifact", required=True)
    parser.add_argument("--rule-artifact", default="")
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--word-features", type=int, default=180000)
    parser.add_argument("--char-features", type=int, default=220000)
    parser.add_argument("--word-min-df", type=int, default=2)
    parser.add_argument("--char-min-df", type=int, default=2)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--weights", default="0,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparse-text", choices=["current_v1", "state_v2", "compact_events_v1", "union"],
                        default="current_v1")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    global SPARSE_TEXT_MODE
    SPARSE_TEXT_MODE = args.sparse_text

    artifact, payloads = load_oof_payloads(args.oof_artifact)
    samples, y, samples_by_id, _ = load_samples_and_labels(args.data_dir)
    sparse_scores, y_true, sparse_ids, fold_summaries = build_sparse_oof(args, payloads, samples, y)
    base_scores, base_ids, base_y, rules = load_transformer_scores(artifact, payloads, args.rule_artifact, samples_by_id)
    if sparse_ids != base_ids:
        raise ValueError("sparse OOF ids and transformer OOF ids differ")
    if not torch.equal(y_true, base_y):
        raise ValueError("sparse OOF labels and transformer OOF labels differ")

    sparse_metrics = score_metrics(y_true, sparse_scores)
    base_metrics = score_metrics(y_true, base_scores)
    print(f"sparse_oof_macro_f1={sparse_metrics['macro_f1']:.6f}", flush=True)
    print(f"base_oof_macro_f1={base_metrics['macro_f1']:.6f}", flush=True)
    weights = [float(value) for value in args.weights.split(",") if value.strip()]
    best = tune_ensemble(base_scores, sparse_scores, y_true, weights, args.tune_bias)

    output_dir = Path("experiments/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.experiment_id}_sparse_svc.json"
    logits_dir = Path("experiments/logits")
    logits_dir.mkdir(parents=True, exist_ok=True)
    sparse_logits_path = logits_dir / f"{args.experiment_id}_sparse_oof_logits.pt"
    torch.save(
        {
            "scores": sparse_scores,
            "ids": sparse_ids,
            "y_true": y_true,
            "classes": ALL_CLASSES,
            "metrics": sparse_metrics,
            "args": vars(args),
        },
        sparse_logits_path,
    )

    payload = {
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_oof_artifact": args.oof_artifact,
        "source_rule_artifact": args.rule_artifact,
        "row_count": len(sparse_ids),
        "classes": ALL_CLASSES,
        "args": vars(args),
        "fold_summaries": fold_summaries,
        "sparse_metrics": sparse_metrics,
        "base_metrics": base_metrics,
        "metrics": best["metrics"],
        "best_sparse_weight": best["sparse_weight"],
        "best_raw_macro_f1": best["macro_f1_raw"],
        "best_class_bias": dict(zip(ALL_CLASSES, best["class_bias"])),
        "sparse_logits_path": str(sparse_logits_path),
        "rule_count": len(rules),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    append_results_csv(
        Path("experiments/results.csv"),
        {
            "experiment_id": args.experiment_id,
            "model_family": "tfidf_linearsvc_oof_ensemble",
            "features": "OOF transformer logits + fold-aware TF-IDF LinearSVC scores",
            "split_type": "session_oof",
            "fold_id": "oof",
            "macro_f1_raw": f"{base_metrics['macro_f1']:.6f}",
            "macro_f1_bias_tuned": f"{best['macro_f1']:.6f}",
            "macro_f1": f"{best['macro_f1']:.6f}",
            "weakest_classes": summarize_weak(best["metrics"]),
            "top_confusions": json.dumps(best["metrics"]["top_confusions"][:8], ensure_ascii=False),
            "prediction_distribution": json.dumps(best["metrics"]["prediction_distribution"], ensure_ascii=False, sort_keys=True),
            "artifact_path": str(output_path),
            "val_logits_path": str(sparse_logits_path),
            "notes": args.notes,
            "decision": "sparse svc oof ensemble diagnostic complete",
        },
    )

    lines = [
        f"## {args.experiment_id}",
        "",
        f"- Date/time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "- Validation setup: fold-aware TF-IDF LinearSVC OOF scores ensembled with current finalist transformer logits.",
        f"- Base Macro-F1: {base_metrics['macro_f1']:.6f}",
        f"- Sparse-only Macro-F1: {sparse_metrics['macro_f1']:.6f}",
        f"- Best sparse weight: {best['sparse_weight']:.3f}",
        f"- Best Macro-F1: {best['macro_f1']:.6f}",
        f"- Weakest classes: {summarize_weak(best['metrics']).replace(';', ', ')}",
        f"- Top confusions: {best['metrics']['top_confusions'][:8]}",
        f"- Artifact: {output_path}",
        f"- Sparse logits: {sparse_logits_path}",
        f"- Decision: {args.notes or 'compare against rule-boost baseline before integration'}",
        "",
    ]
    with Path("research_log.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"best_sparse_weight={best['sparse_weight']:.3f}")
    print(f"best_macro_f1={best['macro_f1']:.6f}")
    print(f"saved {output_path}")
    print(f"saved {sparse_logits_path}")


if __name__ == "__main__":
    main()
