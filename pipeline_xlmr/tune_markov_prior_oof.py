import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from script import ALL_CLASSES, load_jsonl, prompt_intent_tokens, safe_text
from train import append_results_csv, f1_metrics, load_labels, predict_with_bias, session_id, tune_class_bias
from tune_oof_rule_boosts import torch_load
from evaluate_rule_boosts import apply_rules


CLASS_TO_ID = {label: idx for idx, label in enumerate(ALL_CLASSES)}


def load_samples_and_labels(data_dir):
    data_dir = Path(data_dir)
    samples = load_jsonl(data_dir / "train.jsonl")
    labels_by_id = load_labels(data_dir / "train_labels.csv")
    y = [CLASS_TO_ID[labels_by_id[sample["id"]]] for sample in samples]
    samples_by_id = {safe_text(sample.get("id")): sample for sample in samples}
    return samples, y, samples_by_id


def bin_numeric_flag(name, value, bins):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"{name}:missing"
    for label, upper in bins:
        if numeric <= upper:
            return f"{name}:{label}"
    return f"{name}:hi"


def top_language(sample):
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict) and language_mix:
        return safe_text(sorted(language_mix.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]).lower()
    return "missing"


def action_sequence(sample):
    names = []
    for event in sample.get("history") or []:
        if event.get("role") == "assistant_action":
            name = safe_text(event.get("name"))
            if name:
                names.append(name)
    return names


def sample_prior_features(sample):
    names = action_sequence(sample)
    prompt = safe_text(sample.get("current_prompt", ""))
    sm = sample.get("session_meta") or {}
    features = []
    if names:
        features.append(f"last_action:{names[-1]}")
        for name in names[-3:]:
            features.append(f"recent_action:{name}")
        if len(names) >= 2:
            features.append(f"last_pair:{names[-2]}>{names[-1]}")
        if len(names) >= 3:
            features.append(f"last_triple:{names[-3]}>{names[-2]}>{names[-1]}")
    else:
        features.append("last_action:none")

    intents = prompt_intent_tokens(prompt)
    for intent in intents:
        features.append(f"intent:{intent}")
    if names:
        for intent in intents:
            features.append(f"last_action_intent:{names[-1]}|{intent}")
        if len(names) >= 2:
            for intent in intents:
                features.append(f"last_pair_intent:{names[-2]}>{names[-1]}|{intent}")

    features.append(f"top_lang:{top_language(sample)}")
    features.append(bin_numeric_flag("turn", sm.get("turn_index"), [("01", 1), ("02", 2), ("04", 4), ("08", 8), ("12", 12)]))
    features.append(f"hist_len:{min(len(sample.get('history') or []), 12)}")
    return features


def fit_prior_tables(samples, y, train_indices, min_support, smoothing):
    global_counts = torch.full((len(ALL_CLASSES),), float(smoothing), dtype=torch.float32)
    feature_counts = defaultdict(lambda: torch.full((len(ALL_CLASSES),), float(smoothing), dtype=torch.float32))
    raw_support = Counter()
    for idx in train_indices:
        label_id = y[idx]
        global_counts[label_id] += 1.0
        for feature in set(sample_prior_features(samples[idx])):
            feature_counts[feature][label_id] += 1.0
            raw_support[feature] += 1

    global_logp = torch.log(global_counts / global_counts.sum())
    feature_deltas = {}
    for feature, counts in feature_counts.items():
        if raw_support[feature] < min_support:
            continue
        feature_logp = torch.log(counts / counts.sum())
        feature_deltas[feature] = feature_logp - global_logp
    return global_logp, feature_deltas


def predict_prior_scores(samples, ids, samples_by_id, global_logp, feature_deltas, feature_weight):
    rows = []
    missing = []
    for sample_id in ids:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            missing.append(sample_id)
            rows.append(global_logp.clone())
            continue
        score = global_logp.clone()
        used = 0
        for feature in set(sample_prior_features(sample)):
            delta = feature_deltas.get(feature)
            if delta is not None:
                score = score + feature_weight * delta
                used += 1
        rows.append(score)
    if missing:
        raise ValueError(f"missing OOF samples: {missing[:5]}")
    return torch.stack(rows)


def load_oof_payloads(oof_artifact):
    artifact = json.loads(Path(oof_artifact).read_text(encoding="utf-8"))
    payloads = []
    seen_ids = set()
    for path in artifact["fold_logits"]:
        payload = torch_load(path)
        overlap = seen_ids & set(payload["ids"])
        if overlap:
            raise ValueError(f"duplicate OOF ids in {path}: {sorted(overlap)[:5]}")
        seen_ids.update(payload["ids"])
        payloads.append((Path(path), payload))
    return artifact, payloads


def bias_from_artifact(artifact):
    class_bias = artifact.get("class_bias", {})
    return torch.tensor([float(class_bias.get(label, 0.0)) for label in ALL_CLASSES], dtype=torch.float32)


def build_fold_aware_prior(payloads, samples, y, samples_by_id, min_support, smoothing, feature_weight):
    prior_parts = []
    ids_all = []
    y_all = []
    fold_summaries = []
    all_sessions = [session_id(sample.get("id", "")) for sample in samples]
    for path, payload in payloads:
        val_ids = [safe_text(sample_id) for sample_id in payload["ids"]]
        val_sessions = {session_id(sample_id) for sample_id in val_ids}
        train_indices = [idx for idx, sess in enumerate(all_sessions) if sess not in val_sessions]
        global_logp, feature_deltas = fit_prior_tables(samples, y, train_indices, min_support, smoothing)
        prior = predict_prior_scores(samples, val_ids, samples_by_id, global_logp, feature_deltas, feature_weight)
        prior_parts.append(prior)
        ids_all.extend(val_ids)
        y_all.extend(int(label) for label in payload["y_true"])
        fold_summaries.append(
            {
                "path": str(path),
                "fold_id": payload.get("fold_id"),
                "train_rows": len(train_indices),
                "val_rows": len(val_ids),
                "feature_count": len(feature_deltas),
                "val_session_count": len(val_sessions),
            }
        )
    return torch.cat(prior_parts, dim=0), torch.tensor(y_all, dtype=torch.long), ids_all, fold_summaries


def load_transformer_scores(artifact, payloads, rule_artifact_path, samples_by_id):
    logits = torch.cat([payload["logits"].float() for _, payload in payloads], dim=0)
    ids = [sample_id for _, payload in payloads for sample_id in payload["ids"]]
    bias = bias_from_artifact(artifact)
    scores = logits + bias
    rules = []
    if rule_artifact_path:
        rule_payload = json.loads(Path(rule_artifact_path).read_text(encoding="utf-8"))
        rules = rule_payload.get("rules", [])
        scores, _ = apply_rules(scores, ids, samples_by_id, rules)
    return scores, ids, rules


def score_metrics(y_true, scores):
    pred = scores.argmax(dim=1).tolist()
    return f1_metrics(y_true.tolist(), pred)


def tune_prior_weight(base_scores, prior_scores, y_true, weights, tune_bias):
    best = None
    for weight in weights:
        combined = base_scores + weight * prior_scores
        raw_metrics = score_metrics(y_true, combined)
        bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
        metrics = raw_metrics
        if tune_bias:
            bias, _ = tune_class_bias(combined, y_true.tolist(), rounds=2)
            pred = predict_with_bias(combined, bias)
            metrics = f1_metrics(y_true.tolist(), pred)
        candidate = {
            "prior_weight": weight,
            "macro_f1_raw": raw_metrics["macro_f1"],
            "macro_f1": metrics["macro_f1"],
            "class_bias": [float(x) for x in bias.tolist()],
            "metrics": metrics,
        }
        if best is None or candidate["macro_f1"] > best["macro_f1"]:
            best = candidate
        print(
            f"prior_weight={weight:.3f} raw={raw_metrics['macro_f1']:.6f} "
            f"metric={metrics['macro_f1']:.6f}",
            flush=True,
        )
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
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--smoothing", type=float, default=2.0)
    parser.add_argument("--feature-weight", type=float, default=0.35)
    parser.add_argument("--weights", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    artifact, payloads = load_oof_payloads(args.oof_artifact)
    samples, y, samples_by_id = load_samples_and_labels(args.data_dir)
    prior_scores, y_true, prior_ids, fold_summaries = build_fold_aware_prior(
        payloads,
        samples,
        y,
        samples_by_id,
        args.min_support,
        args.smoothing,
        args.feature_weight,
    )
    base_scores, score_ids, rules = load_transformer_scores(artifact, payloads, args.rule_artifact, samples_by_id)
    if prior_ids != score_ids:
        raise ValueError("prior ids and transformer ids differ")

    prior_metrics = score_metrics(y_true, prior_scores)
    base_metrics = score_metrics(y_true, base_scores)
    print(f"prior_only_macro_f1={prior_metrics['macro_f1']:.6f}", flush=True)
    print(f"base_macro_f1={base_metrics['macro_f1']:.6f}", flush=True)
    weights = [float(value) for value in args.weights.split(",") if value.strip()]
    best = tune_prior_weight(base_scores, prior_scores, y_true, weights, args.tune_bias)

    output_dir = Path("experiments/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.experiment_id}_markov_prior.json"
    payload = {
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_oof_artifact": args.oof_artifact,
        "source_rule_artifact": args.rule_artifact,
        "row_count": len(prior_ids),
        "classes": ALL_CLASSES,
        "args": vars(args),
        "fold_summaries": fold_summaries,
        "prior_only_metrics": prior_metrics,
        "base_metrics": base_metrics,
        "metrics": best["metrics"],
        "best_prior_weight": best["prior_weight"],
        "best_raw_macro_f1": best["macro_f1_raw"],
        "best_class_bias": dict(zip(ALL_CLASSES, best["class_bias"])),
        "rule_count": len(rules),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    append_results_csv(
        Path("experiments/results.csv"),
        {
            "experiment_id": args.experiment_id,
            "model_family": "torch_gpu_transformer_oof_markov_prior",
            "features": "OOF transformer logits + fold-aware Markov/action prior",
            "split_type": "session_oof",
            "fold_id": "oof",
            "macro_f1_raw": f"{base_metrics['macro_f1']:.6f}",
            "macro_f1_bias_tuned": f"{best['macro_f1']:.6f}",
            "macro_f1": f"{best['macro_f1']:.6f}",
            "weakest_classes": summarize_weak(best["metrics"]),
            "top_confusions": json.dumps(best["metrics"]["top_confusions"][:8], ensure_ascii=False),
            "prediction_distribution": json.dumps(best["metrics"]["prediction_distribution"], ensure_ascii=False, sort_keys=True),
            "artifact_path": str(output_path),
            "val_logits_path": ";".join(str(path) for path, _ in payloads),
            "notes": args.notes,
            "decision": "markov prior oof diagnostic complete",
        },
    )

    lines = [
        f"## {args.experiment_id}",
        "",
        f"- Date/time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "- Validation setup: fold-aware Markov/action prior evaluated on 3-fold session-aware OOF logits.",
        f"- Base Macro-F1: {base_metrics['macro_f1']:.6f}",
        f"- Prior-only Macro-F1: {prior_metrics['macro_f1']:.6f}",
        f"- Best prior weight: {best['prior_weight']:.3f}",
        f"- Best Macro-F1: {best['macro_f1']:.6f}",
        f"- Weakest classes: {summarize_weak(best['metrics']).replace(';', ', ')}",
        f"- Top confusions: {best['metrics']['top_confusions'][:8]}",
        f"- Artifact: {output_path}",
        f"- Decision: {args.notes or 'compare against rule-boost baseline before integration'}",
        "",
    ]
    with Path("research_log.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"best_prior_weight={best['prior_weight']:.3f}")
    print(f"best_macro_f1={best['macro_f1']:.6f}")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
