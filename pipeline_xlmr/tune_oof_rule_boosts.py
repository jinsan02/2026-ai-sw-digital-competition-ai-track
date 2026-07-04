import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from script import ALL_CLASSES, load_jsonl, prompt_intent_tokens, safe_text
from train import append_results_csv, f1_metrics, load_labels


CLASS_TO_ID = {label: idx for idx, label in enumerate(ALL_CLASSES)}
TARGET_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "web_search",
    "lint_or_typecheck",
    "run_tests",
    "run_bash",
    "ask_user",
    "plan_task",
]


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_oof_logits(paths):
    payloads = []
    seen = set()
    for path in paths:
        payload = torch_load(path)
        dupes = sorted(set(payload["ids"]) & seen)
        if dupes:
            raise ValueError(f"duplicate OOF ids in {path}: {dupes[:5]}")
        seen.update(payload["ids"])
        payloads.append(payload)
    logits = torch.cat([payload["logits"].float() for payload in payloads], dim=0)
    y_true = torch.tensor(
        [int(label) for payload in payloads for label in payload["y_true"]],
        dtype=torch.long,
    )
    ids = [sample_id for payload in payloads for sample_id in payload["ids"]]
    return logits, y_true, ids


def confusion_from_pred(y_true, y_pred):
    num_classes = len(ALL_CLASSES)
    flat = y_true * num_classes + y_pred
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).to(torch.int64)


def macro_from_confusion(confusion):
    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))
    return float(f1.mean().item())


def apply_confusion_delta(confusion, y_true_subset, old_pred, new_pred):
    changed = new_pred != old_pred
    if not bool(changed.any()):
        return confusion, 0
    y_changed = y_true_subset[changed]
    old_changed = old_pred[changed]
    new_changed = new_pred[changed]
    updated = confusion.clone()
    for true_id, pred_id in zip(y_changed.tolist(), old_changed.tolist()):
        updated[true_id, pred_id] -= 1
    for true_id, pred_id in zip(y_changed.tolist(), new_changed.tolist()):
        updated[true_id, pred_id] += 1
    return updated, int(changed.sum().item())


def eval_candidate(scores, y_true, current_pred, current_confusion, indices, target_id, boost):
    idx = indices
    old_pred = current_pred[idx]
    sub_scores = scores[idx].clone()
    sub_scores[:, target_id] += boost
    new_pred = sub_scores.argmax(dim=1)
    new_confusion, changed = apply_confusion_delta(current_confusion, y_true[idx], old_pred, new_pred)
    if changed == 0:
        return None
    return macro_from_confusion(new_confusion), new_confusion, new_pred, changed


def bin_numeric_flag(name, value, bins):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"{name}:missing"
    for label, upper in bins:
        if numeric <= upper:
            return f"{name}:{label}"
    return f"{name}:hi"


def text_has(pattern, text):
    return bool(re.search(pattern, text))


def path_like_tokens(text):
    text = safe_text(text).replace("\\", "/").lower()
    tokens = []
    if not text:
        return tokens
    if "/" in text or re.search(r"\.[a-z0-9]{1,6}\b", text):
        tokens.append("arg_has_path")
    for ext in ("py", "js", "ts", "tsx", "json", "md", "yml", "yaml", "toml", "sh", "txt", "csv"):
        if re.search(rf"\.{ext}\b", text):
            tokens.append(f"arg_ext:{ext}")
    if "*" in text:
        tokens.append("arg_has_glob")
    return tokens


def base_feature_flags(sample):
    prompt = safe_text(sample.get("current_prompt", ""))
    prompt_l = prompt.lower()
    flags = set()
    for token in prompt_intent_tokens(prompt):
        flags.add(f"prompt_{token}")
    if "?" in prompt:
        flags.add("prompt_has_question")
    if text_has(r"\b(open|show|read|inspect|look at|cat|view)\b|열어|보여|읽어", prompt_l):
        flags.add("prompt_read_words")
    if text_has(r"\b(grep|search|find|lookup|occurrence|references?)\b|검색|찾아", prompt_l):
        flags.add("prompt_search_words")
    if text_has(r"\b(list|ls|tree|directory|folder|files?)\b|목록", prompt_l):
        flags.add("prompt_list_words")
    if text_has(r"\b(glob|pattern|\*\.[a-z0-9]+|\*)\b", prompt_l):
        flags.add("prompt_glob_words")
    if text_has(r"\b(web|internet|browser|online|latest|docs?|documentation)\b|웹|인터넷", prompt_l):
        flags.add("prompt_web_words")
    if text_has(r"\b(lint|typecheck|type-check|mypy|pyright|tsc|ruff|eslint)\b", prompt_l):
        flags.add("prompt_lint_words")
    if text_has(r"\b(test|tests|pytest|jest|vitest|spec)\b", prompt_l):
        flags.add("prompt_test_words")
    if text_has(r"\b(run|execute|shell|terminal|build|install|npm|pip|docker)\b", prompt_l):
        flags.add("prompt_run_words")
    for token in path_like_tokens(prompt_l):
        flags.add(f"prompt_{token}")

    history = sample.get("history") or []
    action_names = []
    result_tokens = []
    arg_tokens = []
    for event in history:
        if event.get("role") != "assistant_action":
            continue
        name = safe_text(event.get("name"))
        if not name:
            continue
        action_names.append(name)
        args = event.get("args") or {}
        if isinstance(args, dict):
            for value in list(args.values())[:6]:
                arg_tokens.extend(path_like_tokens(value))
        result = safe_text(event.get("result_summary", "")).lower()
        if result:
            if text_has(r"\b(found|match|matches|occurrences?)\b", result):
                result_tokens.append("result_found")
            if text_has(r"\b(no matches|not found|empty|0 matches)\b", result):
                result_tokens.append("result_none")
            if text_has(r"\b(error|failed|traceback|exception)\b", result):
                result_tokens.append("result_failed")
            if text_has(r"\b(pass|passed|success|ok)\b", result):
                result_tokens.append("result_passed")
            if text_has(r"\b(exit code|return code)\b", result):
                result_tokens.append("result_exit_code")

    flags.add(f"hist_len:{min(len(history), 12)}")
    if action_names:
        flags.add(f"last_action:{action_names[-1]}")
        for name in action_names[-4:]:
            flags.add(f"recent_action:{name}")
        if len(action_names) >= 2:
            flags.add(f"last_pair:{action_names[-2]}>{action_names[-1]}")
    for token in set(arg_tokens):
        flags.add(token)
    for token in set(result_tokens):
        flags.add(token)

    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    flags.add(bin_numeric_flag("turn", sm.get("turn_index"), [("01", 1), ("02", 2), ("04", 4), ("08", 8), ("12", 12)]))
    flags.add(f"dirty:{safe_text(ws.get('git_dirty', 'missing')).lower()}")
    flags.add(f"ci:{safe_text(ws.get('last_ci_status', 'missing')).lower()}")
    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict) and language_mix:
        top_lang = sorted(language_mix.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
        flags.add(f"top_lang:{safe_text(top_lang).lower()}")
    open_files = ws.get("open_files") or []
    if open_files:
        flags.add("has_open_files")
        for path in open_files[:6]:
            for token in path_like_tokens(path):
                flags.add(f"open_{token}")
    return flags


def feature_flags(sample, base_scores_row):
    flags = base_feature_flags(sample)
    top_values, top_indices = torch.topk(base_scores_row, k=2)
    top = ALL_CLASSES[int(top_indices[0])]
    second = ALL_CLASSES[int(top_indices[1])]
    margin = float(top_values[0] - top_values[1])
    logit_flags = {
        f"base_top:{top}",
        f"base_second:{second}",
        f"base_pair:{top}>{second}",
    }
    for threshold in (0.25, 0.50, 1.00):
        if margin <= threshold:
            logit_flags.add(f"margin_le:{threshold:.2f}")
    composite_sources = [
        flag for flag in flags
        if flag.startswith(("prompt_", "last_action:", "recent_action:", "last_pair:", "arg_", "open_arg_", "top_lang:", "result_"))
    ]
    composite_flags = set()
    for flag in composite_sources:
        composite_flags.add(f"{flag}|top:{top}")
        composite_flags.add(f"{flag}|pair:{top}>{second}")
    return flags | logit_flags | composite_flags


def build_feature_index(samples_by_id, ids, base_scores, min_support, max_support_ratio):
    feature_to_indices = defaultdict(list)
    missing = []
    for row_idx, sample_id in enumerate(ids):
        sample = samples_by_id.get(sample_id)
        if sample is None:
            missing.append(sample_id)
            continue
        for feature in feature_flags(sample, base_scores[row_idx]):
            feature_to_indices[feature].append(row_idx)
    if missing:
        raise ValueError(f"OOF ids missing from train.jsonl: {missing[:5]}")

    max_support = int(len(ids) * max_support_ratio)
    kept = {}
    for feature, indices in feature_to_indices.items():
        if min_support <= len(indices) <= max_support:
            kept[feature] = torch.tensor(indices, dtype=torch.long)
    return kept


def rank_feature_index(features, y_true, base_pred, max_features):
    if not max_features or len(features) <= max_features:
        return features
    ranked = []
    errors = y_true != base_pred
    for feature, indices in features.items():
        support = int(indices.numel())
        error_count = int(errors[indices].sum().item())
        error_rate = error_count / max(1, support)
        ranked.append((error_count * error_rate, error_count, support, feature))
    ranked.sort(reverse=True)
    keep = {feature for _, _, _, feature in ranked[:max_features]}
    return {feature: indices for feature, indices in features.items() if feature in keep}


def load_samples_by_id(data_dir):
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    return {safe_text(sample.get("id")): sample for sample in samples}, labels


def choose_candidate_targets(feature_indices, y_true, current_pred, min_target_count):
    targets = set()
    y_counts = Counter(y_true[feature_indices].tolist())
    pred_counts = Counter(current_pred[feature_indices].tolist())
    for class_name in TARGET_CLASSES:
        class_id = CLASS_TO_ID[class_name]
        if y_counts[class_id] >= min_target_count or pred_counts[class_id] >= min_target_count:
            targets.add(class_id)
    return sorted(targets)


def tune_rules(scores, y_true, features, boost_values, max_rules, min_gain, min_target_count):
    current_scores = scores.clone()
    current_pred = current_scores.argmax(dim=1)
    current_confusion = confusion_from_pred(y_true, current_pred)
    current_macro = macro_from_confusion(current_confusion)
    selected = []
    used = set()

    for round_idx in range(1, max_rules + 1):
        best = None
        for feature_name, indices in features.items():
            target_ids = choose_candidate_targets(indices, y_true, current_pred, min_target_count)
            for target_id in target_ids:
                for boost in boost_values:
                    key = (feature_name, target_id, boost)
                    if key in used:
                        continue
                    outcome = eval_candidate(current_scores, y_true, current_pred, current_confusion, indices, target_id, boost)
                    if outcome is None:
                        continue
                    macro, new_confusion, new_pred_subset, changed = outcome
                    gain = macro - current_macro
                    if gain < min_gain:
                        continue
                    if best is None or gain > best["gain"]:
                        best = {
                            "feature": feature_name,
                            "target_id": target_id,
                            "target": ALL_CLASSES[target_id],
                            "boost": boost,
                            "macro_f1": macro,
                            "gain": gain,
                            "changed": changed,
                            "support": int(indices.numel()),
                            "indices": indices,
                            "confusion": new_confusion,
                            "pred_subset": new_pred_subset,
                            "key": key,
                        }
        if best is None:
            break
        used.add(best["key"])
        indices = best["indices"]
        current_scores[indices, best["target_id"]] += best["boost"]
        current_pred[indices] = best["pred_subset"]
        current_confusion = best["confusion"]
        current_macro = best["macro_f1"]
        selected.append(
            {
                "round": round_idx,
                "feature": best["feature"],
                "target": best["target"],
                "boost": best["boost"],
                "support": best["support"],
                "changed": best["changed"],
                "gain": best["gain"],
                "macro_f1": best["macro_f1"],
            }
        )
        print(
            f"rule round={round_idx} target={best['target']} boost={best['boost']:+.2f} "
            f"gain={best['gain']:.6f} macro={best['macro_f1']:.6f} "
            f"support={best['support']} changed={best['changed']} feature={best['feature']}"
            ,
            flush=True,
        )
    return selected, current_scores, current_pred, current_macro


def summarize_weak(metrics, count=5):
    return ";".join(
        f"{name}:{score:.4f}"
        for name, score in sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:count]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-artifact", required=True)
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--min-support", type=int, default=40)
    parser.add_argument("--max-support-ratio", type=float, default=0.60)
    parser.add_argument("--min-target-count", type=int, default=8)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--min-gain", type=float, default=0.00005)
    parser.add_argument("--boost-values", default="-0.60,-0.40,-0.20,0.20,0.40,0.60")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    artifact = json.loads(Path(args.oof_artifact).read_text(encoding="utf-8"))
    fold_paths = [Path(path) for path in artifact["fold_logits"]]
    logits, y_true, ids = load_oof_logits(fold_paths)
    bias_values = [float(artifact.get("class_bias", {}).get(label, 0.0)) for label in ALL_CLASSES]
    bias = torch.tensor(bias_values, dtype=torch.float32)
    base_scores = logits + bias
    base_pred = base_scores.argmax(dim=1)
    base_metrics = f1_metrics(y_true.tolist(), base_pred.tolist())
    print(f"baseline_oof_macro_f1={base_metrics['macro_f1']:.6f}", flush=True)

    samples_by_id, _ = load_samples_by_id(args.data_dir)
    features = build_feature_index(samples_by_id, ids, base_scores, args.min_support, args.max_support_ratio)
    original_feature_count = len(features)
    features = rank_feature_index(features, y_true, base_pred, args.max_features)
    print(
        f"candidate_features={len(features)} original_features={original_feature_count} "
        f"min_support={args.min_support}",
        flush=True,
    )
    boost_values = [float(value) for value in args.boost_values.split(",") if value.strip()]
    rules, boosted_scores, boosted_pred, boosted_macro = tune_rules(
        base_scores,
        y_true,
        features,
        boost_values,
        args.max_rules,
        args.min_gain,
        args.min_target_count,
    )
    boosted_metrics = f1_metrics(y_true.tolist(), boosted_pred.tolist())
    print(f"boosted_oof_macro_f1={boosted_metrics['macro_f1']:.6f}", flush=True)

    output_dir = Path("experiments/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.experiment_id}_rule_boosts.json"
    payload = {
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_oof_artifact": args.oof_artifact,
        "fold_logits": [str(path) for path in fold_paths],
        "row_count": len(ids),
        "classes": ALL_CLASSES,
        "class_bias": dict(zip(ALL_CLASSES, bias_values)),
        "base_metrics": base_metrics,
        "metrics": boosted_metrics,
        "rules": rules,
        "args": vars(args),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    append_results_csv(
        Path("experiments/results.csv"),
        {
            "experiment_id": args.experiment_id,
            "model_family": "torch_gpu_transformer_oof_rule_boost",
            "features": "OOF logits + deterministic sample/logit rule boosts",
            "split_type": "session_oof",
            "fold_id": "oof",
            "macro_f1_raw": f"{base_metrics['macro_f1']:.6f}",
            "macro_f1_bias_tuned": f"{boosted_metrics['macro_f1']:.6f}",
            "macro_f1": f"{boosted_metrics['macro_f1']:.6f}",
            "weakest_classes": summarize_weak(boosted_metrics),
            "top_confusions": json.dumps(boosted_metrics["top_confusions"][:8], ensure_ascii=False),
            "prediction_distribution": json.dumps(boosted_metrics["prediction_distribution"], ensure_ascii=False, sort_keys=True),
            "artifact_path": str(output_path),
            "val_logits_path": ";".join(str(path) for path in fold_paths),
            "notes": args.notes,
            "decision": "rule boost diagnostic complete",
        },
    )

    lines = [
        f"## {args.experiment_id}",
        "",
        f"- Date/time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "- Validation setup: 3-fold session-aware OOF aggregate with deterministic sample/logit rule boosts.",
        f"- Baseline OOF Macro-F1: {base_metrics['macro_f1']:.6f}",
        f"- Boosted OOF Macro-F1: {boosted_metrics['macro_f1']:.6f}",
        f"- Selected rules: {len(rules)}",
        f"- Weakest classes: {summarize_weak(boosted_metrics).replace(';', ', ')}",
        f"- Top confusions: {boosted_metrics['top_confusions'][:8]}",
        f"- Rule artifact: {output_path}",
        f"- Decision: {args.notes or 'inspect OOF gain before integrating rules into final inference'}",
        "",
    ]
    with Path("research_log.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
