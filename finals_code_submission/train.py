import argparse
import csv
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from script import (
    ALL_CLASSES,
    load_jsonl,
)


CLASS_TO_ID = {label: i for i, label in enumerate(ALL_CLASSES)}
VOCAB_CHANNELS = ("word", "char", "meta", "last_user")
LEGACY_HASH_MODEL_MESSAGE = (
    "train.py's legacy hashed/vocab Torch model path depends on symbols removed "
    "from the compacted script.py inference path. Use train_transformer.py for "
    "active transformer experiments or the sparse SVC scripts for ensemble tuning."
)


def require_legacy_hash_model():
    raise RuntimeError(LEGACY_HASH_MODEL_MESSAGE)


def load_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def session_id(sample_id):
    return str(sample_id).split("-step_")[0]


def split_indices(samples, y, split_type, seed=42):
    rng = random.Random(seed)
    if split_type == "random":
        by_label = {}
        for idx, label in enumerate(y):
            by_label.setdefault(label, []).append(idx)
        train_idx = []
        val_idx = []
        for label, indices in by_label.items():
            indices = indices[:]
            rng.shuffle(indices)
            n_val = max(1, int(round(len(indices) * 0.2)))
            val_idx.extend(indices[:n_val])
            train_idx.extend(indices[n_val:])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        return train_idx, val_idx

    if split_type == "session":
        groups = {}
        for idx, sample in enumerate(samples):
            groups.setdefault(session_id(sample.get("id", "")), []).append(idx)
        group_ids = list(groups)
        rng.shuffle(group_ids)
        target = int(round(len(samples) * 0.2))
        val_groups = set()
        val_count = 0
        for group in group_ids:
            if val_count >= target:
                break
            val_groups.add(group)
            val_count += len(groups[group])
        train_idx = []
        val_idx = []
        for group, indices in groups.items():
            if group in val_groups:
                val_idx.extend(indices)
            else:
                train_idx.extend(indices)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        return train_idx, val_idx

    raise ValueError(f"unknown split_type: {split_type}")


def f1_metrics(y_true, y_pred):
    counts = {
        label: {"tp": 0, "fp": 0, "fn": 0}
        for label in range(len(ALL_CLASSES))
    }
    confusions = Counter()
    pred_dist = Counter()
    for true_id, pred_id in zip(y_true, y_pred):
        pred_dist[ALL_CLASSES[pred_id]] += 1
        if true_id == pred_id:
            counts[true_id]["tp"] += 1
        else:
            counts[true_id]["fn"] += 1
            counts[pred_id]["fp"] += 1
            confusions[(ALL_CLASSES[true_id], ALL_CLASSES[pred_id])] += 1
    per_class = {}
    for label_id, values in counts.items():
        denom = 2 * values["tp"] + values["fp"] + values["fn"]
        per_class[ALL_CLASSES[label_id]] = 0.0 if denom == 0 else (2 * values["tp"] / denom)
    macro = sum(per_class.values()) / len(ALL_CLASSES)
    return {
        "macro_f1": macro,
        "per_class_f1": per_class,
        "prediction_distribution": dict(sorted(pred_dist.items())),
        "top_confusions": [(count, true_label, pred_label) for (true_label, pred_label), count in confusions.most_common(20)],
    }


def class_weight_tensor(y, device, power=0.5):
    counts = Counter(y)
    total = len(y)
    weights = []
    for class_id in range(len(ALL_CLASSES)):
        count = max(1, counts[class_id])
        weights.append((total / (len(ALL_CLASSES) * count)) ** power)
    mean = sum(weights) / len(weights)
    weights = [w / mean for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def init_model(model):
    with torch.no_grad():
        torch.nn.init.normal_(model.embedding.weight, mean=0.0, std=0.01)
        if model.model_type == "linear":
            model.bias.zero_()
        else:
            torch.nn.init.xavier_uniform_(model.head.weight)
            model.head.bias.zero_()


def vocab_limits(args):
    return {
        "word": (args.word_features, args.min_df),
        "char": (args.char_features, args.min_df),
        "meta": (args.meta_features, 1),
        "last_user": (args.last_user_features, args.min_df),
    }


def build_vocab_config(samples, fit_idx, args):
    require_legacy_hash_model()
    doc_freq = {channel: Counter() for channel in VOCAB_CHANNELS}
    for row_no, sample_idx in enumerate(fit_idx, 1):
        counts = build_sample_token_counts(samples[sample_idx])
        for channel in VOCAB_CHANNELS:
            doc_freq[channel].update(counts[channel].keys())
        if row_no % 20000 == 0:
            print(f"  vocab docs={row_no}")

    tokens = {}
    idf = {}
    offsets = {}
    cursor = 0
    n_docs = len(fit_idx)
    for channel in VOCAB_CHANNELS:
        max_features, min_df = vocab_limits(args)[channel]
        kept = [
            (token, count)
            for token, count in doc_freq[channel].items()
            if count >= min_df
        ]
        kept.sort(key=lambda kv: (-kv[1], kv[0]))
        kept = kept[:max_features]
        offsets[channel] = cursor
        tokens[channel] = [token for token, _ in kept]
        idf[channel] = [
            1.0 + torch.log(torch.tensor((1.0 + n_docs) / (1.0 + count))).item()
            for _, count in kept
        ]
        cursor += len(kept)
        print(f"  vocab channel={channel:10s} size={len(kept)}")

    config = {
        "feature_mode": "vocab",
        "tokens": tokens,
        "idf": idf,
        "vocab_maps": {
            channel: {token: idx for idx, token in enumerate(tokens[channel])}
            for channel in VOCAB_CHANNELS
        },
        "vocab_offsets": offsets,
        "num_features": cursor,
    }
    return config


def build_hash_config():
    require_legacy_hash_model()
    _, num_features = build_offsets(DEFAULT_BUCKETS)
    return {
        "feature_mode": "hash",
        "buckets": DEFAULT_BUCKETS,
        "num_features": num_features,
    }


def build_features(samples, config):
    require_legacy_hash_model()
    if config["feature_mode"] == "vocab":
        return [extract_feature_items(sample, config) for sample in samples]
    return [extract_feature_indices(sample, config) for sample in samples]


def train_one_model(feature_lists, y, train_idx, args, device, num_features):
    require_legacy_hash_model()
    model = ActionDecisionModel(
        num_features=num_features,
        num_classes=len(ALL_CLASSES),
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
    ).to(device)
    init_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = class_weight_tensor([y[i] for i in train_idx], device, args.class_weight_power)
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = train_idx[:]
        rng.shuffle(shuffled)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_idx = shuffled[start:start + args.batch_size]
            batch_features = [feature_lists[i] for i in batch_idx]
            labels = torch.tensor([y[i] for i in batch_idx], dtype=torch.long, device=device)
            indices, offsets, lengths, feature_weights = make_batch(batch_features, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(indices, offsets, lengths, feature_weights if args.feature_mode == "vocab" else None)
            if args.loss == "ce":
                loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=args.label_smoothing)
            else:
                targets = torch.full_like(logits, -1.0)
                targets.scatter_(1, labels.view(-1, 1), 1.0)
                margins = torch.clamp(1.0 - targets * logits, min=0.0)
                if args.loss == "ovr_squared_hinge":
                    margins = margins * margins
                sample_weights = class_weights[labels]
                loss = (margins.mean(dim=1) * sample_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_idx)
            seen += len(batch_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"  epoch={epoch:02d} train_loss={total_loss / max(1, seen):.5f}")
    return model


def collect_logits(model, feature_lists, indices, batch_size, device):
    require_legacy_hash_model()
    model.eval()
    logits_parts = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_features = [feature_lists[i] for i in batch_idx]
            batch_indices, offsets, lengths, weights = make_batch(batch_features, device)
            logits = model(batch_indices, offsets, lengths, weights).detach().cpu()
            logits_parts.append(logits)
    return torch.cat(logits_parts, dim=0) if logits_parts else torch.empty((0, len(ALL_CLASSES)))


def predict_with_bias(logits, bias):
    return torch.argmax(logits + bias.view(1, -1), dim=1).tolist()


def tune_class_bias(logits, y_true, rounds=3):
    bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
    candidates = [x / 10.0 for x in range(-24, 25)]
    best_pred = predict_with_bias(logits, bias)
    best = f1_metrics(y_true, best_pred)["macro_f1"]
    for round_no in range(rounds):
        improved = False
        for class_id, class_name in enumerate(ALL_CLASSES):
            current = float(bias[class_id])
            local_best = best
            local_value = current
            for candidate in candidates:
                trial = bias.clone()
                trial[class_id] = candidate
                pred = predict_with_bias(logits, trial)
                score = f1_metrics(y_true, pred)["macro_f1"]
                if score > local_best + 1e-7:
                    local_best = score
                    local_value = candidate
            if local_best > best + 1e-7:
                bias[class_id] = local_value
                best = local_best
                improved = True
                print(f"  bias round={round_no + 1} class={class_name} value={local_value:.2f} macro_f1={best:.5f}")
            else:
                bias[class_id] = current
        if not improved:
            break
    return bias, best


def tune_class_bias_two_stage(
    logits,
    y_true,
    initial_bias=None,
    initial_best=None,
    coarse_rounds=2,
    fine_rounds=2,
    fine_window=0.24,
    fine_step=0.02,
):
    if initial_bias is None:
        bias, best = tune_class_bias(logits, y_true, rounds=coarse_rounds)
    else:
        bias = initial_bias.detach().cpu().float().clone()
        if initial_best is None:
            initial_pred = predict_with_bias(logits, bias)
            best = f1_metrics(y_true, initial_pred)["macro_f1"]
        else:
            best = float(initial_best)

    steps_each_side = max(1, int(round(fine_window / fine_step)))
    offsets = [round(step * fine_step, 6) for step in range(-steps_each_side, steps_each_side + 1)]
    for round_no in range(fine_rounds):
        improved = False
        for class_id, class_name in enumerate(ALL_CLASSES):
            current = float(bias[class_id])
            local_best = best
            local_value = current
            for offset in offsets:
                candidate = current + offset
                trial = bias.clone()
                trial[class_id] = candidate
                pred = predict_with_bias(logits, trial)
                score = f1_metrics(y_true, pred)["macro_f1"]
                if score > local_best + 1e-7:
                    local_best = score
                    local_value = candidate
            if local_best > best + 1e-7:
                bias[class_id] = local_value
                best = local_best
                improved = True
                print(f"  bias fine_round={round_no + 1} class={class_name} value={local_value:.2f} macro_f1={best:.5f}")
            else:
                bias[class_id] = current
        if not improved:
            break
    return bias, best


def save_tensor(path, tensor):
    tensor = tensor.detach().cpu().float().contiguous()
    tensor.numpy().tofile(path)


def save_model_artifact(model, output_dir, class_bias, args, validation_metrics, feature_config):
    require_legacy_hash_model()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "raw-torch-float32-v1",
        "feature_mode": feature_config["feature_mode"],
        "classes": ALL_CLASSES,
        "buckets": DEFAULT_BUCKETS,
        "vocab_offsets": feature_config.get("vocab_offsets", {}),
        "num_features": feature_config["num_features"],
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "class_bias": [float(x) for x in class_bias.tolist()],
        "inference_batch_size": args.inference_batch_size,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_split": args.split,
        "trained_with_cuda": torch.cuda.is_available(),
    }
    if model.model_type == "linear":
        save_tensor(output_dir / "embedding.bin", model.embedding.weight)
        save_tensor(output_dir / "bias.bin", model.bias)
    else:
        save_tensor(output_dir / "embedding.bin", model.embedding.weight)
        save_tensor(output_dir / "head_weight.bin", model.head.weight)
        save_tensor(output_dir / "head_bias.bin", model.head.bias)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    if feature_config["feature_mode"] == "vocab":
        with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokens": feature_config["tokens"],
                    "idf": feature_config["idf"],
                },
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )


def append_results_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    default_fieldnames = [
        "experiment_id",
        "model_family",
        "base_model",
        "features",
        "serializer_name",
        "split_type",
        "seed",
        "fold_id",
        "max_length",
        "epochs",
        "learning_rate",
        "batch_size",
        "class_weight_power",
        "label_smoothing",
        "replay_mode",
        "replay_size",
        "macro_f1_raw",
        "macro_f1_bias_tuned",
        "macro_f1_bias_tuned_2stage",
        "macro_f1",
        "weakest_classes",
        "top_confusions",
        "prediction_distribution",
        "artifact_path",
        "val_logits_path",
        "test_logits_path",
        "inference_time_sec",
        "runtime_sec",
        "artifact_size_mb",
        "train_command",
        "notes",
        "decision",
    ]
    row = {key: row.get(key, "") for key in set(default_fieldnames) | set(row)}
    fieldnames = default_fieldnames[:]
    existing_rows = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for fieldname in reader.fieldnames:
                    if fieldname not in fieldnames:
                        fieldnames.append(fieldname)
                existing_rows = list(reader)
    for fieldname in row:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)

    needs_rewrite = bool(existing_rows) and any(fieldname not in existing_rows[0] for fieldname in fieldnames)
    mode = "w" if needs_rewrite or not path.exists() or path.stat().st_size == 0 else "a"
    rows_to_write = existing_rows + [row] if mode == "w" else [row]
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for output_row in rows_to_write:
            writer.writerow({key: output_row.get(key, "") for key in fieldnames})


def append_research_log(path, experiment_id, split_type, metrics, args, decision):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    weak = sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:5]
    strong = sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    lines = [
        f"## {experiment_id}",
        "",
        f"- Date/time: {now}",
        "- Hypothesis: Move the submission pipeline to a CUDA-first Torch classifier while keeping compact hashed prompt/action/meta features.",
        f"- Code/config changes: Torch `{args.model_type}` {args.feature_mode} n-gram model, loss={args.loss}, epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}.",
        f"- Validation setup: {split_type}",
        f"- Overall Macro-F1: {metrics['macro_f1']:.6f}",
        "- Per-class observations:",
        f"  - Weakest: {', '.join(f'{k}={v:.3f}' for k, v in weak)}",
        f"  - Strongest: {', '.join(f'{k}={v:.3f}' for k, v in strong)}",
        f"- Top confusions: {metrics['top_confusions'][:8]}",
        f"- Prediction distribution: {metrics['prediction_distribution']}",
        "- Runtime or package-size concerns: CUDA model inference uses Torch only; JSON/token hashing remains CPU-side preprocessing.",
        f"- Decision: {decision}",
        "- Next suggested experiment: compare linear vs MLP hashed models and tune epochs/lr if GPU validation lags sparse SVC.",
        "",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def train_and_evaluate(args):
    require_legacy_hash_model()
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(requested_device if requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(0)
        print(f"device=cuda name={torch.cuda.get_device_name(0)}")
    else:
        print("device=cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    data_dir = Path(args.data_dir)
    samples = load_jsonl(data_dir / "train.jsonl")
    labels_by_id = load_labels(data_dir / "train_labels.csv")
    y = [CLASS_TO_ID[labels_by_id[sample["id"]]] for sample in samples]
    train_idx, val_idx = split_indices(samples, y, args.split, args.seed)
    print(f"split={args.split} train={len(train_idx)} val={len(val_idx)}")

    if args.feature_mode == "vocab":
        print("fitting validation vocab/idf on train split")
        feature_config = build_vocab_config(samples, train_idx, args)
    else:
        print("using hashed features")
        feature_config = build_hash_config()
    print("extracting features")
    feature_lists = build_features(samples, feature_config)
    avg_features = sum(len(features) for features in feature_lists) / len(feature_lists)
    print(f"features samples={len(feature_lists)} avg_per_sample={avg_features:.1f} num_features={feature_config['num_features']}")

    model = train_one_model(feature_lists, y, train_idx, args, device, feature_config["num_features"])
    val_logits = collect_logits(model, feature_lists, val_idx, args.inference_batch_size, device)
    y_val = [y[i] for i in val_idx]
    bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
    pred = predict_with_bias(val_logits, bias)
    metrics = f1_metrics(y_val, pred)
    print(f"  macro_f1={metrics['macro_f1']:.6f}")
    if args.tune_bias:
        print("  tuning class bias")
        bias, _ = tune_class_bias(val_logits, y_val)
        pred = predict_with_bias(val_logits, bias)
        metrics = f1_metrics(y_val, pred)
        print(f"  tuned_macro_f1={metrics['macro_f1']:.6f}")

    experiment_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_gpu_{args.model_type}_{args.split}"
    decision = "keep as GPU candidate" if metrics["macro_f1"] >= args.keep_threshold else "discard or revisit"
    append_results_csv(
        Path("experiments/results.csv"),
        {
            "experiment_id": experiment_id,
            "model_family": f"torch_gpu_{args.model_type}",
            "features": "hashed prompt word/char ngrams plus action/workspace metadata",
            "split_type": args.split,
            "macro_f1": f"{metrics['macro_f1']:.6f}",
            "notes": args.notes or f"CUDA Torch {args.model_type} model",
            "artifact_path": args.output_dir if args.final_model else "",
        },
    )
    append_research_log(Path("research_log.md"), experiment_id, args.split, metrics, args, decision)

    metrics_path = Path("experiments/artifacts") / f"{experiment_id}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "split": args.split,
                "model_type": args.model_type,
                "metrics": metrics,
                "class_bias": dict(zip(ALL_CLASSES, [float(x) for x in bias.tolist()])),
                "device": str(device),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.final_model:
        print("training final GPU model on all training rows")
        all_idx = list(range(len(samples)))
        if args.feature_mode == "vocab":
            print("fitting final vocab/idf on all training rows")
            final_feature_config = build_vocab_config(samples, all_idx, args)
        else:
            final_feature_config = build_hash_config()
        final_feature_lists = build_features(samples, final_feature_config)
        final_model = train_one_model(final_feature_lists, y, all_idx, args, device, final_feature_config["num_features"])
        save_model_artifact(final_model, args.output_dir, bias, args, metrics, final_feature_config)
        print(f"saved model artifact: {args.output_dir}")

    print("  weakest classes:")
    for label, score in sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:8]:
        print(f"    {label:18s} {score:.4f}")
    print("  top confusions:")
    for count, true_label, pred_label in metrics["top_confusions"][:10]:
        print(f"    {true_label:18s} -> {pred_label:18s} {count}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--split", choices=["random", "session"], default="session")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--model-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--feature-mode", choices=["hash", "vocab"], default="vocab")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--loss", choices=["ce", "ovr_hinge", "ovr_squared_hinge"], default="ce")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--word-features", type=int, default=180000)
    parser.add_argument("--char-features", type=int, default=160000)
    parser.add_argument("--meta-features", type=int, default=50000)
    parser.add_argument("--last-user-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--final-model", action="store_true")
    parser.add_argument("--output-dir", default="model")
    parser.add_argument("--keep-threshold", type=float, default=0.60)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
