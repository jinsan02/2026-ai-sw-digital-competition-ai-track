#!/usr/bin/env python3
"""Build an id-aligned OOF correctness-consensus reliability artifact.

Each repeated ``--model`` argument is one model's complete set of disjoint OOF
fold payloads. The builder validates class order, labels, folds, duplicate ids,
and exact train-set coverage before counting how many models predict each row's
true action. Raw logits are used deliberately; no validation-tuned bias or rule
layer enters the reliability target.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from script import ALL_CLASSES, load_jsonl, safe_text
from train import CLASS_TO_ID, load_labels
from train_transformer import parse_consensus_backbone_weights, torch_load


ARTIFACT_KIND = "oof_correct_consensus_reliability"
SCHEMA_VERSION = 1
USAGE_SCOPE = "full_data_refit_only"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_rows(path):
    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: OOF payload must be a dict")
    if list(payload.get("classes") or []) != ALL_CLASSES:
        raise ValueError(f"{path}: class order does not match ALL_CLASSES")
    if payload.get("split") != "session_oof":
        raise ValueError(f"{path}: expected split='session_oof', got {payload.get('split')!r}")
    ids = [safe_text(sample_id) for sample_id in payload.get("ids") or []]
    logits = torch.as_tensor(payload.get("logits"), dtype=torch.float32)
    labels = torch.as_tensor(payload.get("y_true"), dtype=torch.long).view(-1)
    if logits.ndim != 2 or logits.shape[1] != len(ALL_CLASSES):
        raise ValueError(
            f"{path}: logits must have shape [rows, {len(ALL_CLASSES)}], "
            f"got {tuple(logits.shape)}"
        )
    if len(ids) != logits.shape[0] or len(ids) != len(labels):
        raise ValueError(
            f"{path}: row length mismatch ids={len(ids)} logits={logits.shape[0]} "
            f"labels={len(labels)}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids inside an OOF fold")
    fold_id = payload.get("fold_id")
    n_folds = payload.get("n_folds")
    if fold_id is None or n_folds is None:
        raise ValueError(f"{path}: missing fold_id/n_folds metadata")
    fold_id = int(fold_id)
    n_folds = int(n_folds)
    if fold_id < 0 or fold_id >= n_folds:
        raise ValueError(f"{path}: invalid fold_id={fold_id} for n_folds={n_folds}")
    return payload, ids, labels, logits.argmax(dim=1), fold_id, n_folds


def load_oof_model_group(paths, expected_ids, expected_labels):
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("an OOF model group cannot be empty")
    predictions = {}
    fold_ids = []
    declared_n_folds = set()
    seeds = set()
    base_models = set()
    serializers = set()
    source_files = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"OOF payload is missing: {path}")
        payload, ids, labels, pred, fold_id, n_folds = _payload_rows(path)
        fold_ids.append(fold_id)
        declared_n_folds.add(n_folds)
        seeds.add(payload.get("seed"))
        base_models.add(payload.get("base_model"))
        serializers.add(payload.get("serializer_name"))
        overlap = set(predictions).intersection(ids)
        if overlap:
            raise ValueError(
                f"{path}: OOF folds overlap on ids: {sorted(overlap)[:5]}"
            )
        for sample_id, label, prediction in zip(ids, labels.tolist(), pred.tolist()):
            predictions[sample_id] = (int(label), int(prediction))
        source_files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(ids),
                "fold_id": fold_id,
            }
        )
    if len(declared_n_folds) != 1:
        raise ValueError(f"OOF group has inconsistent n_folds: {sorted(declared_n_folds)}")
    n_folds = next(iter(declared_n_folds))
    if len(paths) != n_folds or sorted(fold_ids) != list(range(n_folds)):
        raise ValueError(
            "OOF group must contain each fold exactly once: "
            f"paths={len(paths)} n_folds={n_folds} fold_ids={sorted(fold_ids)}"
        )
    actual_ids = set(predictions)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:5]
        extra = sorted(actual_ids - expected_ids)[:5]
        raise ValueError(
            "OOF group train coverage mismatch: "
            f"expected={len(expected_ids)} actual={len(actual_ids)} "
            f"missing={missing} extra={extra}"
        )
    mismatches = [
        (sample_id, predictions[sample_id][0], expected_labels[sample_id])
        for sample_id in expected_ids
        if predictions[sample_id][0] != expected_labels[sample_id]
    ]
    if mismatches:
        raise ValueError(f"OOF labels do not match train labels: {mismatches[:5]}")
    accuracy = sum(
        label == prediction for label, prediction in predictions.values()
    ) / len(predictions)
    metadata = {
        "files": source_files,
        "n_folds": n_folds,
        "fold_ids": sorted(fold_ids),
        "seeds": sorted(seeds, key=lambda value: str(value)),
        "base_models": sorted(base_models, key=lambda value: str(value)),
        "serializers": sorted(serializers, key=lambda value: str(value)),
        "rows": len(predictions),
        "raw_accuracy": accuracy,
    }
    return predictions, metadata


def build_consensus_payload(samples, labels_by_id, model_groups, backbone_weights):
    ids = [safe_text(sample.get("id")) for sample in samples]
    if not ids or any(not sample_id for sample_id in ids):
        raise ValueError("train samples must have non-empty ids")
    if len(ids) != len(set(ids)):
        raise ValueError("train samples contain duplicate ids")
    expected_ids = set(ids)
    missing_labels = [sample_id for sample_id in ids if sample_id not in labels_by_id]
    extra_labels = sorted(set(labels_by_id) - expected_ids)
    if missing_labels or extra_labels:
        raise ValueError(
            "train label coverage mismatch: "
            f"missing={missing_labels[:5]} extra={extra_labels[:5]}"
        )
    expected_labels = {}
    for sample_id in ids:
        label = labels_by_id[sample_id]
        if label not in CLASS_TO_ID:
            raise ValueError(f"unknown train label for {sample_id}: {label!r}")
        expected_labels[sample_id] = CLASS_TO_ID[label]

    model_rows = []
    sources = []
    for group in model_groups:
        predictions, metadata = load_oof_model_group(
            group, expected_ids, expected_labels
        )
        model_rows.append(predictions)
        sources.append(metadata)
    if not model_rows:
        raise ValueError("at least one --model OOF group is required")
    weights = parse_consensus_backbone_weights(
        backbone_weights, expected_count=len(model_rows) + 1
    )
    labels = torch.tensor([expected_labels[sample_id] for sample_id in ids], dtype=torch.int16)
    correct_counts = torch.tensor(
        [
            sum(rows[sample_id][1] == expected_labels[sample_id] for rows in model_rows)
            for sample_id in ids
        ],
        dtype=torch.uint8,
    )
    histogram = Counter(int(value) for value in correct_counts.tolist())
    per_class = {}
    for class_id, label in enumerate(ALL_CLASSES):
        row_mask = labels == class_id
        class_counts = correct_counts[row_mask]
        class_hist = Counter(int(value) for value in class_counts.tolist())
        per_class[label] = {
            "rows": int(row_mask.sum()),
            "correct_count_histogram": {
                str(count): int(class_hist.get(count, 0))
                for count in range(len(model_rows) + 1)
            },
            "raw_backbone_weight_mean": (
                sum(weights[int(count)] for count in class_counts.tolist())
                / max(1, len(class_counts))
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "usage_scope": USAGE_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classes": list(ALL_CLASSES),
        "ids": ids,
        "y_true": labels,
        "correct_counts": correct_counts,
        "model_count": len(model_rows),
        "backbone_weights": weights,
        "correct_count_histogram": {
            str(count): int(histogram.get(count, 0))
            for count in range(len(model_rows) + 1)
        },
        "per_class": per_class,
        "sources": sources,
    }


def summary_payload(payload, output_path, train_jsonl, labels_csv):
    return {
        "schema_version": payload["schema_version"],
        "kind": payload["kind"],
        "usage_scope": payload["usage_scope"],
        "created_utc": payload["created_utc"],
        "artifact_path": str(output_path),
        "train_jsonl": str(train_jsonl),
        "labels_csv": str(labels_csv),
        "rows": len(payload["ids"]),
        "model_count": payload["model_count"],
        "backbone_weights": payload["backbone_weights"],
        "correct_count_histogram": payload["correct_count_histogram"],
        "per_class": payload["per_class"],
        "sources": payload["sources"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", default="open/data/train.jsonl")
    parser.add_argument("--labels-csv", default="open/data/train_labels.csv")
    parser.add_argument(
        "--model",
        dest="model_groups",
        action="append",
        nargs="+",
        required=True,
        metavar="OOF_PT",
        help="one model's complete OOF fold payloads; repeat once per model",
    )
    parser.add_argument(
        "--backbone-weights",
        default="0,0.25,0.75,1",
        help="comma-separated raw weights for c=0..number-of-models",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--summary-json",
        default="",
        help="default: <output stem>.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_jsonl = Path(args.train_jsonl)
    labels_csv = Path(args.labels_csv)
    output = Path(args.output)
    samples = load_jsonl(train_jsonl)
    labels_by_id = load_labels(labels_csv)
    payload = build_consensus_payload(
        samples,
        labels_by_id,
        args.model_groups,
        args.backbone_weights,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary_path = Path(args.summary_json) if args.summary_json else output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary_payload(payload, output, train_jsonl, labels_csv)
    summary["artifact_sha256"] = sha256_file(output)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved consensus reliability: {output} rows={len(payload['ids'])} "
        f"models={payload['model_count']} histogram={payload['correct_count_histogram']}"
    )
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
