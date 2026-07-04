import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from script import ALL_CLASSES
from train import append_results_csv, f1_metrics, predict_with_bias, tune_class_bias


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_fold_payloads(paths):
    payloads = []
    seen_ids = set()
    duplicates = []
    for path in paths:
        payload = torch_load(path)
        ids = payload["ids"]
        for sample_id in ids:
            if sample_id in seen_ids:
                duplicates.append(sample_id)
            seen_ids.add(sample_id)
        payloads.append((path, payload))
    if duplicates:
        raise ValueError(f"duplicate validation ids across folds: {duplicates[:5]}")
    return payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no OOF logits matched pattern: {args.pattern}")
    payloads = load_fold_payloads(paths)

    logits = torch.cat([payload["logits"].float() for _, payload in payloads], dim=0)
    y_true = []
    ids = []
    fold_paths = []
    for path, payload in payloads:
        y_true.extend(int(x) for x in payload["y_true"])
        ids.extend(payload["ids"])
        fold_paths.append(str(path))

    raw_pred = torch.argmax(logits, dim=1).tolist()
    raw_metrics = f1_metrics(y_true, raw_pred)
    bias, _ = tune_class_bias(logits, y_true, rounds=3)
    tuned_pred = predict_with_bias(logits, bias)
    tuned_metrics = f1_metrics(y_true, tuned_pred)

    output_dir = Path("experiments/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.experiment_id}_oof_metrics.json"
    payload = {
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold_logits": fold_paths,
        "row_count": len(y_true),
        "unique_id_count": len(set(ids)),
        "classes": ALL_CLASSES,
        "raw_metrics": raw_metrics,
        "metrics": tuned_metrics,
        "class_bias": dict(zip(ALL_CLASSES, [float(x) for x in bias.tolist()])),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    weak = sorted(tuned_metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:5]
    append_results_csv(
        Path("experiments/results.csv"),
        {
            "experiment_id": args.experiment_id,
            "model_family": "torch_gpu_transformer_oof",
            "features": "serialized prompt/action/workspace text",
            "split_type": "session_oof",
            "fold_id": "oof",
            "macro_f1_raw": f"{raw_metrics['macro_f1']:.6f}",
            "macro_f1_bias_tuned": f"{tuned_metrics['macro_f1']:.6f}",
            "macro_f1": f"{tuned_metrics['macro_f1']:.6f}",
            "weakest_classes": ";".join(f"{name}:{score:.4f}" for name, score in weak),
            "top_confusions": json.dumps(tuned_metrics["top_confusions"][:8], ensure_ascii=False),
            "prediction_distribution": json.dumps(tuned_metrics["prediction_distribution"], ensure_ascii=False, sort_keys=True),
            "val_logits_path": ";".join(fold_paths),
            "notes": args.notes,
            "decision": "oof aggregate complete",
        },
    )

    lines = [
        f"## {args.experiment_id}",
        "",
        f"- Date/time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "- Validation setup: 3-fold session-aware OOF aggregate",
        f"- Fold logits: {fold_paths}",
        f"- Raw OOF Macro-F1: {raw_metrics['macro_f1']:.6f}",
        f"- Tuned OOF Macro-F1: {tuned_metrics['macro_f1']:.6f}",
        f"- Weakest classes: {', '.join(f'{k}={v:.3f}' for k, v in weak)}",
        f"- Top confusions: {tuned_metrics['top_confusions'][:8]}",
        f"- Prediction distribution: {tuned_metrics['prediction_distribution']}",
        f"- Decision: {args.notes or 'compare against fixed-session and public calibration before final refit'}",
        "",
    ]
    with Path("research_log.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"raw_oof_macro_f1={raw_metrics['macro_f1']:.6f}")
    print(f"tuned_oof_macro_f1={tuned_metrics['macro_f1']:.6f}")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
