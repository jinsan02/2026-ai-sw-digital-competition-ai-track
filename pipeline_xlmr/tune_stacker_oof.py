"""OOF LogReg stacker (팀 로컬 과제, 노진산 2026-07-02).

트랜스포머 OOF 로짓 + (선택) sparse SVC OOF 마진 + intent/result/action 피처를 입력으로
fold-aware LogisticRegression 스태커를 학습해 OOF Macro-F1을 개선하는지 측정.

fold-aware: 각 fold의 예측은 "그 fold를 제외한 나머지 fold들"로 학습한 스태커에서 나옴 → 누수 없음.

사용:
  python tune_stacker_oof.py --oof-artifact experiments/artifacts/<agg>_oof_metrics.json \
      --data-dir /mnt/c/dacon/open/data --experiment-id stacker_v1 \
      [--sparse-logits experiments/logits/<sparse>_oof_logits.pt] [--tune-bias]
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from script import ALL_CLASSES, load_jsonl, prompt_intent_tokens, rule_base_feature_flags, safe_text
from train import CLASS_TO_ID, f1_metrics, load_labels, predict_with_bias, tune_class_bias
from tune_oof_rule_boosts import torch_load

INTENTS = [
    "intent_run_tests", "intent_lint", "intent_run_bash", "intent_read", "intent_grep",
    "intent_glob", "intent_list", "intent_edit", "intent_write", "intent_plan",
    "intent_ask", "intent_web", "intent_respond",
]
RESULT_FLAGS = ["result_found", "result_none", "result_failed", "result_passed", "result_exit_code"]


def sample_features(sample):
    """intent/result/last-action/컨텍스트 이진 피처 (스태커 보조 입력)."""
    flags = rule_base_feature_flags(sample)
    intents = set(prompt_intent_tokens(safe_text(sample.get("current_prompt", ""))))
    feats = [1.0 if name in intents else 0.0 for name in INTENTS]
    feats += [1.0 if name in flags else 0.0 for name in RESULT_FLAGS]
    last_action = [0.0] * len(ALL_CLASSES)
    for flag in flags:
        if flag.startswith("last_action:"):
            name = flag.split(":", 1)[1]
            if name in CLASS_TO_ID:
                last_action[CLASS_TO_ID[name]] = 1.0
    feats += last_action
    history = sample.get("history") or []
    feats.append(min(len(history), 12) / 12.0)
    feats.append(1.0 if "prompt_has_question" in flags else 0.0)
    return feats


def build_matrix(payloads, samples_by_id, sparse_by_id):
    rows, labels, fold_ids, ids = [], [], [], []
    for fold_no, (path, payload) in enumerate(payloads):
        logits = payload["logits"].float().numpy()
        for row_idx, sample_id in enumerate(payload["ids"]):
            sample_id = safe_text(sample_id)
            sample = samples_by_id.get(sample_id)
            if sample is None:
                raise ValueError(f"missing sample: {sample_id}")
            base = logits[row_idx]
            probs = np.exp(base - base.max())
            probs = probs / probs.sum()
            top2 = np.sort(base)[-2:]
            margin = [top2[1] - top2[0]]
            sparse_part = list(sparse_by_id.get(sample_id, [])) if sparse_by_id else []
            rows.append(list(base) + list(probs) + margin + sparse_part + sample_features(sample))
            labels.append(int(payload["y_true"][row_idx]))
            fold_ids.append(fold_no)
            ids.append(sample_id)
    return np.array(rows, dtype=np.float32), np.array(labels), np.array(fold_ids), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-artifact", required=True)
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--sparse-logits", default="", help="선택: sparse SVC OOF 로짓 .pt (ids/logits)")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    artifact = json.loads(Path(args.oof_artifact).read_text(encoding="utf-8"))
    payloads = []
    seen = set()
    for path in artifact["fold_logits"]:
        payload = torch_load(path)
        dup = seen & set(payload["ids"])
        if dup:
            raise ValueError(f"duplicate ids across folds: {sorted(dup)[:5]}")
        seen |= set(payload["ids"])
        payloads.append((path, payload))

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    samples_by_id = {safe_text(s.get("id")): s for s in samples}

    sparse_by_id = None
    if args.sparse_logits:
        sp = torch_load(args.sparse_logits)
        raw = sp.get("logits", sp.get("scores"))
        sp_logits = raw.float().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
        sparse_by_id = {safe_text(i): sp_logits[k] for k, i in enumerate(sp["ids"])}

    x, y, fold_ids, ids = build_matrix(payloads, samples_by_id, sparse_by_id)
    print(f"stacker matrix: {x.shape} folds={sorted(set(fold_ids))} sparse={'yes' if sparse_by_id else 'no'}")

    # baseline: 트랜스포머 로짓 argmax
    base_logits = torch.tensor(np.concatenate([p["logits"].float().numpy() for _, p in payloads]))
    base_pred = torch.argmax(base_logits, dim=1).tolist()
    base_f1 = f1_metrics(list(y), base_pred)["macro_f1"]
    print(f"baseline_raw_oof_macro_f1={base_f1:.6f}")

    stacked = np.zeros((len(y), len(ALL_CLASSES)), dtype=np.float64)
    for fold in sorted(set(fold_ids)):
        train_mask = fold_ids != fold
        clf = LogisticRegression(C=args.c, max_iter=args.max_iter, n_jobs=-1)
        clf.fit(x[train_mask], y[train_mask])
        proba = clf.predict_proba(x[~train_mask])
        cols = [list(clf.classes_).index(c) if c in clf.classes_ else None for c in range(len(ALL_CLASSES))]
        out = np.zeros((proba.shape[0], len(ALL_CLASSES)))
        for c, col in enumerate(cols):
            if col is not None:
                out[:, c] = proba[:, col]
        stacked[~train_mask] = out
        print(f"  fold={fold} trained on {int(train_mask.sum())} rows, predicted {int((~train_mask).sum())}")

    stacked_t = torch.tensor(np.log(np.clip(stacked, 1e-9, None)))
    stack_pred = torch.argmax(stacked_t, dim=1).tolist()
    stack_f1 = f1_metrics(list(y), stack_pred)["macro_f1"]
    print(f"stacked_raw_oof_macro_f1={stack_f1:.6f} (delta={stack_f1 - base_f1:+.6f})")

    result = {
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "oof_artifact": args.oof_artifact,
        "sparse_logits": args.sparse_logits,
        "feature_dim": int(x.shape[1]),
        "baseline_raw_oof_macro_f1": base_f1,
        "stacked_raw_oof_macro_f1": stack_f1,
        "notes": args.notes,
    }
    if args.tune_bias:
        bias, tuned = tune_class_bias(stacked_t, list(y), rounds=3)
        result["stacked_tuned_oof_macro_f1"] = tuned
        result["stacked_class_bias"] = dict(zip(ALL_CLASSES, [float(v) for v in bias.tolist()]))
        print(f"stacked_tuned_oof_macro_f1={tuned:.6f}")

    out_path = Path("experiments/artifacts") / f"{args.experiment_id}_stacker.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
