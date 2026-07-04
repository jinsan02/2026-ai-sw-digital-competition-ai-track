"""E1: base OOF 로짓 + 탐색 specialist OOF 로짓 블렌딩 튜닝. (노진산, 07-03)

블렌딩: base top-1이 subset이고 margin≤m일 때, subset 열에
  base[cols] += w * (spec[cols] - mean(spec[cols]))   # 중심화 → subset 내 상대 선호만 주입
(m, w) 그리드를 OOF에서 탐색 → 최적 조합에 2단계 bias 재튜닝.

사용:
  python blend_specialist.py --base-artifact experiments/artifacts/oof_focal_replay_p1_oof_metrics.json \
      --spec-pattern "experiments/logits/*E1-specialist-fold*_val_logits.pt" \
      --data-dir /mnt/c/dacon/open/data --experiment-id e1_blend --tune-bias
"""
import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from script import ALL_CLASSES, safe_text
from train import CLASS_TO_ID, f1_metrics, predict_with_bias, tune_class_bias
from tune_oof_rule_boosts import torch_load

SUBSET = ["read_file", "grep_search", "list_directory", "glob_pattern", "web_search"]
SUBSET_IDS = [CLASS_TO_ID[c] for c in SUBSET]


def load_concat(paths):
    ids, logits, y = [], [], []
    seen = set()
    for p in sorted(paths):
        payload = torch_load(p)
        dup = seen & set(payload["ids"])
        if dup:
            raise ValueError(f"duplicate ids: {sorted(dup)[:5]} in {p}")
        seen |= set(payload["ids"])
        ids.extend(safe_text(i) for i in payload["ids"])
        logits.append(payload["logits"].float())
        y.extend(int(v) for v in payload["y_true"])
    return ids, torch.cat(logits, dim=0), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--spec-pattern", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--margins", default="99,1.0,0.5,0.25")
    parser.add_argument("--weights", default="0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--blend-cols", default="",
                        help="블렌딩할 클래스 서브셋(콤마, 기본=SUBSET 전체). 예: read_file,list_directory,web_search")
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    art = json.loads(Path(args.base_artifact).read_text(encoding="utf-8"))
    base_ids, base_logits, y = load_concat(art["fold_logits"])
    spec_paths = glob.glob(args.spec_pattern)
    if len(spec_paths) < 2:
        raise FileNotFoundError(f"specialist logits not found: {args.spec_pattern} -> {spec_paths}")
    spec_ids, spec_logits, spec_y = load_concat(spec_paths)
    spec_map = {i: k for k, i in enumerate(spec_ids)}
    order = [spec_map[i] for i in base_ids]  # base 순서로 정렬
    spec_logits = spec_logits[order]

    blend_names = [s.strip() for s in args.blend_cols.split(",") if s.strip()] or SUBSET
    cols = torch.tensor([CLASS_TO_ID[c] for c in blend_names])
    spec_cols = torch.tensor([SUBSET.index(c) for c in blend_names])  # spec_sub 내 열 위치
    print(f"blend columns: {blend_names}")
    base_pred = torch.argmax(base_logits, dim=1)
    top2 = torch.topk(base_logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    in_subset = torch.zeros(len(y), dtype=torch.bool)
    for c in SUBSET_IDS:
        in_subset |= base_pred == c

    spec_sub = spec_logits[:, torch.tensor(SUBSET_IDS)]  # 중심화는 항상 5-way 전체 기준
    spec_centered_full = spec_sub - spec_sub.mean(dim=1, keepdim=True)
    spec_centered = spec_centered_full[:, spec_cols]

    base_f1 = f1_metrics(y, base_pred.tolist())["macro_f1"]
    print(f"base_raw_oof_macro_f1={base_f1:.6f}")

    best = (base_f1, None, None, base_logits)
    for m in [float(x) for x in args.margins.split(",")]:
        trigger = in_subset & (margin <= m)
        for w in [float(x) for x in args.weights.split(",")]:
            blended = base_logits.clone()
            blended[trigger.nonzero(as_tuple=True)[0][:, None], cols] += w * spec_centered[trigger]
            score = f1_metrics(y, torch.argmax(blended, dim=1).tolist())["macro_f1"]
            marker = " *" if score > best[0] else ""
            print(f"  margin<={m:<5} w={w:<4} triggered={int(trigger.sum())} macro={score:.6f}{marker}")
            if score > best[0]:
                best = (score, m, w, blended)

    best_f1, best_m, best_w, best_scores = best
    print(f"best_raw: macro={best_f1:.6f} margin={best_m} w={best_w} (delta={best_f1 - base_f1:+.6f})")

    result = {
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subset": SUBSET,
        "base_raw_macro_f1": base_f1,
        "blend_raw_macro_f1": best_f1,
        "margin": best_m,
        "weight": best_w,
        "notes": args.notes,
    }
    if args.tune_bias and best_m is not None:
        bias, tuned = tune_class_bias(best_scores, y, rounds=3)
        pred = predict_with_bias(best_scores, bias)
        metrics = f1_metrics(y, pred)
        result["blend_tuned_macro_f1"] = tuned
        result["class_bias"] = dict(zip(ALL_CLASSES, [float(v) for v in bias.tolist()]))
        result["per_class_f1"] = metrics["per_class_f1"]
        print(f"blend_tuned_oof_macro_f1={tuned:.6f}")
        print("  subset per-class:", {c: round(metrics["per_class_f1"][c], 4) for c in SUBSET})

    out = Path("experiments/artifacts") / f"{args.experiment_id}_blend.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
