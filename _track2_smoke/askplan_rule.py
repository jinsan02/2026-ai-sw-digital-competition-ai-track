"""ask/plan 규칙 시험 — 저장 로짓 위, 재학습X.
plan 신호(단계/순서/계획 동사 + turn 이름) → plan 부스트, ask 신호(질문/도움요청) → ask 부스트.
저마진 ask/plan 예측에만 발동. 전체 macro 델타 측정."""
import torch, json, re
from collections import Counter

d = torch.load("/mnt/c/dacon/_track2_smoke/large_val_logits.pt", map_location="cpu")
lg = d["logits"].float()
b = d.get("class_bias")
if b is not None: lg = lg + torch.tensor(b).float()
cls = d["classes"]; ids = d["ids"]; yt = [int(v) for v in d["y_true"]]
idx = {c: i for i, c in enumerate(cls)}
A, P = idx["ask_user"], idx["plan_task"]
pred = lg.argmax(1).tolist()

S = {}
for ln in open("/mnt/c/dacon/open/data/train.jsonl", encoding="utf-8"):
    o = json.loads(ln); S[o["id"]] = o

RE_PLAN = re.compile(r"단계|순서|쪼개|나눠|계획|흐름부터|로드맵|build|step|break.*down|plan|outline|structure", re.I)
RE_ASK = re.compile(r"\?|할지|둘지|맞을지|해야|should i|which|도와줄래|알려줘|애매|헷갈|추천|괜찮을까|어때", re.I)

def macro(pr):
    fs = []
    for c in range(len(cls)):
        tp = sum(1 for p, y in zip(pr, yt) if p == c and y == c)
        fp = sum(1 for p, y in zip(pr, yt) if p == c and y != c)
        fn = sum(1 for p, y in zip(pr, yt) if p != c and y == c)
        pp = tp/(tp+fp) if tp+fp else 0; rr = tp/(tp+fn) if tp+fn else 0
        fs.append(2*pp*rr/(pp+rr) if pp+rr else 0)
    return sum(fs)/len(fs), fs

base_m, base_f = macro(pred)
top2 = torch.topk(lg, 2, 1).values
margin = (top2[:, 0] - top2[:, 1]).tolist()

def run(w, marg, use_turn):
    newlg = lg.clone(); n = Counter()
    for i, ii in enumerate(ids):
        if pred[i] not in (A, P) or margin[i] > marg:
            continue
        s = S.get(ii, {}); p = s.get("current_prompt") or ""
        turn = (s.get("session_meta") or {}).get("turn_index", 99)
        is_plan = bool(RE_PLAN.search(p)) or (use_turn and turn <= 2)
        is_ask = bool(RE_ASK.search(p))
        if is_plan and not is_ask:
            newlg[i, P] += w; n["plan"] += 1
        elif is_ask and not is_plan:
            newlg[i, A] += w; n["ask"] += 1
    pr = newlg.argmax(1).tolist()
    m, f = macro(pr)
    return m, f, n

print(f"baseline macro={base_m:.4f}  ask_F1={base_f[A]:.3f} plan_F1={base_f[P]:.3f}")
print("\n=== ask/plan 규칙 스윕 ===")
best = (base_m, None)
for marg in [1.0, 2.0, 3.0]:
    for w in [0.5, 1.0, 2.0]:
        for ut in [False, True]:
            m, f, n = run(w, marg, ut)
            tag = " *" if m > base_m else ""
            print(f"  m<={marg} w={w} turn={ut}: macro={m:.4f}(Δ{m-base_m:+.4f}) ask={f[A]:.3f} plan={f[P]:.3f} {dict(n)}{tag}")
            if m > best[0]: best = (m, (marg, w, ut, f))
if best[1]:
    marg, w, ut, f = best[1]
    print(f"\nbest: macro={best[0]:.4f} (Δ{best[0]-base_m:+.4f}) @ m<={marg} w={w} turn={ut}")
    print(f"  ask_F1 {base_f[A]:.3f}->{f[A]:.3f}  plan_F1 {base_f[P]:.3f}->{f[P]:.3f}")
else:
    print("\n개선 없음")
