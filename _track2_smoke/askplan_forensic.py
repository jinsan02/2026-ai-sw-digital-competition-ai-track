"""ask_user vs plan_task 를 실제로 가르는 신호 찾기.
분리가능 클러스터(purity 0.80)이므로 규칙이 나올 것으로 기대."""
import torch, json, re, csv
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

# 통계적 신호 탐색: 정답이 ask vs plan 일 때 프롬프트 특성 차이
def prompt_of(ii): return (S.get(ii, {}).get("current_prompt") or "")
qmark = {A: 0, P: 0}; length = {A: [], P: []}; nturn = {A: [], P: []}
words_ask = Counter(); words_plan = Counter()
STOP = set("the a an to i you it is of and or for in on with this that my me can we be do please just".split())
for i, ii in enumerate(ids):
    y = yt[i]
    if y not in (A, P): continue
    p = prompt_of(ii); pl = p.lower()
    length[y].append(len(p))
    s = S.get(ii, {})
    nturn[y].append((s.get("session_meta") or {}).get("turn_index", 0))
    for w in re.findall(r"[a-z가-힣]+", pl):
        if w in STOP or len(w) < 2: continue
        (words_ask if y == A else words_plan)[w] += 1

nA = length[A].count is None
import statistics as st
print("=== ask_user vs plan_task 정답 샘플 프롬프트 특성 ===")
print(f"ask_user  (n={len(length[A])}): 평균길이={st.mean(length[A]):.0f}자  평균turn={st.mean(nturn[A]):.1f}")
print(f"plan_task (n={len(length[P])}): 평균길이={st.mean(length[P]):.0f}자  평균turn={st.mean(nturn[P]):.1f}")

# 어느 클래스에 편중된 단어 (log-odds)
def odds(w):
    a = words_ask[w] + 1; p = words_plan[w] + 1
    return a / p
common = [w for w in set(words_ask) | set(words_plan) if words_ask[w] + words_plan[w] >= 15]
ask_words = sorted(common, key=lambda w: -odds(w))[:15]
plan_words = sorted(common, key=lambda w: odds(w))[:15]
print("\nask_user 편중 단어:", ", ".join(f"{w}({words_ask[w]}/{words_plan[w]})" for w in ask_words))
print("plan_task 편중 단어:", ", ".join(f"{w}({words_ask[w]}/{words_plan[w]})" for w in plan_words))

# 실제 혼동 케이스 예시
print("\n=== true=plan, pred=ask (오분류) 예시 ===")
n = 0
for i, ii in enumerate(ids):
    if yt[i] == P and pred[i] == A:
        print("  ", repr(prompt_of(ii)[:110]))
        n += 1
        if n >= 4: break
print("=== true=ask, pred=plan (오분류) 예시 ===")
n = 0
for i, ii in enumerate(ids):
    if yt[i] == A and pred[i] == P:
        print("  ", repr(prompt_of(ii)[:110]))
        n += 1
        if n >= 4: break
