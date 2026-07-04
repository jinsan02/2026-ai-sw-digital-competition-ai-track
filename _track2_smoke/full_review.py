"""전 14클래스 macro-F1 재검토 — 저장 large val_logits(14k).
각 클래스: support/P/R/F1/macro기여 + 주혼동처 + P-R 불균형(임계값 레버) + 분리가능성(collision)."""
import torch, json, re, csv
from collections import Counter, defaultdict
import statistics as st

d = torch.load("/mnt/c/dacon/_track2_smoke/large_val_logits.pt", map_location="cpu")
lg = d["logits"].float()
b = d.get("class_bias")
if b is not None: lg = lg + torch.tensor(b).float()
cls = d["classes"]; ids = d["ids"]; yt = [int(v) for v in d["y_true"]]
idx = {c: i for i, c in enumerate(cls)}
pred = lg.argmax(1).tolist()
N = len(cls)

def prf(c):
    tp = sum(1 for p, y in zip(pred, yt) if p == c and y == c)
    fp = sum(1 for p, y in zip(pred, yt) if p == c and y != c)
    fn = sum(1 for p, y in zip(pred, yt) if p != c and y == c)
    sup = sum(1 for y in yt if y == c)
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    F = 2*P*R/(P+R) if P+R else 0
    return sup, P, R, F

def top_conf(c):
    cc = Counter(cls[pred[i]] for i in range(len(yt)) if yt[i] == c and pred[i] != c)
    return cc.most_common(2)

# collision ceiling per class (같은 정규화컨텍스트에서 이 클래스 true가 얼마나 순수한가)
S = {}
for ln in open("/mnt/c/dacon/open/data/train.jsonl", encoding="utf-8"):
    o = json.loads(ln); S[o["id"]] = o
def norm(p):
    p = (p or "").lower(); p = re.sub(r"[\w./\\-]+\.[a-z]{1,6}\b", "<F>", p)
    p = re.sub(r"\d+", "<N>", p); p = re.sub(r"[^\w<>가-힣]+", " ", p)
    return " ".join(p.split()[:18])
# 라벨 로드 (train 전체로 collision — val만으론 표본 적음)
lab = {}
with open("/mnt/c/dacon/open/data/train_labels.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ks = list(r.keys()); lab[r[ks[0]]] = r[ks[1]]
grp = defaultdict(Counter)
for i, (sid, o) in enumerate(S.items()):
    y = lab.get(sid)
    if not y: continue
    prev = None
    for e in reversed(o.get("history") or []):
        if e.get("role") == "assistant_action": prev = e.get("name"); break
    grp[(norm(o.get("current_prompt")), prev)][y] += 1
# 각 클래스: 이 클래스 true 샘플이 속한 그룹들의 평균 purity(=그 클래스 관점 상한 근사)
cls_pur = {}
for c in cls:
    num = den = 0
    for k, cnt in grp.items():
        if cnt[c] and sum(cnt.values()) >= 2:
            num += cnt[c] * (cnt[c] / sum(cnt.values())); den += cnt[c]
    cls_pur[c] = num/den if den else 1.0

rows = []
for c in range(N):
    sup, P, R, F = prf(c)
    rows.append((F, cls[c], sup, P, R, top_conf(c), cls_pur[cls[c]]))
rows.sort()

macro = sum(r[0] for r in rows)/N
print(f"=== 전 클래스 (macro-F1={macro:.4f}, 각 클래스 기여 = F1/14) ===")
print(f"{'class':18s}{'sup':>5s}{'P':>6s}{'R':>6s}{'F1':>7s}{'PR불균형':>8s}  주혼동    상한근사")
for F, name, sup, P, R, tc, pur in rows:
    imb = P - R
    conf = " ".join(f"{k}{v}" for k, v in tc)
    flag = "임계값?" if abs(imb) > 0.08 else ""
    print(f"{name:18s}{sup:5d}{P:6.2f}{R:6.2f}{F:7.3f}{imb:+8.2f} {flag:6s} {conf:20s} {pur:.2f}")

# macro 산술: 세 그룹으로 분해
maxed = [r for r in rows if r[0] > 0.88]
floored = [r for r in rows if r[1] in ("read_file","grep_search","list_directory","glob_pattern")]
mid = [r for r in rows if r[0] <= 0.88 and r not in floored]
print(f"\n=== macro 분해 ===")
print(f"이미최대(F1>0.88) {len(maxed)}개: 평균F1 {st.mean(r[0] for r in maxed):.3f} — 기여 {sum(r[0] for r in maxed)/N:.3f} (헤드룸 ~0)")
print(f"탐색바닥(aleatoric) {len(floored)}개: 평균F1 {st.mean(r[0] for r in floored):.3f} 상한근사 {st.mean(r[6] for r in floored):.3f} — 기여 {sum(r[0] for r in floored)/N:.3f} (헤드룸 ~0, 증명됨)")
print(f"중간(mid) {len(mid)}개: 평균F1 {st.mean(r[0] for r in mid):.3f} 상한근사 {st.mean(r[6] for r in mid):.3f} — 기여 {sum(r[0] for r in mid)/N:.3f}")
gap = sum(max(0, r[6]-r[0]) for r in mid)/N
print(f"\n중간 클래스가 각자 상한근사까지 오르면 macro += {gap:.4f} (이론 상한, 실제는 일부만)")
print("→ mid 클래스 목록:", ", ".join(f"{r[1]}({r[0]:.2f}->{r[6]:.2f})" for r in mid))
