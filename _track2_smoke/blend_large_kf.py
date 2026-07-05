"""large ⊕ kf-deberta 블렌딩 판정. 같은 14k session val 정렬 확인 후 softmax 가중평균."""
import torch

L = torch.load("/mnt/c/dacon/_track2_smoke/large_val_logits.pt", map_location="cpu")
K = torch.load("/mnt/c/dacon/_track2_smoke/kf_val_logits.pt", map_location="cpu")
cls = L["classes"]; idx = {c: i for i, c in enumerate(cls)}

def prep(D):
    lg = D["logits"].float(); b = D.get("class_bias")
    ids = [str(i) for i in D["ids"]]; y = [int(v) for v in D["y_true"]]
    return lg, (torch.tensor(b).float() if b is not None else None), ids, y

Llg, Lb, Lids, Ly = prep(L)
Klg, Kb, Kids, Ky = prep(K)
print(f"large ids={len(Lids)} | kf ids={len(Kids)} | classes match={L['classes']==K['classes']}")

# 정렬: large ids 기준으로 kf 재배열
Kmap = {i: n for n, i in enumerate(Kids)}
common = [i for i in Lids if i in Kmap]
print(f"공통 id={len(common)} / {len(Lids)}")
order_L = [Lids.index(i) for i in common]
order_K = [Kmap[i] for i in common]
yt = torch.tensor([Ly[Lids.index(i)] for i in common])

Lp = torch.softmax(Llg[order_L] + (Lb if Lb is not None else 0), dim=1)
Kp = torch.softmax(Klg[order_K] + (Kb if Kb is not None else 0), dim=1)

def macro(prob):
    pred = prob.argmax(1)
    fs = []
    for c in range(len(cls)):
        tp = int(((pred == c) & (yt == c)).sum()); fp = int(((pred == c) & (yt != c)).sum()); fn = int(((pred != c) & (yt == c)).sum())
        p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
        fs.append(2*p*r/(p+r) if p+r else 0)
    return sum(fs)/len(fs)

mL = macro(Lp); mK = macro(Kp)
print(f"\nlarge 단독: {mL:.4f}")
print(f"kf 단독:    {mK:.4f}")
print("\n=== 가중 블렌딩 (w=large 비중) ===")
best = (max(mL, mK), None)
for w in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    m = macro(w * Lp + (1 - w) * Kp)
    tag = " *" if m > max(mL, mK) else ""
    print(f"  w={w:.2f}: {m:.4f} (vs 최고단독 {max(mL,mK):.4f}, Δ{m-max(mL,mK):+.4f}){tag}")
    if m > best[0]: best = (m, w)
if best[1]:
    print(f"\n★ best 블렌드: {best[0]:.4f} @ w={best[1]} — 최고단독 대비 +{best[0]-max(mL,mK):.4f}")
    print(f"   large 단독(0.7733 Public 소스) 대비 fixed +{best[0]-mL:+.4f}")
else:
    print("\n블렌딩 개선 없음 — 상관 높음")
