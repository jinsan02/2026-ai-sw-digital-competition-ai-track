"""탐색4 플러그인 임계값 재배분 — 저장된 large val_logits(14k) 위, 재학습 없음.
현재 2단계 bias 이후 로짓에서, 탐색4 클래스 bias만 coordinate-ascent로 미세조정.
목표: (A) 탐색4 mean-F1 최대화, (B) 전체 14클래스 macro가 유지/개선되는지 확인."""
import torch, glob

d = torch.load("/mnt/c/dacon/_track2_smoke/large_val_logits.pt", map_location="cpu")
lg = d["logits"].float()
b = d.get("class_bias")
if b is not None:
    lg = lg + torch.tensor(b).float()   # 이미 2단계 bias 적용된 상태에서 출발
cls = d["classes"]; yt = torch.tensor([int(v) for v in d["y_true"]])
idx = {c: i for i, c in enumerate(cls)}
EXPL = ["read_file", "grep_search", "list_directory", "glob_pattern"]
EID = [idx[c] for c in EXPL]

def f1_of(pred, c):
    tp = int(((pred == c) & (yt == c)).sum()); fp = int(((pred == c) & (yt != c)).sum()); fn = int(((pred != c) & (yt == c)).sum())
    p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0

def macro_all(bias):
    pred = (lg + bias).argmax(1)
    return sum(f1_of(pred, c) for c in range(len(cls)))/len(cls)

def expl_mean(bias):
    pred = (lg + bias).argmax(1)
    return sum(f1_of(pred, c) for c in EID)/len(EID)

def recalls(bias):
    pred = (lg + bias).argmax(1)
    return {cls[c]: round(f1_of(pred, c), 3) for c in EID}

base_bias = torch.zeros(len(cls))
print(f"출발 (2단계 bias 상태): 전체macro={macro_all(base_bias):.4f}  탐색mean-F1={expl_mean(base_bias):.4f}")
print("  탐색 recall:", recalls(base_bias))

# coordinate ascent — 탐색4 bias만, 전체 macro를 목적함수로 (macro가 바른 목표; 탐색만 올리고 전체 깨지면 무의미)
best = base_bias.clone()
grid = [-1.5, -1.0, -0.6, -0.3, -0.15, 0, 0.15, 0.3, 0.6, 1.0, 1.5]
for it in range(4):
    improved = False
    for c in EID:
        cur = macro_all(best)
        cand = best.clone()
        for delta in grid:
            t = best.clone(); t[c] += delta
            if macro_all(t) > cur:
                cur = macro_all(t); cand = t
        if not torch.equal(cand, best):
            best = cand; improved = True
    if not improved:
        break

print(f"\n최적화 후 (탐색4 bias만 조정): 전체macro={macro_all(best):.4f}  탐색mean-F1={expl_mean(best):.4f}")
print("  탐색 recall:", recalls(best))
print("  적용된 탐색 bias:", {cls[c]: round(float(best[c]), 2) for c in EID})
print(f"\nΔ 전체macro = {macro_all(best)-macro_all(base_bias):+.4f}")
print(f"Δ 탐색mean-F1 = {expl_mean(best)-expl_mean(base_bias):+.4f}")

# 참고: 탐색mean-F1만 목적으로 하면 (전체 무시) 얼마까지 가능한가 (상한 감각)
best2 = base_bias.clone()
for it in range(4):
    for c in EID:
        cur = expl_mean(best2); cand = best2.clone()
        for delta in grid:
            t = best2.clone(); t[c] += delta
            if expl_mean(t) > cur:
                cur = expl_mean(t); cand = t
        best2 = cand
print(f"\n[참고] 탐색mean-F1만 최대화 시: 탐색mean={expl_mean(best2):.4f} 이지만 전체macro={macro_all(best2):.4f} (전체 희생 확인용)")
