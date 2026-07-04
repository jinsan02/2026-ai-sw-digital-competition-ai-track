"""Track2 스모크 — 탐색4종(read/grep/list/glob) 구조신호 rule boost 개념검증.
저장된 large val_logits(14k) 위에서, current_prompt 구조신호로 탐색 클러스터 재랭크.
macro-F1 before/after + 탐색 per-class recall 비교. GPU 무관, CPU."""
import json, re, glob
import torch
from collections import Counter

LOGITS = "/mnt/c/dacon/_track2_smoke/large_val_logits.pt"
TRAIN = "/mnt/c/dacon/open/data/train.jsonl"
EXPL = ["read_file", "grep_search", "list_directory", "glob_pattern"]

d = torch.load(LOGITS, map_location="cpu")
lg = d["logits"].float()
b = d.get("class_bias")
if b is not None:
    lg = lg + torch.tensor(b).float()
cls = d["classes"]; ids = d["ids"]; yt = [int(v) for v in d["y_true"]]
idx = {c: i for i, c in enumerate(cls)}
EXPL_ID = [idx[c] for c in EXPL]

# id -> sample
S = {}
for ln in open(TRAIN, encoding="utf-8"):
    o = json.loads(ln); S[o["id"]] = o

def prev_action(s):
    for e in reversed(s.get("history") or []):
        if e.get("role") == "assistant_action":
            return e.get("name")
    return None

# 구조 신호 정규식
EXT = r"(py|ts|tsx|js|jsx|java|go|rs|md|json|ya?ml|toml|sh|sql|txt|cfg|ini|xml|html|css|vue|cpp?|h|kt|rb|php|dockerfile)"
RE_FILE = re.compile(r"[\w./\\-]+\." + EXT + r"\b", re.I)
RE_WILD = re.compile(r"\*\*?|\?\w|/\*|\.\*")
RE_SEARCH = re.compile(r"찾|검색|어디[서에]|어느|which|where|search|grep|uses?\b|참조|호출|usage|referenc", re.I)
RE_DIR = re.compile(r"폴더|디렉|목록|list\b|directory|contents of|안에.*(뭐|파일|있)", re.I)
RE_READ = re.compile(r"보여|열어|열고|내용|확인|open\b|show\b|read\b|cat\b|봐줘|봐봐", re.I)

def feats(s):
    p = s.get("current_prompt") or ""
    has_file = bool(RE_FILE.search(p)) and not RE_WILD.search(p)
    has_wild = bool(RE_WILD.search(p))
    return has_file, has_wild, bool(RE_SEARCH.search(p)), bool(RE_DIR.search(p))

def macro(pred):
    f = []
    for c in range(len(cls)):
        tp = sum(1 for p, y in zip(pred, yt) if p == c and y == c)
        fp = sum(1 for p, y in zip(pred, yt) if p == c and y != c)
        fn = sum(1 for p, y in zip(pred, yt) if p != c and y == c)
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return sum(f)/len(f), f

base_pred = lg.argmax(1).tolist()
base_macro, base_f = macro(base_pred)
print(f"baseline macro={base_macro:.4f}")
print("  탐색 recall:", {c: round(base_f[idx[c]], 3) for c in EXPL})

top2 = torch.topk(lg, 2, dim=1).values
margin = (top2[:, 0] - top2[:, 1]).tolist()

def run(w, marg, supp):
    """directional 재분배: 신호에 맞는 소수클래스 +w, read는 -supp 억제(read 신호 없을 때)."""
    newlg = lg.clone()
    n = Counter()
    for i, ii in enumerate(ids):
        if base_pred[i] not in EXPL_ID or margin[i] > marg:
            continue
        s = S.get(ii)
        if not s:
            continue
        hf, hw, hs, hd = feats(s)
        hr = bool(RE_READ.search(s.get("current_prompt") or ""))
        if hw:  # 와일드카드 → glob
            newlg[i, idx["glob_pattern"]] += w; n["wild"] += 1
        if hs and not hr:  # 검색동사(&read동사 없음) → grep, read 억제
            newlg[i, idx["grep_search"]] += w; newlg[i, idx["read_file"]] -= supp; n["search"] += 1
        if hd and not hr:  # 디렉토리 동사 → list, read 억제
            newlg[i, idx["list_directory"]] += w; newlg[i, idx["read_file"]] -= supp; n["dir"] += 1
        if hf and hr and not hs and not hd:  # 구체파일+read동사+검색/디렉 없음 → read
            newlg[i, idx["read_file"]] += w; n["read"] += 1
    pred = newlg.argmax(1).tolist()
    m, f = macro(pred)
    return m, f, n

print("\n=== directional 재분배 스윕 ===")
best = (base_macro, None)
for marg in [1.5, 2.5]:
    for w in [1.0, 2.0]:
        for supp in [0.0, 1.0, 2.0]:
            m, f, n = run(w, marg, supp)
            tag = " *" if m > base_macro else ""
            print(f"  margin<={marg} w={w} supp={supp}: macro={m:.4f} (Δ{m-base_macro:+.4f}) {dict(n)}{tag}")
            if m > best[0]:
                best = (m, (marg, w, supp, f))
if best[1]:
    marg, w, supp, f = best[1]
    print(f"\nbest: macro={best[0]:.4f} (Δ{best[0]-base_macro:+.4f}) @ margin<={marg} w={w} supp={supp}")
    print("  탐색 recall after:", {c: round(f[idx[c]], 3) for c in EXPL})
else:
    print("\n개선 없음 — 신호 자체가 부족 (트랜스포머가 이미 다 씀)")
