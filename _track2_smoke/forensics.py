"""탐색4종 포렌식 — 관계형 피처 판별력 + 라벨 결정성 측정 (70k 전체, CPU).
가설: 에이전트는 '경로를 알면 read, 모르면 grep/glob, 구조 파악은 list'.
검증: P(class | 관계형 피처) 가 33%(균등) 대비 얼마나 뾰족한가."""
import json, re, csv
from collections import Counter, defaultdict

TRAIN = "/mnt/c/dacon/open/data/train.jsonl"
LABELS = "/mnt/c/dacon/open/data/train_labels.csv"
EXPL = {"read_file", "grep_search", "list_directory", "glob_pattern"}

lab = {}
with open(LABELS, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        # 컬럼명 자동 탐지
        ks = list(row.keys())
        lab[row[ks[0]]] = row[ks[1]]

RE_PATH = re.compile(r"[\w.\-/\\]+\.[A-Za-z]{1,12}\b|[\w.\-]+/[\w.\-/]+")
RE_WILD = re.compile(r"\*")
RE_IDENT = re.compile(r"\b[a-z]+_[a-z_]+\b|\b[a-z]+[A-Z][A-Za-z]+\b")  # snake/camel
def norm(p):
    return p.lower().lstrip("./").replace("\\", "/").strip(".,!?'\"`")

def paths_in(text):
    return {norm(m.group(0)) for m in RE_PATH.finditer(text or "") if "." in m.group(0) or "/" in m.group(0)}

stats = defaultdict(Counter)          # feature_combo -> class counter
feat_rate = defaultdict(Counter)      # class -> feature counter
n_expl = 0

with open(TRAIN, encoding="utf-8") as f:
    for ln in f:
        o = json.loads(ln)
        y = lab.get(o["id"])
        if y not in EXPL:
            continue
        n_expl += 1
        p = o.get("current_prompt") or ""
        sm = o.get("session_meta") or {}
        ws = sm.get("workspace") or {}
        known = {norm(x) for x in (ws.get("open_files") or [])}
        prev, prev_res = None, ""
        for e in o.get("history") or []:
            if e.get("role") == "assistant_action":
                prev = e.get("name")
                prev_res = (e.get("result_summary") or "").lower()
                for v in (e.get("args") or {}).values():
                    if isinstance(v, str):
                        known |= paths_in(v)
                known |= paths_in(prev_res)
        pp = paths_in(p)
        # 파일경로(확장자 있는 것)만 따로
        pfiles = {x for x in pp if re.search(r"\.[a-z]{1,12}$", x)}
        f_path = bool(pfiles)
        f_known = bool(pfiles & known)
        f_unknown = f_path and not f_known
        f_wild = bool(RE_WILD.search(p))
        f_dir = any(x.endswith("/") or ("." not in x.split("/")[-1]) for x in pp) and not f_path
        f_ident = bool(RE_IDENT.search(p)) and not f_path
        combo = (
            ("K" if f_known else ("U" if f_unknown else "-")),
            ("W" if f_wild else "-"),
            ("D" if f_dir else "-"),
            ("I" if f_ident else "-"),
            f"prev={prev}" if prev in EXPL else "prev=other",
        )
        stats[combo][y] += 1
        for name, v in [("path_any", f_path), ("path_KNOWN", f_known), ("path_UNKNOWN", f_unknown),
                        ("wildcard", f_wild), ("dir_token", f_dir), ("identifier", f_ident)]:
            if v:
                feat_rate[y][name] += 1
        feat_rate[y]["_n"] += 1

print(f"탐색4 총 {n_expl}건\n")
print("=== 클래스별 피처 발생률 ===")
hdr = ["path_any", "path_KNOWN", "path_UNKNOWN", "wildcard", "dir_token", "identifier"]
print(f"{'class':16s}" + "".join(f"{h:>13s}" for h in hdr))
for c in sorted(EXPL):
    n = feat_rate[c]["_n"]
    print(f"{c:16s}" + "".join(f"{feat_rate[c][h]/n:12.1%} " for h in hdr))

print("\n=== 고순도 피처 조합 (support>=200, 최다클래스 비율 내림차순) ===")
rows = []
for combo, cnt in stats.items():
    tot = sum(cnt.values())
    if tot < 200:
        continue
    top_c, top_n = cnt.most_common(1)[0]
    rows.append((top_n / tot, tot, top_c, combo, dict(cnt)))
rows.sort(reverse=True)
for pur, tot, top_c, combo, dist in rows[:18]:
    print(f"  {pur:5.1%} n={tot:5d} -> {top_c:15s} combo={combo}")

# 조합 피처셋 전체의 '도달가능 상한' (각 조합에서 최다 클래스로 찍었을 때 정확도)
tot_all = sum(sum(c.values()) for c in stats.values())
correct = sum(c.most_common(1)[0][1] for c in stats.values())
print(f"\n이 피처셋의 Bayes 상한(조합별 최다클래스 선택 시 acc): {correct/tot_all:.3f}")
print("(현재 large 모델 탐색4 평균 recall ~0.58 과 비교)")
