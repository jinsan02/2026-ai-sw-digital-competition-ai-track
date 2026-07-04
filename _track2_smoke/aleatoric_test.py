"""aleatoric 증명 — 같은/거의같은 컨텍스트에 서로 다른 탐색라벨이 붙나?
컨텍스트 키 = current_prompt 정규화 + prev_action. 같은 키에 라벨이 갈리면 irreducible."""
import json, re, csv
from collections import defaultdict, Counter

TRAIN="/mnt/c/dacon/open/data/train.jsonl"; LABELS="/mnt/c/dacon/open/data/train_labels.csv"
EXPL={"read_file","grep_search","list_directory","glob_pattern"}
lab={}
with open(LABELS,encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ks=list(r.keys()); lab[r[ks[0]]]=r[ks[1]]

def normprompt(p):
    p=(p or "").lower()
    p=re.sub(r"[\w./\\-]+\.[a-z]{1,6}\b","<FILE>",p)   # 파일경로 마스킹
    p=re.sub(r"\d+","<N>",p); p=re.sub(r"[^\w<>]+"," ",p).strip()
    return " ".join(p.split()[:20])

# 키1: 정규화 프롬프트만 / 키2: +prev_action
g1=defaultdict(Counter); g2=defaultdict(Counter)
for ln in open(TRAIN,encoding="utf-8"):
    o=json.loads(ln); y=lab.get(o["id"])
    if y not in EXPL: continue
    prev=None
    for e in reversed(o.get("history") or []):
        if e.get("role")=="assistant_action": prev=e.get("name"); break
    np_=normprompt(o.get("current_prompt"))
    g1[np_][y]+=1; g2[(np_,prev)][y]+=1

def analyze(g,name):
    multi=[c for c in g.values() if sum(c.values())>=3]  # support>=3
    if not multi: print(f"{name}: support>=3 그룹 없음"); return
    # 각 그룹에서 '최다라벨 비율'(purity) — 낮을수록 라벨 갈림=aleatoric
    import statistics as st
    pur=[c.most_common(1)[0][1]/sum(c.values()) for c in multi]
    # 그룹 내 최다라벨로 다 찍었을 때 얻는 정확도(=이 피처의 Bayes 상한)
    tot=sum(sum(c.values()) for c in multi); corr=sum(c.most_common(1)[0][1] for c in multi)
    n_split=sum(1 for c in multi if len(c)>=2)
    print(f"{name}: 그룹 {len(multi)}개(support>=3), 라벨갈림 그룹 {n_split}({n_split*100//len(multi)}%)")
    print(f"  평균 purity={st.mean(pur):.2f}  Bayes상한(이 키로)={corr/tot:.3f}")
    # 예시: 완전 갈린 그룹 3개
    ex=[c for c in multi if len(c)>=3][:3]
    for c in ex: print("   예:", dict(c))

analyze(g1,"프롬프트만(파일마스킹)")
analyze(g2,"프롬프트+prev_action")
