"""탐색 밖 혼동 클러스터가 분리가능(separable)한가 aleatoric인가?
소통(ask/plan), 실행(bash/tests/lint) 클러스터에 동일 aleatoric 테스트.
purity 높으면 = 라벨 안 갈림 = 분리가능 = 진짜 헤드룸."""
import json, re, csv
from collections import defaultdict, Counter
import statistics as st

TRAIN="/mnt/c/dacon/open/data/train.jsonl"; LABELS="/mnt/c/dacon/open/data/train_labels.csv"
lab={}
with open(LABELS,encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ks=list(r.keys()); lab[r[ks[0]]]=r[ks[1]]

def normprompt(p):
    p=(p or "").lower()
    p=re.sub(r"[\w./\\-]+\.[a-z]{1,6}\b","<FILE>",p)
    p=re.sub(r"\d+","<N>",p); p=re.sub(r"[^\w<>]+"," ",p).strip()
    return " ".join(p.split()[:20])

CLUSTERS={
    "탐색(read/grep/list/glob)":{"read_file","grep_search","list_directory","glob_pattern"},
    "소통(ask/plan)":{"ask_user","plan_task"},
    "실행(bash/tests/lint)":{"run_bash","run_tests","lint_or_typecheck"},
    "수정(edit/write/patch)":{"edit_file","write_file","apply_patch"},
}
groups={name:defaultdict(Counter) for name in CLUSTERS}
for ln in open(TRAIN,encoding="utf-8"):
    o=json.loads(ln); y=lab.get(o["id"])
    prev=None
    for e in reversed(o.get("history") or []):
        if e.get("role")=="assistant_action": prev=e.get("name"); break
    key=(normprompt(o.get("current_prompt")),prev)
    for name,members in CLUSTERS.items():
        if y in members: groups[name][key][y]+=1

print(f"{'클러스터':28s} {'그룹':>6s} {'라벨갈림':>8s} {'purity':>7s} {'Bayes상한':>9s}  판정")
for name,members in CLUSTERS.items():
    g=groups[name]
    multi=[c for c in g.values() if sum(c.values())>=3]
    if not multi:
        print(f"{name:28s}  support>=3 그룹 없음"); continue
    pur=[c.most_common(1)[0][1]/sum(c.values()) for c in multi]
    tot=sum(sum(c.values()) for c in multi); corr=sum(c.most_common(1)[0][1] for c in multi)
    nsplit=sum(1 for c in multi if len(c)>=2)
    ncls=len(members)
    verdict="ALEATORIC(막힘)" if corr/tot < (1/ncls+0.25) else "분리가능(헤드룸!)"
    print(f"{name:28s} {len(multi):6d} {nsplit*100//len(multi):6d}% {st.mean(pur):7.2f} {corr/tot:9.3f}  {verdict} (균등={1/ncls:.2f})")
