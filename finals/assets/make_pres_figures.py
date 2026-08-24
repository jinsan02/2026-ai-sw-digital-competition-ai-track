# -*- coding: utf-8 -*-
"""발표용 그림 3종 생성 — 04_제출로그.md 파싱 기반.
① gen_fig1_journey.png  : 점수 여정 (제출 순번 × 점수, 페이즈 색 + 최고점 계단선)
② gen_fig2_teacher.png  : KD 교사 train 일치율 ↔ Public 이득 산점도
③ gen_fig3_arch.png     : 최종 추론 파이프라인 다이어그램
실행: KMP_DUPLICATE_LIB_OK=TRUE python make_pres_figures.py
"""
import re, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, GREY, ORANGE = "#3B7DD8", "#D94F4F", "#999999", "#E8923A"

# ── ① 점수 여정 ──────────────────────────────────────────────
md = open(os.path.join(HERE, "..", "04_제출로그.md"), encoding="utf-8").read()
rows = []  # (id, score, phase)
phase = 0
for line in md.split("\n"):
    if line.startswith("## Phase"):
        phase = int(re.search(r"Phase (\d)", line).group(1))
    m = re.match(r"\|\s*(\d{5})(?:/\d)?\s*\|[^|]*\|[^|]*\|\s*\**(0\.\d+)\**\s*\|", line)
    if m and phase:
        rows.append((int(m.group(1)), float(m.group(2)), phase))
rows.sort()
print(f"파싱된 유효 제출: {len(rows)}건")

xs = list(range(1, len(rows) + 1))
ys = [r[1] for r in rows]
ph = [r[2] for r in rows]
pcolors = {1:"#8E8E8E",2:"#5B9BD5",3:"#70AD47",4:"#FFC000",5:"#ED7D31",6:"#C00000"}
best, steps = 0.0, []
for x, y in zip(xs, ys):
    if y > best: best = y
    steps.append(best)

fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
ax.scatter(xs, ys, c=[pcolors[p] for p in ph], s=34, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4)
ax.step(xs, steps, where="post", color="#222222", lw=1.6, zorder=2, label="당시 최고점")
marks = [(1,0.4358,"TF-IDF 0.436"), None]
ann = {0.7080569737:"XLM-R 0.708", 0.7732655474:"large 0.773", 0.7807288599:"Qwen3-0.6B 0.781",
       0.7851528243:"HCX-0.5B 0.785", 0.7891279075:"KD 0.789", 0.7938816426:"sieve×condα 0.794",
       0.7963584846:"트리오+룰 0.796", 0.797181265:"자기증류 0.797", 0.7976673203:"Weak4-AM 0.7977"}
seen = set()
for x, y in zip(xs, ys):
    if y in ann and y not in seen:
        seen.add(y)
        ax.annotate(ann[y], (x, y), textcoords="offset points", xytext=(6, 10), fontsize=9.5, fontweight="bold")
ax.annotate("TF-IDF 0.436", (1, 0.4358), textcoords="offset points", xytext=(6, 8), fontsize=9.5, fontweight="bold")
ax.set_ylim(0.42, 0.815)
ax.set_xlabel("제출 순번 (2026-07-01 ~ 07-15, 유효 채점)")
ax.set_ylabel("Public Macro-F1")
ax.set_title("점수 여정 — 15일, 제출 114회, 최고점 갱신 24회:  0.436 → 0.7977 (+0.362)", fontsize=14, fontweight="bold", loc="left")
handles = [plt.Line2D([], [], marker="o", ls="", color=pcolors[i], label=l) for i, l in
           [(1,"P1 베이스라인·인코더"),(2,"P2 인코더→디코더"),(3,"P3 HCX+KD 교사축"),(4,"P4 Sieve"),(5,"P5 앙상블+룰"),(6,"P6 자기증류·Weak4-AM")]]
handles.append(plt.Line2D([], [], color="#222222", lw=1.6, label="당시 최고점"))
ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "gen_fig1_journey.png"), bbox_inches="tight")
print("gen_fig1_journey.png 저장")

# ── ② 교사 일치율 ↔ KD 이득 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8.6, 6), dpi=200)
pts = [  # (train 일치율, Public 이득 vs non-KD 0.7852, 라벨, 색)
    (0.811, +0.0039, "m8 (Qwen3.5-0.8B)", BLUE),
    (0.870, +0.0033, "q35 (Qwen3.5-4B)", BLUE),
    (0.891, -0.0108, "t15 (동계열 HCX-1.5B)", RED),
    (0.9428, +0.0001, "m9 (Qwen3.5-9B)", GREY),
]
for x, y, l, c in pts:
    ax.scatter(x, y, s=180, c=c, zorder=3, edgecolors="white", linewidths=1.2)
    ax.annotate(l, (x, y), textcoords="offset points", xytext=(10, 8), fontsize=11)
ax.axhline(0, color="#555555", lw=1, ls="--")
ax.axvspan(0.78, 0.84, alpha=0.10, color=BLUE)
ax.text(0.782, -0.0125, "이상적 구간 (~0.8)\n= 가르칠 이견 보유", fontsize=9.5, color=BLUE)
ax.set_xlabel("교사의 train 라벨 일치율 (암기 정도)")
ax.set_ylabel("KD Public 이득 (vs non-KD 0.7852)")
ax.set_title("KD 교사 법칙 — 일치율 높을수록(암기) 이득 소멸, 동계열은 손해", fontsize=13.5, fontweight="bold", loc="left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "gen_fig2_teacher.png"), bbox_inches="tight")
print("gen_fig2_teacher.png 저장")

# ── ③ 아키텍처 다이어그램 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=200)
ax.axis("off")
def box(x, y, w, h, text, fc="#EAF1FB", ec=BLUE, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2, text="", ty=0.25):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color="#444444", lw=1.5))
    if text: ax.text((x1+x2)/2, max(y1,y2)+ty, text, ha="center", fontsize=8.8, color="#444444")
box(0.2, 2.2, 1.7, 1.3, "test.jsonl\n30,000행", "#F5F5F5", GREY, bold=True)
box(2.5, 2.2, 2.0, 1.3, "직렬화 current_v1\nlen384 · 동적 패딩")
box(5.1, 2.2, 2.3, 1.3, "main\nHCX-0.5B · trio-KD 학생\nINT8 (568MB)", bold=True)
box(8.2, 3.4, 2.3, 1.1, "model_b · INT4 (292MB)")
box(8.2, 1.2, 2.3, 1.1, "model_c · INT4 (292MB)")
box(11.1, 2.2, 1.9, 1.3, "z-centered\n로짓 평균", "#FFF3E2", ORANGE)
box(13.5, 2.2, 2.1, 1.3, "룰 스택 12종\nOOF 5-fold 게이트", "#FDECEC", RED, bold=True)
box(16.2, 2.2, 1.7, 1.3, "submission\n.csv", "#F5F5F5", GREY, bold=True)
arrow(1.9, 2.85, 2.5, 2.85); arrow(4.5, 2.85, 5.1, 2.85)
arrow(7.4, 3.1, 8.2, 3.9, "margin<1.0\n(34.1%만)")
arrow(7.4, 2.6, 8.2, 1.8)
arrow(10.5, 3.9, 11.2, 3.2); arrow(10.5, 1.7, 11.2, 2.4)
arrow(7.4, 2.85, 11.1, 2.85, "~66% main 단독", 0.9)
arrow(13.0, 2.85, 13.5, 2.85); arrow(15.6, 2.85, 16.2, 2.85)
ax.text(0.2, 4.6, "최종 추론 파이프라인 — 0.5B×3 앙상블을 T4 · 7분15초 / 1005.6MB 안에", fontsize=14, fontweight="bold")
ax.text(0.2, 0.3, "동적 패딩: 시간 = f(실제 토큰 수) → 저마진 라우팅으로 멤버 통과 34.1% 제한 = 단일팩 대비 +1:23만 지불  |  Powered by HyperCLOVA X",
        fontsize=9.5, color="#555555")
ax.set_xlim(0, 18.2); ax.set_ylim(0, 5.2)
fig.savefig(os.path.join(HERE, "gen_fig3_arch.png"), bbox_inches="tight")
print("gen_fig3_arch.png 저장")

# ── ④ 룰 스택 기여 막대 (슬라이드 8) ─────────────────────────
# Public 실측 델타 (04_제출로그 도입 순서). *표 = 멤버 변경 동시 도입이라 순수 룰 기여 아님.
rules = [  # (라벨, Public 델타, 혼입 여부)
    ("R1  budget<5k: ws→ask_user",        0.00108, False),
    ("R1b apply_patch→edit_file",          0.00012, False),
    ("R1c budget존 au 로짓 부스트 *",      0.00029, True),
    ("R1d *",                              0.00006, True),
    ("R1e",                                0.00001, False),
    ("R1f+g+h 스택 (h: 83플립 0-harm)",    0.00033, False),
    ("R1h-wide + garnish",                 0.00021, False),
    ("R1i + seq-exec (transfer gate)",     0.00025, False),
]
fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
ys = list(range(len(rules)))[::-1]
for y, (lab, d, mixed) in zip(ys, rules):
    ax.barh(y, d * 1e5, color=ORANGE if mixed else BLUE, alpha=0.9,
            hatch="//" if mixed else None, edgecolor="white")
    ax.text(d * 1e5 + 1.2, y, f"+{d:.5f}", va="center", fontsize=10, fontweight="bold")
    ax.text(-1.2, y, lab, va="center", ha="right", fontsize=10.5)
ax.set_yticks([])
ax.set_xlim(0, 125)
ax.set_xlabel("Public Macro-F1 기여 (단위: 0.00001)")
ax.set_title("룰 스택 12종 — Public 실측 기여 (도입 순서, 합계 약 +0.0023: 0.7943→0.7966)",
             fontsize=13.5, fontweight="bold", loc="left")
ax.text(0, -2.45, "* 주황 빗금 = 멤버 변경과 동시 도입(순수 룰 기여 아님)  |  전 룰: 5-fold × 교차시드 rescue>harm 게이트 통과분만 배포 · 룰 간 충돌(2룰 이상 터치 행) 0건",
        fontsize=9, color="#555555")
ax.grid(axis="x", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "gen_fig4_rules.png"), bbox_inches="tight")
print("gen_fig4_rules.png 저장")

# ── ⑤ 추론 속도 여정 막대 (슬라이드 12) ──────────────────────
# 전부 채점 서버 실측 (04_제출로그 시간 열).
speed = [  # (라벨, 초, 점수 라벨, 색)
    ("Qwen3-0.6B 단독 (m7)",              535, "0.7807", GREY),
    ("Qwen3.5-0.8B probe",                600, "시간초과", RED),
    ("HCX-0.5B 단일 (sieve×condα)",       358, "0.7939", BLUE),
    ("2모델 앙상블 + R1",                 447, "0.7953", BLUE),
    ("3모델 트리오 + 룰 12종",            449, "0.7966", BLUE),
    ("amw4 트리오 @1.25 (39.7% 라우팅)",  475, "0.7960", ORANGE),
    ("최종 amhyb @1.0 (34.1% 라우팅)",    435, "0.7977", "#2E7D32"),
]
fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
xs2 = list(range(len(speed)))
for x, (lab, s, sc, c) in zip(xs2, speed):
    ax.bar(x, s, color=c, alpha=0.9, width=0.62,
           hatch="//" if sc == "시간초과" else None, edgecolor="white")
    ax.text(x, s + 12, f"{s//60}:{s%60:02d}" if sc != "시간초과" else "DNF",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(x, s / 2, sc, ha="center", fontsize=9.5, color="white", fontweight="bold", rotation=90)
ax.axhline(600, color=RED, lw=1.6, ls="--")
ax.text(len(speed) - 0.4, 608, "제한 10:00", color=RED, fontsize=10, ha="right", fontweight="bold")
ax.set_xticks(xs2)
ax.set_xticklabels([l for l, *_ in speed], rotation=18, ha="right", fontsize=9.5)
ax.set_ylabel("추론 시간 (초, T4 채점 서버 실측)")
ax.set_ylim(0, 660)
ax.set_title("속도 여정 — 0.6B 단독보다 빠른 0.5B×3 앙상블: 저마진 라우팅이 점수와 속도를 동시에",
             fontsize=13.5, fontweight="bold", loc="left")
ax.text(-0.4, -170, "0.8B는 물리적 불가(DNF) → 시간예산 산식으로 0.5B 확정  |  라우팅 1.25→1.0 축소 = 점수 +0.0017·속도 -40초 동시 개선(@1.25는 시드 교란 s909 포함)  |  본선 속도 10% = 최종팩 7:15",
        fontsize=9, color="#555555")
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "gen_fig5_speed.png"), bbox_inches="tight")
print("gen_fig5_speed.png 저장")
