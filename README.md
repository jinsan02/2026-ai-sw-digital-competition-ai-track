# 2026 AI·SW중심대학 디지털 경진대회 — AI부문 (팀 토큰강도)

AI 코딩 에이전트의 다음 행동(action)을 14개 클래스 중 하나로 예측하는 코드 제출형 대회.
**예선 Macro-F1 0.7977 · 예선 8위로 본선 진출 · 대회 최종 9위/269팀 · AI부문 후원기업상** (한신대학교 팀 토큰강도)

## 태스크

- 입력: `session_meta`(세션/작업공간 메타) + `history`(0~12개 user/assistant_action 교대) + `current_prompt`
- 출력: 다음 행동 1개 — `read_file` / `grep_search` / `list_directory` / `glob_pattern` / `edit_file` / `write_file` / `apply_patch` / `run_bash` / `run_tests` / `lint_or_typecheck` / `ask_user` / `plan_task` / `web_search` / `respond_only`
- 평가지표: **Macro-F1** (14개 클래스) · Public = 최종 점수 (private holdout 없음)
- 제약: 패키지 ≤ 1GB · 추론 ≤ 10분 · 인터넷 불가 · T4 16GB

## 대회 결과

| 항목 | 값 |
|---|---|
| 예선 최종 Macro-F1 | **0.7977** (베이스라인 0.4358 → **+0.3619**) |
| 예선 순위 | **8위**로 본선 진출 (1위와 0.00112 / 7위와 0.00019) |
| 대회 최종 순위 | **9위 / 269팀** |
| 수상 | **AI부문 후원기업상(팀스파르타 상)** |
| 추론 시간 | **7분 15초** / 10분 |
| 패키지 | **1,005.6MB** / 1GB (0.5B 모델 3개) |
| 제출 | Dacon 공식 115회 / 내부 로그 120회(유효 114 · 오류 6), 최고점 갱신 24회 |

> **순위 구분:** `finals/*`의 8위·12팀 표기는 2026-07-15 예선 리더보드와 본선 진출선에 관한 당시 기록이다. 발표·심사를 포함한 대회 전체 결과는 최종 **9위/269팀**이다.

### 최종 아키텍처

```
입력 (test.jsonl 30,000행) → current_v1 직렬화 (max_len 384)
  ↓
[main]     HCX-0.5B · amw4 recipe s7070 · INT8 (568MB)  ← 전 행 추론
  ↓  저마진 라우팅: top1-top2 margin < 1.0 인 행만 (34.1%)
[model_b]  구 recipe s909 · INT4 (292MB)  ┐
[model_c]  amw4 recipe s42 · INT4 (292MB) ┘ → z-centered 로짓 평균
  ↓
후처리 룰 12종 (전부 OOF 5-fold × 교차시드 통과분)
  ↓
output/submission.csv
```

### 레버별 Public 기여 (실측)

| 단계 | Public | Δ |
|---|---|---|
| HCX-0.5B 단독 (non-KD refit) | 0.7852 | — |
| + KD (m8 교사, α0.5·T3) | 0.7891 | +0.0039 |
| + 조건부 α (Weak4 행만 0.7) | 0.7896 | +0.0005 |
| + consensus sieve | 0.7939 | +0.0043 |
| + 앙상블 · 룰 스택 · 자기증류 | 0.7972 | +0.0033 |
| + Weak4 action-margin KD (margin 1.0, s7070) | **0.7977** | +0.0005 |

## 핵심 기술 기여 (팀 성과)

> 아래 6건은 **4인 팀(토큰강도)의 공동 성과**다. 개인별 기여 범위는 [CONTRIBUTIONS.md](CONTRIBUTIONS.md)에 파일 단위로 구분해 두었다.

1. **KD 교사 선택 법칙** — 교사의 train 암기율과 증류 이득이 완전 단조 역상관. "강한 교사"가 아니라 "학생이 모르는 걸 아는 교사"가 좋은 교사 (9B 교사 이득 ±0.000, 동계열 1.5B는 −0.011).
2. **Consensus Sieve × 조건부 α** — 독립 모델 3개의 행 단위 합의 수로 backbone gradient를 0/0.25/0.75/1 스케일. 라벨 노이즈가 표현 학습을 손상시키는 것을 차단 (단독 +0.0043, 최대 레버). **임준현이 방법·코드·실행·해석을 담당**했고, 노진산이 팀장으로 검토해 최종 시스템에 통합·채택했다.
3. **Action-Margin KD** — 교사 기준 hard-negative top-3에 대한 마진을 SmoothL1로 정렬. λ는 gradient 노름 비율 0.10으로 1회 자동 보정 (탐색 비용 0).
4. **저장 전용 양자화 코덱** — 임준현이 int8 row-wise 초안을 만들고, 노진산이 int4 group-128로 확장해 최종 3모델 팩에 적용했다. 로드 시 fp16으로 복원해 양자화 추론 커널의 속도 영향을 피했고, main은 int8 일치율 게이트를 통과한 반면 int4는 보조 모델에만 제한해 1GB를 충족했다.
5. **저마진 라우팅** — 확신이 낮은 34.1%만 앙상블 통과. 점수와 속도를 동시에 개선한 유일한 레버 (3점 dose-response로 임계값 실측 확정). 최종 `script.py`/`pack/script.py` 추론 파이프라인은 **노진산이 주도 구현**했다.
6. **시간 예산 역산** — 노진산이 서버 제출을 운영하며 로그 회귀로 추론시간 = f(실제 토큰 수)를 규명했다. 동적 패딩이라 길이 캡 축소는 효과 0이라는 반직관 결론, 제출 전 서버 시간 예측 가능.

### 노진산 개인 기여 요약

- **직접 구현** — `train_transformer.py` 공동 구현(임준현과 공동), 최종 추론 파이프라인 주도,
  int4 코덱 확장·적용, `export_teacher_logits.py`·`package_submission.py`·`train.py`,
  인코더·초기 제출·기각 실험 축 구현
- **직접 운영·결정** — 본인 서버에서 실험과 공식 115회 제출을 운영하고, Public 결과를 바탕으로
  다음 실험의 우선순위·채택·중단 및 최종 제출본을 결정
- **공동 작업** — 데이터 분석은 목원주와 공동, 발표자료·그림은 김태연과 공동
- **구분해야 할 팀원 기여** — Consensus Sieve와 `colab/*`은 임준현 담당. 팀장 관할·통합을
  직접 설계·구현으로 바꿔 말하지 않음

## 팀 구성과 이 저장소의 성격

이 대회는 **한신대학교 4인 팀 「토큰강도」**(노진산·임준현·목원주·김태연)로 참가했다. 그런데 이 저장소의 커밋은 `jinsan02` 단독 5건뿐이다. 이유는 다음과 같다.

- 이 저장소는 팀 공용 원격이 아니라 **팀장(노진산)의 개인 작업 저장소**였다. 팀원은 각자 로컬·Colab에서 작업하고 산출물을 **Google Drive와 Slack으로 교환**했다.
- `.gitignore`에 `teammate_output/`이 `# 팀원 작업물 (참고용 로컬 보관, 이 레포에 커밋 금지)`로 명시되어 있다 — 팀원 산출물을 의도적으로 제외했다.
- 본선 자료와 제출 코드는 대회 종료 후(`b53f027`) **팀장이 일괄 정리해 한 번에 커밋**했다. 커밋 이력은 작성 주체가 아니라 **아카이브 주체**를 나타낸다.

따라서 **커밋 로그는 기여도의 근거가 되지 않는다.** 누가 무엇을 했는지는 다음 두 문서에 정리했다.

| 문서 | 내용 |
|---|---|
| [CONTRIBUTIONS.md](CONTRIBUTIONS.md) | 본인 직접 작성 / 본인 실험·전략 결정 / 팀원 산출물을 파일 단위로 구분하고, 남은 미확정 2건을 별도 표시. 과대 서술 정정표, 수치↔근거 매핑, Public 리더보드 사용 방식, ±0.002의 정확한 정체 포함 |
| [docs/interview_evidence.md](docs/interview_evidence.md) | 결정 · 근거(파일:행) · 결과 · 본인 역할 표와 파일별 구현 분업 |

### 검증 방법론에 대한 정확한 서술

이 저장소의 일부 발표용 문서(`finals/14_발표대본_초안.md`)에 "OOF↔Public 캘리브레이션 오차 ±0.002"라는 표현이 있으나, **저장소 근거로는 뒷받침되지 않는다.**

- `0.002`는 캘리브레이션 오차가 아니라 **"단일 Public 델타가 이 값 미만이면 노이즈로 버린다"는 판정 임계값**이다. (`finals/ref_슬랙로그_태연백업.md:394`, `finals/13_QnA_통합본.md:49`)
- 실제로 **측정된** OOF↔Public 편차는 `CLAUDE.md:31` 기준 **CE 라인 OOF≈Public / focal 라인 OOF→Public +0.013**이며, 로컬 KD 스크린은 fold-교사 누수로 **+0.012 인플레**였다. (`finals/03` §1.5)
- 승격 판정도 전 기간 동일하지 않았다. 07-04 이후 **Public-gated 프로모션으로 전환**하고 OOF는 앙상블·룰 튜닝 도구로 강등했다. **"제출 없이 판정"은 룰에 대해서만 사실**이다.

자세한 추적 결과는 [CONTRIBUTIONS.md §9](CONTRIBUTIONS.md)에 있다.

## 디렉토리 구조

```
dacon/
├── CONTRIBUTIONS.md           # 개인/팀 기여 범위 구분 · 과대 서술 정정 · 수치↔근거 매핑
├── docs/
│   └── interview_evidence.md  # 결정·근거·결과·본인 역할 표 + 파일별 구현 분업
├── finals/                    # 본선 준비 자료 (아키텍처·기술기여·검증문화·제출로그·QnA 등)
│   ├── 01_솔루션_아키텍처.md
│   ├── 02_핵심기술_기여.md
│   ├── 03_검증문화_기각축.md
│   ├── 05_학습코드_재현성.md
│   ├── 13_QnA_통합본.md      # 심사 대비 32문 QnA 뱅크
│   ├── assets/                # 발표용 그림
│   └── junhyun/               # 최종 챔피언(amw4) 학습 명세
├── finals_code_submission/    # 본선 학습·추론 코드 아카이브 (데이터·가중치·교사 로짓 제외)
│   ├── train_transformer.py   # 학습 본체 (노진산·임준현 공동)
│   ├── quantize_checkpoint.py # int8 row-wise 초안 (임준현)
│   ├── quantize_int4.py       # int4 group-128 확장·적용 (노진산)
│   ├── build_oof_consensus.py # consensus sieve 방법·코드·실행·해석 (임준현)
│   ├── export_teacher_logits.py # 교사 로짓 추출 (노진산)
│   ├── pack/script.py         # 최종 추론 파이프라인 주도 (노진산)
│   └── colab/                 # 시드별 학습 plan JSON (임준현)
├── open/                      # 대회 배포 원본 (데이터는 .gitignore 제외)
├── notebooks/ · src/ · submit/  # 초기 TF-IDF 계열 (인코더 라인 종료)
└── submissions/               # 제출 이력 (zip은 git 제외)
```

> 모델 가중치(`*.safetensors`/`*.pt`), 교사 로짓, 대회 데이터, 제출 zip은 `.gitignore`로 제외됐다.
> 저장소는 학습·패키징·추론 코드를 감사할 수 있게 하지만, 제외된 입력 없이 최종 점수의 완전 재현을 보장하지 않는다.

## 재현 범위

아래 명령과 레시피는 보존돼 있지만, 이 저장소만으로 최종 0.7977을 즉시 재현할 수는 없다.
대회 데이터, 베이스 모델, 교사 로짓/가중치와 최종 제출 가중치를 별도로 확보하고 동일 환경을
구성해야 한다.

```bash
# 챔피언 레시피 (amw4) — 상세는 finals/05_학습코드_재현성.md
python train_transformer.py \
  --base-model naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
  --lr 2e-5 --split session --serializer current_v1 --max-length 384 \
  --epochs 3 --batch-size 16 --loss focal --focal-gamma 2.0 \
  --class-weight-power 0.5 --label-smoothing 0.02 \
  --distill-logits m8_qwen35_refit_train70k_fp16.pt \
  --distill-alpha 0.5 --distill-alpha-weak 0.7 --distill-temp 3.0 \
  --consensus-reliability 20260710_m7_m8_v6_oof_consensus.pt \
  --consensus-backbone-weights 0,0.25,0.75,1 \
  --action-margin-kd-target-grad-ratio 0.10 --action-margin-kd-topk 3 \
  --action-margin-kd-label-scope weak4 \
  --seed 7070 --save-fp16 --final-model --final-only

# 양자화 (배포 팩 조립)
python quantize_checkpoint.py quantize --input model.safetensors --output model.int8.safetensors
python quantize_int4.py quantize --input model.safetensors --output model.int4.safetensors
```

학습 환경: Python 3.10~3.11 / torch 2.5~2.7 (cu121·cu128) / transformers 4.51.3
추론 환경(평가 서버): Ubuntu 22.04.5 / T4 16GB / Python 3.11.15 / transformers 4.46.3

## 자원 출처 및 라이선스

- **베이스 모델**: [naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B)
  HyperCLOVA X SEED Model is licensed under the HyperCLOVA X SEED Model License Agreement, Copyright © NAVER Corp.
- **교사 모델**: Qwen3.5 계열 (Apache-2.0) — 로짓 추출에만 사용, 배포 패키지 미포함
- 외부 데이터·유료 API 미사용
