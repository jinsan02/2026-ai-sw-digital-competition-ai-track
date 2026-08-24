# 2026 AI·SW중심대학 디지털 경진대회 — AI부문 (팀 토큰강도)

AI 코딩 에이전트의 다음 행동(action)을 14개 클래스 중 하나로 예측하는 코드 제출형 대회.
**최종 Macro-F1 0.7977 · 12팀 중 8위 · 본선 진출** (한신대학교 팀 토큰강도)

## 태스크

- 입력: `session_meta`(세션/작업공간 메타) + `history`(0~12개 user/assistant_action 교대) + `current_prompt`
- 출력: 다음 행동 1개 — `read_file` / `grep_search` / `list_directory` / `glob_pattern` / `edit_file` / `write_file` / `apply_patch` / `run_bash` / `run_tests` / `lint_or_typecheck` / `ask_user` / `plan_task` / `web_search` / `respond_only`
- 평가지표: **Macro-F1** (14개 클래스) · Public = 최종 점수 (private holdout 없음)
- 제약: 패키지 ≤ 1GB · 추론 ≤ 10분 · 인터넷 불가 · T4 16GB

## 최종 결과

| 항목 | 값 |
|---|---|
| 최종 Macro-F1 | **0.7977** (베이스라인 0.4358 → **+0.3619**) |
| 순위 | 12팀 중 **8위** (1위와 0.00112 / 7위와 0.00019) |
| 추론 시간 | **7분 15초** / 10분 |
| 패키지 | **1,005.6MB** / 1GB (0.5B 모델 3개) |
| 제출 | 15일간 115회, 최고점 갱신 24회 |

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

## 핵심 기여

1. **KD 교사 선택 법칙** — 교사의 train 암기율과 증류 이득이 완전 단조 역상관. "강한 교사"가 아니라 "학생이 모르는 걸 아는 교사"가 좋은 교사 (9B 교사 이득 ±0.000, 동계열 1.5B는 −0.011).
2. **Consensus Sieve × 조건부 α** — 독립 모델 3개의 행 단위 합의 수로 backbone gradient를 0/0.25/0.75/1 스케일. 라벨 노이즈가 표현 학습을 손상시키는 것을 차단 (단독 +0.0043, 최대 레버).
3. **Action-Margin KD** — 교사 기준 hard-negative top-3에 대한 마진을 SmoothL1로 정렬. λ는 gradient 노름 비율 0.10으로 1회 자동 보정 (탐색 비용 0).
4. **저장 전용 양자화 코덱** — 자체 int8 row-wise / int4 group-128 구현. 로드 시 fp16 복원이라 추론은 그대로 → 속도·정확도 트레이드오프 없이 1GB 충족.
5. **저마진 라우팅** — 확신이 낮은 34.1%만 앙상블 통과. 점수와 속도를 동시에 개선한 유일한 레버 (3점 dose-response로 임계값 실측 확정).
6. **시간 예산 역산** — 서버 로그 회귀로 추론시간 = f(실제 토큰 수) 규명. 동적 패딩이라 길이 캡 축소는 효과 0이라는 반직관 결론, 제출 전 서버 시간 예측 가능.

## 디렉토리 구조

```
dacon/
├── finals/                    # 본선 준비 자료 (아키텍처·기술기여·검증문화·제출로그·QnA 등)
│   ├── 01_솔루션_아키텍처.md
│   ├── 02_핵심기술_기여.md
│   ├── 03_검증문화_기각축.md
│   ├── 05_학습코드_재현성.md
│   ├── 13_QnA_통합본.md      # 심사 대비 32문 QnA 뱅크
│   ├── assets/                # 발표용 그림
│   └── junhyun/               # 최종 챔피언(amw4) 학습 명세
├── finals_code_submission/    # 본선 제출 학습코드 (재현 가능 패키지)
│   ├── train_transformer.py   # 학습 본체 (KD·sieve·action-margin)
│   ├── quantize_checkpoint.py # int8 row-wise 저장 코덱
│   ├── quantize_int4.py       # int4 group-128 저장 코덱
│   ├── build_oof_consensus.py # consensus sieve 아티팩트 생성
│   ├── export_teacher_logits.py
│   ├── pack/script.py         # 추론 파이프라인 (직렬화→3모델→라우팅→룰)
│   └── colab/                 # 시드별 학습 plan JSON
├── open/                      # 대회 배포 원본 (데이터는 .gitignore 제외)
├── notebooks/ · src/ · submit/  # 초기 TF-IDF 계열 (인코더 라인 종료)
└── submissions/               # 제출 이력 (zip은 git 제외)
```

> 모델 가중치(`*.safetensors`/`*.pt`), 대회 데이터, 제출 zip은 `.gitignore`로 제외 — 학습 코드로 재생성.

## 재현

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
