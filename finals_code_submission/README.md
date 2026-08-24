# 학습 코드 제출 — 팀 토큰강도 (Agent Action Decision Prediction)

최종 제출물: `rfinal_amhyb_m10.zip` — **Public/Private 0.7976673203** (추론 7분 15초 / T4).
이 패키지는 위 점수를 기록한 최종 팩의 **모든 모델 가중치를 재생산하는 학습 코드·설정·의존 아티팩트**다.
기준 커밋: `070338a`(action-margin KD 구현) ~ `f1af1e1`(마감 기록).

---

## 0. 검증자 빠른 시작 (처음 보는 분께)

**읽는 순서**: 이 README §1(무엇을 재현하나) → §2(재현 체인) → §3(정확한 학습 CLI) → §5(환경) → §6(자원 출처). 나머지 디렉토리는 아래 지도의 근거 자료다.

```
(zip 루트)
├── README.md                  ← 이 문서 (재현의 단일 진입점)
├── train_transformer.py       ← 학습 본체 — §3 CLI로 main/model_b/model_c 재생산
├── export_teacher_logits.py, build_oof_consensus.py, quantize_*.py, package_submission.py
│                              ← 교사 로짓·consensus·양자화·팩 조립 도구 (§2 체인의 각 단계)
├── train.py, script.py        ← build_oof_consensus.py 의존 모듈 (라벨 로더·클래스 순서)
├── requirements.txt / requirements_qwen35.txt  ← 학생(HCX) / 교사(Qwen3.5) 환경
├── artifacts/                 ← 재현 필수 입력 2종 (m8 교사 로짓 · consensus)
├── pack/                      ← 최종 제출 zip의 배포 script.py·requirements 원본 (해시 검증본)
├── colab/                     ← 최종 3시드 학습의 실행 원본 (노트북 4·플랜 JSON 3·스테이징/지원 스크립트)
├── experiments/               ← 실측 증빙 원장:
│   ├── results.csv            ← 전 실험 기록 (학습 커맨드 원문 포함)
│   ├── artifacts/*.json       ← 런별 metrics·consensus 명세
│   ├── logits/                ← m7/m8/v6 OOF fold 로짓 9개 (consensus 재생성 재료) + m8 교사 로짓 원본
│   └── manifests/             ← 체크포인트 SHA·λ·행수 기록 (lane a/b, 복원 manifest)
├── research_log_final12h.md   ← 마감 12시간 의사결정 로그 (챔피언 선택 근거)
└── finals_amw4_handoff.md     ← 최종 레시피 기술 명세 (손실식·팩 구성·dose-response 상세)
```

**전제**: 대회 배포 데이터(train.jsonl, train_labels.csv)를 `open/data/`에 배치. 최소 재현 = §3 CLI 1회 실행(교사 로짓·consensus는 `artifacts/` 동봉분 사용) → §2-[3] 양자화 → §2-[4] 팩 조립. 교사·consensus까지 처음부터 재생산하려면 §2.5.

---

## 1. 최종 팩 구성과 재현 대상

| 구성요소 | 모델 | 학습 레시피 | 저장 형식 |
|---|---|---|---|
| `model/` (main) | HCX-0.5B, seed **7070** | **amw4** (§3) | INT8 row-wise (`quantize_checkpoint.py`) |
| `model_b/` | HCX-0.5B, seed **909** | 기존 챔피언 recipe (amw4에서 AM 플래그 3개 제외) | INT4 group-128 (`quantize_int4.py`) |
| `model_c/` | HCX-0.5B, seed **42** | **amw4** (§3, seed만 42) | INT4 group-128 |
| `pack/script.py` | — (추론, 최종 제출 zip 배포 원본 — Drive 복원본에서 추출·해시 검증 동봉) | 저마진 라우팅 margin<1.0 (34.1% 행) + 룰 스택 | — |

## 2. 재현 체인

```
[0] 베이스: naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B (HF)
[1] 교사 학습·로짓 (§2.5): m8 교사 학습(train_transformer.py, Qwen3.5-0.8B base)
    → 로짓 추출(export_teacher_logits.py) → artifacts/m8_qwen35_refit_train70k_fp16.pt (동봉)
    consensus 생성(build_oof_consensus.py, m7/m8/v6 OOF 합의)
    → artifacts/20260710_m7_m8_v6_oof_consensus.pt (동봉)
[2] 학생 학습: train_transformer.py — §3 CLI (seed 7070/42/909)
[3] 양자화: main → quantize_checkpoint.py quantize(+verify) / 멤버 → quantize_int4.py (group128)
[4] 팩 조립: model/·model_b/·model_c/ + pack/script.py + pack/requirements.txt → zip (루트 5항목, ≤1GiB)
    → 클린 추출 스모크: TRANSFORMERS_OFFLINE=1 python script.py
```

## 2.5 교사 학습 (동봉 아티팩트의 생성 경로)

**m8 교사 (Qwen3.5-0.8B full-data refit)** — 학생과 동일한 `train_transformer.py`로 학습 (별도 트레이너 없음). 실측 기록 원문 (experiments/results.csv `20260705_183536_..._m8_qwen35_refit`, A100 40GB, 학습 8.2h):

```bash
python train_transformer.py \
  --base-model igorktech/Qwen3.5-0.8B-Base-LM \
  --lr 2e-5 --device cuda --split session --serializer current_v1 \
  --max-length 400 --epochs 3 --batch-size 8 --grad-accum-steps 2 \
  --eval-batch-size 64 --class-weight-power 0.5 --label-smoothing 0.02 \
  --loss focal --focal-gamma 2.0 \
  --replay-mode last1 --max-replay-samples 10000 --replay-sample-weight 0.5 \
  --tune-bias --keep-threshold 0.0 --tokenize-batch-size 1024 \
  --no-research-log --seed 42 --save-fp16 --final-model --final-only
```

- ⚠️ Qwen3.5 계열은 **전용 환경** 필요: `requirements_qwen35.txt` (transformers>=5.13,<5.14 — qwen3_5_text seq-cls auto mapping). 학생(HCX) 학습 환경(4.46.3~4.51.3)과 분리.
- 교사 로짓 추출: `export_teacher_logits.py` (train 70k 전행, fp16, 배포 codec 직렬화와 bit-faithful).
- **consensus 아티팩트 생성**: `build_oof_consensus.py` — m7/m8/v6 세 모델의 session-OOF fold 로짓(payload)을 id-정렬 검증 후 행별 "정답 맞힌 모델 수"(0~3)로 집계 (`usage_scope: full_data_refit_only`). 의존: `train.py`(라벨 로더)·`script.py`(클래스 순서) — 동봉. OOF payload 자체는 동일 트레이너의 `--split session` fold 런 산출물.
- m1~m9 교사 사다리 전체의 스크린·기각 이력은 발표자료·research log 참조 (최종 채택 교사는 m8 단독).

## 3. 챔피언 학습 CLI (main, seed 7070)

```bash
python train_transformer.py \
  --base-model naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
  --lr 2e-5 --device cuda --split session --serializer current_v1 \
  --max-length 384 --epochs 3 --batch-size 16 --grad-accum-steps 1 \
  --eval-batch-size 64 --gradient-checkpointing \
  --class-weight-power 0.5 --label-smoothing 0.02 \
  --loss focal --focal-gamma 2.0 \
  --replay-mode last1 --max-replay-samples 10000 --replay-sample-weight 0.5 \
  --distill-logits artifacts/m8_qwen35_refit_train70k_fp16.pt \
  --distill-alpha 0.5 --distill-alpha-weak 0.7 --distill-temp 3.0 \
  --consensus-reliability artifacts/20260710_m7_m8_v6_oof_consensus.pt \
  --consensus-backbone-weights 0,0.25,0.75,1 \
  --action-margin-kd-target-grad-ratio 0.10 \
  --action-margin-kd-topk 3 \
  --action-margin-kd-label-scope weak4 \
  --tokenize-batch-size 1024 \
  --amp-init-scale 64 --amp-growth-interval 1000000 --require-zero-amp-skips \
  --seed 7070 --no-research-log --save-fp16 \
  --final-model --final-only
```

- **model_c (seed 42)**: `--seed 42`만 교체.
- **model_b (seed 909)**: `--seed 909` + `--action-margin-kd-*` 3개 및 `--amp-*` 3개 플래그 제거 (기존 챔피언 recipe — AM·AMP 커스텀 플래그는 이후 amw4 단계에서 도입됨). **실측 근거 확보 (07-17)**: ① 동일 배치·동일 recipe인 s202 런의 학습 커맨드 원문(팀 A100 results.csv)이 KD(M8 distill α0.5/weak0.7/T3)·consensus 사용, AMP 플래그 부재를 실측으로 기록 — model_b는 해당 커맨드에서 `--seed 909`와 suffix만 교체 ② model_b 원본 팩(`kd_sieve_ca_s909.zip`)의 학습 manifest(`hf_meta.json`): 전 하이퍼파라미터 상기 CLI와 일치, consensus enabled(sha256 `040d4772…`), `transformers_version` 4.51.3, 원본 INT8 가중치 sha256 `9d7dc9d9…` (최종 팩 int4 `95f3762a…`는 이것의 int8→fp16→int4(group128) 변환본) — hf_meta 원문 동봉 가능. (동봉 `colab/*_s909_*_plan.json`은 AM 포함 amw4 변형으로 최종 팩 model_b와 다른 런임에 유의.)
- 손실: `L_total = L_base + λ·L_AM`. L_AM은 Weak4 true-label 행에서 teacher-기준 hard-negative top-3에 대한 SmoothL1 마진 매칭(T=3). λ는 첫 eligible 배치에서 gradient-노름 비율 0.10으로 1회 결정론 보정(시드별 실측: s42 0.1436 / s909 0.1239 / s7070 0.1312 — 학습 manifest에 기록됨).
- Colab 실행 원본: `colab/colab_runner*.ipynb` + 플랜 JSON 3종 (`colab/action_margin_weak4_refit_*_plan.json`) + 프로토콜 `colab/COLAB.md`. 오프라인 스테이징: `colab/stage_hcx05b_base.py` (기본 경로는 Colab Drive — `--asset-dir`로 오버라이드).
- ※ 경로 규약: 루트의 학습 코드(`train_transformer.py` 외 7종)는 **전부 상대 경로**만 사용한다. `colab/` 하위 스크립트에 보이는 `/content/...` 경로는 Colab 런타임 고유 마운트 경로(해당 환경의 표준 경로)로, 실행 기록 보존을 위한 원본이며 로컬 재현 시에는 루트 스크립트를 사용하면 된다.

## 4. 파일 맵

| 파일 | 역할 |
|---|---|
| `train_transformer.py` | 학습 본체 (action-margin KD 함수 3종 포함) |
| `export_teacher_logits.py` | 교사 로짓 추출 (M8 payload 생성) |
| `colab/export_trio_teacher.py` | 트리오 앙상블 teacher 추출 (중간 세대 mainT 계보용 — 최종 amw4는 M8 교사 사용) |
| `quantize_checkpoint.py` | INT8 코덱 (quantize/verify, argmax fidelity 511/512) |
| `quantize_int4.py` | INT4 group-128 코덱 인코더 (멤버용) |
| `package_submission.py` | 단독팩 조립기 (트리오팩은 §2-[4] 수동 조립) |
| `build_oof_consensus.py` | consensus 아티팩트 생성기 (m7/m8/v6 OOF 합의) |
| `train.py` / `script.py` | consensus 생성기 의존 모듈 (라벨 로더 / 클래스 순서·직렬화 — 단일모델용 로컬 스크립트; 배포 스크립트 아님) |
| `pack/script.py` · `pack/requirements.txt` | 최종 제출 zip(rfinal_amhyb_m10.zip) 배포 원본 — Drive 복원본에서 추출, sha256 `ea2225dd…`/`4ddce960…` (복원 manifest `20260716_final_package_restoration.json` common_payload와 일치) |
| `requirements_qwen35.txt` | m8 교사(Qwen3.5) 학습 전용 환경 |
| `artifacts/*.pt` | 재현 필수 의존 아티팩트 2종 (교사 로짓·consensus) |
| `requirements.txt` | 학습 의존성 |
| `experiments/results.csv` | 전 실험 원장 — 각 행에 **실행된 학습 커맨드 원문** 포함 (m8 교사 refit 행 `20260705_183536_…` 등) |
| `experiments/logits/` | m7/m8/v6 session-OOF fold 로짓 9개 = `build_oof_consensus.py` 입력 (consensus 처음부터 재생성 가능) + m8 교사 로짓 원본(.pt/.npz) |
| `experiments/manifests/` | 최종 3시드 런의 manifest (체크포인트 SHA·보정 λ·적용 행수) + 배포 원본 복원 manifest(`20260716_final_package_restoration.json` — pack/ 해시 근거) |
| `experiments/artifacts/` | 런별 metrics JSON (m8 refit·amw4 s42 등 실측 기록) |
| `colab/` (지원 스크립트) | `aadp_colab.py`·`run_plan_remote.py`·`vm_agent.py` 등은 Colab 세션 부트스트랩/동기화 도구, `m8_*_probe.py`는 교사 스크리닝 기록 — 재현 필수 아님, 실행 이력 보존용 |
| `research_log_final12h.md` / `finals_amw4_handoff.md` | 마감 12시간 의사결정 로그 / 최종 레시피 기술 명세 — 재현 서사의 근거 문서 |

## 5. 개발 환경

| 머신 | OS | GPU | Python | torch | transformers |
|---|---|---|---|---|---|
| 학교 A100 서버 | Ubuntu | A100 40GB | 3.10 | 2.x cu121 | 4.51.3 |
| 데스크톱 | Win11 + WSL2 | RTX 4070 Ti S | 3.11 | 2.7.1 cu128 | 4.51.3 |
| Colab (s7070·s42 최종 시드) | — | A100 | 3.11 | 2.x | 4.46.3 (모델 `hf_model/config.json` `transformers_version` 실측) |
| 평가 서버 (추론) | Ubuntu 22.04.5 | T4 16GB | 3.11.15 | 2.7.1+cu128 | 4.46.3 (서버 기본) |

추론은 평가 서버 기본 패키지로 동작 (제출 zip의 requirements 최소화). bit-exact 재현은 GPU/드라이버 의존이므로 "동일 레시피·시드" 수준을 보장하며, 클린 런 요건(AMP skip=0 감사)과 학습 로그·manifest로 이를 문서화한다.

## 6. 자원 출처 (외부 요소)

| 자원 | 출처 | 활용 범위 | 라이선스 |
|---|---|---|---|
| HyperCLOVAX-SEED-Text-Instruct-0.5B | HuggingFace (naver-hyperclovax) | 최종 배포 모델 (파인튜닝) | HyperCLOVA X SEED Model License — 사용·수정·파생·배포 허용(§2.2), attribution 이행: 본 제출물은 "HyperCLOVA X SEED Model is licensed under the HyperCLOVA X SEED Model License Agreement, Copyright © NAVER Corp." 고지를 포함하며 발표 자료에 "Powered by HyperCLOVA X"를 표기함 |
| Qwen3.5-0.8B 외 교사 모델 | HuggingFace (Qwen 등) | KD 교사 (로짓만 사용, 배포 미포함) | Apache-2.0 등 각 라이선스 |
| torch / transformers / scikit-learn 등 | PyPI | 학습·추론 | OSS |
| 외부 데이터 | **없음** (대회 제공 데이터만 사용) | — | — |
| 유료 API | **없음** | — | — |
