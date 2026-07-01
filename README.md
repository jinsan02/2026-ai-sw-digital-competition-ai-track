# 2026 AI·SW중심대학 디지털 경진대회 — AI부문

AI 코딩 에이전트의 다음 행동(action)을 14개 클래스 중 하나로 예측하는 코드 제출형 대회.

## 태스크

- 입력: `session_meta`(세션/작업공간 메타) + `history`(0~12개 user/assistant_action 교대) + `current_prompt`
- 출력: 다음 행동 1개 — `read_file` / `grep_search` / `list_directory` / `glob_pattern` / `edit_file` / `write_file` / `apply_patch` / `run_bash` / `run_tests` / `lint_or_typecheck` / `ask_user` / `plan_task` / `web_search` / `respond_only`
- 평가지표: **Macro-F1** (14개 클래스)

## 현황 (2026-07-01)

| 접근 | 로컬 | Public | 비고 |
|---|---|---|---|
| baseline TF-IDF + LogReg (current_prompt만) | 0.437 | 0.4375 | 배포 베이스라인 |
| + 피처엔지니어링 v2~v5 (history/seq/args pseudo-token) | 0.645 | 0.6436(v4) | TF-IDF 계열 천장 |
| 트랜스포머 파인튜닝 (팀 별도 파이프라인) | 0.719 | 0.7081 | 현재 팀 최고 |

**핵심 교훈**: 피처 엔지니어링은 로컬 0.64대에서 천장, 분류기 교체(sklearn 12종·CatBoost)로는 못 넘음. 실제 도약은 사전학습 트랜스포머 인코더에서 나옴. 트랜스포머는 local→public 격차가 큼(-0.01대)이라 **session split(id의 `-step_` 앞 세션 단위 그룹 분할)을 1차 검증 지표**로 사용.

> 트랜스포머 파인튜닝 코드는 팀원이 별도 관리하는 파이프라인이라 이 레포에는 포함하지 않음.

## 디렉토리 구조

```
dacon/
├── open/                # 대회 배포 원본 (데이터는 .gitignore 제외)
│   └── data/            # train.jsonl(70k) / train_labels.csv / test.jsonl(샘플5) / sample_submission.csv
├── notebooks/           # 베이스라인 노트북 (TF-IDF + LogReg 학습/추론)
├── src/                 # v2 피처엔지니어링 학습 코드 (train.py)
├── submit/              # v2 제출 구조 (model/ + script.py + requirements.txt)
└── submissions/         # 로컬 제출 이력 (zip은 git 제외)
```

> 모델 가중치(`*.safetensors`/`*.pkl` 등), 대회 데이터, 제출 zip은 `.gitignore`로 제외 — 학습 코드로 재생성.

## 제출 규칙 요약

- `submit/`(또는 트랜스포머 산출물) 구조를 zip으로 묶어 제출: `model/` + `script.py` + `requirements.txt`.
- 평가 서버가 `data/`, `output/`을 자동 주입 — `script.py`는 `./data/test.jsonl`을 읽고 `./output/submission.csv`를 생성.
- zip ≤ 1GB, 패키지 설치 ≤ 10분, 추론 ≤ 10분, 인터넷 불가(사전학습 가중치는 `model/`에 미리 포함).
- 평가 서버: Ubuntu 22.04.5 / **T4 16GB** / Python 3.11.15 / CUDA 12.8 / `transformers==4.46.3`(고정). 기본 설치 패키지는 `requirements.txt`에서 제외 권장.
- 1일 최대 제출 10회. 예선 마감 07.15(수) 10:00.

> ⚠️ `transformers==4.46.3` 고정이라 ModernBERT 계열(mmBERT 등, 4.48+ 필요)은 제출 시 로드 실패. xlm-roberta-base는 호환 확인됨.
> ⚠️ xlm-roberta-base는 fp32 저장 시 1.11GB(제한 초과) → **fp16 저장으로 556MB**(추론이 어차피 fp16이라 무손실).

## 로컬 개발 환경

```bash
# TF-IDF 계열 (CPU)
pip install -r requirements-dev.txt
python src/train.py
```
