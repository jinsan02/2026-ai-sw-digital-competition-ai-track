# 2026 AI·SW중심대학 디지털 경진대회 — AI부문

AI 코딩 에이전트의 다음 행동(action)을 14개 클래스 중 하나로 예측하는 코드 제출형 대회.

## 태스크

- 입력: `session_meta`(세션/작업공간 메타) + `history`(0~12개 user/assistant_action 교대) + `current_prompt`
- 출력: 다음 행동 1개 — `read_file` / `grep_search` / `list_directory` / `glob_pattern` / `edit_file` / `write_file` / `apply_patch` / `run_bash` / `run_tests` / `lint_or_typecheck` / `ask_user` / `plan_task` / `web_search` / `respond_only`

## 디렉토리 구조

```
dacon/
├── open/                # 대회 배포 원본 (읽기 전용, .gitignore로 데이터 제외)
│   ├── data/
│   │   ├── train.jsonl          # 학습 입력 70,000건
│   │   ├── train_labels.csv     # 학습 정답 (id, action)
│   │   ├── test.jsonl           # 형식 확인용 샘플 5건 (실 평가 데이터는 비공개)
│   │   └── sample_submission.csv
│   └── baseline_submit.zip      # 참고용 베이스라인 제출 예시
├── notebooks/           # 실험 노트북 (베이스라인: TF-IDF + LogReg)
├── src/                 # 재사용 학습/전처리 코드
├── scripts/             # 유틸 스크립트 (제출 zip 패키징 등)
├── submit/              # 실제 제출 zip과 동일한 구조 (여기를 그대로 zip)
│   ├── model/           # 모델 가중치 (git 제외, 학습 노트북으로 재생성)
│   ├── script.py        # 평가 서버가 실행하는 추론 코드
│   └── requirements.txt
└── submissions/         # 로컬 제출 이력 zip (git 제외)
```

## 제출 규칙 요약

- `submit/` 그대로 zip으로 묶어 제출 (`model/`, `script.py`, `requirements.txt`).
- 평가 서버가 `data/`, `output/`을 자동 주입 — `script.py`는 `./data/test.jsonl`을 읽고 `./output/submission.csv`를 생성해야 함.
- 인터넷 접속 불가 (패키지 설치 제외) — 사전학습 가중치는 `model/`에 미리 포함.
- zip 용량 ≤ 1GB, 패키지 설치 ≤ 10분, 추론 실행 ≤ 10분.
- 평가 서버: Ubuntu 22.04.5, T4 16GB, Python 3.11.15, CUDA 12.8. 기본 설치된 패키지(torch/pandas/transformers 등)는 `requirements.txt`에서 제외 권장.
- 1일 최대 제출 10회.

## 로컬 개발 환경

```bash
pip install -r requirements-dev.txt
```
