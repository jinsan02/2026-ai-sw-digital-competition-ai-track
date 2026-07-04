# CLAUDE.md — DACON 236694 (2026 AI·SW 디지털 경진대회, AI부문)
# ⚠️ 모든 세션 시작 시 Claude Code가 가장 먼저 읽습니다. 간결 유지 (토큰 = 비용).

---

## 🎯 프로젝트

**과제**: AI 코딩 에이전트 세션의 다음 행동을 14-class로 예측
(read_file / grep_search / list_directory / glob_pattern / edit_file / write_file / apply_patch /
run_bash / run_tests / lint_or_typecheck / ask_user / plan_task / web_search / respond_only).
입력 = `session_meta` + `history`(0~12턴) + `current_prompt`.

**평가**: Macro-F1. **코드 제출 대회** — 예선 Private LB 100%, 마감 **07-15(수) 10:00**.
**스택**: Python 3.11 · transformers 4.46.3 · torch · scikit-learn 1.8.0. **JS/TS 도입 불가.**

## 🚨 제출 제약 (먼저 읽어라)

- **submit.zip ≤ 1GB**, 오프라인(인터넷 차단), 설치/추론 각 10분, T4 16GB / Python 3.11.15.
- zip 루트 구조: `script.py`, `requirements.txt`, `model/`. `data/`·`output/`은 평가서버가 주입.
- **zip은 반드시 python `zipfile`로 생성** — PowerShell `Compress-Archive` 금지(엔트리를 백슬래시로
  써서 Linux 서버가 `model/`을 못 읽음 → 제출 실패 전례). 생성 후 `unzip -l`로 백슬래시 0개 확인 +
  Linux 클린폴더 unzip → 오프라인 CPU 스모크(id/action 5행) 필수.
- **매 제출 전 `python verify_zip.py <zip> --smoke` 필수** (CRC·백슬래시·필수파일·1GB·SHA256+스모크).
  출력된 SHA256을 업로드 직전 파일과 대조 — 부분복사/오업로드로 "./model missing" 재발 방지 (07-04 실전 1회).
- large 모델은 int8 저장코덱(`teammate_output/.../quantize_checkpoint.py`)으로 1GB 캡 회피(로드 시 fp16 복원).

## 📏 실험 / 평가 규칙

- **승격 판단은 3-fold session OOF.** fixed 단일 split은 낙관적(+0.009) → 스크린 전용.
- **OOF는 최종 후보 1개에만** 돌린다(large는 6h/fold). 스크린·A/B는 fixed로.
- 캘리브레이션: CE 라인 OOF≈Public / focal 라인 OOF→Public +0.013(전체 refit에서 회복).
- 시드 분산 ±0.011 — 단일 run <0.01 차이는 노이즈. **코드베이스 간 점수 직접 비교 금지**(하니스 차이).
- 다양성 앙상블은 정확도가 아니라 **오류 비상관**으로 고른다(예: kf-deberta > mBERT).

## 🖥️ 작업 환경 (2대, Tailscale)

- 노트북: hostname `rohjinsan`, RTX 5060 8GB, WSL `~/dacon-venv`.
- 데탑: `DESKTOP-1E5JAJD`(100.71.102.28), RTX 3060 12GB, **Windows venv** `C:\dacon\venv`(WSL 아님).
- 노트북→데탑 무비번 SSH: `ssh desktop-3060 "..."`. **데탑 기본 SSH 셸 = PowerShell** →
  원격 명령에 `&&` 금지, `;` 사용. 학습 로그 `C:\dacon\runs\*.log`.

## 🚫 데이터 / 모델 취급

- **운영 데이터(`open/data/*.jsonl`, `train_labels.csv`)는 git 제외** — clone에 안 딸려옴, 별도 전송(scp).
- `*.pkl` · `model*/` · `venv/`는 git 제외. 대형 fp32 아티팩트(1.1GB/개)는 실험 끝나면 즉시 삭제(디스크 관리).
- `src/train.py`와 `submit/script.py`의 `build_input_text()`는 **동일 로직 유지**(학습·추론 입력 불일치 = 성능 붕괴).

## ✅ 커밋 규칙

- 커밋 메시지에 `Co-Authored-By` 줄 넣지 않음.
- `feature/*` 브랜치엔 커밋 금지(참고 전용). 제출 zip은 `submissions/`에 보관.

## 🤖 행동 규칙 (Gather → Act → Verify)

1. **Gather**: 스키마·필드명은 문서 말고 **실제 코드를 Read**한 뒤 사용(필드명 오류 방지).
2. **Act**: 요청된 것만. 투기적 코드·불필요한 추상화 금지. 외과적 변경.
3. **Verify**: 목표를 검증 가능하게. 제출물은 반드시 오프라인 스모크 통과 후 보고.
