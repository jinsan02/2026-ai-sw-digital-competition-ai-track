# DACON 236694 — 다음 방향 (2026-07-04)

## 현재 위치
- **Public 0.7732655474** (신규 팀 최고, 직전 0.7499 대비 **+0.023**), 추론 **6분 38초** / 제한 10분
- 제출물: `submissions/submit.zip` (528MB) = large + focal γ2.0 + **len384** + current_v1 + replay_last1 + rules 12 + 2stage bias + sparse SVC(w=0.05), int8 코덱
- **0.77 목표 이미 달성.** 목표가 "0.77 도달"에서 **"순위 극대화 + 마진 확보"**로 전환

### 재현 레시피 (0.7733)
```
train_transformer.py --base-model xlm-roberta-large --serializer current_v1 --split session \
  --max-length 384 --epochs 5 --batch-size 4 --grad-accum-steps 4 --lr 2e-5 \
  --loss focal --focal-gamma 2.0 --replay-mode last1 --max-replay-samples 10000 \
  --gradient-checkpointing --tune-bias --per-epoch-eval --save-checkpoints \
  --final-model --final-only --rule-boosts-path <rules.json> --class-bias-artifact <bias.json>
```
+ quantize_checkpoint.py int8 (1068→536MB), sparse SVC(C=0.05, w=0.05) 동봉.
grad-accum/per-epoch-eval/checkpoints는 로컬 train_transformer.py 커스텀 추가분(원본엔 없음).
⚠️ zip은 **반드시 python zipfile**로 생성(Compress-Archive 금지 — 백슬래시 엔트리로 서버가 model/ 못 읽음, 실제 제출 1회 날림).

## 이번 결과에서 배운 것 (전략에 반영)
1. **fixed-val 예측(0.757)이 실제(0.7733)를 0.016 과소평가.** → 단일 split은 신뢰 불가. 다음 레버 승격 판단은 반드시 OOF로.
2. **추론 6:38 / 제한 10:00 → 마진 3:22.** large 2개 앙상블은 시간 초과로 불가. 추가 인코더는 base급 1개, 길이 제한해야 겨우 진입. **이게 앞으로 모든 앙상블 설계의 하드 제약.**
3. **seed 분산 ±0.011** → 0.7733 자체가 노이즈 포함(재런 시 대략 0.762~0.784). 단일 숫자 과신 금지, 차이 <0.01은 노이즈.

## 남은 천장 = 탐색계열 혼동
- 약점 클래스(val): list_directory 0.486, read_file 0.549, grep_search 0.623, glob_pattern 0.636
- 최다 혼동: read↔grep↔list↔glob. Macro-F1은 약한 클래스가 지배 → **이 혼동을 풀어야 점수가 오름**
- 혼동을 가르는 정보(직전 결과가 파일목록인지, 인자에 경로가 있는지)는 history/state에 있고, current_v1은 이를 일부 버림. len384가 도움됐지만(+0.010 실증) 아직 current_prompt 중심

## 우선순위 레버

### P0 — 이번 승리 분해 (✅ 완료 2026-07-04)
- Full 0.7732655(6:38) vs Transformer-only 0.7716474(6:05) → **rules+SVC = +0.0016181 (결정론적, 노이즈 아님), +33초**
- **결론**: 승리의 97.9%가 트랜스포머 본체. large+len384+focal 단독으로 이미 0.7716 > 0.77. rules는 in-sample +0.0098→Public +0.0016 수축(팀 2.7배 낙관 패턴 일치). SVC는 val +0.0004로 사실상 무효 + 33초·56MB 소모 → **가성비 나쁨**
- **액션**: 다양성 모델 넣을 때 SVC부터 제거(추론 33초 회수, rules는 거의 공짜로 유지). OOF rules 재튜닝 기대이익 낮음(천장 ~+0.002) → OOF는 state_v2/다양성 승격 판정에 투입

### P1 — state_v2 × len384 (최우선 업사이드, 로컬 밤런)
- 탐색 혼동을 **직접 타격**: state_v2는 최신 user/action/result/args를 앞에 배치 → read/grep/list/glob 결정 정보 보존
- 과거 state_v2 실패는 "길이 무용"이 아니라 "직렬화가 메타 버린 채 늘린 혼입 실험"이었음(임준현 재해석). 아무도 안 돌린 조합 = "정보 더 담기 × 충분한 길이"
- 명령: `--serializer state_v2 --max-length 384` (나머지 focal 레시피 동일). 토큰 분포 보고 len512도 후보
- 판정: OOF 나오기 전엔 fixed로 방향만, 승격은 OOF로

### P2 — large len384 3-fold OOF (Colab A100, 팀 분업)
- fixed-val이 신뢰 불가로 판명 → 다음 레버 승격 판단에 OOF 필수
- 로컬 16h 대신 Colab 몇 시간. OOF 로짓 생기면 rules/bias/SVC-weight를 정식 재튜닝(현재 fixed 근사 대체)
- 조율: 임준현님께 의뢰 + 진산님 3세션 script.py(SVC 추론) 공유(이미 2회 요청받음)

### P3 — 이종 인코더 다양성 (예산 제약 하에서만)
- kf-deberta-base(ask_user에서 large보다 강함 — 한국어 프롬프트 신호) 또는 mBERT를 int8 동봉(zip 여유 ~470MB)
- **반드시 추론 예산 실측**: 6:38 + 추가 인코더 < 10:00. base급 + len256 정도로 제한
- model soup 금지(가중평균 붕괴 실증), **로짓 블렌딩만**, 가중치는 OOF로

## D-11 일정 (07-15 10:00 마감)
| 시기 | 작업 |
|---|---|
| 07-04(오늘) | 0.7733 제출 완료. transformer-only 분해 제출(P0). state_v2 밤런 세팅(P1) |
| 07-05~07 | state_v2×len384 스크린(로컬) / large OOF(Colab) / 다양성 모델 준비 |
| 07-08~11 | OOF 기반 rules·bias·블렌딩 재튜닝 + 통합 (추론 예산 내) |
| 07-12~14 | 패키징, 오프라인+클린unzip 스모크, 캘리브레이션 제출 |

## 손절선
- 단일 런 <0.01 차이는 노이즈 → 승격 금지
- state_v2가 OOF에서 +0.005 이상 안 나오면 다양성/추가 실험으로 전환
- 추론 10분 초과 위험 앙상블은 스모크에서 시간 실측 후에만 채택
- 제출 전 항상 python zipfile 생성 + 백슬래시 0개 검증 + 클린폴더 unzip 스모크
