# 본선 인수인계 — amw4 (Weak4 Action-Margin KD) 챔피언 계열

> [보관 메모 — 진산] 준현이 07-16 05:45 팀채널에 업로드 (F0BH62R8E4F). 명세 요청 5건 전부 답변.
> ⚠️ 원문의 "1위와 0.00096"은 마감 전 스냅샷 — 최종 확정은 1위 0.79878 / 갭 0.00112.

팀원 요청 5건 정리. 최종 챔피언 **Public `0.7976673203`** 은 amw4 계열이며, 동일
점수를 두 팩이 보유하고 동률 중 더 빠른 **`rfinal_amhyb_m10.zip` (7:15)이 최종
챔피언 팩으로 등재**되었다 (`rfinal_amw4_7070m10.zip` 7:23이 스코어 최초 달성).
코드 기준 커밋: `070338a`(action-margin KD 구현) ~ `f1af1e1`(마감 기록).

---

## 1. amw4 손실식 + 정확한 학습 CLI

### 손실식 (한 문단)

챔피언 recipe의 총손실은 기존 sieve×condalpha 챔피언 손실에 **Weak4-스코프
action-margin 보조항** 하나를 더한 것이다:

```
L_total = L_base + λ · L_AM

L_base  = 기존 챔피언 손실 그대로
          (focal γ=2 CE, class-weight^0.5, label smoothing 0.02,
           M8 teacher 로짓 KD α=0.5 / Weak4 true-label 행 α=0.7, T=3,
           m7/m8/v6 consensus 신뢰도 sieve(행 가중 0/0.25/0.75/1), replay last1 1만·w0.5)

L_AM    = mean over covered rows, k∈top-3 of
          SmoothL1( (s_true − s_k)/T , (t_true − t_k)/T ),   T = 3.0
```

여기서 `t`는 detach된 M8 teacher 로짓, `s`는 학생 로짓이며, hard negative
`k`는 **teacher 로짓에서 true 클래스를 −∞ 마스킹한 뒤의 top-3**로 한 번만
선택된다(teacher 기준 선택이므로 학생 쪽으로 이동 타깃이 생기지 않음). 적용
범위(coverage)는 "일반 KD teacher mask > 0인 original 행 ∩ **true label ∈
Weak4** (read_file/grep_search/list_directory/glob_pattern)"로, replay 행과
teacher 미커버 행은 자동 제외된다 — 즉 두 번째 teacher 표면을 만들지 않는다.
가중치 λ는 하이퍼파라미터가 아니라 **첫 eligible 배치에서 로짓-gradient 노름
비율이 0.10이 되도록 1회 결정론적으로 보정 후 고정**:

```
λ = 0.10 · ‖∇_logits L_base‖ / ‖∇_logits L_AM‖
```

시드별 실측 λ (checkpoint_state/hf_meta에 기록됨):

| seed | λ (calibrated) | epoch당 AM 적용 행 |
|---|---|---|
| 42 | 0.143567735 | 28,782 |
| 909 | 0.123874354 | 28,782 |
| 7070 (**챔피언**) | 0.131212791 | 28,782 |

구현 위치: `train_transformer.py` — `action_margin_kd_loss`,
`action_margin_kd_coverage_mask`, `calibrate_action_margin_weight`
(커밋 `070338a`).

### 정확한 학습 CLI (챔피언 seed 7070)

원본 플랜: `colab/action_margin_weak4_refit_s7070_lane_a_plan.json`
(스테이징 arm: `colab/stage_hcx05b_base.py --asset-dir <hcx05b_base>`, 오프라인
환경변수 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`).

```bash
python train_transformer.py \
  --base-model naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
  --lr 2e-5 --device cuda --split session --serializer current_v1 \
  --max-length 384 --epochs 3 --batch-size 16 --grad-accum-steps 1 \
  --eval-batch-size 64 --gradient-checkpointing \
  --class-weight-power 0.5 --label-smoothing 0.02 \
  --loss focal --focal-gamma 2.0 \
  --replay-mode last1 --max-replay-samples 10000 --replay-sample-weight 0.5 \
  --distill-logits <경로>/m8_qwen35_refit_train70k_fp16.pt \
  --distill-alpha 0.5 --distill-alpha-weak 0.7 --distill-temp 3.0 \
  --consensus-reliability experiments/20260710_m7_m8_v6_oof_consensus.pt \
  --consensus-backbone-weights 0,0.25,0.75,1 \
  --action-margin-kd-target-grad-ratio 0.10 \
  --action-margin-kd-topk 3 \
  --action-margin-kd-label-scope weak4 \
  --tokenize-batch-size 1024 \
  --amp-init-scale 64 --amp-growth-interval 1000000 --require-zero-amp-skips \
  --seed 7070 --no-research-log --save-fp16 \
  --epoch-checkpoint-dir <출력>/kd_sieve_ca_amw4_t010_k3_refit_s7070_ckpt \
  --output-dir <출력>/kd_sieve_ca_amw4_t010_k3_refit_s7070 \
  --experiment-suffix kd_sieve_ca_amw4_t010_k3_refit_s7070 \
  --final-model --final-only
```

- 기존 챔피언 recipe 대비 **품질 변수는 AM 3개 플래그뿐**. AMP 3개 플래그는
  안정화(스크린 크래시의 원인이던 skip-0 감사와 세트, 세 refit 모두 skip=0 달성).
- seed 42/909 재현은 `--seed`와 suffix만 교체 (플랜 파일:
  `colab/action_margin_weak4_refit_lane_b_plan.json`(s42),
  `colab/action_margin_weak4_refit_s909_lane_c_plan.json`).
- 근거 스크린(발표용): 크래시 런의 epoch-3 체크포인트 복구 평가 — 동일 고정분할
  14,001행에서 raw `+0.0018513` / Weak4 `+0.0084416` vs 짝 대조군
  (`experiments/artifacts/20260715_weak4_am_ckpt_recovery_eval.json`).

---

## 2. `rfinal_amw4_7070m10` vs `rfinal_amhyb_m10` — "트리오 변경"의 정체

두 팩은 Public **10자리 완전 동률**(`0.7976673203`) — 히든셋에서 예측 플립 0.

| 구성요소 | `rfinal_amw4_7070m10.zip` (7:23) | `rfinal_amhyb_m10.zip` (7:15) |
|---|---|---|
| `model/` (main, INT8) | amw4 **s7070** (동일) | amw4 **s7070** (동일) |
| `model_b/` (INT4) | 구 recipe s909 (동일) | 구 recipe s909 (동일) |
| `model_c/` (INT4) | **구 recipe s7070** (수신 rfinal 아카이브 바이트 원본) | **amw4 s42** (자체 int4 인코딩) |
| `script.py` | margin<1.0 + 룰 스택 (동일) | 동일 |
| zip 크기 / SHA256 앞자리 | 1,002,966,933 B / `47e50cef` | 1,005,612,893 B / `72b608d4` |

즉 차이는 **model_c 한 자리**뿐이다. 하이브리드의 의도는 (a) main(amw4-s7070)과
member(구-s7070)의 같은-시드 상관 제거, (b) Weak4 개선을 routed ensemble에 한 번
더 공급, (c) generalist(구-s909) 1개는 완충으로 유지. 결과가 완전 동률이라는
것은 **centered 저마진 평균 안에서 member 정체성은 학습 목적함수 계열이 달라도
예측-불활성**이라는 판정(같은 결론 2회째: c0a8 동률, 본 건). 발표 시 "구조를
바꿨는데 예측이 1행도 안 변한 통제실험"으로 쓸 수 있고, 동률 중 더 빠른
**amhyb(7:15)가 최종 챔피언 팩으로 등재** — 점수 50점과 속도 10점이 같은 팩에
귀속된다.

---

## 3. 최종팩 `model_b` / `model_c` 명세

| 디렉토리 | 모델 | 훈련 | 저장 형식 |
|---|---|---|---|
| `model_b/` (두 팩 공통) | seed **909**, 기존 챔피언 recipe(sieve×condalpha, M8 teacher KD, AM 없음) full refit | 팀측 (07-13~14 트리오 조립기; 커맨드는 팀측 기록) | `int4-group128-v1` (per-row group128, scale=absmax/7, fp16 스케일) |
| `model_c/` (7070m10) | seed **7070**, 위와 동일 recipe | 팀측 | 동일 INT4, 수신 아카이브 바이트 원본 |
| `model_c/` (amhyb) | seed **42**, **amw4 recipe** (§1 CLI에서 seed만 42) | 07-15 lane B 클린 런 (AMP skip 0, fp16 fidelity 512/512) | 동일 INT4 코덱, 자체 인코더로 재인코딩 (기존 파일과 코덱 패리티 170/170 텐서 바이트·스케일 일치 검증) |
| main `model/` (두 팩 공통) | seed **7070**, amw4 recipe | 07-15 lane A 클린 런 (AMP skip 0) | INT8 row-wise (`quantize_checkpoint.py`, argmax fidelity 511/512) |

멤버는 추론 시 **main raw-logit margin < 1.0인 행(~34%)에서만** forward되어
행-centered 로짓의 등가중 3-평균에 참여한다. 룰 스택(ask-boost + 10룰 + R1i +
seq-exec)은 그 뒤에 적용.

---

## 4. 학습코드 패키징 체크리스트 (Private 복원 서사용)

레포 07-14~15 커밋(`070338a`…`f1af1e1`) 기준, 포함할 것:

**핵심 코드**
- `train_transformer.py` — amw4 학습 본체 (AM 함수 3종 포함)
- `export_teacher_logits.py` — teacher 로짓 추출 (M8 계열 payload 생성 도구;
  pack의 codec 로더/직렬화기를 재사용해 배포와 bit-faithful)
- `colab/export_trio_teacher.py` — **trioT teacher 추출** (배포-충실 트리오
  앙상블 teacher: main raw + margin<1.25 행만 centered 3-평균, 룰 제외).
  ※ 주의: trioT는 **mainT 챔피언(0.797181265) 계보**의 teacher이고, 최종
  챔피언 amw4는 **원본 M8 teacher**를 쓴다. 복원 서사에서 두 계보를 섞지 말 것.
- `quantize_checkpoint.py` — main INT8 코덱 (quantize/verify)
- `quantize_int4.py` — 멤버 INT4 코덱 인코더(팀측 보유 원본; 레포의 pack
  `script.py` 내 `load_int4_state_dict`가 디코더 사양의 기준)
- `package_submission.py` — 단독팩 조립기 (트리오팩은 수동 조립: 아래)
- `colab/stage_hcx05b_base.py` — 오프라인 HCX 베이스 스테이징 (SHA 검증 설치)

**Colab 노트북 원본**: `colab/colab_runner.ipynb` (+`_b`/`_c`/`_d`),
프로토콜 문서 `colab/COLAB.md`, 실행 플랜 JSON 3종
(`colab/action_margin_weak4_refit_{lane_b,s909_lane_c,s7070_lane_a}_plan.json`).

**의존 아티팩트 (코드 아님, 재현에 필수)**
- `experiments/20260710_m7_m8_v6_oof_consensus.pt` (3,126,629 B) — sieve payload, 레포 내
- `m8_qwen35_refit_train70k_fp16.pt` (6,858,261 B) — M8 teacher 로짓, Drive
  `AADP_exchange_b/teacher/` (및 `_c/teacher/`); 레포 미포함이므로 제출물에 동봉 필요
- 원본 데이터 `open/data/` (train.jsonl + train_labels.csv)

**트리오팩 수동 조립 순서** (7070m10 기준):
fp16 refit → `quantize_checkpoint.py quantize`(+`verify`) → `model/` 교체,
`model_b`/`model_c`/`requirements.txt`는 원본 유지, `script.py`의 routing 상수
확인(line 3063, `< 1.0`) → `zip -r -X` (루트 5항목) → 클린 추출 스모크
(`TRANSFORMERS_OFFLINE=1 python script.py`) → 1GiB 이하 확인.

---

## 5. (보너스) margin 1.25→1.0 재축소가 amw4에서 이긴 해석 — 3점 실측 곡선

**한 줄**: action-margin 보조항이 정답-대-경쟁자 로짓 마진 자체를 조형하다 보니
amw4 main은 마진 분포가 압축되어, 구 threshold 1.25로는 라우팅이 31.9%→39.7%로
불어나는데 그 추가 밴드([1.0,1.25))의 main 단독 오류율이 19.4%뿐이라(구 main은
41.5%로 손익분기) "이미 맞는 행"들이 멤버 평균의 간섭에 노출되었고, 1.0 복원은
라우팅 볼륨을 튜닝된 작동점(~34%)으로 되돌린 것이다.

발표용 강점 — **threshold를 양쪽에서 브래키팅한 3점 dose-response가 전부 Public
실측**이다 (마지막 제출로 0.75까지 확보):

| routing threshold | 라우팅 비율 | Public | 해석 |
|---|---|---|---|
| margin < 1.25 | 39.7% | `0.7960` (s909, 7:55) | 과다 라우팅 — 저오류(19.4%) 밴드에 멤버 간섭 |
| **margin < 1.0** | 34.1% | **`0.7976673203`** (s7070, 7:23) | **최적 작동점** |
| margin < 0.75 | ~28.5% | `0.79739` (s7070, 최종 제출) | 과소 라우팅 — 33.4%-오류 밴드의 멤버 구제 상실 |

특히 1.0↔0.75 쌍은 **같은 main·같은 시드의 순수 단일변수 비교**(−0.00028)라,
"[0.75,1.0) 밴드에서는 멤버가 실제로 순구제한다"는 것까지 정량 확인된다 —
1.0이 우연한 끝점이 아니라 실측으로 둘러싸인 최적점이라는 서사가 완성된다.
(@1.25 점은 시드 교란(s909)이 섞여 있음을 발표에서 한 줄 언급 권장.)

---

*근거 문서: `research_log_final12h.md`(엔드게임 전 과정),
`leaderboard_calibration.md`(제출 원장), 스크린/게이트 아티팩트
`experiments/artifacts/20260715_weak4_am_*.json`.*
