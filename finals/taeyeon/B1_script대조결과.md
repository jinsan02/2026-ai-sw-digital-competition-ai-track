# B-1 script.py 실물 대조 결과 (2026-07-15, 태연)

> 대상: `script.py` (145KB, 3342줄) — 진산님 제공 실물 추론 코드
> 목적: ① 초안 서술과 실물 일치 여부 ② pptx vs md 불일치 진실 판정 ③ 발표 슬라이드 5 확정
> **결론: 실물이 초안(pptx·md 둘 다)과 여러 핵심에서 다름. 특히 pptx는 실물과 심하게 어긋남. 그리고 판단 필요한 사안 1건 발견(아래 §4).**

---

## 1. 실물 파이프라인 실제 실행 순서 (코드 기준 확정)

```
main() → run_hf_inference(model_dir="./model", ...)
  │
① test.jsonl 로드 + id 파싱
② leak_overrides 계산 ← train_lookup(leak_lookup.json.gz) 있으면 활성 (§4 주목)
③ current_v1 직렬화 (max_length는 meta 기본 192, 실측 384)
④ base_scores 산출: 3가지 경로 중 meta가 지정
     · cascade (base+secondary 인코더 blend)  또는
     · encoders (sequential softmax 평균)      또는
     · 단일 transformer (기본) → model_logits_sorted (동적 정렬 배칭)
⑤ sparse SVC 앙상블 결합 ← sparse_svc.pkl 있으면 sparse_weight로 가산 (§3 주목)
⑥ 저마진 라우팅: margin < 1.25 행만 model_b 추가 forward   ← ⚠️ 1.0 아님!
     · model_c도 있으면 추가 (시간 가드 430초 내)
     · z-centered logit 평균 (통과 행만)
⑦ prior_calibration bias + class_bias + rule_boosts 가산
⑧ 룰 스택 12종 순차 적용 (argmax 후 예측 flip)
⑨ leak_overrides로 최종 예측 덮어쓰기 ← pred_map.update(leak_overrides)
⑩ submission.csv 저장
```

## 2. pptx vs md vs 실물 — 핵심 대조

| 항목 | 실물 script.py | md 초안 | pptx | 판정 |
|---|---|---|---|---|
| **저마진 라우팅 임계값** | **`< 1.25`** (L3060) | 1.0 | (언급) | **둘 다 틀림 → 1.25 확정** |
| **룰 개수** | **12종** (R1/b/c/d/e/f/g/h/i + seq-exec + twig×2) | 12종 ✓ | 10종 | **md 맞음, pptx 틀림** |
| **시간 가드** | **430.0초** (L3056/3081) | 430초 ✓ | — | 일치 |
| **앙상블 구조** | main + model_b + model_c (조건부) + **sparse SVC** | main+b/c (sparse 미언급) | 2-seed→trio | **실물에 sparse SVC 추가 존재** |
| **양자화** | int8-rowwise-v1 (donor+patch 공유) + int4 지원 | int8 568 + int4 292×2 | int8 512/512 | 실물은 **donor 공유 코덱** (용량은 model 폴더 봐야 확정) |
| **직렬화** | current_v1 (기본) | current_v1 ✓ | current_v1 ✓ | 일치 |
| **leak_lookup** | **존재·기본 활성** (train→test override) | **미언급** | **미언급** | **초안 양쪽 다 누락 (§4)** |

→ **결론: pptx는 한 세대 이상 뒤쳐진 서사(10룰, trio만).** md가 실물에 더 가까우나 **md도 두 가지를 빠뜨림: ① 라우팅 1.25(1.0 아님) ② sparse SVC 앙상블 ③ leak_lookup override.**

## 3. 실물에만 있는 구성요소 (초안 미반영)

**(a) Sparse SVC 앙상블** — `sparse_svc.pkl` + `sparse_meta.json`
- TF-IDF 벡터라이저 + SVC decision_function → sparse_weight로 main logit에 가산
- `apply_sparse_controls`로 게이팅(gate_margin/topk/class_mask) 후 결합
- **즉 최종 팩은 "LLM 단독"이 아니라 "LLM + TF-IDF SVC 하이브리드"**. 발표에서 이걸 안 밝히면 코드검증 때 "이 sparse_svc.pkl은 뭐냐" 질문 나옴.

**(b) 저마진 라우팅 임계값 1.25** — md가 말한 1.0이 아님
- md 초안·05 재현체인이 "margin 1.0 + seed7070"을 amw4의 정의로 적었는데, **이 script.py는 1.25**.
- **✅ 진산님 확정(07-15)**: 이 파일은 **`script_mgn125.py`** (SHA `f286f21d…`, 145,762B) — **mgn125 팩 코드**. 최종 선택본은 `rfinal_amhyb_m10.zip`(7:15)이고, **B-1 대조 기준 = mgn125 script 구조 + margin 1.0 + Weak4-AM 게이트**(margin 1.0과 Weak4-AM 부분은 준현 명세 도착 시 확정).
- 즉 **파이프라인 뼈대(직렬화→sparse→라우팅→룰12종→leak)는 이 mgn125 코드가 최종과 동일**, 차이는 라우팅 임계값(1.25→1.0)과 Weak4-AM 게이트 추가뿐.

**(c) leak_lookup override** — §4에서 별도.

## 4. ⚠️ 판단 필요 사안 — leak_lookup (train→test override)

**코드가 실제로 하는 것** (`compute_leak_overrides`, L2239~):

leak_lookup은 **4개 tier**로 나뉨:

| tier | 무엇 | train→test 유출? |
|---|---|---|
| `positional` | test 세트 **내부** 같은 세션의 다른 step 히스토리에 답이 노출된 걸 복구 (step 산술) | ❌ 아님 (test 자기완결, "train 60,553 recovered 0 wrong") |
| `aligned` | id-free 변형, test 내부 히스토리 정합만으로 복구 | ❌ 아님 (test 내부) |
| `train_prompt_last` | **train 데이터**로 만든 `(prompt, last action)→label` 룩업으로 test 예측 덮어쓰기 | ✅ **train→test** |
| `train_prompt` | **train 데이터**로 만든 `prompt→label` 룩업 | ✅ **train→test** |

- 코드 주석 원문: train 룩업은 "holdout precision 0.978 / 0.918 **vs model ~0.74**". 즉 **모델 예측을 train 정답 룩업으로 갈아끼우면 정확도가 크게 오름.**
- 최종 결합: `pred_map.update(leak_overrides)` — **leak override가 모델+룰의 최종 예측을 무조건 덮어씀.**
- 활성 조건: `model/leak_lookup.json.gz` 파일이 팩에 **동봉돼 있으면 자동 활성.** (없으면 "Leak overrides disabled" 출력하고 model-only)

**왜 이게 판단 필요한가:**
- positional/aligned tier는 **test 세트 자체에 답이 들어있는 걸 읽는 것** → 정당한 추론 기법으로 볼 여지 큼 (외부 정보 안 씀).
- 하지만 train_prompt/train_prompt_last tier는 **train 라벨을 test에 직접 매핑** → 이게 대회 규칙상 허용되는 "학습 데이터 활용"인지, 아니면 문제 소지가 있는지는 **규칙 대조 + 팀 판단 필요.** 
- 변수명이 `leak_`이고 주석이 "Recover labels", "holdout precision vs model"인 점에서, **작성자(준현?)도 이걸 leakage 계열로 인식**하고 있었음.
- **태연 입장**: 이건 제가 "괜찮다/문제다" 단정할 사안이 아님. 다만 **발표·코드제출 전에 팀이 반드시 명시적으로 결정**해야 함. 최종 제출본에 `leak_lookup.json.gz`가 **동봉돼 있었는지**부터 확인해야 함 (동봉됐으면 우리 0.79766 점수에 train override가 이미 기여한 것).

**확인 질문 (진산·준현):**
1. 최종 선택본(amhyb_m10) 팩에 `leak_lookup.json.gz`가 동봉돼 있나? (있으면 점수에 반영됨)
2. train_prompt/train_prompt_last tier가 실제로 test에서 override를 발생시켰나? (제출 로그에 "Leak overrides: total=N ... train_prompt_last=X train_prompt=Y" 출력이 남아있을 것)
3. 대회 규칙상 train 라벨의 prompt 매핑이 허용 범위인지 — 이건 규칙 원문 대조 필요.

→ **이 사안은 발표 슬라이드 서사(특히 슬라이드 11 "실패 원장"에 leak 관련 언급이 있는지)와 코드제출 재현성 양쪽에 걸림.** 준현님 명세 요청에 "leak_lookup tier별로 최종 제출본에서 뭐가 활성이었는지" 항목 추가 권장.

## 5. 발표 슬라이드 5 (B-1) — 실물 기준 확정본

script.py 대조로 슬라이드 5 추론 파이프라인을 실물에 맞춰 확정:

```
① current_v1 직렬화 (session_meta+history+prompt → 텍스트, len 384)
② 모델 로드: main (int8-rowwise) — cuda면 half()
③ base_scores: 단일 transformer forward (동적 정렬 배칭 model_logits_sorted)
④ + sparse SVC(TF-IDF) 앙상블 가산 (sparse_weight 게이팅)   ← 실물에만
⑤ 저마진 라우팅: margin < 1.25 행만 model_b/c 추가 forward → z-centered 평균  ← 1.25!
⑥ prior calibration + class_bias + rule_boosts
⑦ 룰 스택 12종 (R1/b/c/d/e/f/g/h/i + seq-exec + twig photo1-h/garnish)
⑧ leak_overrides 덮어쓰기 (동봉 시)   ← 팀 판단 대기(§4)
⑨ submission.csv
```

**시간 분해**: `time.perf_counter()`가 코드에 **실제로 있음** (inference_start, wall_seconds 반환). 단, forward/룰/sparse **단계별 분해 로그는 없음** — 총 wall_seconds만 반환. 그래도 md가 말한 "perf_counter 없음"은 부정확 → **총 wall time은 측정됨, 단계별은 없음.**

## 6. 진산님 액션 (우선순위) — 07-15 회신 반영 후 갱신

1. **[🔴 최우선·미해결] leak_lookup 판단** — 최종 팩(amhyb_m10)에 `leak_lookup.json.gz` 동봉 여부 확인 + 규칙 대조 + 팀 결정. 준현 명세에 tier별 활성 내역 추가 요청. **← 유일하게 남은 미결 리스크.**
2. ~~[🔴] 이 script.py가 mgn125인지 amhyb인지~~ **✅ 해소: mgn125 확정** (진산 07-15). B-1 기준 = mgn125 구조 + margin 1.0 + Weak4-AM(준현 대기).
3. **[🟡] model 폴더 용량** — 512MB냐 1005MB냐. amhyb 최종팩 기준 `ls -la ./model ./model_b ./model_c`. (pptx 512 vs md 1005 통일용)
4. **[🟡] pptx 전면 갱신** — 10룰→12룰, sparse SVC 추가, 라우팅(1.25→최종 1.0), 갭 0.00112. §2 대조표 기준.
5. **[🟡] 슬라이드 5** — 위 §5 확정본으로 (leak 사안 결정 후 ⑧ 포함 여부 정리).

---

### 태연 메모
- 이 script.py는 라우팅 1.25 + 12룰 + sparse SVC + leak_lookup 구조. **mgn125 계열로 강하게 추정** (amw4는 margin 1.0이어야 함).
- **가장 중요한 건 §4 leak_lookup.** 나머지는 숫자·서사 정리지만, 이건 점수 정당성·규칙 준수에 직결. 발표 나가기 전에 팀이 명시적으로 다뤄야 함. 제가 임의 판단 안 하고 그대로 올립니다.
