# Research Log — Final 12 Hours

Continuation of `research_log.md` (frozen at the 2026-07-14 s202-transfer-gate
entry). All new entries from 2026-07-14 ~23:00 KST onward go here. Deadline:
**2026-07-15 (Wed) 10:00 KST**; the final day has the full 10 submission slots.
Dacon retains the team's highest score, so lower-scoring probes cost only slots.

## 2026-07-15 — Local weight cleanup

- Per user instruction, Drive was left untouched and local cleanup was limited
  to the working tree; `.git` history was not rewritten or pruned.
- Removed all local model bundles and weight-bearing submission archives,
  three weak4 full-CV learned checkpoints, and the trained starter
  TF-IDF/logistic model. Approximate removed file content: 95.62 GB.
- Preserved non-weight tensor artifacts (logits, OOF/consensus predictions,
  hidden features, datasets, and caches). Historical artifact paths in this
  log document provenance and may no longer exist locally.
- Follow-up audit found `kd_condalpha_refit` FP16 in an unreachable Git tar
  object. The complete nine-file HF bundle was recovered and uploaded to the
  Drive project `materials/checkpoints/` tree; model SHA256 is
  `552868fe63711dc542c1951023513f46b5301b477eaefe99aa3a2390fb294478`
  and Drive MD5 is `9ff579148caf2c14bc0422de099ffae3`. A non-reproducible
  31 KB OOF logistic-referee parameter file was also preserved under
  `materials/classical/`.
- After remote verification, reproducible random probe heads and all
  unreachable/reflog-only weight objects were deleted. `.git` shrank from
  about 4.5 GB to 131 MB without rewriting reachable commit history. Five
  learned-artifact objects (four initial baseline objects plus the uploaded
  OOF referee; 71,250,950 bytes total) remain reachable in published history
  under the prior 3A exclusion. Full hashes and decisions are in
  `experiments/manifests/20260715_local_weight_cleanup.json`.

## State snapshot (2026-07-14 ~23:00 KST)

- **Team Public champion: `submissions/rfinal_r1i_seqx.zip`, displayed
  `0.7966`, runtime `7:29`.** Exact rfinal trio archive (models byte-identical)
  plus two s202-transfer-validated hard rules (R1i, sequence-exec) in
  `script.py`. Public 1st is `0.79862`; **remaining gap ~`0.00202`**.
- Previous champion `kd_ens3_trio_rfinal` (`0.7963584846`) and its exact
  archive, the ens2 R1b archive, and seed202 OOF folds 1/2 are local
  (`experiments/manifests/20260714_team_champion_assets.json`).
- Local reproducible fallback: `kd_sieve_ca_s42.zip` (`0.7938816426`).

### Standing protocol adopted today

- **s202 transfer gate (validated end-to-end):** any post-hoc
  calibration/prior/rule lever must be re-measured on the deployed main
  model's own OOF surface (seed202 folds, local) before shipping. It admitted
  R1i + seq-exec (landed `+0.00024` Public) and rejected the P1 soft prior,
  which was `+0.0005` on the seed42 surface but negative on s202 — the same
  failure mode as the rejected `read_file -0.14` bias.
- Rejected/fail-closed today: P1 soft prior (transfer), A4 hidden-kNN,
  relational hidden-KD, and exact-predecessor replay metadata.

### Active lanes

- **Lane C (this session): terminal-teacher KD student screen**
  (`kd_terminal_teacher_m8_screen_s42`) — the last unresolved orthogonal
  training lever. Teacher (terminal-token M8, train argmax `0.809`) is done and
  verified; the student screen was stalled ~51 min by an HF Xet download hang
  (CLOSE-WAIT socket, cache stuck at 200 MB), the hung child was killed via
  the Drive cmd channel, and a staged-base fix (`colab/stage_hcx05b_base.py`,
  SHA-verified install from `AADP_exchange_b/assets/hcx05b_base`) plus a plan
  relaunch are queued on the daemon. Caveat: the lane C CLI session state was
  wiped by a transient 404/401 and keep-alive is dead — the VM survives on the
  Drive control plane only and may idle out; if reclaimed, remount is needed.
  Screen gate: control raw/bias/2stage `0.785381/0.789876/0.790594`.
- **Lanes A/B completed and released:** relational-KD and replay-predecessor
  screens both failed the matched A100 control and their conditional refits
  are closed. Results were auto-collected and pulled; the 1.1 GB screen models
  remain on Drive and are intentionally not downloaded. See
  `experiments/artifacts/20260714_breakthrough_lane_ab_screen_decision.json`
  and `experiments/manifests/20260714_lane_[ab]_*.json`.

### If a screen passes

Full sieve×condalpha refit with the single validated variable (~1.5 h A100)
→ package int8 (explicit `--hf-dir`; never the stale default `model/`) →
offline smoke → Public. R1i + seq-exec can be re-appended to any new pack's
script only after revalidation on that model's own OOF surface (implementation
in `rfinal_r1i_seqx.zip` / this repo's diagnostic
`experiments/artifacts/20260714_s202_transfer_p1_r1i_seqexec.py`). The
all-class action-margin screen later flipped R1i from positive to negative, so
automatic rule inheritance across changed model surfaces is forbidden.

---

## Entries

### 2026-07-15 ~10:00 — CLOSE-OUT: leaderboard shut at 0.7976673203; final slot = champion@0.75 coin-flip (queued)

- Final-hour sequence after the s7070-AM champion: `rfinal_amw4_42m10`
  (seed42@1.0) read **0.7960642287** — killing the "seed42 always wins"
  heuristic (same-config seed spread 0.0016 peak-to-peak). The hybrid trio
  `rfinal_amhyb_m10` (model_c old-7070 -> AM-s42 via the parity-verified
  int4 encoder) read an **exact 10-decimal tie** `0.7976673203` at `7:15` —
  second exact tie of the day; member identity in the centered low-margin
  average is prediction-inert across training-objective families. The
  **member axis is closed for good.**
- Rejected in the final hour, all on leak-free band evidence: selective
  threshold w4-1.0/else-1.25 (its target band non-w4 error is only 16.9% —
  members there are net interference), sub-1.0 grid generally, and the
  member-drop/fp16-hybrid precision card (both halves separately refuted by
  Public anchors: mlp6 −0.00006, ens2 +0.0005).
- **s777-AM race lost to the clock by ~3 minutes**: launched 08:50 on warm
  lane A (after an 07:40->08:50 wall-clock accounting error on my side),
  trained clean (AMP skips 0, epoch ckpts 09:11/09:32, final artifact
  09:54), but the irreducible int8+zip+upload pipeline (~6 min) could not
  fit before 10:00. Model archived at
  `experiments/incoming/models/kd_sieve_ca_amw4_t010_k3_refit_s777` — an
  unread @1.0 main instance, usable if any post-deadline evaluation ever
  matters.
- **Final slot (10/10)**: `rfinal_amw4_7070m075.zip` — champion pack with
  routing `<0.75` (one line). Rationale: with no experiments after it,
  interpretability is worthless and Dacon-max makes zero-mean variance
  EV-positive; the [0.75,1.0) band (33.4% err) sits below the
  weakly-identified 41.5% break-even anchor, so direction is a genuine coin
  flip. Submitted 09:57; scoring queued behind ~70 entries. **Post-close
  result: `0.79739` (`-0.00028`) — champion unchanged.** The read completes
  the three-point routing dose-response (@1.25 `0.7960` / @1.0
  `0.7976673203` / @0.75 `0.79739`), bracketing 1.0 as the measured optimum;
  the 1.0-vs-0.75 pair is a clean same-seed single-variable comparison
  confirming net member rescue in the [0.75,1.0) band.
- **Finals context (user-provided)**: 본선 = presentation 40 / score 50 /
  inference speed 10, with speed expected to bind to the top-score solution.
  The 7:15 exact-tie pack is the fastest holder of the top score; if the
  0.75 gamble lands ≥ champion it would take that role at ~7:0x.
- Day summary: 0.7966244725 -> **0.7976673203** (+0.0010428478) across 10
  slots; gap to 1st shrank 0.0020 -> **0.00096**. The decisive chain:
  weak4-AM checkpoint recovery eval (+0.00185 screen) -> routing-band audit
  (@1.25 over-routes AM mains) -> s7070@1.0. Ops lessons for the record:
  `AADP_EXCHANGE` vs `AADP_EXCHANGE_DIR` env-var fallback caused every
  cross-lane incident tonight; wall-clock arithmetic errors (twice) nearly
  cost the final two cards — timestamps beat mental clocks.

### 2026-07-15 ~07:20 — s7070-AM@1.0 Public 0.7976673203: NEW CHAMPION; gap to 1st 0.00096

- `rfinal_amw4_7070m10.zip` scored **0.7976673203**, runtime `7:23` —
  `+0.0004860553` over mainT-s42, `+0.00167` over the s909-AM@1.25 sibling.
  **The routing-correction hypothesis paid**: −32s runtime matches the
  39.7%→34.1% routing cut, and the pre-registered band-audit direction
  (19.4%-error marginal band tilts negative) is consistent with the split
  between the two AM reads. Seed draw (±0.0006) and threshold effect remain
  mixed in the +0.00167, but both AM cards agree the axis is champion-grade
  at @1.0.
- Gap to 1st (`0.79863`): **`0.0009626797`** — sub-0.001 for the first time.
- Next: `rfinal_amw4_42m10.zip` ready (s42-AM, fidelity 512/512, @1.0,
  SHA `8eb6a931…`, smoke OK) — best-of-N at the winning configuration.
  Slots after it: 2 (candidates: selective threshold on the best AM pack,
  all-AM member trio via the validated INT4 encoder, reserve).

### 2026-07-15 ~06:38 — Weak4-AM seed42 clean full refit COMPLETE; artifact pulled

- Lane B rerun completed all 15,000 optimizer steps with **AMP skips 0** and
  auto-collected successfully (`release_safe=true`). The original crashed run
  was discarded for promotion; its epoch-2 checkpoint remains archived as
  `kd_sieve_ca_amw4_t010_k3_refit_s42_ckpt_crash_20260715_0503`.
- Clean run: `20260714_212744_gpu_transformer_session_current_v1_len384_replay-last1_kd_sieve_ca_amw4_t010_k3_refit_s42`, runtime `3667.87s`, final fp16
  artifact `1090.64 MiB`. Action-margin calibration was deterministic and
  matched the contract: Weak4 true-label scope, teacher top-3, target gradient
  ratio `0.10`, fixed weight `0.143567735`; active rows `28,782/epoch` and
  mean auxiliary loss `0.05691 -> 0.01138 -> 0.00651`.
- Model pulled to
  `experiments/incoming/models/kd_sieve_ca_amw4_t010_k3_refit_s42`; weights
  SHA256 `890d05c3a546e0085df8c33270e5102c34ecae62de01a538fff750d0dbd2331d`.
  This was a `--final-only` refit, so its ledger Macro-F1 `0` is a sentinel,
  not an evaluation. Promotion evidence remains the matched fixed-session
  recovery eval: raw `+0.0018513`, Weak4 Macro `+0.0084416` versus the seed42
  control. Next operation is INT8 quantization and a main-only swap into the
  current mainT pack; no Dacon submission has been made from this run.

### 2026-07-15 ~06:45 — s909-AM Public 0.7960 (-0.00118): axis undecided; over-routing suspect confirmed by runtime

- `rfinal_amw4_909.zip` scored **0.7960**, runtime `7:55` — below champion by
  `-0.00118`, below mgn125 by `-0.0006`. Not a clean kill: the run mixes AM
  recipe + seed909 + the untreated over-routing (39.7% at @1.25 on AM
  margins; runtime +18s vs mainT matches the prediction). The [1.00,1.25)
  band on the AM surface is only 19.4% error, so @1.25 exposes mostly-correct
  rows to member interference — exactly what @1.0 removes.
- Decision: proceed with **s7070-AM@1.0** (tests the routing correction;
  artifact ~06:25) and the lane-B **s42-AM** rerun (~07:30, resolves the seed
  axis). Kill criterion: if s7070-AM@1.0 also reads ≤ mgn125 (0.7966), the
  AM-ensemble lane closes and s42-AM submission becomes optional. Slots
  remaining 4.

### 2026-07-15 ~06:05 — s909-AM axis pack ready; geometry-lock REJECTED at the exact-op gate; selective threshold survives

- **`submissions/rfinal_amw4_909.zip` ready** (SHA `4c605173…`,
  1,002,964,322 B): lane-B VM death promoted s909-AM (clean run, **AMP
  skips 0**, INT8 fidelity 511/512, zero bias, λ 0.1239) to the axis card;
  mainT pack with only `model/` swapped, script @1.25 byte-kept. Smoke OK.
- **Geometry-lock gate (opinion.md card 2): negative.** Members (INT4
  model_b/c) were forwarded on the 14,001 val rows (leak-inflated: val
  macro 0.816/0.817) and the exact deployed op was replicated. lock
  `-0.00348` macro and **`-0.0122` Weak4** vs the ens125 baseline —
  the metric it was built to improve dropped hardest. Formally inconclusive
  (leak favors member-heavy arms), but with zero exact-op positive evidence
  and decision-shaped levers 0-for-4 tonight, **no slot for the lock**. The
  opinion's `+0.0012` citation was a different operation (control-blend
  graft on the screen manifest), not this ensemble lock.
- **Selective threshold (weak4-internal 1.0 / else 1.25): alive.**
  `-0.00014` ≈ neutral on a surface biased AGAINST member-reach reduction →
  plausibly positive on the honest surface; keeps priority over global 1.0,
  after the seed cards. Artifacts:
  `experiments/artifacts/20260715_geometry_lock_gate.json`,
  `experiments/logits/20260715_pack_model_{b,c}_val_logits.pt`.
- Sequence: ① s909-AM@1.25 submit now → ② positive: s7070-AM@1.0 (~06:25
  artifact) → ③ s42-AM lane-B rerun → ④ selective-threshold card → ⑤ reserve.

### 2026-07-15 ~05:30 — s777 swap Public 0.7969: lower draw; champion holds

- `rfinal_mainT_s777.zip` scored **0.7969** (`7:36`) — about `-0.0003` vs the
  mainT-s42 champion `0.797181265`. Not promoted; slot spent as priced.
- Draw-band read: two trioT main draws now exist (s42 `+0.00056`, s777
  `-0.0003` relative) — consistent with ~±0.0006. Plain trioT rerolls are
  deprioritized; all three lanes are on the weak4-AM axis (B s42 main card,
  C s909, A s7070). Slots remaining ~5.

### 2026-07-15 ~05:00 — s777 swap pack ready; "rfinal_a1main" is actually an a0-class main

- **`submissions/rfinal_mainT_s777.zip` ready** (1,002,965,591 B, SHA
  `d4d2b46b…`): trioT-s777 full refit (15,000 steps, AMP skips 9, artifact
  1090.6 MB) INT8-quantized (fidelity **512/512**), swapped as `model/` into
  the mainT pack; script/model_b/model_c byte-kept. Clean-extraction smoke
  OK. Provenance verified (hf_meta seed 777, zero bias) — the earlier
  dual-writer exposure affected only the unused `_ckpt` dir; the final
  artifact was single-writer.
- **`rfinal_a1main.zip` is mispacked**: its main is `seed 202, replay_mode
  last1`, trained on old code (no `replay_meta_mode` key), weights distinct
  from both the deployed s202 main and B-fast — profile matches **a0**
  (slightly-old nested champion), not a1 (replay-all; a1 likely never had a
  full refit). Structure/script are current champion (SHA `f286f21d…`),
  clean-extraction smoke passed (pack SHA `1bee02be…`). Re-labeled mentally
  as an a0-class reserve draw (slight negative tilt, last priority);
  teammate asked to confirm which artifact was packed.
- Slot sequence unchanged: s777 swap → lane-B seed42-AM swap → (positive
  read) lane-C s909-AM swap → a0-class reserve only if slots remain.

### 2026-07-15 ~04:15 — Weak4-AM checkpoint recovery eval: POSITIVE (+0.00185 macro, +0.00844 Weak4)

- Damage assessment of the fail-closed Weak4 action-margin screen: the
  epoch-3 checkpoint is **intact** (219 tensors, zero non-finite;
  `checkpoint_state.json` epoch 3 complete, calibrated weight `0.2619077`,
  23,038 active rows/epoch — matches the manifest). Timeline correction: the
  trainer never stalled; the "GPU 0%" that led to the parent-kill was the
  idle lane C heartbeat read through the env-var fallback. The only loss was
  the unexecuted validation step.
- Recovery eval (local GPU, exact screen protocol: seed42 fixed session
  split, `current_v1`, len384, `train_transformer.evaluate` path; 14,001 val
  rows, 115s): **raw macro `0.7853020` vs control `0.7834507` =
  `+0.0018513`; Weak4 macro `0.6123472` vs `0.6039056` = `+0.0084416`.**
  Per-class: read_file 0.623, grep 0.643, list_dir 0.523, glob 0.660.
- Reading: the Weak4-only scope kept (and enlarged) the Weak4 gain that the
  rejected all-class variant showed (`+0.0084` vs `+0.0047`) while flipping
  the aggregate from negative to positive — the same-seed paired comparison
  isolates the single scope variable. **This is the only positive
  training-recipe screen of the endgame window.**
- Status: diagnostic (8 AMP skips tripped a plan bug, not weight damage — the
  eval itself is the damage check). Promotion path: full refit with the
  champion recipe + weak4-scoped AM + stabilized AMP flags
  (`--amp-init-scale 64 --amp-growth-interval 1000000`), then main-swap into
  the mainT pack. Artifact:
  `experiments/artifacts/20260715_weak4_am_ckpt_recovery_eval.json`.

### 2026-07-15 ~03:55 — ops backfill: s777 sole-writer on lane A; s909 held; lane C stale

- seed777 (trioT main draw) launched 03:12 KST intending lane C but executed
  on lane A via the `AADP_EXCHANGE` vs `AADP_EXCHANGE_DIR` env-var fallback
  documented in the incident entry below. After the orphan SIGKILL at
  03:37 the tracked trainer (PID 63117) is the **single writer**; ETA ~04:45
  (the 03:12–03:37 dual-trainer contention cost ~10-15 min). The final
  artifact dir is written once at completion by the survivor; only the
  `_ckpt` dir had overlapping writers and it is not used for packaging.
  Packaging gate: hf_meta seed==777 + INT8 quantize-verify fidelity.
- **User directive: seed909 is on hold** (plan exists locally and on the lane
  A VM; not launched anywhere). Priority shifted to Weak4 checkpoint damage
  assessment/recovery. Lane C VM heartbeat went stale (500s+, idle since
  seed42 completed) — treat as reclaimed; no revive unless a training need
  reappears.
- Monitoring of s777 moved to passive rclone reads of the lane A heartbeat;
  no further cmd-queue writes toward lane A from this session.

### 2026-07-15 ~03:05 — a1 replay-all screen: 2-fold REJECTED; trajectory axis closed (backfill)

- Team-side final verdict (single-model OOF, replay `all` vs champion
  `last1`, seed42, same folds, replay mode the only variable):
  fold0 A1 `0.78782` vs regenerated A0 control `0.78704` = `+0.00078`;
  fold1 A1 `0.79202` vs original `champ_oof_f1` `0.79363` = `-0.00161`;
  two-fold mean `-0.00042`. Standing rule (any negative fold rejects) →
  **a1-refit rejected; trajectory-densification axis closed.**
- The earlier preliminary `+0.00208` on fold0 was retracted: its control file
  had been deleted in a disk cleanup, and the clean regeneration moved the
  baseline `0.78574 -> 0.78704`, shrinking the gain to noise level.
- Consistent with the lane-B replay-predecessor rejection: densifying replay
  trajectories does not beat champion `last1` under the current recipe.
- Team-side reserve: `rfinal_a1main.zip` (1005.7 MB, structure-verified,
  unsubmitted) is a main-swap draw with a slight negative tilt
  (`-0.00042` OOF mean); hold unless slots remain after neutral draws.

### 2026-07-15 ~03:30 — B-fast Public 0.79614: lower instance; no mainT/trioT was applied

- `kd_sieve_ca_bfast_s202` completed the reconstructed seed202 direct full
  refit in `3,621.3s` with zero AMP skips. Its fail-closed replay audit
  passed: 10,000 legacy current-metadata replay rows, 9,575 exact predecessor
  links, and 7,112 consensus-`c>=2` replay rows receiving predecessor KD;
  serialized replay text/order stayed invariant and active teacher top-1
  accuracy was `0.982705`.
- **Teacher/composition clarification:** B-fast used the original M8 train70k
  logits, not mainT/trioT. The submitted pack contains B-fast seed202 INT8 as
  the sole `model/` main. `model_b`, `model_c`, and `script.py` are
  byte-identical to both mgn125 and mainT packs, but the mainT-s42 weights are
  absent because `model/` was replaced rather than stacked.
- `submissions/rfinal_bfast_s202.zip` (1,005,616,302 B, SHA256
  `9d3b47b9...b816d`) scored **`0.79614`**, runtime **`7:51`**. That is about
  `-0.001041` versus the `0.797181265` mainT champion and `-0.000484` versus
  mgn125, using the user-reported rounded score.
- **Decision: do not promote.** The champion remains `rfinal_mainT_s42`.
  The delta is still inside the `0.002` interpretation band, so this is a
  lower full-refit instance, not strong causal evidence that trusted replay
  KD is harmful. Exact run/package audit lives in
  `experiments/manifests/20260715_lane_b_bfast_trusted_replay_kd_refit.json`.

### 2026-07-15 ~03:20 — Weak4 follow-up invalid; A/C command-plane incident reconstructed

- The true-label-Weak4 action-margin follow-up reached epoch 3 and wrote a
  checkpoint (`scope=weak4`, calibrated weight `0.261907703`, 23,038 active
  rows per epoch), but it is **not a completed screen**. The plan parent was
  killed while the trainer child continued orphaned; final audit then found
  8 skipped AMP optimizer steps out of 12,375 and failed closed before
  validation, final model save, results row, or auto-collect.
- The immediate plan bug was requiring zero AMP skips without the stabilized
  `--amp-init-scale 64 --amp-growth-interval 1000000` settings. Therefore no
  positive/negative quality verdict is allowed. The epoch-3 checkpoint may be
  evaluated as a diagnostic, but promotion still requires a clean run.
- Control-plane correction: some intended lane-C manual commands used
  `AADP_EXCHANGE=AADP_exchange_c`; `cloud_sync.py` recognizes
  `AADP_EXCHANGE_DIR`, so those calls fell back to `AADP_exchange` and acted
  on lane A. A later exhaustive `/proc/*/environ` scan found only the proper
  A daemon (`PID 1150`, `AADP_exchange`) and the separate idle C daemon
  (`PID 1573`, `AADP_exchange_c`); no duplicate exchange-C daemon was found
  or killed. The actual `AADP_exchange_c/cmd/done` directory had no new
  command after `15:52Z`, confirming that the later s909 refusal was written
  to the default A queue rather than claimed from the C queue.
- The same scan found two identical seed777 trainers (`PID 60035` and `63117`)
  writing the same checkpoint/output destinations. After explicit user
  authorization, the first orphan was identity-checked and SIGKILLed at
  `18:37:20Z`; tracked trainer `63117` remained alive and A GPU memory returned
  from ~23.4 GiB to ~11.7 GiB. Because the two writers overlapped earlier, the
  final seed777 artifact remains collision-exposed and must pass
  provenance/integrity verification before use. Full incident state is
  retained in the Weak4 manifest.

### 2026-07-15 ~02:25 — all-class action-margin KD rejected; rules cannot recover it

- Against the same-code seed42 control, teacher-top3 action-margin KD changed
  raw/bias/2-stage Macro-F1 by
  **`-0.000679/-0.001175/-0.000744`**. Raw Weak4 mean improved
  **`+0.004693`**, but the aggregate loss came from non-Weak4 behavior,
  especially the `ask_user`/`plan_task` boundary.
- Post-hoc rescue failed: ten rules still left `-0.000431`; adding R1i and
  sequence-exec widened the gap to `-0.001287`; tuned bias plus all rules was
  `-0.001861`. R1i itself flipped from `+0.000452` on the control to
  `-0.000267` on the action-margin surface, and no new ask/plan rule was
  positive on all three subfolds.
- **Decision:** reject the all-class auxiliary and do not force it back with
  rules. The Weak4-only scope remains a logically separate hypothesis because
  its motivating class slice improved; its first attempted run above is void,
  not a negative result. Detailed paired metrics and rule audit are stored in
  `experiments/manifests/20260714_lane_a_action_margin_kd_pair.json`.

### 2026-07-15 ~02:50 — mainT swap 0.797181265: NEW CHAMPION (+0.00056); gap to 1st now 0.00144

- `rfinal_mainT_s42.zip` scored **0.797181265**, runtime `7:37` —
  `+0.0005567925` over mgn125. **New team Public champion.** Gap to 1st
  (`0.79863`, user-updated 07-15): `0.001448735`.
- Read discipline: sub-0.002 instance win, not recipe evidence. The same
  weights read neutral solo (`0.79368`), so this is a favorable main-draw
  inside the ensemble — empirically, main swaps move ~±0.0006 at champion
  scale (the correlation-shrinkage worry did not bind on this instance).
- Consequences: every remaining main instance is a legitimate 1-slot
  best-of-N draw around the new champion — teammate seed202-trioT (~03:30)
  gets re-promoted from "optional" to "submit when ready" (as a mainT-pack
  main swap), and an a1-refit (if the ~02:15 verdict passed) doubles as both
  a hypothesis test and another draw. Base pack for all further main swaps is
  now `rfinal_mainT_s42`.
- Docs updated: `final_summary.md` champion section,
  `leaderboard_calibration.md` ledger.

### 2026-07-15 ~02:35 — trio-KD solo Public 0.79368: neutral tie; gen-2 dead, seed202 demoted

- `kd_trioT_s42.zip` scored **0.79368**, runtime `5:57` — `-0.0002` vs the
  M8-teacher single champion `0.7938816426`. Exact-tie class: the trio
  teacher's ensemble dark knowledge did not convert to Macro-F1 at seed42.
- Consequences: **gen-2 KD closed** (no gain to re-distill); teammate
  seed202-trioT is no longer a gap-scale card — if its run completes, using
  it is an optional best-of-N main-swap variance draw (~±0.001), not a
  hypothesis test. The last genuine lever standing is the a1 replay-all axis
  (same-fold verdict due ~02:15): if a1 wins both folds, the A100 should
  preempt seed202 for the a1-refit rather than queue behind it.
- `rfinal_mainT_s42` (main-swap probe) reads separately; expectation is
  champion ± instance noise, with downside risk from higher student-member
  correlation on routed rows.

### 2026-07-15 ~02:15 — trio-KD refit done; solo + main-swap packs built and smoked

- Lane C run `run_20260714_155225` completed: teacher export (rows=70000,
  routed=21589, main_agreement=0.9868) + full refit (15,000 steps, 3,627s,
  AMP skips 9, artifact 1090.6 MB). INT8 fidelity **512/512 argmax (100%)**,
  weight mean_rel 0.97%.
- Two packs ready under the no-gate endgame policy (slots 9, packaging-first):
  - `submissions/kd_trioT_s42.zip` (512 MB, SHA `57968ba2…`) — solo
    single-model pack, packager smoke OK. Read baseline: champion single
    `0.7938816426`; also gates teammate seed202 usage and gen-2 KD.
  - `submissions/rfinal_mainT_s42.zip` (1,002,963,703 B, headroom 70.8 MB,
    SHA `847e04e0…`) — exact mgn125 champion pack with only `model/` (main)
    swapped s202-INT8 -> trioT-s42-INT8; model_b/model_c/script byte-kept.
    Clean-extraction offline smoke OK (5 rows, all rules executed, ID
    order/labels valid). Note: vs the deployed champion this changes seed
    (202->42) and teacher (M8->trio) together — a Public instance probe, not
    a causal read. Caveat: the trio teacher was distilled FROM
    model_b/model_c, so the routed-row ensemble gain may shrink (higher
    student-member correlation).
- Meta check: trioT trained at HEAD carries `replay_meta_mode: current`
  (default; lane-B controls reproduced champion numbers at HEAD, so behavior
  is unchanged) and zero class bias.

### 2026-07-15 ~01:45 — xlm-r-large member-survival gate: FAIL; cross-family member axis closed

- Question (user-prompted): for the Wave-2 diversity-member card, isn't
  xlm-r-large better than mbert? Answer: yes on the old evidence
  (07-07 blend probe vs non-KD HCX: xlm-r-large `+0.0073` > xlm-r-base
  `+0.0041`; mbert's qv600 pairing evidence was anchored on xlm-r mains, not
  HCX) — but the KD-absorption risk (m7 precedent: `+0.0088` -> `-0.0027`
  after KD) had to be re-measured on the deployed surface first.
- Free gate, no GPU/slot: p2 xlm-r-large len384 3-fold OOF logits (07-04)
  give honest OOF for every s202 fold-1/2 row (23,332/23,334 rows, 0 dropped;
  routed fraction 0.3187/0.3189 matches deployment). Deployed pipeline
  replicated (margin<1.25 routing, ask-boost, all rules); blends tested on
  routed rows only: centered-mean, z-mean, softmax w50, softmax w70-HCX.
- **All variants negative on both folds.** Pooled: centered `-0.0044`,
  z `-0.0045`, prob-w50 `-0.0052`, prob-w70 `-0.0013`; harm > rescue in every
  cell (e.g. fold2 centered 255 rescue / 361 harm). xlmr raw OOF macro
  0.734/0.747 vs s202 0.783/0.796 — the KD students absorbed/surpassed what
  xlm-r added to the pre-KD HCX; the m7 pattern generalizes across
  architecture families.
- **Decision: no cross-family member card — neither mbert nor xlm-r-large.**
  Wave-2 negative-branch fallback is KD seed production / free variance, not
  member diversity. Artifacts:
  `experiments/artifacts/20260715_xlmr_member_survival_gate.{py,json}`.

### 2026-07-15 ~01:10 — s202v sieve refit Public 0.7919; lane closed, teammate seed plan cancelled

- `kd_s202v_s42.zip` scored **0.7919**, runtime `6:09`. One variable vs
  `kd_sieve_ca_s42` (`0.7938816426`): the consensus payload voter swap
  (v6 -> deployed s202 on the 46,666 fold-1/2-covered rows). Delta about
  `-0.0020`.
- Read: at the edge of the noise band, so not causal evidence the swap hurts —
  but there is no trace of the `+0.002`-class sieve gain this lane was priced
  on. The payload moved only 4,039/70,000 rows' correct-counts, and aligning
  sieve weights to the deployed main model's own errors bought nothing on this
  instance.
- **Decision: close the s202v-voter lane.** Cancel the teammate's s202v
  seed202 pre-training instruction (no positive seed42 read to justify 3x seed
  cost with ~9 hours left); redirect that lane to a0/a1 follow-ups. Champion
  unchanged: `rfinal_mgn125` `0.7966244725`, gap to 1st `0.0019955275`.
- Trio-KD refit on lane C remains the live gap-scale card: teacher export
  succeeded (`rows=70000 routed=21589 main_agreement=0.9868` — the teacher
  disagrees with the s202 main on ~924 rows, the dark knowledge being
  distilled), training healthy at step 2500/15000, ETA ~02:00 KST.

### 2026-07-15 ~00:45 (clock corrected; logged as 02:10) — mgn125: NEW CHAMPION by 1.2e-5; widening axis closed

- `rfinal_mgn125.zip` Public **`0.7966244725`**, runtime `7:46` (projection
  7:57). `+0.0000118616` over `rfinal_r1i_seqx` — noise-level, but retained by
  Dacon max, so the team standing improves and the gap to 1st is now
  **`0.0019955275`**. Unlike the c0a8 tie, predictions did change: the net
  ensemble rescue in the widened [1.00,1.25) band is ~break-even, i.e. the
  ensemble-gain gradient is flat beyond margin 1.0. **Do not spend a slot on a
  1.5 widening.** The s202-voter sieve refit (`kd_sieve_ca_s202v_refit_s42`)
  is training on lane C as the remaining gap-scale card.

### 2026-07-15 ~00:15 (clock corrected; logged as 01:40) — c0a8 swap probe: exact tie; margin-1.25 probe submitted

- `rfinal_c0a8_swap.zip` Public **`0.7966126109` — identical to the champion
  to 10 decimals**, runtime `7:17`. The member swap flipped zero hidden-test
  predictions: same-recipe-family members are prediction-equivalent inside the
  low-margin z-centered trio average. **Member-swap axis closed** absent a
  genuinely diverse strong member (none exists locally — the cross-family
  candidates are all ~0.70-0.75 class). Side value: the champion's exact
  score digits are now known (`rfinal_r1i_seqx` = `0.7966126109`; gap to 1st
  `0.79862` = `0.0019473891`).
- Submitted `rfinal_mgn125.zip` (last slot of the window): single variable =
  routing margin `<1.0 -> <1.25` (24.4% -> 31.9% routed). Basis: s202
  margin-band audit — [1.00,1.25) carries a 41.5% error rate, matching the
  routed band's density, while [1.25,1.50) drops to 24.5%; projected runtime
  `7:57`. Result pending. Duplicate build `rfinal_margin125.zip` (bit-identical
  script) removed; `rfinal_mgn125.zip` is the canonical artifact.
- Lane 1 (s202-voter consensus sieve refit) proceeds in the parallel session;
  payload `20260715_m7_m8_s202v_oof_consensus.pt` was reconstruction-verified
  here before handoff (4,039/70,000 counts changed; c=3 48,607 -> 49,325).

### 2026-07-14 ~23:25 (clock corrected; logged as 07-15 00:50) — terminal-teacher screen REJECTED; c0a8 member-swap probe ready

- `kd_terminal_teacher_m8_screen_s42` (run `20260714_142018`, staged-base fix,
  full 3-arm plan, rc=0): raw/bias/2stage `0.782026/0.786332/0.787495` vs
  control `0.785381/0.789876/0.790594` — **deltas
  `-0.003355/-0.003544/-0.003099`, all tiers clearly negative**, no weak-class
  compensation (list `0.5223`, read `0.6281`, grep `0.6392`, glob `0.6508`).
- **Decision: reject; do not launch the staged refit; no slot spent. The
  terminal-token lane is now closed on both sides** (student pooling
  `-0.0020`, teacher signal `-0.0031..-0.0035`), consistent with every prior
  teacher-signal replacement failing (tm8 `0.7867`). The soft-target-quality
  thesis is answered negatively for this family.
- **`submissions/rfinal_c0a8_swap.zip` is submission-ready** (SHA256
  `f552bfe2...c92b29`, 1,004,646,637 B): champion pack with only the model_c
  weights swapped s7070-int4 -> c0a8-int4 (`int4-group128-v1` codec parity
  verified; fidelity 128/128 argmax, TV mean `0.0176`; clean-extraction CPU
  smoke ran the full 3-member ensemble + 12 rules). Pure Public probe —
  no honest local surface exists for member quality; seed-swap precedent
  ~`+0.0004`; costs one slot, zero score risk (Dacon keeps the max).
- Lane C A100 session remains up with a healthy keep-alive (release deferred
  to the user: teammate's corrected A/B re-runs may want a warm A100 with the
  HCX base already staged).

- Reopened the read_file logit bias with the one condition its earlier
  rejection named: tuning on the deployment model's own OOF (s202 folds, now
  local), on the full deployed pipeline (rules + R1i + seq-exec) with
  two-direction LOFO. Result: f1-tuned `-0.26` confirms `-0.000947` on f2;
  f2-tuned `-0.02` confirms `-0.000046` on f1; the preregistered team value
  `-0.14` is `+0.000055/-0.000648`. Fold2's entire grid is flat.
- **Decision: the bias signal is fold noise, not model calibration — closed
  for good; no slot spent.** Fourth decision-shaped lever rejected today.
- Prepared and hotfixed onto the lane C VM the single-variable full-refit plan
  `colab/terminal_teacher_refit_lane_c_plan.json` (exact kd_sieve_ca_s42
  command with only `--distill-logits` swapped to the terminal-M8 payload;
  output `kd_sieve_ca_tterm_refit_s42`) so the refit can launch the moment the
  screen reads out. Slot posture (user): teammate lanes are consuming slots —
  submit in priority order refit > (bias: dead) > member-swap probe, gated
  cards only, and lenient screen reading per user directive.

### 2026-07-14 ~23:00 — A4 hidden-kNN rejected at the s202 transfer gate

- Teammate handoff `s202_hidden_train70k_fp16.pt` (70,000 x 1024 fp16,
  "s202 kd_sieve_ca last-layer last-token pre-score hidden") landed at the
  repo root, unblocking the previously fail-closed A4 gate. Built A4-compatible
  caches per s202 OOF fold (fold rows = queries with honest OOF logits;
  datastore = the other 46.7k rows; label alignment cross-checked) and ran the
  unmodified `audit_hidden_knn_memory.py`.
- **Confirm-fold deltas: fold1 `-0.000408` macro / `-0.001427` Weak4; fold2
  `+0.000206` / `+0.000720`.** Sign-inconsistent across folds, pooled ~zero,
  rescue:harm 1.17/1.24, and the selected config is unstable (k=5,w=0.4 vs
  k=31,w=0.8). All of this under an *optimistic* bias: query hiddens come from
  the full-refit model that trained on those rows, a flattery deployment will
  not enjoy.
- **Decision: do not ship the kNN.** The clean champion-surface `+0.001062`
  (5/5 folds) evaporates on the deployed model's own surface. Third
  decision-shaped lever to die at this gate (read_file bias, P1, now A4) —
  consistent with the standing conclusion that only distribution-shaped
  mechanisms have ever paid on Public. Artifacts:
  `experiments/artifacts/20260714_s202_knn_gate_f{1,2}.json`.
- Lane C screen meanwhile healthy on the fresh session: staging + verifier
  arms passed, training at step 1500 with falling loss.

> **2026-07-15 correction (user-reported):** an implementation mistake was
> found in the lane A/B experiments below; the teammate plans to re-run both.
> Treat the two rejections as VOID pending the corrected re-experiments, not
> as closed lanes.

### 2026-07-14 ~22:42 — A/B breakthrough screens completed; both refits closed

- Both fixed-session seed42 A100 screens completed in ~51 minutes, saved all
  epoch checkpoints, auto-collected successfully, and were pulled by explicit
  run name after normal VM release. The SIGKILL cleanup-bypass recovery did not
  interrupt either detached trainer; no replacement VM or extra seed was used.
- **Lane A relational hidden-KD:** raw/bias/2-stage
  `0.784092/0.787127/0.787876`, versus matched control
  `0.785381/0.789876/0.790594`; deltas
  `-0.001289/-0.002749/-0.002718`. The implementation gate passed: all 70,000
  original rows aligned, 10,000 replay rows were relation-masked, and the
  hidden payload matched canonical teacher logits at `0.997129` argmax
  agreement (`max_abs=0.107422`).
- **Lane B exact-predecessor replay metadata:** raw/bias/2-stage
  `0.782022/0.784410/0.785709`; deltas
  `-0.003358/-0.005466/-0.004885`. Its audit exactly matched the card: 48,853
  tail candidates, 46,775 exact predecessors, 2,078 fail-closed missing drops,
  and the unchanged class-balanced cap selected 10,000.
- Priority classes make the rejection directional rather than a near-tie:
  isolated `read_file`/`glob_pattern` gains could not offset losses in
  `grep_search`, `web_search`, and `lint_or_typecheck`; predecessor replay also
  hurt `run_bash` and `run_tests`. **Decision:** launch neither champion refit,
  pull neither screen model, and spend no Public slot. Full per-class deltas:
  `experiments/artifacts/20260714_breakthrough_lane_ab_screen_decision.json`.
