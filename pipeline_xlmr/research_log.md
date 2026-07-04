## 20260701_191727_gpu_transformer_session_current_v1_len256_B1-focal-len256-5ep-batch12

- Date/time: 2026-07-01 19:17:28 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=none, max_length=256, epochs=5, lr=2e-05, batch=12, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.732274
- Overall Macro-F1: 0.736614
- Per-class observations:
  - Weakest: list_directory=0.472, read_file=0.572, web_search=0.583, grep_search=0.599, glob_pattern=0.625
  - Strongest: respond_only=1.000, write_file=0.995, edit_file=0.969, apply_patch=0.936, run_bash=0.815
- Top confusions: [(598, 'grep_search', 'read_file'), (345, 'read_file', 'list_directory'), (262, 'list_directory', 'read_file'), (258, 'grep_search', 'list_directory'), (234, 'glob_pattern', 'read_file'), (233, 'read_file', 'grep_search'), (145, 'glob_pattern', 'list_directory'), (134, 'ask_user', 'plan_task')]
- Prediction distribution: {'apply_patch': 935, 'ask_user': 469, 'edit_file': 2244, 'glob_pattern': 747, 'grep_search': 1437, 'lint_or_typecheck': 480, 'list_directory': 1259, 'plan_task': 563, 'read_file': 2292, 'respond_only': 1052, 'run_bash': 1001, 'run_tests': 893, 'web_search': 321, 'write_file': 308}
- Runtime or package-size concerns: runtime=3843.4s, tokenize=11.1s, train=3761.4s, eval=25.4s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260701_191727_gpu_transformer_session_current_v1_len256_B1-focal-len256-5ep-batch12_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260701_201619_gpu_transformer_session_current_v1_len192_replay-last1_B2-focal-len192-replay_last1-cap10000-5ep-batch1

- Date/time: 2026-07-01 20:16:20 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.735109
- Overall Macro-F1: 0.744391
- Per-class observations:
  - Weakest: list_directory=0.450, read_file=0.568, grep_search=0.591, glob_pattern=0.623, ask_user=0.630
  - Strongest: respond_only=0.999, write_file=0.993, edit_file=0.966, apply_patch=0.930, run_bash=0.828
- Top confusions: [(650, 'grep_search', 'read_file'), (325, 'read_file', 'list_directory'), (311, 'list_directory', 'read_file'), (263, 'glob_pattern', 'read_file'), (240, 'grep_search', 'list_directory'), (206, 'read_file', 'grep_search'), (190, 'ask_user', 'plan_task'), (130, 'glob_pattern', 'list_directory')]
- Prediction distribution: {'apply_patch': 924, 'ask_user': 461, 'edit_file': 2253, 'glob_pattern': 747, 'grep_search': 1370, 'lint_or_typecheck': 380, 'list_directory': 1159, 'plan_task': 664, 'read_file': 2471, 'respond_only': 1053, 'run_bash': 1042, 'run_tests': 944, 'web_search': 226, 'write_file': 307}
- Runtime or package-size concerns: runtime=3456.8s, tokenize=10.8s, train=3365.5s, eval=20.8s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260701_201619_gpu_transformer_session_current_v1_len192_replay-last1_B2-focal-len192-replay_last1-cap10000-5ep-batch1_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260701_222706_gpu_transformer_session_current_v1_len192_S1-1-focal-len192-5ep-FGM-eps1.0

- Date/time: 2026-07-01 22:27:07 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=none, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.718962
- Overall Macro-F1: 0.724287
- Per-class observations:
  - Weakest: list_directory=0.473, web_search=0.524, read_file=0.571, grep_search=0.583, lint_or_typecheck=0.614
  - Strongest: respond_only=0.999, write_file=0.995, edit_file=0.972, apply_patch=0.944, run_bash=0.789
- Top confusions: [(642, 'grep_search', 'read_file'), (352, 'read_file', 'list_directory'), (277, 'grep_search', 'list_directory'), (270, 'list_directory', 'read_file'), (241, 'glob_pattern', 'read_file'), (177, 'read_file', 'grep_search'), (160, 'run_bash', 'run_tests'), (154, 'glob_pattern', 'list_directory')]
- Prediction distribution: {'apply_patch': 916, 'ask_user': 530, 'edit_file': 2261, 'glob_pattern': 829, 'grep_search': 1235, 'lint_or_typecheck': 559, 'list_directory': 1307, 'plan_task': 515, 'read_file': 2377, 'respond_only': 1051, 'run_bash': 896, 'run_tests': 907, 'web_search': 310, 'write_file': 308}
- Runtime or package-size concerns: runtime=4845.4s, tokenize=2.6s, train=4761.1s, eval=21.6s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260701_222706_gpu_transformer_session_current_v1_len192_S1-1-focal-len192-5ep-FGM-eps1.0_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260701_230939_gpu_transformer_session_oof_current_v1_len192_fold0-of3_S1-2-OOF-fold0

- Date/time: 2026-07-01 23:09:39 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=none, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=0/3
- Raw Macro-F1: 0.720907
- Overall Macro-F1: 0.720907
- Per-class observations:
  - Weakest: list_directory=0.469, read_file=0.534, web_search=0.559, lint_or_typecheck=0.589, grep_search=0.599
  - Strongest: respond_only=0.998, write_file=0.990, edit_file=0.963, apply_patch=0.926, run_bash=0.792
- Top confusions: [(787, 'grep_search', 'read_file'), (636, 'read_file', 'list_directory'), (627, 'read_file', 'grep_search'), (473, 'grep_search', 'list_directory'), (307, 'list_directory', 'read_file'), (294, 'run_bash', 'run_tests'), (274, 'glob_pattern', 'read_file'), (253, 'ask_user', 'plan_task')]
- Prediction distribution: {'apply_patch': 1695, 'ask_user': 720, 'edit_file': 3649, 'glob_pattern': 1433, 'grep_search': 2908, 'lint_or_typecheck': 720, 'list_directory': 2219, 'plan_task': 968, 'read_file': 3019, 'respond_only': 1730, 'run_bash': 1515, 'run_tests': 1740, 'web_search': 517, 'write_file': 501}
- Runtime or package-size concerns: runtime=2419.5s, tokenize=2.3s, train=2373.2s, eval=34.1s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260701_230939_gpu_transformer_session_oof_current_v1_len192_fold0-of3_S1-2-OOF-fold0_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260701_235001_gpu_transformer_session_oof_current_v1_len192_fold1-of3_S1-2-OOF-fold1

- Date/time: 2026-07-01 23:50:01 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=none, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=1/3
- Raw Macro-F1: 0.716689
- Overall Macro-F1: 0.716689
- Per-class observations:
  - Weakest: list_directory=0.466, read_file=0.528, web_search=0.571, lint_or_typecheck=0.590, grep_search=0.601
  - Strongest: respond_only=1.000, write_file=0.984, edit_file=0.958, apply_patch=0.915, run_bash=0.777
- Top confusions: [(799, 'grep_search', 'read_file'), (641, 'read_file', 'list_directory'), (619, 'read_file', 'grep_search'), (459, 'grep_search', 'list_directory'), (317, 'list_directory', 'read_file'), (278, 'glob_pattern', 'read_file'), (269, 'run_bash', 'run_tests'), (249, 'run_tests', 'lint_or_typecheck')]
- Prediction distribution: {'apply_patch': 1685, 'ask_user': 756, 'edit_file': 3662, 'glob_pattern': 1476, 'grep_search': 2895, 'lint_or_typecheck': 889, 'list_directory': 2195, 'plan_task': 833, 'read_file': 3022, 'respond_only': 1725, 'run_bash': 1548, 'run_tests': 1526, 'web_search': 617, 'write_file': 504}
- Runtime or package-size concerns: runtime=2418.6s, tokenize=2.3s, train=2370.1s, eval=34.1s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260701_235001_gpu_transformer_session_oof_current_v1_len192_fold1-of3_S1-2-OOF-fold1_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260702_003021_gpu_transformer_session_oof_current_v1_len192_fold2-of3_S1-2-OOF-fold2

- Date/time: 2026-07-02 00:30:21 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=none, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=2/3
- Raw Macro-F1: 0.718151
- Overall Macro-F1: 0.718151
- Per-class observations:
  - Weakest: list_directory=0.465, read_file=0.532, web_search=0.568, lint_or_typecheck=0.587, grep_search=0.620
  - Strongest: respond_only=0.999, write_file=0.986, edit_file=0.960, apply_patch=0.925, run_bash=0.777
- Top confusions: [(763, 'grep_search', 'read_file'), (655, 'read_file', 'grep_search'), (570, 'read_file', 'list_directory'), (402, 'grep_search', 'list_directory'), (394, 'list_directory', 'read_file'), (292, 'run_bash', 'run_tests'), (291, 'glob_pattern', 'read_file'), (229, 'glob_pattern', 'grep_search')]
- Prediction distribution: {'apply_patch': 1739, 'ask_user': 770, 'edit_file': 3611, 'glob_pattern': 1404, 'grep_search': 3026, 'lint_or_typecheck': 843, 'list_directory': 2005, 'plan_task': 847, 'read_file': 3127, 'respond_only': 1731, 'run_bash': 1494, 'run_tests': 1630, 'web_search': 610, 'write_file': 496}
- Runtime or package-size concerns: runtime=2416.1s, tokenize=2.3s, train=2369.4s, eval=34.1s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260702_003021_gpu_transformer_session_oof_current_v1_len192_fold2-of3_S1-2-OOF-fold2_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## oof_focal_len192_ep5_s1

- Date/time: 2026-07-02 00:32:51 UTC
- Validation setup: 3-fold session-aware OOF aggregate
- Fold logits: ['experiments/logits/20260701_230939_gpu_transformer_session_oof_current_v1_len192_fold0-of3_S1-2-OOF-fold0_val_logits.pt', 'experiments/logits/20260701_235001_gpu_transformer_session_oof_current_v1_len192_fold1-of3_S1-2-OOF-fold1_val_logits.pt', 'experiments/logits/20260702_003021_gpu_transformer_session_oof_current_v1_len192_fold2-of3_S1-2-OOF-fold2_val_logits.pt']
- Raw OOF Macro-F1: 0.718657
- Tuned OOF Macro-F1: 0.722769
- Weakest classes: list_directory=0.472, read_file=0.553, web_search=0.568, lint_or_typecheck=0.591, grep_search=0.598
- Top confusions: [(2878, 'grep_search', 'read_file'), (1943, 'read_file', 'list_directory'), (1407, 'grep_search', 'list_directory'), (1348, 'read_file', 'grep_search'), (1113, 'list_directory', 'read_file'), (1048, 'glob_pattern', 'read_file'), (851, 'run_bash', 'run_tests'), (757, 'ask_user', 'plan_task')]
- Prediction distribution: {'apply_patch': 4771, 'ask_user': 2094, 'edit_file': 11304, 'glob_pattern': 3854, 'grep_search': 7548, 'lint_or_typecheck': 2138, 'list_directory': 6728, 'plan_task': 2919, 'read_file': 10586, 'respond_only': 5181, 'run_bash': 4744, 'run_tests': 5033, 'web_search': 1620, 'write_file': 1480}
- Decision: S1-2 OOF aggregate: focal len192 5ep finalist
## oof_rules_focal_s1

- Date/time: 2026-07-02 00:42:42 UTC
- Validation setup: 3-fold session-aware OOF aggregate with deterministic sample/logit rule boosts.
- Baseline OOF Macro-F1: 0.722769
- Boosted OOF Macro-F1: 0.734126
- Selected rules: 12
- Weakest classes: list_directory:0.4717, read_file:0.5528, grep_search:0.5982, lint_or_typecheck:0.6119, web_search:0.6396
- Top confusions: [(2878, 'grep_search', 'read_file'), (1943, 'read_file', 'list_directory'), (1407, 'grep_search', 'list_directory'), (1349, 'read_file', 'grep_search'), (1113, 'list_directory', 'read_file'), (1048, 'glob_pattern', 'read_file'), (745, 'ask_user', 'plan_task'), (735, 'glob_pattern', 'list_directory')]
- Rule artifact: experiments/artifacts/oof_rules_focal_s1_rule_boosts.json
- Decision: S1-3 rule boosts on focal OOF
## 20260702_013201_gpu_transformer_session_current_v1_len192_S1-4-final-focal-len192-5ep-full-70k-refit

- Date/time: 2026-07-02 03:44:53 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=none, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.730498
- Overall Macro-F1: 0.735537
- Per-class observations:
  - Weakest: list_directory=0.475, read_file=0.562, web_search=0.596, grep_search=0.596, ask_user=0.612
  - Strongest: respond_only=0.999, write_file=0.993, edit_file=0.968, apply_patch=0.933, run_bash=0.819
- Top confusions: [(532, 'grep_search', 'read_file'), (370, 'read_file', 'list_directory'), (277, 'grep_search', 'list_directory'), (256, 'read_file', 'grep_search'), (222, 'list_directory', 'read_file'), (194, 'glob_pattern', 'read_file'), (180, 'ask_user', 'plan_task'), (147, 'glob_pattern', 'list_directory')]
- Prediction distribution: {'apply_patch': 922, 'ask_user': 373, 'edit_file': 2264, 'glob_pattern': 885, 'grep_search': 1473, 'lint_or_typecheck': 442, 'list_directory': 1328, 'plan_task': 652, 'read_file': 2058, 'respond_only': 1051, 'run_bash': 1034, 'run_tests': 891, 'web_search': 325, 'write_file': 303}
- Runtime or package-size concerns: runtime=2911.5s, tokenize=2.3s, train=2844.1s, eval=20.4s, artifact_size_mb=1077.0.
- Validation logits: experiments/logits/20260702_013201_gpu_transformer_session_current_v1_len192_S1-4-final-focal-len192-5ep-full-70k-refit_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260702_044710_gpu_transformer_session_current_v1_len192_replay-last1_S1-4b-final-focal-replay-len192-5ep-full-70k-ref

- Date/time: 2026-07-02 07:33:01 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.724546
- Overall Macro-F1: 0.731747
- Per-class observations:
  - Weakest: list_directory=0.474, read_file=0.557, web_search=0.560, grep_search=0.596, lint_or_typecheck=0.614
  - Strongest: respond_only=1.000, write_file=0.997, edit_file=0.969, apply_patch=0.940, run_bash=0.815
- Top confusions: [(535, 'grep_search', 'read_file'), (397, 'read_file', 'list_directory'), (294, 'grep_search', 'list_directory'), (246, 'read_file', 'grep_search'), (224, 'list_directory', 'read_file'), (209, 'glob_pattern', 'read_file'), (158, 'glob_pattern', 'list_directory'), (132, 'plan_task', 'ask_user')]
- Prediction distribution: {'apply_patch': 955, 'ask_user': 578, 'edit_file': 2224, 'glob_pattern': 832, 'grep_search': 1435, 'lint_or_typecheck': 458, 'list_directory': 1395, 'plan_task': 499, 'read_file': 2080, 'respond_only': 1051, 'run_bash': 1001, 'run_tests': 908, 'web_search': 280, 'write_file': 305}
- Runtime or package-size concerns: runtime=3433.0s, tokenize=2.1s, train=3353.8s, eval=20.5s, artifact_size_mb=1077.0.
- Validation logits: experiments/logits/20260702_044710_gpu_transformer_session_current_v1_len192_replay-last1_S1-4b-final-focal-replay-len192-5ep-full-70k-ref_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## sparse_text_current_v1_dev

- Date/time: 2026-07-02 07:40:04 UTC
- Validation setup: fold-aware TF-IDF LinearSVC OOF scores ensembled with current finalist transformer logits.
- Base Macro-F1: 0.722769
- Sparse-only Macro-F1: 0.536345
- Best sparse weight: 1.000
- Best Macro-F1: 0.729494
- Weakest classes: list_directory:0.4820, read_file:0.5573, grep_search:0.5994, web_search:0.5996, lint_or_typecheck:0.6105
- Top confusions: [(2723, 'grep_search', 'read_file'), (1894, 'read_file', 'list_directory'), (1467, 'grep_search', 'list_directory'), (1422, 'read_file', 'grep_search'), (1011, 'list_directory', 'read_file'), (988, 'glob_pattern', 'read_file'), (786, 'ask_user', 'plan_task'), (766, 'glob_pattern', 'list_directory')]
- Artifact: experiments/artifacts/sparse_text_current_v1_dev_sparse_svc.json
- Sparse logits: experiments/logits/sparse_text_current_v1_dev_sparse_oof_logits.pt
- Decision: sparse text A/B: current_v1 on no-replay OOF
## sparse_text_state_v2_dev

- Date/time: 2026-07-02 07:42:44 UTC
- Validation setup: fold-aware TF-IDF LinearSVC OOF scores ensembled with current finalist transformer logits.
- Base Macro-F1: 0.722769
- Sparse-only Macro-F1: 0.541921
- Best sparse weight: 1.000
- Best Macro-F1: 0.726900
- Weakest classes: list_directory:0.4750, read_file:0.5475, web_search:0.5913, grep_search:0.5997, lint_or_typecheck:0.6075
- Top confusions: [(2702, 'grep_search', 'read_file'), (2012, 'read_file', 'list_directory'), (1474, 'grep_search', 'list_directory'), (1441, 'read_file', 'grep_search'), (1028, 'list_directory', 'read_file'), (994, 'glob_pattern', 'read_file'), (780, 'ask_user', 'plan_task'), (767, 'run_bash', 'run_tests')]
- Artifact: experiments/artifacts/sparse_text_state_v2_dev_sparse_svc.json
- Sparse logits: experiments/logits/sparse_text_state_v2_dev_sparse_oof_logits.pt
- Decision: sparse text A/B: state_v2 on no-replay OOF
## 20260702_082600_gpu_transformer_session_oof_current_v1_len192_fold0-of3_replay-last1_P1-OOF-fold0-focal-replay-len192-5ep-best-epoch

- Date/time: 2026-07-02 08:26:00 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=0/3
- Raw Macro-F1: 0.723298
- Overall Macro-F1: 0.723298
- Per-class observations:
  - Weakest: list_directory=0.456, read_file=0.549, web_search=0.582, ask_user=0.599, grep_search=0.609
  - Strongest: respond_only=0.999, write_file=0.993, edit_file=0.967, apply_patch=0.936, run_bash=0.786
- Top confusions: [(764, 'grep_search', 'read_file'), (644, 'read_file', 'grep_search'), (543, 'read_file', 'list_directory'), (421, 'grep_search', 'list_directory'), (349, 'list_directory', 'read_file'), (300, 'glob_pattern', 'read_file'), (265, 'ask_user', 'plan_task'), (265, 'run_bash', 'run_tests')]
- Prediction distribution: {'apply_patch': 1686, 'ask_user': 738, 'edit_file': 3665, 'glob_pattern': 1414, 'grep_search': 3053, 'lint_or_typecheck': 830, 'list_directory': 1976, 'plan_task': 939, 'read_file': 3137, 'respond_only': 1729, 'run_bash': 1528, 'run_tests': 1605, 'web_search': 538, 'write_file': 496}
- Runtime or package-size concerns: runtime=3110.3s, tokenize=12.5s, train=3052.2s, eval=34.5s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260702_082600_gpu_transformer_session_oof_current_v1_len192_fold0-of3_replay-last1_P1-OOF-fold0-focal-replay-len192-5ep-best-epoch_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260702_091750_gpu_transformer_session_oof_current_v1_len192_fold1-of3_replay-last1_P1-OOF-fold1

- Date/time: 2026-07-02 09:17:51 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=1/3
- Raw Macro-F1: 0.731395
- Overall Macro-F1: 0.731395
- Per-class observations:
  - Weakest: list_directory=0.436, read_file=0.552, grep_search=0.595, ask_user=0.618, lint_or_typecheck=0.638
  - Strongest: respond_only=0.999, write_file=0.986, edit_file=0.963, apply_patch=0.924, run_bash=0.787
- Top confusions: [(959, 'grep_search', 'read_file'), (538, 'read_file', 'grep_search'), (467, 'read_file', 'list_directory'), (454, 'list_directory', 'read_file'), (345, 'grep_search', 'list_directory'), (335, 'glob_pattern', 'read_file'), (229, 'list_directory', 'grep_search'), (224, 'ask_user', 'plan_task')]
- Prediction distribution: {'apply_patch': 1749, 'ask_user': 813, 'edit_file': 3598, 'glob_pattern': 1520, 'grep_search': 2751, 'lint_or_typecheck': 893, 'list_directory': 1674, 'plan_task': 849, 'read_file': 3618, 'respond_only': 1726, 'run_bash': 1543, 'run_tests': 1537, 'web_search': 564, 'write_file': 498}
- Runtime or package-size concerns: runtime=3106.7s, tokenize=11.0s, train=3049.6s, eval=34.2s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260702_091750_gpu_transformer_session_oof_current_v1_len192_fold1-of3_replay-last1_P1-OOF-fold1_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260702_101002_gpu_transformer_session_oof_current_v1_len192_fold2-of3_replay-last1_P1-OOF-fold2

- Date/time: 2026-07-02 10:10:02 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=2/3
- Raw Macro-F1: 0.733906
- Overall Macro-F1: 0.733906
- Per-class observations:
  - Weakest: list_directory=0.440, read_file=0.539, grep_search=0.609, ask_user=0.611, glob_pattern=0.631
  - Strongest: respond_only=1.000, write_file=0.992, edit_file=0.968, apply_patch=0.936, run_bash=0.793
- Top confusions: [(845, 'grep_search', 'read_file'), (624, 'read_file', 'grep_search'), (530, 'read_file', 'list_directory'), (474, 'list_directory', 'read_file'), (377, 'grep_search', 'list_directory'), (318, 'glob_pattern', 'read_file'), (285, 'ask_user', 'plan_task'), (232, 'run_bash', 'run_tests')]
- Prediction distribution: {'apply_patch': 1694, 'ask_user': 749, 'edit_file': 3634, 'glob_pattern': 1385, 'grep_search': 2934, 'lint_or_typecheck': 866, 'list_directory': 1839, 'plan_task': 953, 'read_file': 3413, 'respond_only': 1727, 'run_bash': 1555, 'run_tests': 1554, 'web_search': 530, 'write_file': 500}
- Runtime or package-size concerns: runtime=3112.0s, tokenize=11.0s, train=3056.2s, eval=34.4s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260702_101002_gpu_transformer_session_oof_current_v1_len192_fold2-of3_replay-last1_P1-OOF-fold2_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## oof_focal_replay_p1

- Date/time: 2026-07-02 10:12:31 UTC
- Validation setup: 3-fold session-aware OOF aggregate
- Fold logits: ['experiments/logits/20260702_082600_gpu_transformer_session_oof_current_v1_len192_fold0-of3_replay-last1_P1-OOF-fold0-focal-replay-len192-5ep-best-epoch_val_logits.pt', 'experiments/logits/20260702_091750_gpu_transformer_session_oof_current_v1_len192_fold1-of3_replay-last1_P1-OOF-fold1_val_logits.pt', 'experiments/logits/20260702_101002_gpu_transformer_session_oof_current_v1_len192_fold2-of3_replay-last1_P1-OOF-fold2_val_logits.pt']
- Raw OOF Macro-F1: 0.729590
- Tuned OOF Macro-F1: 0.735002
- Weakest classes: list_directory=0.454, read_file=0.562, grep_search=0.596, ask_user=0.619, lint_or_typecheck=0.628
- Top confusions: [(3010, 'grep_search', 'read_file'), (1754, 'read_file', 'list_directory'), (1301, 'list_directory', 'read_file'), (1297, 'grep_search', 'list_directory'), (1249, 'read_file', 'grep_search'), (1105, 'glob_pattern', 'read_file'), (860, 'ask_user', 'plan_task'), (778, 'run_bash', 'run_tests')]
- Prediction distribution: {'apply_patch': 4864, 'ask_user': 2388, 'edit_file': 11187, 'glob_pattern': 3959, 'grep_search': 7368, 'lint_or_typecheck': 2005, 'list_directory': 6140, 'plan_task': 3024, 'read_file': 11226, 'respond_only': 5178, 'run_bash': 4804, 'run_tests': 5120, 'web_search': 1260, 'write_file': 1477}
- Decision: P1 aggregate: focal+replay best-epoch 3-fold
## oof_rules_p1

- Date/time: 2026-07-02 10:23:02 UTC
- Validation setup: 3-fold session-aware OOF aggregate with deterministic sample/logit rule boosts.
- Baseline OOF Macro-F1: 0.735002
- Boosted OOF Macro-F1: 0.741554
- Selected rules: 12
- Weakest classes: list_directory:0.4545, read_file:0.5615, grep_search:0.5962, lint_or_typecheck:0.6346, glob_pattern:0.6420
- Top confusions: [(3010, 'grep_search', 'read_file'), (1753, 'read_file', 'list_directory'), (1301, 'list_directory', 'read_file'), (1296, 'grep_search', 'list_directory'), (1249, 'read_file', 'grep_search'), (1105, 'glob_pattern', 'read_file'), (738, 'ask_user', 'plan_task'), (677, 'glob_pattern', 'list_directory')]
- Rule artifact: experiments/artifacts/oof_rules_p1_rule_boosts.json
- Decision: P2 rules on P1 focal+replay OOF
## sparse_p1

- Date/time: 2026-07-02 10:48:20 UTC
- Validation setup: fold-aware TF-IDF LinearSVC OOF scores ensembled with current finalist transformer logits.
- Base Macro-F1: 0.741554
- Sparse-only Macro-F1: 0.536345
- Best sparse weight: 1.000
- Best Macro-F1: 0.743218
- Weakest classes: list_directory:0.4705, read_file:0.5588, grep_search:0.6065, lint_or_typecheck:0.6342, ask_user:0.6402
- Top confusions: [(2653, 'grep_search', 'read_file'), (1689, 'read_file', 'grep_search'), (1581, 'read_file', 'list_directory'), (1228, 'grep_search', 'list_directory'), (1175, 'list_directory', 'read_file'), (1013, 'glob_pattern', 'read_file'), (795, 'ask_user', 'plan_task'), (658, 'glob_pattern', 'list_directory')]
- Artifact: experiments/artifacts/sparse_p1_sparse_svc.json
- Sparse logits: experiments/logits/sparse_p1_sparse_oof_logits.pt
- Decision: P2 sparse SVC on P1 OOF + rules
## 20260702_111646_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed42-splitfix42

- Date/time: 2026-07-02 11:16:54 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.736998
- Overall Macro-F1: 0.746271
- Per-class observations:
  - Weakest: list_directory=0.454, read_file=0.560, grep_search=0.600, glob_pattern=0.621, ask_user=0.655
  - Strongest: respond_only=1.000, write_file=0.995, edit_file=0.970, apply_patch=0.938, run_bash=0.816
- Top confusions: [(544, 'grep_search', 'read_file'), (332, 'read_file', 'grep_search'), (293, 'read_file', 'list_directory'), (284, 'list_directory', 'read_file'), (249, 'grep_search', 'list_directory'), (226, 'glob_pattern', 'read_file'), (176, 'ask_user', 'plan_task'), (133, 'glob_pattern', 'grep_search')]
- Prediction distribution: {'apply_patch': 943, 'ask_user': 477, 'edit_file': 2231, 'glob_pattern': 740, 'grep_search': 1664, 'lint_or_typecheck': 377, 'list_directory': 1140, 'plan_task': 644, 'read_file': 2211, 'respond_only': 1050, 'run_bash': 994, 'run_tests': 1001, 'web_search': 225, 'write_file': 304}
- Runtime or package-size concerns: runtime=3537.1s, tokenize=3.3s, train=3465.9s, eval=20.6s, artifact_size_mb=1077.0.
- Validation logits: experiments/logits/20260702_111646_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed42-splitfix42_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260702_121611_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed43-splitfix42

- Date/time: 2026-07-02 12:16:20 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.740143
- Overall Macro-F1: 0.743936
- Per-class observations:
  - Weakest: list_directory=0.473, read_file=0.559, grep_search=0.604, glob_pattern=0.625, lint_or_typecheck=0.642
  - Strongest: respond_only=0.999, write_file=0.987, edit_file=0.969, apply_patch=0.938, run_bash=0.817
- Top confusions: [(495, 'grep_search', 'read_file'), (339, 'read_file', 'list_directory'), (324, 'read_file', 'grep_search'), (267, 'grep_search', 'list_directory'), (235, 'list_directory', 'read_file'), (204, 'glob_pattern', 'read_file'), (143, 'plan_task', 'ask_user'), (143, 'glob_pattern', 'list_directory')]
- Prediction distribution: {'apply_patch': 933, 'ask_user': 558, 'edit_file': 2267, 'glob_pattern': 794, 'grep_search': 1643, 'lint_or_typecheck': 453, 'list_directory': 1268, 'plan_task': 514, 'read_file': 2032, 'respond_only': 1053, 'run_bash': 1002, 'run_tests': 903, 'web_search': 280, 'write_file': 301}
- Runtime or package-size concerns: runtime=3542.5s, tokenize=2.6s, train=3465.9s, eval=20.6s, artifact_size_mb=1077.0.
- Validation logits: experiments/logits/20260702_121611_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed43-splitfix42_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260702_131530_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed44-splitfix42

- Date/time: 2026-07-02 13:15:38 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.725632
- Overall Macro-F1: 0.732493
- Per-class observations:
  - Weakest: list_directory=0.480, web_search=0.542, read_file=0.558, grep_search=0.601, glob_pattern=0.619
  - Strongest: respond_only=1.000, write_file=0.995, edit_file=0.971, apply_patch=0.941, run_bash=0.820
- Top confusions: [(550, 'grep_search', 'read_file'), (405, 'read_file', 'list_directory'), (302, 'grep_search', 'list_directory'), (246, 'read_file', 'grep_search'), (236, 'glob_pattern', 'read_file'), (211, 'list_directory', 'read_file'), (165, 'glob_pattern', 'list_directory'), (128, 'ask_user', 'plan_task')]
- Prediction distribution: {'apply_patch': 920, 'ask_user': 464, 'edit_file': 2264, 'glob_pattern': 733, 'grep_search': 1444, 'lint_or_typecheck': 405, 'list_directory': 1438, 'plan_task': 556, 'read_file': 2126, 'respond_only': 1051, 'run_bash': 1002, 'run_tests': 957, 'web_search': 335, 'write_file': 306}
- Runtime or package-size concerns: runtime=3533.0s, tokenize=2.5s, train=3467.9s, eval=20.5s, artifact_size_mb=1077.0.
- Validation logits: experiments/logits/20260702_131530_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed44-splitfix42_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260702_141914_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed43b-init42

- Date/time: 2026-07-02 14:19:23 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.718760
- Overall Macro-F1: 0.724926
- Per-class observations:
  - Weakest: list_directory=0.464, web_search=0.548, read_file=0.559, grep_search=0.601, lint_or_typecheck=0.617
  - Strongest: respond_only=0.998, write_file=0.985, edit_file=0.970, apply_patch=0.941, run_bash=0.790
- Top confusions: [(548, 'grep_search', 'read_file'), (326, 'read_file', 'list_directory'), (268, 'list_directory', 'read_file'), (263, 'read_file', 'grep_search'), (244, 'grep_search', 'list_directory'), (227, 'glob_pattern', 'read_file'), (155, 'run_bash', 'run_tests'), (132, 'read_file', 'glob_pattern')]
- Prediction distribution: {'apply_patch': 940, 'ask_user': 459, 'edit_file': 2258, 'glob_pattern': 876, 'grep_search': 1496, 'lint_or_typecheck': 489, 'list_directory': 1174, 'plan_task': 525, 'read_file': 2190, 'respond_only': 1055, 'run_bash': 951, 'run_tests': 930, 'web_search': 358, 'write_file': 300}
- Runtime or package-size concerns: runtime=3529.1s, tokenize=2.5s, train=3451.2s, eval=20.6s, artifact_size_mb=1077.0.
- Validation logits: experiments/logits/20260702_141914_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed43b-init42_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260702_151840_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed44b-init42

- Date/time: 2026-07-02 15:18:48 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session
- Raw Macro-F1: 0.740544
- Overall Macro-F1: 0.745077
- Per-class observations:
  - Weakest: list_directory=0.514, web_search=0.584, grep_search=0.611, read_file=0.612, glob_pattern=0.631
  - Strongest: respond_only=1.000, write_file=0.993, edit_file=0.980, apply_patch=0.965, run_bash=0.802
- Top confusions: [(568, 'grep_search', 'read_file'), (328, 'read_file', 'list_directory'), (287, 'grep_search', 'list_directory'), (223, 'glob_pattern', 'read_file'), (205, 'list_directory', 'read_file'), (172, 'run_bash', 'run_tests'), (161, 'glob_pattern', 'list_directory'), (154, 'read_file', 'grep_search')]
- Prediction distribution: {'apply_patch': 964, 'ask_user': 551, 'edit_file': 2215, 'glob_pattern': 813, 'grep_search': 1291, 'lint_or_typecheck': 373, 'list_directory': 1365, 'plan_task': 449, 'read_file': 2288, 'respond_only': 1050, 'run_bash': 942, 'run_tests': 1044, 'web_search': 347, 'write_file': 309}
- Runtime or package-size concerns: runtime=3539.5s, tokenize=3.5s, train=3457.3s, eval=20.5s, artifact_size_mb=1077.0.
- Validation logits: experiments/logits/20260702_151840_gpu_transformer_session_current_v1_len192_replay-last1_soup-seed44b-init42_val_logits.pt
- Decision: keep as GPU candidate
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260703_044240_gpu_transformer_session_oof_current_v1_len192_fold0-of3_replay-last1_E1-specialist-fold0-explore5

- Date/time: 2026-07-03 04:42:40 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=0/3
- Raw Macro-F1: 0.120348
- Overall Macro-F1: 0.120348
- Per-class observations:
  - Weakest: edit_file=0.000, write_file=0.000, apply_patch=0.000, run_bash=0.000, run_tests=0.000
  - Strongest: glob_pattern=0.429, grep_search=0.352, list_directory=0.349, read_file=0.346, web_search=0.208
- Top confusions: [(1299, 'edit_file', 'read_file'), (1055, 'edit_file', 'grep_search'), (821, 'grep_search', 'read_file'), (815, 'ask_user', 'web_search'), (808, 'apply_patch', 'grep_search'), (807, 'plan_task', 'web_search'), (737, 'read_file', 'list_directory'), (710, 'run_tests', 'grep_search')]
- Prediction distribution: {'glob_pattern': 2969, 'grep_search': 6564, 'list_directory': 3952, 'read_file': 6378, 'web_search': 3471}
- Runtime or package-size concerns: runtime=1253.4s, tokenize=2.4s, train=1205.6s, eval=34.3s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260703_044240_gpu_transformer_session_oof_current_v1_len192_fold0-of3_replay-last1_E1-specialist-fold0-explore5_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260703_050337_gpu_transformer_session_oof_current_v1_len192_fold1-of3_replay-last1_E1-specialist-fold1-explore5

- Date/time: 2026-07-03 05:03:37 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=1/3
- Raw Macro-F1: 0.125881
- Overall Macro-F1: 0.125881
- Per-class observations:
  - Weakest: edit_file=0.000, write_file=0.000, apply_patch=0.000, run_bash=0.000, run_tests=0.000
  - Strongest: glob_pattern=0.438, grep_search=0.361, read_file=0.359, list_directory=0.359, web_search=0.245
- Top confusions: [(1375, 'edit_file', 'read_file'), (1203, 'edit_file', 'grep_search'), (893, 'apply_patch', 'grep_search'), (794, 'grep_search', 'read_file'), (783, 'plan_task', 'web_search'), (780, 'ask_user', 'web_search'), (654, 'run_tests', 'grep_search'), (624, 'edit_file', 'glob_pattern')]
- Prediction distribution: {'glob_pattern': 3069, 'grep_search': 7014, 'list_directory': 3832, 'read_file': 6640, 'web_search': 2778}
- Runtime or package-size concerns: runtime=1251.7s, tokenize=2.3s, train=1204.7s, eval=34.3s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260703_050337_gpu_transformer_session_oof_current_v1_len192_fold1-of3_replay-last1_E1-specialist-fold1-explore5_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## 20260703_054654_gpu_transformer_session_oof_current_v1_len192_fold2-of3_replay-last1_E1-specialist-fold2-explore5

- Date/time: 2026-07-03 05:46:54 UTC
- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.
- Code/config changes: `xlm-roberta-base`, serializer=current_v1, replay=last1, max_length=192, epochs=5, lr=2e-05, batch=16, bucket_multiplier=8.
- Validation setup: session_oof, fold=2/3
- Raw Macro-F1: 0.122665
- Overall Macro-F1: 0.122665
- Per-class observations:
  - Weakest: edit_file=0.000, write_file=0.000, apply_patch=0.000, run_bash=0.000, run_tests=0.000
  - Strongest: glob_pattern=0.419, grep_search=0.359, list_directory=0.353, read_file=0.341, web_search=0.245
- Top confusions: [(1388, 'edit_file', 'read_file'), (1214, 'edit_file', 'grep_search'), (882, 'grep_search', 'read_file'), (787, 'ask_user', 'web_search'), (773, 'plan_task', 'web_search'), (722, 'apply_patch', 'grep_search'), (664, 'edit_file', 'glob_pattern'), (657, 'run_tests', 'grep_search')]
- Prediction distribution: {'glob_pattern': 3169, 'grep_search': 6801, 'list_directory': 3303, 'read_file': 7226, 'web_search': 2834}
- Runtime or package-size concerns: runtime=1268.5s, tokenize=2.3s, train=1221.6s, eval=34.3s, artifact_size_mb=0.0.
- Validation logits: experiments/logits/20260703_054654_gpu_transformer_session_oof_current_v1_len192_fold2-of3_replay-last1_E1-specialist-fold2-explore5_val_logits.pt
- Decision: oof fold complete; aggregate before decision
- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.
## oof_rules_p1_v2

- Date/time: 2026-07-03 06:02:56 UTC
- Validation setup: 3-fold session-aware OOF aggregate with deterministic sample/logit rule boosts.
- Baseline OOF Macro-F1: 0.735002
- Boosted OOF Macro-F1: 0.741554
- Selected rules: 12
- Weakest classes: list_directory:0.4545, read_file:0.5615, grep_search:0.5962, lint_or_typecheck:0.6346, glob_pattern:0.6420
- Top confusions: [(3010, 'grep_search', 'read_file'), (1753, 'read_file', 'list_directory'), (1301, 'list_directory', 'read_file'), (1296, 'grep_search', 'list_directory'), (1249, 'read_file', 'grep_search'), (1105, 'glob_pattern', 'read_file'), (738, 'ask_user', 'plan_task'), (677, 'glob_pattern', 'list_directory')]
- Rule artifact: experiments/artifacts/oof_rules_p1_v2_rule_boosts.json
- Decision: E2: expanded features (test-path, question, last_user)
