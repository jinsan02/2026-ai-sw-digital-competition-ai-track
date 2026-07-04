## 20260701_165055_gpu_transformer_session

- Date/time: 2026-07-01 16:50:55 UTC
- Hypothesis: A multilingual transformer fine-tuned on GPU should recover semantic prompt/action cues that sparse GPU models missed.
- Code/config changes: `xlm-roberta-base`, max_length=192, epochs=5, lr=2e-05, batch=16, loss=ce, fgm=False, ema=False.
- Validation setup: session
- Overall Macro-F1: 0.738857
- Per-class observations:
  - Weakest: list_directory=0.474, read_file=0.561, grep_search=0.595, web_search=0.610, glob_pattern=0.626
  - Strongest: respond_only=1.000, write_file=0.992, edit_file=0.971, apply_patch=0.939, run_bash=0.808
- Top confusions: [(564, 'grep_search', 'read_file'), (360, 'read_file', 'list_directory'), (283, 'grep_search', 'list_directory'), (251, 'read_file', 'grep_search'), (243, 'list_directory', 'read_file'), (216, 'glob_pattern', 'read_file'), (149, 'glob_pattern', 'list_directory'), (148, 'ask_user', 'plan_task')]
- Prediction distribution: {'apply_patch': 932, 'ask_user': 501, 'edit_file': 2257, 'glob_pattern': 790, 'grep_search': 1452, 'lint_or_typecheck': 456, 'list_directory': 1319, 'plan_task': 592, 'read_file': 2166, 'respond_only': 1051, 'run_bash': 1028, 'run_tests': 891, 'web_search': 256, 'write_file': 310}
- Runtime or package-size concerns: GPU inference uses packaged HuggingFace weights; package remains under the 1 GB limit.
- Decision: keep as GPU candidate
- Next suggested experiment: tune max_length/epochs or ensemble with sparse GPU logits if transformer under-recognizes file-operation classes.
## 20260701_174758_gpu_transformer_session

- Date/time: 2026-07-01 17:47:58 UTC
- Hypothesis: A multilingual transformer fine-tuned on GPU should recover semantic prompt/action cues that sparse GPU models missed.
- Code/config changes: `xlm-roberta-base`, max_length=192, epochs=5, lr=2e-05, batch=16, loss=focal, fgm=False, ema=False.
- Validation setup: session
- Overall Macro-F1: 0.744883
- Per-class observations:
  - Weakest: list_directory=0.467, read_file=0.553, grep_search=0.595, glob_pattern=0.625, lint_or_typecheck=0.639
  - Strongest: respond_only=1.000, write_file=0.993, edit_file=0.966, apply_patch=0.930, run_bash=0.814
- Top confusions: [(550, 'grep_search', 'read_file'), (400, 'read_file', 'list_directory'), (296, 'grep_search', 'list_directory'), (267, 'read_file', 'grep_search'), (219, 'list_directory', 'read_file'), (204, 'glob_pattern', 'read_file'), (167, 'glob_pattern', 'list_directory'), (161, 'ask_user', 'plan_task')]
- Prediction distribution: {'apply_patch': 955, 'ask_user': 477, 'edit_file': 2223, 'glob_pattern': 758, 'grep_search': 1504, 'lint_or_typecheck': 419, 'list_directory': 1404, 'plan_task': 608, 'read_file': 2080, 'respond_only': 1052, 'run_bash': 944, 'run_tests': 1000, 'web_search': 268, 'write_file': 309}
- Runtime or package-size concerns: GPU inference uses packaged HuggingFace weights; package remains under the 1 GB limit.
- Decision: keep as GPU candidate
- Next suggested experiment: tune max_length/epochs or ensemble with sparse GPU logits if transformer under-recognizes file-operation classes.
