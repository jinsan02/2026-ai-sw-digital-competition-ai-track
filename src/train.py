"""[v2] TF-IDF + 로지스틱 회귀 — 학습 (src/train.py)

notebooks/[Baseline_Train]_TF-IDF+LogReg...ipynb 를 스크립트로 변환한 baseline(current_prompt만
사용, val Macro-F1 0.4367)에, history/session_meta에서 뽑은 카테고리 피처를 pseudo-token으로
텍스트에 붙이는 v2 피처를 추가한 것 (2026-07-01 ablation: 동일 파이프라인에서 val Macro-F1
0.4367 -> 0.5435, 자세한 내용은 Notion 운영일지 참고).

submit/script.py의 build_input_text()와 반드시 동일한 로직을 유지해야 한다
(학습·추론 입력 전처리가 다르면 모델이 깨짐).

실행 (프로젝트 루트에서):
    python src/train.py
"""
import csv
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

DATA_DIR = "open/data"
MODEL_DIR = "submit/model"
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg.pkl")


def last_action_name(sample):
    for turn in reversed(sample.get("history") or []):
        if turn.get("role") == "assistant_action":
            return turn.get("name")
    return "NONE"


def turn_bucket(turn_index):
    if turn_index == 0:
        return "first"
    if turn_index <= 3:
        return "early"
    if turn_index <= 8:
        return "mid"
    return "late"


def build_input_text(sample):
    """current_prompt + history/session_meta 카테고리 피처(pseudo-token). 학습·추론 공통 로직."""
    meta = sample.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    parts = [
        sample.get("current_prompt") or "",
        "__PREV_" + str(last_action_name(sample)),
        "__CI_" + str(ws.get("last_ci_status")),
        "__DIRTY_" + str(ws.get("git_dirty")),
        "__TURN_" + turn_bucket(meta.get("turn_index", 0)),
        "__OPENFILES_" + str(bool(ws.get("open_files"))),
    ]
    return " ".join(parts)


def main():
    print("Load data...")
    samples = [json.loads(line)
               for line in open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8")
               if line.strip()]
    labels = {row["id"]: row["action"]
              for row in csv.DictReader(open(os.path.join(DATA_DIR, "train_labels.csv"), encoding="utf-8"))}

    X = [build_input_text(s) for s in samples]
    y = [labels[s["id"]] for s in samples]
    print(f" samples={len(X)} classes={len(set(y))}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print(f" train={len(X_train)} val={len(X_val)}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=80_000,
            sublinear_tf=True, lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=500, class_weight="balanced", C=2.0,
        )),
    ])

    print("Fit (train split)...")
    pipe.fit(X_train, y_train)

    val_pred = pipe.predict(X_val)
    macro_f1 = f1_score(y_val, val_pred, labels=ALL_CLASSES, average="macro", zero_division=0)
    print(f"Validation Macro-F1: {macro_f1:.4f}")

    print("Refit on full data...")
    pipe.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH, compress=3)
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
