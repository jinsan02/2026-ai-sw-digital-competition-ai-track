import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from script import (
    ALL_CLASSES,
    load_jsonl,
    safe_text,
    serialize_transformer_sample,
    tokenize_texts_with_terminal,
)
from train import (
    CLASS_TO_ID,
    append_results_csv,
    f1_metrics,
    load_labels,
    predict_with_bias,
    session_id,
    split_indices,
    tune_class_bias,
    tune_class_bias_two_stage,
)

EXPLORER4_CLASSES = [
    "list_directory",
    "read_file",
    "grep_search",
    "glob_pattern",
]
EXPLORER4_CLASS_IDS = [ALL_CLASSES.index(label) for label in EXPLORER4_CLASSES]
EXPLORER4_CLASS_ID_SET = set(EXPLORER4_CLASS_IDS)
WEAK4_CLASSES = ALL_CLASSES[:4]
WEAK4_CLASS_IDS = list(range(4))
WEAK4_CLASS_ID_SET = set(WEAK4_CLASS_IDS)
RELATIONAL_HIDDEN_SCHEMA_VERSION = 1
RELATIONAL_HIDDEN_KIND = "pooled_classifier_hidden_v1"
RELATIONAL_HIDDEN_USAGE_SCOPE = "full_data_refit_teacher_train_rows"
RELATIONAL_HIDDEN_POOLING = "last_nonpadding_classifier_input"
RELATIONAL_MIN_REFERENCE_ARGMAX_AGREEMENT = 0.995
REPLAY_ORIGINAL_ID_RE = re.compile(r"^(?P<session>.+)-step_(?P<step>\d+)$")
REPLAY_SAMPLE_ID_RE = re.compile(
    r"^(?P<source>.+)::replay_(?P<history_index>\d+)_(?P<label>.+)$"
)


def safe_slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return value or "none"


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def class_weights(y, device, power):
    counts = Counter(y)
    total = len(y)
    present = sorted(class_id for class_id, count in counts.items() if count > 0)
    if not present:
        raise ValueError("cannot compute class weights from an empty label set")
    weights = [0.0] * len(ALL_CLASSES)
    for class_id in present:
        weights[class_id] = (total / (len(present) * counts[class_id])) ** power
    mean = sum(weights[class_id] for class_id in present) / len(present)
    for class_id in present:
        weights[class_id] /= mean
    return torch.tensor(weights, dtype=torch.float32, device=device)


def classification_loss_values(
    logits,
    labels,
    weights,
    label_smoothing,
    loss_name,
    focal_gamma,
    class_ids=None,
):
    loss_logits = logits.float()
    if class_ids is not None:
        loss_logits = loss_logits[:, class_ids]
    loss_values = F.cross_entropy(
        loss_logits,
        labels,
        weight=weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    if loss_name == "focal":
        pt = F.log_softmax(loss_logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1).exp()
        loss_values = (1.0 - pt) ** focal_gamma * loss_values
    elif loss_name != "ce":
        raise ValueError(f"unknown loss: {loss_name}")
    return loss_values


def select_balanced_subset(indices, y, size, seed):
    if not size or size >= len(indices):
        return indices
    rng = random.Random(seed)
    by_label = {}
    for idx in indices:
        by_label.setdefault(y[idx], []).append(idx)
    for label_indices in by_label.values():
        rng.shuffle(label_indices)

    selected = []
    labels = list(by_label)
    cursor = 0
    while len(selected) < size and labels:
        label = labels[cursor % len(labels)]
        if by_label[label]:
            selected.append(by_label[label].pop())
        labels = [label_id for label_id in labels if by_label[label_id]]
        cursor += 1
    rng.shuffle(selected)
    return selected


def session_oof_split_indices(samples, y, n_folds, fold_id, seed):
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2 for session_oof")
    if fold_id < 0 or fold_id >= n_folds:
        raise ValueError(f"--fold-id must be in [0, {n_folds - 1}] for session_oof")

    groups = {}
    for idx, sample in enumerate(samples):
        groups.setdefault(session_id(sample.get("id", "")), []).append(idx)

    rng = random.Random(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda group_id: len(groups[group_id]), reverse=True)

    total_label_counts = Counter(y)
    target_label_counts = {
        label_id: count / float(n_folds)
        for label_id, count in total_label_counts.items()
    }
    target_size = len(samples) / float(n_folds)
    fold_label_counts = [Counter() for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]
    fold_groups = [set() for _ in range(n_folds)]

    for group_id in group_ids:
        indices = groups[group_id]
        group_counts = Counter(y[idx] for idx in indices)
        group_size = len(indices)
        best_fold = None
        best_score = None
        for fold in range(n_folds):
            size_before = fold_sizes[fold]
            size_after = fold_sizes[fold] + group_size
            size_before_score = ((size_before - target_size) ** 2) / max(1.0, target_size)
            size_after_score = ((size_after - target_size) ** 2) / max(1.0, target_size)
            label_before_score = 0.0
            label_after_score = 0.0
            for label_id, target in target_label_counts.items():
                before = fold_label_counts[fold][label_id]
                after = fold_label_counts[fold][label_id] + group_counts[label_id]
                label_before_score += ((before - target) ** 2) / max(1.0, target)
                label_after_score += ((after - target) ** 2) / max(1.0, target)
            score = (label_after_score - label_before_score) + 0.2 * (size_after_score - size_before_score)
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold
        fold_groups[best_fold].add(group_id)
        fold_sizes[best_fold] += group_size
        fold_label_counts[best_fold].update(group_counts)

    val_groups = fold_groups[fold_id]
    train_idx = []
    val_idx = []
    for group_id, indices in groups.items():
        if group_id in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def split_for_args(samples, y, args):
    if args.split == "session_oof":
        return session_oof_split_indices(samples, y, args.n_folds, args.fold_id, args.seed)
    return split_indices(samples, y, args.split, args.seed)


def balanced_cap_replay(examples, max_count, seed):
    if not max_count or len(examples) <= max_count:
        return examples
    rng = random.Random(seed)
    by_label = {}
    for label_id, replay_sample in examples:
        by_label.setdefault(label_id, []).append((label_id, replay_sample))
    for label_examples in by_label.values():
        rng.shuffle(label_examples)

    selected = []
    labels = list(by_label)
    cursor = 0
    while len(selected) < max_count and labels:
        label = labels[cursor % len(labels)]
        if by_label[label]:
            selected.append(by_label[label].pop())
        labels = [label_id for label_id in labels if by_label[label_id]]
        cursor += 1
    rng.shuffle(selected)
    return selected


def build_replay_predecessor_index(samples, y, train_idx):
    """Index original training rows by exact ``(session, step)``.

    The index is deliberately restricted to ``train_idx``.  This keeps a
    malformed future split from borrowing validation metadata even though the
    supported session-aware splits normally keep whole sessions together.
    Rows with malformed IDs or time metadata are excluded; duplicate keys are
    fatal because choosing either row would not be fail-closed.
    """

    if len(samples) != len(y):
        raise ValueError("replay predecessor index inputs have inconsistent row counts")
    if len(set(train_idx)) != len(train_idx):
        raise ValueError("replay predecessor index received duplicate train indices")

    index = {}
    audit = Counter()
    for sample_idx in train_idx:
        if sample_idx < 0 or sample_idx >= len(samples):
            raise ValueError(f"replay predecessor train index out of range: {sample_idx}")
        sample = samples[sample_idx]
        audit["index_rows"] += 1
        match = REPLAY_ORIGINAL_ID_RE.fullmatch(safe_text(sample.get("id")))
        if match is None:
            audit["index_invalid_id"] += 1
            continue
        step = int(match.group("step"))
        if step < 1:
            audit["index_invalid_step"] += 1
            continue
        meta = sample.get("session_meta")
        if not isinstance(meta, dict):
            audit["index_invalid_meta"] += 1
            continue
        turn = meta.get("turn_index")
        if isinstance(turn, bool):
            audit["index_turn_mismatch"] += 1
            continue
        try:
            turn = int(turn)
        except (TypeError, ValueError):
            audit["index_turn_mismatch"] += 1
            continue
        if turn != step:
            audit["index_turn_mismatch"] += 1
            continue

        key = (match.group("session"), step)
        if key in index:
            other = index[key]
            raise ValueError(
                "ambiguous replay predecessor key "
                f"{key!r}: indices {other['sample_idx']} and {sample_idx}"
            )
        index[key] = {
            "sample_idx": sample_idx,
            "sample_id": safe_text(sample.get("id")),
            "current_prompt": safe_text(sample.get("current_prompt")),
            "label_id": int(y[sample_idx]),
            "session_meta": meta,
        }
        audit["index_valid_rows"] += 1
    return index, audit


def replay_examples_for_sample(
    sample,
    pair_limit,
    replay_meta_mode="current",
    predecessor_index=None,
    audit=None,
):
    if replay_meta_mode not in ("current", "predecessor"):
        raise ValueError(f"unknown replay metadata mode: {replay_meta_mode}")
    if replay_meta_mode == "predecessor" and predecessor_index is None:
        raise ValueError("predecessor replay metadata mode requires an exact predecessor index")
    if audit is None:
        audit = Counter()

    history = sample.get("history") or []
    candidates = []
    for idx, event in enumerate(history[:-1]):
        if event.get("role") != "user":
            continue
        next_event = history[idx + 1]
        if next_event.get("role") == "assistant_action":
            label = safe_text(next_event.get("name"))
            if label in CLASS_TO_ID:
                candidates.append((idx, event, next_event, label))

    replay_samples = []
    selected = candidates[-pair_limit:]
    audit["source_rows"] += 1
    audit["history_candidates"] += len(candidates)
    audit["tail_candidates"] += len(selected)

    source_match = None
    source_step = None
    history_turns = None
    if replay_meta_mode == "predecessor" and selected:
        source_match = REPLAY_ORIGINAL_ID_RE.fullmatch(safe_text(sample.get("id")))
        if source_match is None:
            audit["dropped_source_id"] += len(selected)
            return replay_samples
        source_step = int(source_match.group("step"))
        if len(history) % 2 or any(
            event.get("role") != ("user" if idx % 2 == 0 else "assistant_action")
            for idx, event in enumerate(history)
        ):
            audit["dropped_history_shape"] += len(selected)
            return replay_samples
        history_turns = len(history) // 2
        if source_step <= history_turns:
            audit["dropped_history_step_range"] += len(selected)
            return replay_samples

    for idx, user_event, target_event, label in selected:
        session_meta = sample.get("session_meta") or {}
        predecessor = None
        if replay_meta_mode == "predecessor":
            # History contains the immediately preceding turns, capped at six.
            # With alternating user/action events, raw event index ``idx`` maps
            # exactly to this original step within the same session.
            predecessor_step = source_step - history_turns + (idx // 2)
            key = (source_match.group("session"), predecessor_step)
            predecessor = predecessor_index.get(key)
            if predecessor is None:
                audit["dropped_predecessor_missing"] += 1
                continue
            if predecessor["current_prompt"] != safe_text(user_event.get("content")):
                audit["dropped_prompt_mismatch"] += 1
                continue
            if predecessor["label_id"] != CLASS_TO_ID[label]:
                audit["dropped_label_mismatch"] += 1
                continue
            session_meta = copy.deepcopy(predecessor["session_meta"])

        replay_sample = {
            "id": f"{safe_text(sample.get('id'))}::replay_{idx}_{label}",
            "_is_replay": True,
            "session_meta": session_meta,
            "history": history[:idx],
            "current_prompt": safe_text(user_event.get("content")),
            # Training-only target metadata.  Serializers intentionally
            # ignore private keys; only the privileged mode label
            # builder may read this event.
            "_privileged_target_event": {
                "name": label,
                "args": copy.deepcopy(target_event.get("args") or {}),
                "result_summary": safe_text(target_event.get("result_summary")),
            },
        }
        if predecessor is not None:
            replay_sample["_replay_predecessor_id"] = predecessor["sample_id"]
            replay_sample["_replay_predecessor_step"] = predecessor_step
        replay_samples.append(
            (
                CLASS_TO_ID[label],
                replay_sample,
            )
        )
        audit["matched_candidates"] += 1
    return replay_samples


def _ordered_text_sha256(values):
    payload = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def attach_replay_kd_predecessors(
    samples,
    y,
    train_idx,
    replay_examples,
    args,
):
    """Attach exact predecessor IDs to an already selected legacy replay cap.

    This deliberately runs *after* ``balanced_cap_replay``.  It never filters,
    reorders, or rematerializes a replay row, and it does not touch
    ``session_meta``.  The only mutations are private training-only predecessor
    keys which every serializer ignores.  Missing or non-exact links remain in
    the cap and are later kept hard-label-only.
    """

    source = safe_text(getattr(args, "replay_kd_source", "none")) or "none"
    if source == "none":
        args.replay_kd_audit = None
        return replay_examples
    if source != "predecessor":
        raise ValueError(f"unknown replay KD source: {source}")
    if getattr(args, "replay_meta_mode", "current") != "current":
        raise ValueError(
            "trusted predecessor replay KD requires legacy current replay metadata"
        )
    if getattr(args, "serializer", "current_v1") != "current_v1":
        raise ValueError(
            "trusted predecessor replay KD is registered only for serializer current_v1"
        )

    selected_ids_before = [
        safe_text(replay_sample.get("id")) for _, replay_sample in replay_examples
    ]
    selected_texts_before = [
        serialize_transformer_sample(replay_sample, args.serializer)
        for _, replay_sample in replay_examples
    ]
    replay_id_sha256 = _ordered_text_sha256(selected_ids_before)
    replay_text_sha256 = _ordered_text_sha256(selected_texts_before)

    predecessor_index, index_audit = build_replay_predecessor_index(
        samples, y, train_idx
    )
    source_by_id = {}
    for sample_idx in train_idx:
        sample_id = safe_text(samples[sample_idx].get("id"))
        if sample_id in source_by_id:
            raise ValueError(f"duplicate replay source id in training split: {sample_id}")
        source_by_id[sample_id] = samples[sample_idx]

    audit = Counter()
    audit.update(index_audit)
    audit["selected_replay_rows"] = len(replay_examples)
    for label_id, replay_sample in replay_examples:
        replay_id = safe_text(replay_sample.get("id"))
        replay_match = REPLAY_SAMPLE_ID_RE.fullmatch(replay_id)
        if replay_match is None:
            audit["unmatched_replay_id"] += 1
            continue
        source_id = replay_match.group("source")
        source_sample = source_by_id.get(source_id)
        if source_sample is None:
            audit["unmatched_source_id"] += 1
            continue
        # Legacy current-mode replay must retain the source row metadata.  An
        # equality check (rather than identity) also covers an empty metadata
        # dict without weakening the serialized-input invariant below.
        if replay_sample.get("session_meta") != (source_sample.get("session_meta") or {}):
            raise AssertionError(
                "trusted replay KD observed non-legacy replay session_meta"
            )

        source_match = REPLAY_ORIGINAL_ID_RE.fullmatch(source_id)
        if source_match is None:
            audit["unmatched_source_shape"] += 1
            continue
        history = source_sample.get("history") or []
        if len(history) % 2 or any(
            event.get("role") != ("user" if idx % 2 == 0 else "assistant_action")
            for idx, event in enumerate(history)
        ):
            audit["unmatched_history_shape"] += 1
            continue
        history_index = int(replay_match.group("history_index"))
        if history_index < 0 or history_index + 1 >= len(history):
            audit["unmatched_history_index"] += 1
            continue
        user_event = history[history_index]
        target_event = history[history_index + 1]
        label = replay_match.group("label")
        if (
            user_event.get("role") != "user"
            or target_event.get("role") != "assistant_action"
            or safe_text(user_event.get("content"))
            != safe_text(replay_sample.get("current_prompt"))
        ):
            audit["unmatched_replay_surface"] += 1
            continue
        if (
            safe_text(target_event.get("name")) != label
            or CLASS_TO_ID.get(label) != int(label_id)
        ):
            audit["unmatched_replay_label"] += 1
            continue

        source_step = int(source_match.group("step"))
        history_turns = len(history) // 2
        if source_step <= history_turns:
            audit["unmatched_history_step_range"] += 1
            continue
        predecessor_step = source_step - history_turns + (history_index // 2)
        predecessor = predecessor_index.get(
            (source_match.group("session"), predecessor_step)
        )
        if predecessor is None:
            audit["unmatched_predecessor_missing"] += 1
            continue
        if predecessor["current_prompt"] != safe_text(user_event.get("content")):
            audit["unmatched_predecessor_prompt"] += 1
            continue
        if predecessor["label_id"] != int(label_id):
            audit["unmatched_predecessor_label"] += 1
            continue

        replay_sample["_replay_predecessor_id"] = predecessor["sample_id"]
        replay_sample["_replay_predecessor_step"] = predecessor_step
        audit["exact_predecessors"] += 1

    selected_ids_after = [
        safe_text(replay_sample.get("id")) for _, replay_sample in replay_examples
    ]
    selected_texts_after = [
        serialize_transformer_sample(replay_sample, args.serializer)
        for _, replay_sample in replay_examples
    ]
    if selected_ids_after != selected_ids_before:
        raise AssertionError("trusted replay KD changed selected replay IDs or order")
    if selected_texts_after != selected_texts_before:
        raise AssertionError("trusted replay KD changed serialized replay text")

    exact = int(audit["exact_predecessors"])
    unmatched = len(replay_examples) - exact
    if exact <= 0:
        raise ValueError("trusted replay KD found zero exact predecessors")
    expected_exact = int(getattr(args, "replay_kd_expected_predecessors", -1))
    expected_unmatched = int(getattr(args, "replay_kd_expected_unmatched", -1))
    expected_id_sha = safe_text(
        getattr(args, "replay_kd_expected_replay_id_sha256", "")
    ).lower()
    if expected_exact >= 0 and exact != expected_exact:
        raise AssertionError(
            "trusted replay KD predecessor-count mismatch: "
            f"expected={expected_exact} actual={exact}"
        )
    if expected_unmatched >= 0 and unmatched != expected_unmatched:
        raise AssertionError(
            "trusted replay KD unmatched-count mismatch: "
            f"expected={expected_unmatched} actual={unmatched}"
        )
    if expected_id_sha and replay_id_sha256 != expected_id_sha:
        raise AssertionError(
            "trusted replay KD selected replay ID digest mismatch: "
            f"expected={expected_id_sha} actual={replay_id_sha256}"
        )

    args.replay_kd_audit = {
        "source": source,
        "attach_stage": "after_legacy_balanced_cap",
        "selected_replay_rows": len(replay_examples),
        "exact_predecessors": exact,
        "unmatched_predecessors": unmatched,
        "selected_replay_id_sha256": replay_id_sha256,
        "serialized_replay_text_sha256_before": replay_text_sha256,
        "serialized_replay_text_sha256_after": _ordered_text_sha256(
            selected_texts_after
        ),
        "input_and_order_invariant": True,
        **{name: int(count) for name, count in sorted(audit.items())},
    }
    print(
        "trusted replay KD links: "
        f"exact={exact}/{len(replay_examples)} unmatched={unmatched} "
        f"id_sha256={replay_id_sha256} text_sha256={replay_text_sha256}"
    )
    return replay_examples


def add_replay_examples(samples, y, train_idx, args):
    sample_weights = [1.0] * len(samples)
    if args.replay_mode == "none":
        args.replay_predecessor_audit = None
        args.replay_kd_audit = None
        return samples, y, train_idx, sample_weights, 0
    if args.split not in ("session", "session_oof"):
        raise ValueError("Replay augmentation is only enabled for session-aware splits to avoid session leakage.")

    pair_limit = {"last1": 1, "last2": 2}[args.replay_mode]
    replay_meta_mode = getattr(args, "replay_meta_mode", "current")
    audit = Counter()
    predecessor_index = None
    if replay_meta_mode == "predecessor":
        predecessor_index, index_audit = build_replay_predecessor_index(samples, y, train_idx)
        audit.update(index_audit)
    replay_examples = []
    for sample_idx in train_idx:
        replay_examples.extend(
            replay_examples_for_sample(
                samples[sample_idx],
                pair_limit,
                replay_meta_mode=replay_meta_mode,
                predecessor_index=predecessor_index,
                audit=audit,
            )
        )
    pre_cap_count = len(replay_examples)
    replay_examples = balanced_cap_replay(replay_examples, args.max_replay_samples, args.seed + 101)
    replay_examples = attach_replay_kd_predecessors(
        samples, y, train_idx, replay_examples, args
    )
    audit["matched_before_cap"] = pre_cap_count
    audit["selected_after_cap"] = len(replay_examples)

    dropped = sum(
        count for name, count in audit.items() if name.startswith("dropped_")
    )
    if replay_meta_mode == "predecessor":
        if audit["matched_candidates"] != pre_cap_count:
            raise AssertionError(
                "replay predecessor audit mismatch: "
                f"matched={audit['matched_candidates']} examples={pre_cap_count}"
            )
        if audit["tail_candidates"] != pre_cap_count + dropped:
            raise AssertionError(
                "replay predecessor audit does not account for every tail candidate: "
                f"tail={audit['tail_candidates']} matched={pre_cap_count} dropped={dropped}"
            )
        if any(
            int(replay_sample["session_meta"]["turn_index"])
            != int(replay_sample["_replay_predecessor_step"])
            for _, replay_sample in replay_examples
        ):
            raise AssertionError("selected replay metadata turn does not match predecessor step")

    args.replay_predecessor_audit = {
        "mode": replay_meta_mode,
        **{name: int(count) for name, count in sorted(audit.items())},
        "dropped_candidates": int(dropped),
        "drop_policy": (
            "drop_before_balanced_cap_on_missing_or_nonexact_predecessor"
            if replay_meta_mode == "predecessor"
            else "legacy_current_source_metadata"
        ),
    }

    start_idx = len(samples)
    replay_samples = [sample for _, sample in replay_examples]
    replay_y = [label_id for label_id, _ in replay_examples]
    new_samples = samples + replay_samples
    new_y = y + replay_y
    replay_idx = list(range(start_idx, start_idx + len(replay_samples)))
    new_train_idx = train_idx + replay_idx
    sample_weights.extend([args.replay_sample_weight] * len(replay_samples))
    print(
        f"replay mode={args.replay_mode} generated={len(replay_samples)} "
        f"cap={args.max_replay_samples} weight={args.replay_sample_weight} "
        f"meta_mode={replay_meta_mode} matched_before_cap={pre_cap_count} dropped={dropped}"
    )
    if replay_meta_mode == "predecessor":
        print(
            "replay predecessor audit="
            + json.dumps(args.replay_predecessor_audit, ensure_ascii=False, sort_keys=True)
        )
    return new_samples, new_y, new_train_idx, sample_weights, len(replay_samples)


def apply_sim_early_turn_loss_scale(samples, y, train_idx, sample_weights, args):
    """Reweight the complete mixed loss of early original SIM rows.

    The raw multiplier is applied to non-replay ``sess_sim`` rows at turns 1-2,
    then normalized to mean one inside each SIM true class on the actual
    training split. AU rows and replay rows are deliberately untouched.
    """

    scale = float(getattr(args, "sim_early_turn_loss_scale", 1.0))
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("sim early-turn loss scale must be finite and in (0, 1]")
    if len(samples) != len(y) or len(samples) != len(sample_weights):
        raise ValueError("SIM early-turn loss inputs have inconsistent row counts")

    updated = list(sample_weights)
    if scale == 1.0:
        args.sim_early_turn_loss_meta = {
            "enabled": False,
            "raw_scale": 1.0,
            "turn_max": 2,
        }
        return updated

    by_class = {}
    turn_by_index = {}
    for idx in train_idx:
        sample = samples[idx]
        if sample.get("_is_replay"):
            continue
        sample_id = safe_text(sample.get("id"))
        if not sample_id.startswith("sess_sim_"):
            continue
        meta = sample.get("session_meta") or {}
        turn_value = meta.get("turn_index")
        try:
            turn = int(turn_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"SIM early-turn loss requires integer turn_index: id={sample_id!r} "
                f"value={turn_value!r}"
            ) from exc
        label = int(y[idx])
        by_class.setdefault(label, []).append(idx)
        turn_by_index[idx] = turn

    if not by_class:
        raise ValueError("SIM early-turn loss found no original sess_sim training rows")

    per_class = {}
    target_rows = 0
    for label, indices in sorted(by_class.items()):
        early_count = sum(turn_by_index[idx] <= 2 for idx in indices)
        raw_mean = (early_count * scale + len(indices) - early_count) / len(indices)
        if raw_mean <= 0.0 or not math.isfinite(raw_mean):
            raise ValueError(f"invalid SIM early-turn class mean for label={label}: {raw_mean}")
        early_multiplier = scale / raw_mean
        later_multiplier = 1.0 / raw_mean
        for idx in indices:
            multiplier = early_multiplier if turn_by_index[idx] <= 2 else later_multiplier
            updated[idx] *= multiplier
        target_rows += early_count
        per_class[ALL_CLASSES[label]] = {
            "rows": len(indices),
            "early_rows": early_count,
            "raw_mean": raw_mean,
            "early_multiplier": early_multiplier,
            "later_multiplier": later_multiplier,
        }

    meta = {
        "enabled": True,
        "raw_scale": scale,
        "turn_max": 2,
        "normalization": "mean_one_within_sim_true_class",
        "sim_rows": sum(len(indices) for indices in by_class.values()),
        "target_rows": target_rows,
        "per_class": per_class,
    }
    args.sim_early_turn_loss_meta = meta
    print(
        "SIM early-turn mixed-loss scale: "
        f"raw={scale} target={target_rows}/{meta['sim_rows']} "
        "normalization=SIM-true-class-mean1 AU/replay=unchanged"
    )
    return updated


def filter_train_indices(train_idx, y, mode):
    if mode == "none":
        return list(train_idx)
    if mode != "weak4":
        raise ValueError(f"unknown train label filter: {mode}")
    filtered = [idx for idx in train_idx if y[idx] in WEAK4_CLASS_ID_SET]
    if not filtered:
        raise ValueError("--train-label-filter weak4 selected zero training rows")
    counts = Counter(y[idx] for idx in filtered)
    missing = [label for label_id, label in enumerate(WEAK4_CLASSES) if counts[label_id] == 0]
    if missing:
        raise ValueError(f"weak4 training split is missing labels: {missing}")
    print(
        "train label filter=weak4 "
        f"kept={len(filtered)}/{len(train_idx)} "
        f"counts={{{', '.join(f'{WEAK4_CLASSES[i]}:{counts[i]}' for i in WEAK4_CLASS_IDS)}}}"
    )
    return filtered


def specialist_optimizer_groups(model, weight_decay, train_label_filter):
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if train_label_filter != "weak4":
        return [param for _, param in named]
    score_params = [param for name, param in named if ".score." in f".{name}."]
    other_params = [param for name, param in named if ".score." not in f".{name}."]
    if not score_params:
        raise ValueError("weak4 specialist found no trainable score parameters")
    groups = []
    if other_params:
        groups.append({"params": other_params, "weight_decay": weight_decay})
    groups.append({"params": score_params, "weight_decay": 0.0})
    return groups


def validate_specialist_warmstart(args):
    if args.train_label_filter != "weak4":
        return
    if not args.resume_from:
        raise ValueError("Weak4 specialist training requires --resume-from")
    resume_dir = Path(args.resume_from)
    meta_path = resume_dir.parent / "hf_meta.json" if resume_dir.name == "hf_model" else resume_dir / "hf_meta.json"
    if not meta_path.is_file():
        raise ValueError(f"Weak4 warm-start metadata is missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if list(meta.get("classes") or []) != ALL_CLASSES:
        raise ValueError("Weak4 warm-start class order does not match ALL_CLASSES")
    if meta.get("base_model") != args.base_model:
        raise ValueError(
            f"Weak4 warm-start base_model mismatch: {meta.get('base_model')!r} != {args.base_model!r}"
        )
    if not bool(meta.get("saved_fp16", False)):
        raise ValueError("Weak4 warm-start must be a saved fp16 artifact")
    is_final_refit = bool(meta.get("final_refit", False))
    if args.final_only:
        if not is_final_refit:
            raise ValueError("Weak4 final-only training must warm-start from the final refit")
    else:
        if is_final_refit:
            raise ValueError("Weak4 screen must not warm-start from a final refit that saw validation")
        if meta.get("validation_split") != "session":
            raise ValueError("Weak4 screen warm-start must use the fixed session validation split")
    print(
        f"Weak4 warm-start OK: {resume_dir} final_refit={is_final_refit} "
        f"saved_fp16={meta.get('saved_fp16')}"
    )


def assert_validation_anchor(samples, y, train_idx, val_idx, args):
    if not args.assert_val_ids:
        return
    payload = torch_load(args.assert_val_ids)
    anchor_classes = list(payload.get("classes") or [])
    if anchor_classes != ALL_CLASSES:
        raise ValueError(
            f"validation anchor classes mismatch: expected={ALL_CLASSES} actual={anchor_classes}"
        )
    if payload.get("split") != args.split:
        raise ValueError(
            f"validation anchor split mismatch: expected={args.split} actual={payload.get('split')}"
        )
    if int(payload.get("seed", -1)) != int(args.seed):
        raise ValueError(
            f"validation anchor seed mismatch: expected={args.seed} actual={payload.get('seed')}"
        )

    anchor_ids = [safe_text(sample_id) for sample_id in payload.get("ids") or []]
    val_ids = [safe_text(samples[idx].get("id")) for idx in val_idx]
    if len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError("validation anchor contains duplicate ids")
    if len(val_ids) != len(set(val_ids)):
        raise ValueError("computed validation split contains duplicate ids")
    if set(anchor_ids) != set(val_ids):
        missing = sorted(set(anchor_ids) - set(val_ids))[:5]
        extra = sorted(set(val_ids) - set(anchor_ids))[:5]
        raise ValueError(
            "validation ids do not match anchor: "
            f"anchor={len(anchor_ids)} computed={len(val_ids)} missing={missing} extra={extra}"
        )

    train_ids = {safe_text(samples[idx].get("id")) for idx in train_idx}
    overlap = train_ids & set(anchor_ids)
    if overlap:
        raise ValueError(f"training ids overlap anchor validation ids: {sorted(overlap)[:5]}")

    anchor_y = payload.get("y_true")
    if anchor_y is not None:
        anchor_labels = {sample_id: int(label) for sample_id, label in zip(anchor_ids, anchor_y)}
        mismatched = [
            sample_id
            for sample_id, idx in zip(val_ids, val_idx)
            if anchor_labels.get(sample_id) != int(y[idx])
        ]
        if mismatched:
            raise ValueError(f"validation labels do not match anchor for ids: {mismatched[:5]}")
    print(
        f"validation anchor OK: ids={len(val_ids)} train_disjoint={len(train_ids)} "
        f"split={args.split} seed={args.seed}"
    )


def make_batches(indices, batch_size, rng=None, lengths=None, bucket_multiplier=1):
    indices = indices[:]
    if rng is not None:
        rng.shuffle(indices)
    if lengths is None or bucket_multiplier <= 1:
        for start in range(0, len(indices), batch_size):
            yield indices[start:start + batch_size]
        return

    bucket_size = max(batch_size, batch_size * bucket_multiplier)
    for bucket_start in range(0, len(indices), bucket_size):
        bucket = indices[bucket_start:bucket_start + bucket_size]
        bucket.sort(key=lambda idx: lengths[idx], reverse=True)
        for start in range(0, len(bucket), batch_size):
            yield bucket[start:start + batch_size]


def cache_path(args, source_path, sample_count, kind, cache_scope="train"):
    source_path = Path(source_path)
    try:
        stamp = source_path.stat().st_mtime_ns
    except FileNotFoundError:
        stamp = 0
    base_model = safe_slug(args.base_model)
    serializer = safe_slug(args.serializer)
    terminal = ""
    if getattr(args, "terminal_token", ""):
        terminal = f"_terminal-{safe_slug(args.terminal_token)}"
    replay = ""
    if getattr(args, "replay_mode", "none") != "none":
        replay = (
            f"_replay-{safe_slug(args.replay_mode)}-n{args.max_replay_samples}"
            f"-w{safe_slug(args.replay_sample_weight)}-scope-{safe_slug(cache_scope)}"
            f"-seed{args.seed}"
        )
        # Keep the legacy/current cache key byte-compatible.  The corrected
        # predecessor mode must never reuse text/token caches serialized with
        # future/current-row metadata.
        if getattr(args, "replay_meta_mode", "current") != "current":
            replay += f"-meta-{safe_slug(args.replay_meta_mode)}"
        if getattr(args, "split", "") == "session_oof":
            replay += f"-oof{args.fold_id}of{args.n_folds}"
    return (
        Path(args.cache_dir)
        / f"{kind}_{base_model}_{serializer}{terminal}{replay}_{source_path.stem}_n{sample_count}_m{stamp}_len{args.max_length}.pt"
    )


def build_serialized_texts(
    samples, args, source_path, cache_scope="train", tokenizer=None
):
    path = cache_path(args, source_path, len(samples), "texts", cache_scope)
    if not args.no_text_cache and path.exists() and not args.rebuild_cache:
        payload = torch_load(path)
        if payload.get("serializer_name") == args.serializer and len(payload.get("texts", [])) == len(samples):
            print(f"loaded serialized text cache: {path}")
            return payload["texts"], path

    start = time.perf_counter()
    texts = [
        serialize_transformer_sample(
            sample, args.serializer, tokenizer=tokenizer
        )
        for sample in samples
    ]
    elapsed = time.perf_counter() - start
    print(f"serialized texts={len(texts)} serializer={args.serializer} elapsed={elapsed:.2f}s")
    if not args.no_text_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "serializer_name": args.serializer,
                "texts": texts,
                "sample_count": len(samples),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            path,
        )
        print(f"saved serialized text cache: {path}")
    return texts, path if not args.no_text_cache else ""


def tokenize_texts(tokenizer, texts, args, source_path, cache_scope="train"):
    path = cache_path(args, source_path, len(texts), "tokens", cache_scope)
    if not args.no_token_cache and path.exists() and not args.rebuild_cache:
        payload = torch_load(path)
        meta = payload.get("meta", {})
        if (
            meta.get("base_model") == args.base_model
            and meta.get("serializer_name") == args.serializer
            and int(meta.get("max_length", -1)) == args.max_length
            and safe_text(meta.get("terminal_token"))
            == safe_text(getattr(args, "terminal_token", ""))
            and len(payload.get("features", [])) == len(texts)
        ):
            print(f"loaded token cache: {path}")
            return payload["features"], payload["lengths"], path

    start = time.perf_counter()
    features = []
    chunk_size = max(1, args.tokenize_batch_size)
    total_chunks = math.ceil(len(texts) / chunk_size) if texts else 0
    for chunk_no, chunk_start in enumerate(range(0, len(texts), chunk_size), 1):
        chunk_texts = texts[chunk_start:chunk_start + chunk_size]
        encoded = tokenize_texts_with_terminal(
            tokenizer,
            chunk_texts,
            args.max_length,
            safe_text(getattr(args, "terminal_token", "")),
        )
        keys = list(encoded.keys())
        features.extend(
            {key: encoded[key][idx] for key in keys}
            for idx in range(len(chunk_texts))
        )
        if total_chunks > 1 and (chunk_no == 1 or chunk_no == total_chunks or chunk_no % 10 == 0):
            print(f"  tokenized chunk {chunk_no}/{total_chunks} samples={len(features)}")
    lengths = [len(feature["input_ids"]) for feature in features]
    elapsed = time.perf_counter() - start
    print(f"tokenized samples={len(features)} max_length={args.max_length} elapsed={elapsed:.2f}s")
    if not args.no_token_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": features,
                "lengths": lengths,
                "meta": {
                    "base_model": args.base_model,
                    "serializer_name": args.serializer,
                    "max_length": args.max_length,
                    "terminal_token": safe_text(getattr(args, "terminal_token", "")),
                    "tokenize_batch_size": args.tokenize_batch_size,
                    "sample_count": len(texts),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
            },
            path,
        )
        print(f"saved token cache: {path}")
    return features, lengths, path if not args.no_token_cache else ""


def make_encoded_batch(tokenizer, encoded_features, batch_idx, args, device):
    features = [encoded_features[i] for i in batch_idx]
    encoded = tokenizer.pad(
        features,
        padding=True,
        pad_to_multiple_of=args.pad_to_multiple_of if args.pad_to_multiple_of > 1 else None,
        return_tensors="pt",
    )
    return {key: value.to(device, non_blocking=True) for key, value in encoded.items()}


def evaluate(model, tokenizer, encoded_features, lengths, y, indices, args, device):
    model.eval()
    logits_parts = []
    ordered_indices = []
    with torch.inference_mode():
        for batch_idx in make_batches(
            indices,
            args.eval_batch_size,
            lengths=lengths,
            bucket_multiplier=args.eval_bucket_multiplier,
        ):
            encoded = make_encoded_batch(tokenizer, encoded_features, batch_idx, args, device)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.bfloat16 if args.bf16 else torch.float16):
                logits = model(**encoded).logits.float()
            logits_parts.append(logits.detach().cpu())
            ordered_indices.extend(batch_idx)
    logits = torch.cat(logits_parts, dim=0)
    y_true = [y[i] for i in ordered_indices]
    pred = torch.argmax(logits, dim=1).tolist()
    metrics = f1_metrics(y_true, pred)
    return logits, y_true, metrics, ordered_indices


def build_teacher_targets(samples, args):
    """Align an OOF teacher payload (ids + log-prob logits) to `samples` by id.
    Returns (logprobs, mask) CPU tensors or None when --distill-logits is unset.
    Replay pseudo-samples and unmatched ids get mask 0 (pure hard-label loss),
    so OOF teachers stay leak-free by construction.

    The mask is a per-row alpha SCALE, not just 0/1: the loss uses
    alpha_row = --distill-alpha * mask, so with --distill-alpha-weak set,
    teacher-matched original rows whose true label is Weak4 carry
    mask = alpha_weak/alpha (team condalpha semantics: matched Weak4-true rows
    get alpha_weak, other matched rows alpha, replay/unmatched stay 0)."""
    if not getattr(args, "distill_logits", None):
        return None
    payload = torch.load(args.distill_logits, map_location="cpu", weights_only=False)
    teacher_rows = payload["logits"].float()
    by_id = {safe_text(sample_id): row for sample_id, row in zip(payload["ids"], teacher_rows)}
    logprobs = torch.zeros(len(samples), teacher_rows.shape[1])
    mask = torch.zeros(len(samples))
    for i, sample in enumerate(samples):
        row = by_id.get(safe_text(sample.get("id")))
        if row is not None:
            logprobs[i] = row
            mask[i] = 1.0
    matched = int(mask.sum())
    weak_alpha = getattr(args, "distill_alpha_weak", None)
    weak_count = 0
    if weak_alpha is not None:
        labels_by_id = load_labels(Path(args.data_dir) / "train_labels.csv")
        weak_labels = set(ALL_CLASSES[:4])
        scale = float(weak_alpha) / float(args.distill_alpha)
        for i, sample in enumerate(samples):
            if mask[i] > 0 and labels_by_id.get(safe_text(sample.get("id"))) in weak_labels:
                mask[i] = scale
                weak_count += 1
    print(
        f"distill: matched {matched}/{len(samples)} rows from {args.distill_logits} "
        f"(alpha={args.distill_alpha} T={args.distill_temp}"
        + (f" alpha_weak={weak_alpha} weak_rows={weak_count}" if weak_alpha is not None else "")
        + ")"
    )
    return logprobs, mask


def build_relational_teacher_targets(samples, y, args, teacher=None):
    """Load and ID-align a pooled teacher-hidden cache for relational KD.

    The cache must cover every original training row exactly once. Replay rows
    are deliberately absent and receive a false mask. When ordinary logit KD
    is active, both masks must agree exactly so relational KD cannot leak onto
    replay or another unmatched surface.
    """
    artifact_value = safe_text(getattr(args, "relational_teacher_hidden", ""))
    if not artifact_value:
        return None
    path = Path(artifact_value)
    if not path.is_file():
        raise ValueError(f"relational teacher hidden cache is missing: {path}")
    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError("relational teacher hidden cache must be a dict payload")
    if int(payload.get("schema_version", -1)) != RELATIONAL_HIDDEN_SCHEMA_VERSION:
        raise ValueError(
            "unsupported relational hidden schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("kind") != RELATIONAL_HIDDEN_KIND:
        raise ValueError(f"unexpected relational hidden kind: {payload.get('kind')!r}")
    if payload.get("usage_scope") != RELATIONAL_HIDDEN_USAGE_SCOPE:
        raise ValueError(
            "unexpected relational hidden usage_scope: "
            f"{payload.get('usage_scope')!r}"
        )
    if list(payload.get("classes") or []) != ALL_CLASSES:
        raise ValueError("relational teacher hidden class order does not match ALL_CLASSES")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("relational teacher hidden metadata is missing")
    if metadata.get("serializer_name") != args.serializer:
        raise ValueError(
            "relational teacher serializer mismatch: "
            f"cache={metadata.get('serializer_name')!r} student={args.serializer!r}"
        )
    expected_teacher = safe_text(
        getattr(args, "relational_teacher_base_model", "")
    )
    if not expected_teacher:
        raise ValueError("relational KD requires an explicit expected teacher base model")
    if safe_text(metadata.get("base_model")) != expected_teacher:
        raise ValueError(
            "relational teacher base_model mismatch: "
            f"cache={metadata.get('base_model')!r} expected={expected_teacher!r}"
        )
    expected_terminal = safe_text(
        getattr(args, "relational_teacher_terminal_token", "")
    )
    if safe_text(metadata.get("terminal_token")) != expected_terminal:
        raise ValueError(
            "relational teacher terminal token mismatch: "
            f"cache={metadata.get('terminal_token')!r} expected={expected_terminal!r}"
        )
    expected_max_length = int(
        getattr(args, "relational_teacher_max_length", 0) or 0
    )
    if expected_max_length <= 0:
        raise ValueError("relational KD requires an explicit teacher max length")
    if int(metadata.get("max_length", -1)) != expected_max_length:
        raise ValueError(
            "relational teacher max_length mismatch: "
            f"cache={metadata.get('max_length')!r} expected={expected_max_length}"
        )
    if metadata.get("pooling") != RELATIONAL_HIDDEN_POOLING:
        raise ValueError(
            "relational teacher pooling mismatch: "
            f"{metadata.get('pooling')!r}"
        )
    model_sha = safe_text(metadata.get("model_weights_sha256"))
    if not re.fullmatch(r"[0-9a-f]{64}", model_sha):
        raise ValueError("relational teacher cache has no valid model_weights_sha256")
    if metadata.get("prefer_fp16_weights") is not True:
        raise ValueError("relational teacher cache was not exported from preferred fp16 weights")
    model_weight_files = metadata.get("model_weight_files")
    if (
        not isinstance(model_weight_files, list)
        or not model_weight_files
        or any(not safe_text(name) for name in model_weight_files)
        or any(".int8." in name or ".int4." in name for name in model_weight_files)
    ):
        raise ValueError(
            "relational teacher cache has invalid standard-fp16 model weight provenance"
        )
    if safe_text(metadata.get("dtype")) != "fp16":
        raise ValueError("relational teacher cache metadata dtype must be fp16")
    if not safe_text(metadata.get("classifier_head")):
        raise ValueError("relational teacher cache has no classifier_head provenance")

    fidelity = metadata.get("asserted_reference_logits")
    if not isinstance(fidelity, dict):
        raise ValueError("relational teacher cache has no asserted logit fidelity")
    agreement = float(fidelity.get("argmax_agreement", -1.0))
    max_abs = float(fidelity.get("max_abs", math.inf))
    if (
        not math.isfinite(agreement)
        or agreement < RELATIONAL_MIN_REFERENCE_ARGMAX_AGREEMENT
    ):
        raise ValueError(
            "relational teacher logit-fidelity agreement is below "
            f"{RELATIONAL_MIN_REFERENCE_ARGMAX_AGREEMENT}: {agreement}"
        )
    if not math.isfinite(max_abs) or max_abs > 0.25:
        raise ValueError(
            "relational teacher logit-fidelity max_abs exceeds 0.25: "
            f"{max_abs}"
        )
    reference_sha = safe_text(fidelity.get("sha256"))
    if not re.fullmatch(r"[0-9a-f]{64}", reference_sha):
        raise ValueError("relational teacher cache has no valid reference-logit SHA256")
    distill_path = Path(safe_text(getattr(args, "distill_logits", "")))
    if not distill_path.is_file():
        raise ValueError(
            f"relational KD cannot verify missing ordinary teacher logits: {distill_path}"
        )
    actual_reference_sha = _sha256_file(distill_path)
    if actual_reference_sha != reference_sha:
        raise ValueError(
            "relational teacher reference-logit fingerprint mismatch: "
            f"cache={reference_sha!r} current={actual_reference_sha!r}"
        )

    train_path = Path(args.data_dir) / "train.jsonl"
    if not train_path.is_file():
        raise ValueError(f"relational KD cannot verify missing training data: {train_path}")
    data_sha = safe_text(metadata.get("data_sha256"))
    actual_data_sha = _sha256_file(train_path)
    if data_sha != actual_data_sha:
        raise ValueError(
            "relational teacher data fingerprint mismatch: "
            f"cache={data_sha!r} current={actual_data_sha!r}"
        )

    artifact_ids = [safe_text(sample_id) for sample_id in payload.get("ids") or []]
    if not artifact_ids:
        raise ValueError("relational teacher hidden cache has no ids")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("relational teacher hidden cache contains duplicate ids")
    hidden = payload.get("hidden")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
        raise ValueError(
            "relational teacher hidden must be a rank-2 tensor, got "
            f"{getattr(hidden, 'shape', None)}"
        )
    if hidden.dtype != torch.float16:
        raise ValueError(f"relational teacher hidden must be fp16, got {hidden.dtype}")
    if hidden.shape[0] != len(artifact_ids) or hidden.shape[1] < 1:
        raise ValueError(
            "relational teacher hidden shape mismatch: "
            f"ids={len(artifact_ids)} hidden={tuple(hidden.shape)}"
        )
    if int(metadata.get("hidden_size", -1)) != int(hidden.shape[1]):
        raise ValueError(
            "relational teacher hidden_size metadata mismatch: "
            f"meta={metadata.get('hidden_size')!r} tensor={hidden.shape[1]}"
        )
    if int(metadata.get("row_count", -1)) != len(artifact_ids):
        raise ValueError(
            "relational teacher row_count metadata mismatch: "
            f"meta={metadata.get('row_count')!r} ids={len(artifact_ids)}"
        )
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("relational teacher hidden cache contains non-finite values")
    artifact_y = torch.as_tensor(payload.get("y_true"), dtype=torch.long).view(-1)
    if artifact_y.shape[0] != len(artifact_ids):
        raise ValueError(
            "relational teacher y_true row mismatch: "
            f"ids={len(artifact_ids)} y_true={artifact_y.shape[0]}"
        )

    if len(samples) != len(y):
        raise ValueError(f"samples/y length mismatch: {len(samples)} != {len(y)}")
    original_rows = [
        idx for idx, sample in enumerate(samples) if not sample.get("_is_replay", False)
    ]
    replay_rows = [
        idx for idx, sample in enumerate(samples) if sample.get("_is_replay", False)
    ]
    original_ids = [safe_text(samples[idx].get("id")) for idx in original_rows]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("current original samples contain duplicate ids")
    artifact_id_set = set(artifact_ids)
    original_id_set = set(original_ids)
    if artifact_id_set != original_id_set:
        missing = sorted(original_id_set - artifact_id_set)[:5]
        extra = sorted(artifact_id_set - original_id_set)[:5]
        raise ValueError(
            "relational teacher id coverage mismatch: "
            f"cache={len(artifact_ids)} original={len(original_ids)} "
            f"missing={missing} extra={extra}"
        )

    artifact_pos = {sample_id: idx for idx, sample_id in enumerate(artifact_ids)}
    aligned_hidden = torch.zeros(
        (len(samples), hidden.shape[1]), dtype=hidden.dtype
    )
    mask = torch.zeros(len(samples), dtype=torch.bool)
    label_mismatches = []
    for sample_idx in original_rows:
        sample_id = safe_text(samples[sample_idx].get("id"))
        source_idx = artifact_pos[sample_id]
        source_label = int(artifact_y[source_idx])
        if source_label != int(y[sample_idx]):
            label_mismatches.append((sample_id, source_label, int(y[sample_idx])))
            continue
        aligned_hidden[sample_idx] = hidden[source_idx]
        mask[sample_idx] = True
    if label_mismatches:
        raise ValueError(
            "relational teacher labels do not match current training labels: "
            f"{label_mismatches[:5]}"
        )
    if replay_rows and bool(mask[replay_rows].any()):
        raise ValueError("relational teacher mask must exclude every replay row")
    if teacher is not None:
        if not isinstance(teacher, (tuple, list)) or len(teacher) != 2:
            raise ValueError("ordinary teacher targets must be a (logits, mask) pair")
        logit_mask = torch.as_tensor(teacher[1]).view(-1) > 0
        if logit_mask.shape != mask.shape or not torch.equal(logit_mask, mask):
            raise ValueError(
                "relational teacher mask does not exactly match logit-KD coverage"
            )

    meta = {
        "enabled": True,
        "artifact_path": str(path),
        "artifact_sha256": _sha256_file(path),
        "schema_version": RELATIONAL_HIDDEN_SCHEMA_VERSION,
        "kind": RELATIONAL_HIDDEN_KIND,
        "usage_scope": RELATIONAL_HIDDEN_USAGE_SCOPE,
        "teacher_base_model": expected_teacher,
        "serializer_name": args.serializer,
        "teacher_terminal_token": expected_terminal,
        "teacher_max_length": expected_max_length,
        "pooling": RELATIONAL_HIDDEN_POOLING,
        "hidden_size": int(hidden.shape[1]),
        "matched_original_rows": int(mask.sum()),
        "masked_replay_rows": len(replay_rows),
        "weight": float(args.relational_kd_weight),
        "loss": "one_minus_centered_cosine_gram_alignment",
        "model_weights_sha256": model_sha,
        "model_weight_files": list(model_weight_files),
        "reference_logits_sha256": reference_sha,
        "reference_argmax_agreement": agreement,
        "reference_max_abs": max_abs,
        "data_sha256": data_sha,
    }
    args.relational_kd_meta = meta
    print(
        "relational KD cache: "
        f"matched={int(mask.sum())}/{len(samples)} replay_masked={len(replay_rows)} "
        f"hidden={tuple(hidden.shape)} weight={args.relational_kd_weight} "
        f"teacher={expected_teacher} serializer={args.serializer}"
    )
    return {"hidden": aligned_hidden, "mask": mask, "meta": meta}


def parse_consensus_backbone_weights(value, expected_count=None):
    """Parse the raw c=0..N backbone-gradient scales used by the sieve."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("consensus backbone weights are empty")
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                f"invalid consensus backbone weights: {value!r}"
            ) from exc
    else:
        values = [float(item) for item in value]
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            "consensus backbone weights length mismatch: "
            f"expected={expected_count} actual={len(values)}"
        )
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in values):
        raise ValueError("consensus backbone weights must be finite values in [0, 1]")
    if any(left > right for left, right in zip(values, values[1:])):
        raise ValueError("consensus backbone weights must be nondecreasing")
    return values


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_consensus_reliability(samples, y, train_idx, args):
    """Align and normalize an OOF-consensus reliability artifact by row id.

    Original rows must have exact artifact coverage and matching labels. Replay
    rows are intentionally outside the OOF artifact and retain scale 1.0. When
    class normalization is enabled, raw scales are divided by their mean within
    each true class on the current training split. This preserves the baseline
    class-level hard-loss mass while moving its backbone gradient away from
    rows on which all OOF teachers failed.
    """
    artifact_path = safe_text(getattr(args, "consensus_reliability", ""))
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.is_file():
        raise ValueError(f"consensus reliability artifact is missing: {path}")
    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError("consensus reliability artifact must be a dict payload")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(
            "unsupported consensus reliability schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("kind") != "oof_correct_consensus_reliability":
        raise ValueError(f"unexpected consensus reliability kind: {payload.get('kind')!r}")
    usage_scope = payload.get("usage_scope")
    if usage_scope != "full_data_refit_only":
        raise ValueError(
            "unsupported consensus reliability usage_scope: "
            f"{usage_scope!r}"
        )
    if list(payload.get("classes") or []) != ALL_CLASSES:
        raise ValueError("consensus reliability class order does not match ALL_CLASSES")

    artifact_ids = [safe_text(sample_id) for sample_id in payload.get("ids") or []]
    if not artifact_ids:
        raise ValueError("consensus reliability artifact has no ids")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("consensus reliability artifact contains duplicate ids")
    artifact_labels = torch.as_tensor(payload.get("y_true"), dtype=torch.long).view(-1)
    correct_counts = torch.as_tensor(payload.get("correct_counts"), dtype=torch.long).view(-1)
    if len(artifact_ids) != len(artifact_labels) or len(artifact_ids) != len(correct_counts):
        raise ValueError(
            "consensus reliability row length mismatch: "
            f"ids={len(artifact_ids)} y_true={len(artifact_labels)} "
            f"correct_counts={len(correct_counts)}"
        )
    model_count = int(payload.get("model_count", -1))
    if model_count < 1:
        raise ValueError(f"invalid consensus model_count: {model_count}")
    if bool(((correct_counts < 0) | (correct_counts > model_count)).any()):
        raise ValueError(f"correct_counts must lie in [0, {model_count}]")
    configured_weights = safe_text(getattr(args, "consensus_backbone_weights", ""))
    weight_source = configured_weights or payload.get("backbone_weights")
    if weight_source is None:
        raise ValueError(
            "consensus artifact has no backbone_weights and no CLI override was supplied"
        )
    raw_weight_values = parse_consensus_backbone_weights(
        weight_source, expected_count=model_count + 1
    )
    raw_weight_table = torch.tensor(raw_weight_values, dtype=torch.float32)

    if len(samples) != len(y):
        raise ValueError(f"samples/y length mismatch: {len(samples)} != {len(y)}")
    original_rows = [idx for idx, sample in enumerate(samples) if not sample.get("_is_replay", False)]
    replay_rows = [idx for idx, sample in enumerate(samples) if sample.get("_is_replay", False)]
    original_ids = [safe_text(samples[idx].get("id")) for idx in original_rows]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("current original samples contain duplicate ids")
    artifact_id_set = set(artifact_ids)
    original_id_set = set(original_ids)
    if artifact_id_set != original_id_set:
        missing = sorted(original_id_set - artifact_id_set)[:5]
        extra = sorted(artifact_id_set - original_id_set)[:5]
        raise ValueError(
            "consensus reliability id coverage mismatch: "
            f"artifact={len(artifact_ids)} original={len(original_ids)} "
            f"missing={missing} extra={extra}"
        )

    artifact_pos = {sample_id: idx for idx, sample_id in enumerate(artifact_ids)}
    aligned_counts = torch.full((len(samples),), -1, dtype=torch.long)
    raw_scales = torch.ones(len(samples), dtype=torch.float32)
    label_mismatches = []
    for sample_idx in original_rows:
        sample_id = safe_text(samples[sample_idx].get("id"))
        source_idx = artifact_pos[sample_id]
        source_label = int(artifact_labels[source_idx])
        if source_label != int(y[sample_idx]):
            label_mismatches.append((sample_id, source_label, int(y[sample_idx])))
            continue
        count = int(correct_counts[source_idx])
        aligned_counts[sample_idx] = count
        raw_scales[sample_idx] = raw_weight_table[count]
    if label_mismatches:
        raise ValueError(
            "consensus reliability labels do not match current training labels: "
            f"{label_mismatches[:5]}"
        )

    train_set = set(int(idx) for idx in train_idx)
    if any(idx < 0 or idx >= len(samples) for idx in train_set):
        raise ValueError("train_idx contains an out-of-range row")
    heldout_original = [idx for idx in original_rows if idx not in train_set]
    if heldout_original:
        raise ValueError(
            "full-data OOF consensus reliability cannot be used with held-out "
            "validation rows: its source OOF models may have trained on those "
            "labels. Use it only for --final-model --final-only, or build a "
            "nested reliability artifact from the outer training split. "
            f"heldout_original={len(heldout_original)}"
        )
    class_normalize = bool(getattr(args, "consensus_class_normalize", True))
    class_rows = {}
    for class_id, label in enumerate(ALL_CLASSES):
        class_rows[label] = (
            [idx for idx in original_rows if idx in train_set and int(y[idx]) == class_id],
            [idx for idx in replay_rows if idx in train_set and int(y[idx]) == class_id],
            [idx for idx in original_rows if int(y[idx]) == class_id],
        )

    def normalize_scales(raw, branch):
        effective = raw.clone()
        stats = {}
        for label, (
            class_original_train,
            class_replay_train,
            class_all_original,
        ) in class_rows.items():
            if not class_original_train and not class_replay_train:
                continue
            normalization_factor = 1.0
            raw_mean = None
            if class_original_train:
                raw_mean = float(raw[class_original_train].mean())
                if class_normalize:
                    if raw_mean <= 0.0:
                        raise ValueError(
                            f"consensus raw {branch} weights have zero mean "
                            f"for training class {label}"
                        )
                    normalization_factor = 1.0 / raw_mean
                    effective[class_all_original] *= normalization_factor
            stats[label] = {
                "original_train_rows": len(class_original_train),
                "replay_train_rows": len(class_replay_train),
                "raw_mean": raw_mean,
                "normalization_factor": normalization_factor,
                "effective_original_mean": (
                    float(effective[class_original_train].mean())
                    if class_original_train
                    else None
                ),
            }
        # Replay pseudo-targets describe older actions and cannot inherit the
        # current row's OOF correctness. Keep their baseline gradient (the KD
        # branch additionally masks them out entirely via the teacher mask).
        if replay_rows:
            effective[replay_rows] = 1.0
        return effective, stats

    effective_scales, class_stats = normalize_scales(raw_scales, "backbone")

    kd_config = safe_text(getattr(args, "consensus_kd_weights", ""))
    kd_raw_scales = None
    kd_effective_scales = None
    kd_class_stats = None
    kd_weight_values = None
    if kd_config:
        kd_weight_values = parse_consensus_backbone_weights(
            kd_config, expected_count=model_count + 1
        )
        kd_weight_table = torch.tensor(kd_weight_values, dtype=torch.float32)
        kd_raw_scales = torch.ones(len(samples), dtype=torch.float32)
        for sample_idx in original_rows:
            kd_raw_scales[sample_idx] = kd_weight_table[int(aligned_counts[sample_idx])]
        kd_effective_scales, kd_class_stats = normalize_scales(kd_raw_scales, "kd")

    train_original = [idx for idx in original_rows if idx in train_set]
    train_replay = [idx for idx in replay_rows if idx in train_set]
    count_histogram = Counter(int(aligned_counts[idx]) for idx in train_original)
    meta = {
        "enabled": True,
        "artifact_path": str(path),
        "artifact_sha256": _sha256_file(path),
        "artifact_schema_version": 1,
        "usage_scope": usage_scope,
        "model_count": model_count,
        "backbone_weights": raw_weight_values,
        "weights_source": "cli" if configured_weights else "artifact",
        "class_normalize": class_normalize,
        "train_original_rows": len(train_original),
        "train_replay_rows": len(train_replay),
        "correct_count_histogram": {
            str(count): int(count_histogram.get(count, 0))
            for count in range(model_count + 1)
        },
        "raw_train_mean": (
            float(raw_scales[train_original].mean()) if train_original else None
        ),
        "effective_train_mean": (
            float(effective_scales[train_original].mean()) if train_original else None
        ),
        "class_stats": class_stats,
        "kd_weights": kd_weight_values,
        "kd_raw_train_mean": (
            float(kd_raw_scales[train_original].mean())
            if kd_raw_scales is not None and train_original
            else None
        ),
        "kd_effective_train_mean": (
            float(kd_effective_scales[train_original].mean())
            if kd_effective_scales is not None and train_original
            else None
        ),
        "kd_class_stats": kd_class_stats,
        "sources": payload.get("sources") or [],
    }
    args.consensus_reliability_meta = meta
    print(
        "consensus sieve: "
        f"artifact={path} models={model_count} original_train={len(train_original)} "
        f"replay_train={len(train_replay)} weights={raw_weight_values} "
        f"kd_weights={kd_weight_values} "
        f"class_normalize={class_normalize} histogram={meta['correct_count_histogram']}"
    )
    return {
        "gradient_scales": effective_scales,
        "raw_scales": raw_scales,
        "kd_gradient_scales": kd_effective_scales,
        "kd_raw_scales": kd_raw_scales,
        "correct_counts": aligned_counts,
        "meta": meta,
    }


def apply_consensus_distill_alpha(teacher, consensus, args):
    """Raise matched c=0 rows to the configured per-row KD-alpha floor.

    ``build_teacher_targets`` has already encoded the base/Weak4 alpha in its
    mask, while ``build_consensus_reliability`` has already aligned correctness
    counts by sample id.  Combining those two aligned tensors here avoids any
    positional join against either source artifact.  Replay and teacher-
    unmatched rows retain mask zero even if their aligned count is zero.
    """
    alpha_c0 = getattr(args, "distill_alpha_c0", None)
    if alpha_c0 is None:
        return teacher
    if teacher is None:
        raise ValueError("--distill-alpha-c0 requires teacher targets")
    if consensus is None:
        raise ValueError("--distill-alpha-c0 requires consensus reliability")
    if not isinstance(teacher, (tuple, list)) or len(teacher) != 2:
        raise ValueError("teacher targets must be a (logits, mask) pair")
    if not isinstance(consensus, dict) or consensus.get("correct_counts") is None:
        raise ValueError("consensus reliability is missing aligned correct_counts")

    base_alpha = float(args.distill_alpha)
    alpha_c0 = float(alpha_c0)
    if not math.isfinite(base_alpha) or base_alpha <= 0.0:
        raise ValueError("--distill-alpha-c0 requires --distill-alpha > 0")
    if not math.isfinite(alpha_c0) or alpha_c0 < 0.0 or alpha_c0 > 1.0:
        raise ValueError("--distill-alpha-c0 must be a finite value in [0, 1]")

    teacher_logits, teacher_mask = teacher
    if not isinstance(teacher_logits, torch.Tensor) or teacher_logits.ndim < 1:
        raise ValueError("teacher logits must be a tensor with a row dimension")
    if not isinstance(teacher_mask, torch.Tensor) or teacher_mask.ndim != 1:
        raise ValueError("teacher mask must be a one-dimensional tensor")
    correct_counts = consensus["correct_counts"]
    if not isinstance(correct_counts, torch.Tensor) or correct_counts.ndim != 1:
        raise ValueError("aligned consensus correct_counts must be one-dimensional")
    row_count = int(teacher_mask.shape[0])
    if int(teacher_logits.shape[0]) != row_count or int(correct_counts.shape[0]) != row_count:
        raise ValueError(
            "distill/consensus row length mismatch: "
            f"logits={teacher_logits.shape[0]} mask={row_count} "
            f"correct_counts={correct_counts.shape[0]}"
        )
    if not bool(torch.isfinite(teacher_mask).all()) or bool((teacher_mask < 0).any()):
        raise ValueError("teacher mask must contain finite nonnegative values")

    boosted_mask = teacher_mask.clone()
    matched_c0 = (teacher_mask > 0) & (correct_counts == 0)
    current_alpha = base_alpha * boosted_mask[matched_c0]
    target_alpha = torch.full_like(current_alpha, alpha_c0)
    boosted_alpha = torch.maximum(current_alpha, target_alpha)
    boosted_mask[matched_c0] = boosted_alpha / base_alpha
    boosted_rows = int((boosted_alpha > current_alpha).sum())
    matched_c0_rows = int(matched_c0.sum())

    meta = consensus.get("meta")
    if isinstance(meta, dict):
        meta["distill_alpha_c0"] = {
            "target": alpha_c0,
            "matched_c0_rows": matched_c0_rows,
            "boosted_rows": boosted_rows,
        }
        args.consensus_reliability_meta = meta
    print(
        "distill c0 alpha: "
        f"target={alpha_c0} matched_c0_rows={matched_c0_rows} "
        f"boosted_rows={boosted_rows}"
    )
    return teacher_logits, boosted_mask


def apply_consensus_conditioned_weak_alpha(
    teacher, consensus, y, train_idx, args
):
    """Redistribute Weak4 KD alpha by consensus reliability, preserving mass.

    For each true Weak4 class, reliability is ``r = 1 - correct_count / M``
    and alpha is centered around ``--distill-alpha-weak`` so its class mean is
    unchanged.  Because the champion hard-backbone sieve is itself correlated
    with ``correct_count``, the already-normalized hard scale is then multiplied
    by one class-level factor so ``mean((1 - alpha) * scale)`` also stays at the
    champion baseline ``1 - alpha_weak``.  Replay, non-Weak4 rows, raw sieve
    tensors, and the optional KD-branch sieve remain untouched.

    The option is deliberately fail-closed: no clipping/recentering is hidden,
    every original Weak4 row must be teacher-matched, and the incoming hard
    scales must already have class mean one.
    """
    slope = float(getattr(args, "distill_alpha_weak_consensus_lambda", 0.0) or 0.0)
    if slope == 0.0:
        return teacher, consensus
    if not math.isfinite(slope) or slope < 0.0:
        raise ValueError(
            "--distill-alpha-weak-consensus-lambda must be finite and >= 0"
        )
    if getattr(args, "distill_alpha_c0", None) is not None:
        raise ValueError(
            "--distill-alpha-weak-consensus-lambda cannot be combined with "
            "--distill-alpha-c0"
        )
    if teacher is None:
        raise ValueError(
            "--distill-alpha-weak-consensus-lambda requires teacher targets"
        )
    if consensus is None:
        raise ValueError(
            "--distill-alpha-weak-consensus-lambda requires consensus reliability"
        )
    if not bool(getattr(args, "consensus_class_normalize", True)):
        raise ValueError(
            "--distill-alpha-weak-consensus-lambda requires class-normalized "
            "consensus scales"
        )
    if not isinstance(teacher, (tuple, list)) or len(teacher) != 2:
        raise ValueError("teacher targets must be a (logits, mask) pair")
    if not isinstance(consensus, dict):
        raise ValueError("consensus reliability must be a dict")

    base_alpha = float(args.distill_alpha)
    weak_alpha_value = getattr(args, "distill_alpha_weak", None)
    if weak_alpha_value is None:
        raise ValueError(
            "--distill-alpha-weak-consensus-lambda requires --distill-alpha-weak"
        )
    weak_alpha = float(weak_alpha_value)
    if not math.isfinite(base_alpha) or base_alpha <= 0.0:
        raise ValueError(
            "--distill-alpha-weak-consensus-lambda requires --distill-alpha > 0"
        )
    if not math.isfinite(weak_alpha) or not 0.0 < weak_alpha < 1.0:
        raise ValueError(
            "--distill-alpha-weak-consensus-lambda requires a finite "
            "--distill-alpha-weak in (0, 1)"
        )

    teacher_logits, teacher_mask = teacher
    correct_counts = consensus.get("correct_counts")
    gradient_scales = consensus.get("gradient_scales")
    meta = consensus.get("meta")
    if not isinstance(teacher_logits, torch.Tensor) or teacher_logits.ndim < 1:
        raise ValueError("teacher logits must be a tensor with a row dimension")
    if not isinstance(teacher_mask, torch.Tensor) or teacher_mask.ndim != 1:
        raise ValueError("teacher mask must be a one-dimensional tensor")
    if not isinstance(correct_counts, torch.Tensor) or correct_counts.ndim != 1:
        raise ValueError("consensus correct_counts must be one-dimensional")
    if not isinstance(gradient_scales, torch.Tensor) or gradient_scales.ndim != 1:
        raise ValueError("consensus gradient_scales must be one-dimensional")
    if not isinstance(meta, dict):
        raise ValueError("consensus reliability metadata is missing")
    model_count = int(meta.get("model_count", -1))
    if model_count < 1:
        raise ValueError("consensus reliability metadata has invalid model_count")

    row_count = int(teacher_mask.shape[0])
    lengths = {
        "teacher_logits": int(teacher_logits.shape[0]),
        "correct_counts": int(correct_counts.shape[0]),
        "gradient_scales": int(gradient_scales.shape[0]),
        "labels": len(y),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(
            "conditioned-alpha row length mismatch: "
            f"mask={row_count} "
            + " ".join(f"{name}={length}" for name, length in lengths.items())
        )
    if not bool(torch.isfinite(teacher_mask).all()) or bool((teacher_mask < 0).any()):
        raise ValueError("teacher mask must contain finite nonnegative values")
    if not bool(torch.isfinite(gradient_scales).all()) or bool((gradient_scales < 0).any()):
        raise ValueError("consensus gradient scales must be finite and nonnegative")

    train_rows = sorted(set(int(idx) for idx in train_idx))
    if any(idx < 0 or idx >= row_count for idx in train_rows):
        raise ValueError("train_idx contains an out-of-range row")

    updated_mask = teacher_mask.clone()
    updated_scales = gradient_scales.clone()
    class_meta = {}
    touched_rows = 0
    coefficient_target = 1.0 - weak_alpha
    tolerance = 1e-6

    for class_id in WEAK4_CLASS_IDS:
        rows = [
            idx
            for idx in train_rows
            if int(y[idx]) == class_id and int(correct_counts[idx]) >= 0
        ]
        if not rows:
            raise ValueError(
                f"conditioned-alpha found no original training rows for {ALL_CLASSES[class_id]}"
            )
        row_idx = torch.tensor(rows, dtype=torch.long)
        if bool((teacher_mask[row_idx] <= 0).any()):
            missing = int((teacher_mask[row_idx] <= 0).sum())
            raise ValueError(
                "conditioned-alpha requires full teacher coverage for original Weak4 "
                f"rows: class={ALL_CLASSES[class_id]} missing={missing}"
            )
        class_counts = correct_counts[row_idx].to(torch.long)
        if bool(((class_counts < 0) | (class_counts > model_count)).any()):
            raise ValueError(
                f"conditioned-alpha counts out of range for {ALL_CLASSES[class_id]}"
            )
        current_alpha = base_alpha * teacher_mask[row_idx].float()
        expected_alpha = torch.full_like(current_alpha, weak_alpha)
        if not torch.allclose(current_alpha, expected_alpha, rtol=0.0, atol=tolerance):
            raise ValueError(
                "conditioned-alpha expected every original Weak4 row to carry "
                f"alpha={weak_alpha}: class={ALL_CLASSES[class_id]} "
                f"observed_min={float(current_alpha.min()):.9f} "
                f"observed_max={float(current_alpha.max()):.9f}"
            )

        reliability = 1.0 - class_counts.float() / float(model_count)
        reliability_mean = reliability.mean()
        conditioned_alpha = weak_alpha + slope * (reliability - reliability_mean)
        if not bool(torch.isfinite(conditioned_alpha).all()):
            raise ValueError("conditioned-alpha produced non-finite alpha values")
        if bool(((conditioned_alpha < 0.0) | (conditioned_alpha > 1.0)).any()):
            raise ValueError(
                "conditioned-alpha would leave [0, 1]; reduce "
                "--distill-alpha-weak-consensus-lambda"
            )
        if not math.isclose(
            float(conditioned_alpha.mean()), weak_alpha, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(
                f"conditioned-alpha mean drift for {ALL_CLASSES[class_id]}"
            )

        base_scales = gradient_scales[row_idx].float()
        base_scale_mean = float(base_scales.mean())
        if not math.isclose(base_scale_mean, 1.0, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                "conditioned-alpha requires incoming class-normalized scales: "
                f"class={ALL_CLASSES[class_id]} mean={base_scale_mean:.9f}"
            )
        uncompensated_mass = float(((1.0 - conditioned_alpha) * base_scales).mean())
        if not math.isfinite(uncompensated_mass) or uncompensated_mass <= 0.0:
            raise ValueError(
                f"conditioned-alpha has invalid hard-backbone mass for {ALL_CLASSES[class_id]}"
            )
        compensation = coefficient_target / uncompensated_mass
        if not math.isfinite(compensation) or compensation <= 0.0 or compensation > 1.0 + tolerance:
            raise ValueError(
                "conditioned-alpha compensation is outside the pre-registered "
                f"(0, 1] range: class={ALL_CLASSES[class_id]} factor={compensation}"
            )
        compensated_scales = base_scales * compensation
        compensated_mass = float(
            ((1.0 - conditioned_alpha) * compensated_scales).mean()
        )
        if not math.isclose(
            compensated_mass, coefficient_target, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(
                f"conditioned-alpha compensation drift for {ALL_CLASSES[class_id]}"
            )

        updated_mask[row_idx] = conditioned_alpha / base_alpha
        updated_scales[row_idx] = compensated_scales.to(updated_scales.dtype)
        by_count = {}
        for count in range(model_count + 1):
            count_mask = class_counts == count
            count_rows = int(count_mask.sum())
            by_count[str(count)] = {
                "rows": count_rows,
                "alpha": (
                    float(conditioned_alpha[count_mask].mean()) if count_rows else None
                ),
                "scale": (
                    float(compensated_scales[count_mask].mean()) if count_rows else None
                ),
            }
        label = ALL_CLASSES[class_id]
        class_meta[label] = {
            "rows": len(rows),
            "reliability_mean": float(reliability_mean),
            "alpha_mean": float(conditioned_alpha.mean()),
            "alpha_min": float(conditioned_alpha.min()),
            "alpha_max": float(conditioned_alpha.max()),
            "base_scale_mean": base_scale_mean,
            "uncompensated_hard_backbone_mass": uncompensated_mass,
            "compensation_factor": compensation,
            "compensated_scale_mean": float(compensated_scales.mean()),
            "compensated_scale_max": float(compensated_scales.max()),
            "compensated_hard_backbone_mass": compensated_mass,
            "by_correct_count": by_count,
        }
        touched_rows += len(rows)

    updated_consensus = dict(consensus)
    updated_meta = copy.deepcopy(meta)
    updated_meta["distill_alpha_weak_consensus"] = {
        "formula_version": "class-centered-v1-hard-backbone-mass-preserved",
        "lambda": slope,
        "weak_alpha_baseline": weak_alpha,
        "hard_backbone_mass_target": coefficient_target,
        "touched_original_weak4_rows": touched_rows,
        "train_replay_rows_untouched": int(
            sum(int(correct_counts[idx]) < 0 for idx in train_rows)
        ),
        "classes": class_meta,
    }
    updated_consensus["gradient_scales"] = updated_scales
    updated_consensus["meta"] = updated_meta
    args.consensus_reliability_meta = updated_meta
    print(
        "consensus-conditioned Weak4 alpha: "
        f"lambda={slope} weak_alpha={weak_alpha} touched={touched_rows} "
        + " ".join(
            f"{label}:k={values['compensation_factor']:.6f},"
            f"alpha={values['alpha_min']:.6f}-{values['alpha_max']:.6f}"
            for label, values in class_meta.items()
        )
    )
    return (teacher_logits, updated_mask), updated_consensus


def apply_trusted_replay_kd(
    teacher,
    consensus,
    samples,
    y,
    train_idx,
    args,
):
    """Give trusted replay rows their exact predecessor's M8 KD target.

    Trust is determined exclusively by the predecessor original row's aligned
    OOF ``correct_count``.  Replay rows do not inherit consensus backbone/KD
    gradient scales, and their own aligned count remains ``-1``.  Thus this
    changes only the hard/KD mixture for the preselected legacy replay rows.
    """

    source = safe_text(getattr(args, "replay_kd_source", "none")) or "none"
    if source == "none":
        return teacher, consensus
    if source != "predecessor":
        raise ValueError(f"unknown replay KD source: {source}")
    if teacher is None:
        raise ValueError("trusted predecessor replay KD requires teacher targets")
    if consensus is None:
        raise ValueError("trusted predecessor replay KD requires consensus reliability")
    if not isinstance(teacher, (tuple, list)) or len(teacher) != 2:
        raise ValueError("teacher targets must be a (logits, mask) pair")
    if not isinstance(consensus, dict):
        raise ValueError("consensus reliability must be a dict")

    teacher_logits, teacher_mask = teacher
    correct_counts = consensus.get("correct_counts")
    gradient_scales = consensus.get("gradient_scales")
    kd_gradient_scales = consensus.get("kd_gradient_scales")
    meta = consensus.get("meta")
    row_count = len(samples)
    lengths = {
        "labels": len(y),
        "teacher_logits": int(teacher_logits.shape[0]),
        "teacher_mask": int(teacher_mask.shape[0]),
        "correct_counts": int(correct_counts.shape[0]),
        "gradient_scales": int(gradient_scales.shape[0]),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(
            "trusted replay KD row length mismatch: "
            f"samples={row_count} "
            + " ".join(f"{name}={length}" for name, length in lengths.items())
        )
    if not isinstance(meta, dict):
        raise ValueError("trusted replay KD requires consensus metadata")
    model_count = int(meta.get("model_count", -1))
    min_consensus = int(getattr(args, "replay_kd_min_consensus", 2))
    if model_count < 1 or not 1 <= min_consensus <= model_count:
        raise ValueError(
            "trusted replay KD min consensus must lie in "
            f"[1, {model_count}], got {min_consensus}"
        )
    base_alpha = float(args.distill_alpha)
    weak_alpha_value = getattr(args, "distill_alpha_weak", None)
    if not math.isfinite(base_alpha) or not 0.0 < base_alpha <= 1.0:
        raise ValueError("trusted replay KD requires --distill-alpha in (0, 1]")
    if weak_alpha_value is None:
        raise ValueError("trusted replay KD requires --distill-alpha-weak")
    weak_alpha = float(weak_alpha_value)
    if not math.isfinite(weak_alpha) or not 0.0 < weak_alpha <= 1.0:
        raise ValueError(
            "trusted replay KD requires --distill-alpha-weak in (0, 1]"
        )
    weak_scale = weak_alpha / base_alpha

    train_set = set(int(idx) for idx in train_idx)
    if any(idx < 0 or idx >= row_count for idx in train_set):
        raise ValueError("trusted replay KD received an out-of-range train index")
    original_rows = [
        idx for idx, sample in enumerate(samples) if not sample.get("_is_replay", False)
    ]
    replay_rows = [
        idx for idx, sample in enumerate(samples) if sample.get("_is_replay", False)
    ]
    if not replay_rows:
        raise ValueError("trusted replay KD requires replay rows")
    if any(idx not in train_set for idx in replay_rows):
        raise ValueError("trusted replay KD found a replay row outside the training split")
    replay_index = torch.tensor(replay_rows, dtype=torch.long)
    if bool((teacher_mask[replay_index] != 0).any()):
        raise ValueError(
            "trusted replay KD requires every replay row to start hard-label-only"
        )
    if bool((correct_counts[replay_index] != -1).any()):
        raise ValueError(
            "trusted replay KD requires replay consensus counts to remain unaligned (-1)"
        )
    if not torch.equal(
        gradient_scales[replay_index], torch.ones_like(gradient_scales[replay_index])
    ):
        raise ValueError("trusted replay KD requires replay hard-backbone scale 1.0")
    if kd_gradient_scales is not None and not torch.equal(
        kd_gradient_scales[replay_index],
        torch.ones_like(kd_gradient_scales[replay_index]),
    ):
        raise ValueError("trusted replay KD requires replay KD-backbone scale 1.0")

    original_by_id = {}
    for idx in original_rows:
        sample_id = safe_text(samples[idx].get("id"))
        if sample_id in original_by_id:
            raise ValueError(f"duplicate original sample id: {sample_id}")
        original_by_id[sample_id] = idx

    updated_logits = teacher_logits.clone()
    updated_mask = teacher_mask.clone()
    original_logits_before = teacher_logits[original_rows].clone()
    original_mask_before = teacher_mask[original_rows].clone()
    count_histogram = Counter()
    active_ids = []
    active = 0
    active_weak4 = 0
    linked = 0
    teacher_top1_correct = 0
    below_threshold = 0
    for replay_idx in replay_rows:
        predecessor_id = safe_text(
            samples[replay_idx].get("_replay_predecessor_id")
        )
        if not predecessor_id:
            continue
        linked += 1
        predecessor_idx = original_by_id.get(predecessor_id)
        if predecessor_idx is None or predecessor_idx not in train_set:
            raise ValueError(
                "trusted replay KD predecessor is not an original training row: "
                f"replay={samples[replay_idx].get('id')} predecessor={predecessor_id}"
            )
        if int(y[predecessor_idx]) != int(y[replay_idx]):
            raise ValueError(
                "trusted replay KD predecessor label mismatch: "
                f"replay={samples[replay_idx].get('id')} predecessor={predecessor_id}"
            )
        count = int(correct_counts[predecessor_idx])
        if count < 0 or count > model_count:
            raise ValueError(
                f"trusted replay KD predecessor has invalid correct_count={count}"
            )
        count_histogram[count] += 1
        if count < min_consensus:
            below_threshold += 1
            continue
        if float(teacher_mask[predecessor_idx]) <= 0.0:
            raise ValueError(
                "trusted replay KD active predecessor has no M8 teacher target: "
                f"{predecessor_id}"
            )

        updated_logits[replay_idx] = teacher_logits[predecessor_idx]
        is_weak4 = int(y[replay_idx]) in WEAK4_CLASS_ID_SET
        updated_mask[replay_idx] = weak_scale if is_weak4 else 1.0
        active += 1
        active_weak4 += int(is_weak4)
        active_ids.append(safe_text(samples[replay_idx].get("id")))
        teacher_top1_correct += int(
            int(torch.argmax(teacher_logits[predecessor_idx]).item())
            == int(y[replay_idx])
        )

    if active <= 0:
        raise ValueError("trusted replay KD activated zero replay rows")
    if linked != int(
        (getattr(args, "replay_kd_audit", {}) or {}).get(
            "exact_predecessors", linked
        )
    ):
        raise AssertionError(
            "trusted replay KD linked-count drift between replay construction and targets"
        )
    if not torch.equal(updated_logits[original_rows], original_logits_before):
        raise AssertionError("trusted replay KD changed original teacher logits")
    if not torch.equal(updated_mask[original_rows], original_mask_before):
        raise AssertionError("trusted replay KD changed original teacher mask")

    expected_active = int(getattr(args, "replay_kd_expected_active", -1))
    if expected_active >= 0 and active != expected_active:
        raise AssertionError(
            "trusted replay KD active-count mismatch: "
            f"expected={expected_active} actual={active}"
        )
    audit = copy.deepcopy(getattr(args, "replay_kd_audit", None) or {})
    audit.update(
        {
            "min_consensus": min_consensus,
            "consensus_model_count": model_count,
            "linked_predecessors": linked,
            "active_replay_kd_rows": active,
            "active_weak4_rows": active_weak4,
            "active_nonweak_rows": active - active_weak4,
            "below_consensus_threshold": below_threshold,
            "linked_correct_count_histogram": {
                str(count): int(count_histogram.get(count, 0))
                for count in range(model_count + 1)
            },
            "base_alpha": base_alpha,
            "weak4_alpha": weak_alpha,
            "active_teacher_top1_correct": teacher_top1_correct,
            "active_teacher_top1_accuracy": teacher_top1_correct / active,
            "active_replay_id_sha256": _ordered_text_sha256(active_ids),
            "replay_hard_backbone_scale": 1.0,
            "replay_consensus_counts_unchanged": True,
            "original_teacher_tensors_unchanged": True,
        }
    )
    args.replay_kd_audit = audit
    print(
        "trusted replay KD targets: "
        f"active={active}/{len(replay_rows)} linked={linked} "
        f"below_threshold={below_threshold} weak4={active_weak4} "
        f"teacher_top1={teacher_top1_correct}/{active} "
        f"counts={dict(sorted(count_histogram.items()))}"
    )
    return (updated_logits, updated_mask), consensus


def build_training_signals(samples, y, train_idx, args):
    """Build ID-aligned teacher/consensus/relational tensors once."""
    teacher = build_teacher_targets(samples, args)
    consensus = build_consensus_reliability(samples, y, train_idx, args)
    teacher = apply_consensus_distill_alpha(teacher, consensus, args)
    teacher, consensus = apply_consensus_conditioned_weak_alpha(
        teacher, consensus, y, train_idx, args
    )
    teacher, consensus = apply_trusted_replay_kd(
        teacher, consensus, samples, y, train_idx, args
    )
    relational = build_relational_teacher_targets(samples, y, args, teacher=teacher)
    return teacher, consensus, relational


def find_terminal_classifier_head(model):
    """Find the final Linear producing the canonical 14 action logits."""
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and int(module.out_features) == len(ALL_CLASSES):
            if name == "score" or name.endswith(".score"):
                rank = 0
            elif name == "classifier" or name.endswith(".classifier"):
                rank = 1
            elif name.endswith(".classifier.out_proj") or name.endswith(".out_proj"):
                rank = 2
            else:
                rank = 3
            candidates.append((rank, name, module))
    if not candidates:
        raise ValueError(
            "could not find a terminal Linear classifier with "
            f"out_features={len(ALL_CLASSES)}"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_rank = candidates[0][0]
    best = [item for item in candidates if item[0] == best_rank]
    if len(best) != 1:
        raise ValueError(
            "found ambiguous terminal classifier heads: "
            f"{[name for _, name, _ in best]}"
        )
    return best[0][2], best[0][1]


def forward_with_classifier_input(model, encoded, classifier_head=None):
    """Run one ordinary forward and capture the final classifier input."""
    if classifier_head is None:
        classifier_head, _ = find_terminal_classifier_head(model)
    captured_inputs = []

    def capture_head_input(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise ValueError("terminal classifier did not receive a positional tensor input")
        captured_inputs.append(inputs[0])

    handle = classifier_head.register_forward_pre_hook(capture_head_input)
    try:
        outputs = model(**encoded)
    finally:
        handle.remove()
    if len(captured_inputs) != 1:
        raise ValueError(
            "terminal classifier must run exactly once in the sequence-classifier forward; "
            f"observed={len(captured_inputs)}"
        )
    return outputs, captured_inputs[0]


def pool_classifier_hidden(hidden, reference_logits, encoded):
    """Pool the hidden state that produced each sequence-classification logit."""
    if hidden.ndim == 2 and reference_logits.ndim == 2:
        if hidden.shape[0] != reference_logits.shape[0]:
            raise ValueError(
                "classifier hidden/logit batch mismatch: "
                f"hidden={tuple(hidden.shape)} logits={tuple(reference_logits.shape)}"
            )
        return hidden
    if (
        hidden.ndim == 3
        and reference_logits.ndim == 2
        and hidden.shape[0] == reference_logits.shape[0]
    ):
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError(
                "classifier hidden pooling needs a matching attention_mask"
            )
        positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).view(1, -1)
        positions = positions.expand_as(attention_mask)
        sequence_end = positions.masked_fill(
            ~attention_mask.bool(), -1
        ).max(dim=1).values
        if bool((sequence_end < 0).any()):
            raise ValueError("classifier hidden pooling encountered an all-padding sequence")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, sequence_end]
    raise ValueError(
        "cannot pool classifier hidden to sequence logits: "
        f"hidden={tuple(hidden.shape)} logits={tuple(reference_logits.shape)}"
    )


def centered_cosine_gram_loss(student_hidden, teacher_hidden, mask=None, eps=1e-8):
    """Coordinate/scale-invariant relational loss via centered cosine Grams.

    Each representation is row-normalized, converted to a sample-by-sample
    cosine Gram matrix, double-centered over the matched samples, and compared
    with normalized kernel alignment. Independent orthogonal rotations and
    global positive rescaling of either representation leave the loss intact.
    """
    if student_hidden.ndim != 2 or teacher_hidden.ndim != 2:
        raise ValueError(
            "relational hidden tensors must both be rank 2: "
            f"student={tuple(student_hidden.shape)} teacher={tuple(teacher_hidden.shape)}"
        )
    if student_hidden.shape[0] != teacher_hidden.shape[0]:
        raise ValueError(
            "relational hidden batch mismatch: "
            f"student={student_hidden.shape[0]} teacher={teacher_hidden.shape[0]}"
        )
    if mask is None:
        mask = torch.ones(
            student_hidden.shape[0], dtype=torch.bool, device=student_hidden.device
        )
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=student_hidden.device)
        if mask.ndim != 1 or mask.shape[0] != student_hidden.shape[0]:
            raise ValueError(
                "relational mask batch mismatch: "
                f"mask={tuple(mask.shape)} hidden={student_hidden.shape[0]}"
            )
    if int(mask.sum()) < 2:
        # Keep a differentiable zero attached to the student graph.
        return student_hidden.sum() * 0.0

    student = F.normalize(student_hidden[mask].float(), dim=-1, eps=eps)
    teacher = F.normalize(
        teacher_hidden.to(student_hidden.device)[mask].float(), dim=-1, eps=eps
    ).detach()
    student_gram = student @ student.transpose(0, 1)
    teacher_gram = teacher @ teacher.transpose(0, 1)

    def center(gram):
        return (
            gram
            - gram.mean(dim=0, keepdim=True)
            - gram.mean(dim=1, keepdim=True)
            + gram.mean()
        )

    student_gram = center(student_gram)
    teacher_gram = center(teacher_gram)
    numerator = (student_gram * teacher_gram).sum()
    denominator = torch.linalg.vector_norm(student_gram) * torch.linalg.vector_norm(
        teacher_gram
    )
    if float(denominator.detach()) <= eps:
        return student_hidden.sum() * 0.0
    alignment = numerator / denominator.clamp_min(eps)
    return (1.0 - alignment).clamp(min=0.0, max=2.0)


def action_margin_kd_loss(
    student_logits,
    teacher_logits,
    labels,
    mask,
    *,
    topk=3,
    temperature=3.0,
):
    """Match teacher true-vs-hard-negative action margins on covered rows.

    Hard negatives are selected once from the detached teacher scores. Replay
    and unmatched rows are excluded by the ordinary KD coverage mask, so this
    auxiliary cannot introduce a second teacher surface.
    """
    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("action-margin student and teacher logits must be rank 2")
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "action-margin student/teacher shape mismatch: "
            f"student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
        )
    if labels.ndim != 1 or labels.shape[0] != student_logits.shape[0]:
        raise ValueError("action-margin labels must match the logit batch")
    if not 1 <= int(topk) < student_logits.shape[1]:
        raise ValueError(
            f"action-margin topk must be in [1, {student_logits.shape[1] - 1}]"
        )
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("action-margin temperature must be finite and > 0")

    coverage = torch.as_tensor(mask, device=student_logits.device).view(-1) > 0
    if coverage.shape[0] != student_logits.shape[0]:
        raise ValueError("action-margin mask must match the logit batch")
    if not bool(coverage.any()):
        return student_logits.sum() * 0.0

    student = student_logits[coverage].float() / float(temperature)
    teacher = (
        teacher_logits.to(student_logits.device, non_blocking=True)[coverage].float()
        / float(temperature)
    ).detach()
    targets = labels.to(student_logits.device)[coverage]
    true_mask = F.one_hot(targets, num_classes=student.shape[1]).bool()
    hard_negative = teacher.masked_fill(true_mask, -torch.inf).topk(
        int(topk), dim=1
    ).indices
    student_true = student.gather(1, targets[:, None])
    teacher_true = teacher.gather(1, targets[:, None])
    student_margin = student_true - student.gather(1, hard_negative)
    teacher_margin = teacher_true - teacher.gather(1, hard_negative)
    return F.smooth_l1_loss(
        student_margin,
        teacher_margin,
        reduction="none",
    ).mean()


def action_margin_kd_coverage_mask(teacher_mask, labels, label_scope="all"):
    """Build the action-margin row mask without changing ordinary KD coverage."""
    labels = torch.as_tensor(labels)
    coverage = torch.as_tensor(teacher_mask, device=labels.device).view(-1) > 0
    if labels.ndim != 1 or labels.shape[0] != coverage.shape[0]:
        raise ValueError("action-margin labels must match the teacher mask")
    if label_scope == "all":
        return coverage
    if label_scope == "weak4":
        weak4_ids = torch.tensor(WEAK4_CLASS_IDS, device=labels.device)
        return coverage & torch.isin(labels, weak4_ids)
    raise ValueError(f"unknown action-margin label scope: {label_scope}")


def calibrate_action_margin_weight(
    base_loss,
    action_margin_loss,
    student_logits,
    target_grad_ratio,
):
    """Choose one fixed auxiliary weight from logit-gradient norms."""
    target = float(target_grad_ratio)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("action-margin target gradient ratio must be finite and > 0")
    base_grad = torch.autograd.grad(
        base_loss, student_logits, retain_graph=True, create_graph=False
    )[0]
    margin_grad = torch.autograd.grad(
        action_margin_loss, student_logits, retain_graph=True, create_graph=False
    )[0]
    base_norm = torch.linalg.vector_norm(base_grad.detach().float())
    margin_norm = torch.linalg.vector_norm(margin_grad.detach().float())
    base_value = float(base_norm.cpu())
    margin_value = float(margin_norm.cpu())
    if not math.isfinite(base_value) or base_value <= 0.0:
        raise ValueError(f"action-margin base gradient norm is invalid: {base_value}")
    if not math.isfinite(margin_value) or margin_value <= 0.0:
        raise ValueError(f"action-margin auxiliary gradient norm is invalid: {margin_value}")
    weight = target * base_value / margin_value
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"action-margin calibrated weight is invalid: {weight}")
    return weight, {
        "base_logit_grad_norm": base_value,
        "unweighted_margin_logit_grad_norm": margin_value,
        "calibrated_weight": weight,
        "achieved_grad_ratio": weight * margin_value / base_value,
    }


def _pool_recomputed_head_logits(head_logits, reference_logits, encoded):
    if tuple(head_logits.shape) == tuple(reference_logits.shape):
        return head_logits
    if (
        head_logits.ndim == 3
        and reference_logits.ndim == 2
        and head_logits.shape[0] == reference_logits.shape[0]
        and head_logits.shape[2] == reference_logits.shape[1]
    ):
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None or tuple(attention_mask.shape) != tuple(head_logits.shape[:2]):
            raise ValueError(
                "consensus sieve needs a matching attention_mask to pool token-level logits"
            )
        positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).view(1, -1)
        positions = positions.expand_as(attention_mask)
        sequence_end = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        if bool((sequence_end < 0).any()):
            raise ValueError("consensus sieve encountered an all-padding sequence")
        rows = torch.arange(head_logits.shape[0], device=head_logits.device)
        return head_logits[rows, sequence_end]
    raise ValueError(
        "consensus sieve cannot map terminal head output to model logits: "
        f"head={tuple(head_logits.shape)} model={tuple(reference_logits.shape)}"
    )


def forward_with_consensus_sieve(
    model,
    encoded,
    backbone_scales,
    classifier_head=None,
    kd_backbone_scales=None,
    return_hidden=False,
):
    """Return (ordinary logits, hard-label logits with gated backbone gradient).

    Numerically, the recomputed hard logits equal the ordinary model logits.
    Autograd sees `stopgrad(h) + w * (h - stopgrad(h))` at the final classifier
    input, so classifier parameters receive the full hard-label gradient while
    the gradient entering the backbone is scaled per row. The ordinary logits
    remain available for an untouched KD branch.

    With ``kd_backbone_scales`` the same gate is applied a second time for the
    KD branch (KD head gradient stays full, KD backbone gradient is scaled per
    row) and a third element is returned: (ordinary, hard, kd).
    """
    if classifier_head is None:
        classifier_head, _ = find_terminal_classifier_head(model)
    outputs, hidden = forward_with_classifier_input(
        model, encoded, classifier_head=classifier_head
    )

    def gated_logits(row_scales, branch):
        scales = torch.as_tensor(row_scales, dtype=hidden.dtype, device=hidden.device)
        if scales.ndim != 1 or scales.shape[0] != hidden.shape[0]:
            raise ValueError(
                f"consensus {branch} scale batch mismatch: "
                f"scales={tuple(scales.shape)} hidden={tuple(hidden.shape)}"
            )
        scale_shape = [hidden.shape[0]] + [1] * (hidden.ndim - 1)
        scales = scales.view(scale_shape)
        gated_hidden = hidden.detach() + scales * (hidden - hidden.detach())
        return _pool_recomputed_head_logits(
            classifier_head(gated_hidden), outputs.logits, encoded
        )

    hard_logits = gated_logits(backbone_scales, "backbone")
    if kd_backbone_scales is None:
        if return_hidden:
            return (
                outputs.logits,
                hard_logits,
                pool_classifier_hidden(hidden, outputs.logits, encoded),
            )
        return outputs.logits, hard_logits
    kd_logits = gated_logits(kd_backbone_scales, "kd-backbone")
    if return_hidden:
        return (
            outputs.logits,
            hard_logits,
            kd_logits,
            pool_classifier_hidden(hidden, outputs.logits, encoded),
        )
    return outputs.logits, hard_logits, kd_logits


def load_sequence_classifier(args, tokenizer, label_kwargs):
    lora_r = int(getattr(args, "lora_r", 0) or 0)
    resume_from = getattr(args, "resume_from", "") or ""
    resume_path = Path(resume_from) if resume_from else None
    resume_is_adapter = bool(resume_path and (resume_path / "adapter_config.json").exists())
    load_source = args.base_model if resume_is_adapter else (resume_from or args.base_model)
    load_dtype = torch.float16 if lora_r > 0 else (torch.bfloat16 if args.bf16 else torch.float32)
    model_class = getattr(args, "model_class", "auto")

    if model_class == "gemma4custom":
        if args.dropout is not None:
            raise ValueError("--dropout is not wired for --model-class gemma4custom")
        from gemma4_seqcls import build_gemma4_seqcls

        model = build_gemma4_seqcls(load_source, **label_kwargs, torch_dtype=load_dtype)
    elif model_class == "qwen35text":
        from transformers import Qwen3_5TextForSequenceClassification

        model = Qwen3_5TextForSequenceClassification.from_pretrained(
            load_source,
            **label_kwargs,
            torch_dtype=load_dtype,
        )
    elif model_class == "gemma3text":
        from transformers import Gemma3TextForSequenceClassification

        model = Gemma3TextForSequenceClassification.from_pretrained(
            load_source,
            **label_kwargs,
            torch_dtype=load_dtype,
        )
    else:
        dtype_kwargs = {"torch_dtype": load_dtype}
        if args.dropout is not None:
            # decoder configs (Llama/Qwen family) default all dropout to 0.0, which
            # makes R-Drop's two passes identical -- override before weight load
            config = AutoConfig.from_pretrained(load_source, **label_kwargs)
            touched = []
            for attr in (
                "attention_dropout",
                "hidden_dropout",
                "hidden_dropout_prob",
                "attention_probs_dropout_prob",
                "classifier_dropout",
                "resid_pdrop",
                "embd_pdrop",
            ):
                if hasattr(config, attr):
                    setattr(config, attr, args.dropout)
                    touched.append(attr)
            print(f"dropout override={args.dropout} on: {', '.join(touched) if touched else 'NO MATCHING CONFIG ATTRS'}")
            model = AutoModelForSequenceClassification.from_pretrained(load_source, config=config, **dtype_kwargs)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(load_source, **label_kwargs, **dtype_kwargs)

    ensure_model_pad_token(model, tokenizer.pad_token_id)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if lora_r > 0:
        from peft import LoraConfig, PeftModel, get_peft_model

        if resume_is_adapter:
            print(f"loading trainable LoRA adapter from {resume_from}")
            model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=2 * lora_r,
                    lora_dropout=0.05,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    modules_to_save=["score"],
                    task_type="SEQ_CLS",
                ),
            )
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        for param in model.parameters():
            if param.requires_grad and param.dtype == torch.float16:
                param.data = param.data.float()
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    return model


def train_model(
    tokenizer,
    encoded_features,
    lengths,
    y,
    sample_weights,
    train_idx,
    args,
    device,
    teacher=None,
    consensus=None,
    relational=None,
):
    label_kwargs = dict(
        num_labels=len(ALL_CLASSES),
        id2label={i: label for i, label in enumerate(ALL_CLASSES)},
        label2id={label: i for i, label in enumerate(ALL_CLASSES)},
    )
    model = load_sequence_classifier(args, tokenizer, label_kwargs).to(device)
    consensus_classifier_head = None
    if consensus is not None:
        consensus_classifier_head, consensus_head_name = find_terminal_classifier_head(model)
        consensus["meta"]["classifier_head"] = consensus_head_name
        args.consensus_reliability_meta = consensus["meta"]
        print(f"consensus sieve classifier head: {consensus_head_name}")
    relational_classifier_head = consensus_classifier_head
    if relational is not None:
        if relational_classifier_head is None:
            relational_classifier_head, relational_head_name = find_terminal_classifier_head(
                model
            )
        else:
            _, relational_head_name = find_terminal_classifier_head(model)
        relational["meta"]["classifier_head"] = relational_head_name
        args.relational_kd_meta = relational["meta"]
        print(f"relational KD classifier head: {relational_head_name}")

    weights = class_weights([y[i] for i in train_idx], device, args.class_weight_power)
    weak4_ids = torch.tensor(WEAK4_CLASS_IDS, dtype=torch.long, device=device)
    weak4_weights = weights[weak4_ids]
    if args.train_label_filter == "weak4":
        weak4_weights = weak4_weights / torch.clamp(weak4_weights.mean(), min=1e-8)
    explorer4_ids = torch.tensor(EXPLORER4_CLASS_IDS, dtype=torch.long, device=device)
    explorer4_local_targets = torch.full((len(ALL_CLASSES),), -1, dtype=torch.long, device=device)
    explorer4_local_targets[explorer4_ids] = torch.arange(len(EXPLORER4_CLASS_IDS), device=device)
    explorer4_weights = None
    if args.explorer4_loss_balance:
        explorer4_weights = weights[explorer4_ids]
        explorer4_weights = explorer4_weights / torch.clamp(explorer4_weights.mean(), min=1e-8)
    optimizer_params = specialist_optimizer_groups(
        model, args.weight_decay, args.train_label_filter
    )
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            optimizer_params, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_params, lr=args.lr, weight_decay=args.weight_decay
        )
    accum = max(1, args.grad_accum_steps)
    batches_per_epoch = math.ceil(len(train_idx) / args.batch_size)
    total_steps = math.ceil(batches_per_epoch / accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # bf16 needs no loss scaling; a disabled scaler passes scale/unscale_/step through
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=args.amp_init_scale,
        growth_interval=args.amp_growth_interval,
        enabled=device.type == "cuda" and not args.bf16,
    )
    amp_step_stats = {
        "attempted": 0,
        "skipped": 0,
        "enabled": bool(scaler.is_enabled()),
        "init_scale": float(args.amp_init_scale),
        "growth_interval": int(args.amp_growth_interval),
    }
    rng = random.Random(args.seed)
    action_margin_requested = bool(
        getattr(args, "action_margin_kd_weight", 0.0) > 0.0
        or getattr(args, "action_margin_kd_target_grad_ratio", 0.0) > 0.0
    )
    if action_margin_requested and teacher is None:
        raise ValueError("action-margin KD requires aligned ordinary teacher targets")
    action_margin_weight = (
        float(args.action_margin_kd_weight)
        if getattr(args, "action_margin_kd_weight", 0.0) > 0.0
        else None
    )
    action_margin_meta = None
    if action_margin_requested:
        action_margin_meta = {
            "enabled": True,
            "loss": "teacher_topk_true_vs_negative_smooth_l1",
            "topk": int(args.action_margin_kd_topk),
            "temperature": float(args.distill_temp),
            "coverage": "ordinary_logit_kd_mask_gt_zero",
            "label_scope": str(args.action_margin_kd_label_scope),
            "target_grad_ratio": (
                float(args.action_margin_kd_target_grad_ratio)
                if args.action_margin_kd_target_grad_ratio > 0.0
                else None
            ),
            "fixed_weight_input": (
                float(args.action_margin_kd_weight)
                if args.action_margin_kd_weight > 0.0
                else None
            ),
            "calibrated_weight": action_margin_weight,
            "calibration": None,
            "epochs": [],
        }
        args.action_margin_kd_meta = action_margin_meta

    def optimizer_step():
        amp_step_stats["attempted"] += 1
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not scaler.is_enabled() and not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite grad norm: {float(grad_norm.detach().cpu())}")
        scale_before = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        # GradScaler lowers its scale exactly when it found non-finite gradients
        # and skipped optimizer.step(). Keep this observable for cloud-run audits.
        if scaler.is_enabled() and scale_after < scale_before:
            amp_step_stats["skipped"] += 1
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def per_row_loss(logits, labels, batch_idx, hard_logits=None, kd_logits=None):
        supervised_logits = logits if hard_logits is None else hard_logits
        if args.train_label_filter == "weak4":
            if bool(((labels < 0) | (labels >= len(WEAK4_CLASS_IDS))).any()):
                raise ValueError(f"weak4 specialist batch has labels outside 0-3: {labels.tolist()}")
            loss_weights = weak4_weights
            loss_class_ids = weak4_ids
        else:
            loss_weights = weights
            loss_class_ids = None
        loss_values = classification_loss_values(
            supervised_logits,
            labels,
            loss_weights,
            args.label_smoothing,
            args.loss,
            args.focal_gamma,
            class_ids=loss_class_ids,
        )
        if args.explorer4_loss_weight > 0 and args.train_label_filter == "none":
            explorer4_targets = explorer4_local_targets[labels]
            explorer4_mask = explorer4_targets >= 0
            if bool(explorer4_mask.any()):
                explorer4_loss_values = torch.zeros_like(loss_values)
                explorer4_loss_values[explorer4_mask] = F.cross_entropy(
                    supervised_logits.float()[explorer4_mask][:, explorer4_ids],
                    explorer4_targets[explorer4_mask],
                    weight=explorer4_weights,
                    reduction="none",
                )
                loss_values = loss_values + args.explorer4_loss_weight * explorer4_loss_values
        if teacher is not None:
            temp = args.distill_temp
            teacher_p = F.softmax(teacher[0][batch_idx].to(device) / temp, dim=-1)
            # KD student logits: the kd-gated recompute when the KD-branch
            # sieve is active (numerically identical, backbone gradient scaled)
            kd_student_logits = logits if kd_logits is None else kd_logits
            student_lp = F.log_softmax(kd_student_logits.float() / temp, dim=-1)
            kd = F.kl_div(student_lp, teacher_p, reduction="none").sum(-1) * (temp * temp)
            # per-row alpha: replay/unmatched rows (mask 0) keep the pure hard-label loss
            alpha = args.distill_alpha * teacher[1][batch_idx].to(device)
            loss_values = (1.0 - alpha) * loss_values + alpha * kd
        return loss_values

    start_epoch = 1
    if args.resume_from:
        state_path = Path(args.resume_from) / "checkpoint_state.json"
        last_done = 0
        checkpoint_state = {}
        if state_path.exists():
            checkpoint_state = json.loads(state_path.read_text(encoding="utf-8"))
            last_done = int(checkpoint_state.get("last_completed_epoch", 0))
        if (
            action_margin_requested
            and args.action_margin_kd_target_grad_ratio > 0.0
            and last_done > 0
        ):
            saved = checkpoint_state.get("action_margin_kd")
            if not isinstance(saved, dict):
                raise ValueError(
                    "action-margin epoch resume is missing calibrated checkpoint state"
                )
            expected = {
                "topk": int(args.action_margin_kd_topk),
                "temperature": float(args.distill_temp),
                "target_grad_ratio": float(args.action_margin_kd_target_grad_ratio),
                "label_scope": str(args.action_margin_kd_label_scope),
            }
            for key, value in expected.items():
                if saved.get(key) != value:
                    raise ValueError(
                        f"action-margin resume {key} mismatch: "
                        f"checkpoint={saved.get(key)!r} current={value!r}"
                    )
            restored_weight = float(saved.get("calibrated_weight", 0.0))
            if not math.isfinite(restored_weight) or restored_weight <= 0.0:
                raise ValueError(
                    "action-margin epoch resume has no valid calibrated weight"
                )
            action_margin_weight = restored_weight
            action_margin_meta.update(copy.deepcopy(saved))
            action_margin_meta.setdefault("epochs", [])
            args.action_margin_kd_meta = action_margin_meta
            print(
                "action-margin KD restored: "
                f"weight={action_margin_weight:.8f} "
                f"target_ratio={args.action_margin_kd_target_grad_ratio:.6f}"
            )
        start_epoch = last_done + 1
        steps_per_opt_epoch = math.ceil(batches_per_epoch / accum)
        for _ in range(steps_per_opt_epoch * last_done):
            scheduler.step()
        for _ in range(last_done):
            list(make_batches(train_idx, args.batch_size, rng, lengths, args.bucket_multiplier))
        print(
            f"resume: {args.resume_from} last_completed_epoch={last_done} "
            f"start_epoch={start_epoch} lr={scheduler.get_last_lr()[0]:.3e}"
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_kl = 0.0
        total_relational = 0.0
        relational_batches = 0
        relational_rows = 0
        total_action_margin = 0.0
        action_margin_batches = 0
        action_margin_rows = 0
        seen = 0
        step = 0
        for step, batch_idx in enumerate(
            make_batches(train_idx, args.batch_size, rng, lengths, args.bucket_multiplier),
            1,
        ):
            labels = torch.tensor([y[i] for i in batch_idx], dtype=torch.long, device=device)
            weights_for_samples = torch.tensor([sample_weights[i] for i in batch_idx], dtype=torch.float32, device=device)
            encoded = make_encoded_batch(tokenizer, encoded_features, batch_idx, args, device)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.bfloat16 if args.bf16 else torch.float16):
                if consensus is None:
                    if relational is None:
                        logits = model(**encoded).logits
                        student_hidden = None
                    else:
                        outputs, classifier_input = forward_with_classifier_input(
                            model,
                            encoded,
                            classifier_head=relational_classifier_head,
                        )
                        logits = outputs.logits
                        student_hidden = pool_classifier_hidden(
                            classifier_input, logits, encoded
                        )
                    hard_logits = None
                    kd_logits = None
                elif consensus["kd_gradient_scales"] is None:
                    batch_scales = consensus["gradient_scales"][batch_idx].to(device)
                    if relational is None:
                        logits, hard_logits = forward_with_consensus_sieve(
                            model,
                            encoded,
                            batch_scales,
                            classifier_head=consensus_classifier_head,
                        )
                        student_hidden = None
                    else:
                        logits, hard_logits, student_hidden = forward_with_consensus_sieve(
                            model,
                            encoded,
                            batch_scales,
                            classifier_head=consensus_classifier_head,
                            return_hidden=True,
                        )
                    kd_logits = None
                else:
                    batch_scales = consensus["gradient_scales"][batch_idx].to(device)
                    kd_batch_scales = consensus["kd_gradient_scales"][batch_idx].to(device)
                    if relational is None:
                        logits, hard_logits, kd_logits = forward_with_consensus_sieve(
                            model,
                            encoded,
                            batch_scales,
                            classifier_head=consensus_classifier_head,
                            kd_backbone_scales=kd_batch_scales,
                        )
                        student_hidden = None
                    else:
                        (
                            logits,
                            hard_logits,
                            kd_logits,
                            student_hidden,
                        ) = forward_with_consensus_sieve(
                            model,
                            encoded,
                            batch_scales,
                            classifier_head=consensus_classifier_head,
                            kd_backbone_scales=kd_batch_scales,
                            return_hidden=True,
                        )
                loss_values = per_row_loss(
                    logits, labels, batch_idx, hard_logits=hard_logits, kd_logits=kd_logits
                )
                if args.rdrop_alpha > 0:
                    # R-Drop (arXiv:2106.14448): second dropout-perturbed pass +
                    # symmetric KL; needs --dropout > 0 or both passes are identical
                    logits2 = model(**encoded).logits
                    if args.train_label_filter == "weak4":
                        rdrop_logits1 = logits.float()[:, weak4_ids]
                        rdrop_logits2 = logits2.float()[:, weak4_ids]
                    else:
                        rdrop_logits1 = logits.float()
                        rdrop_logits2 = logits2.float()
                    lp1 = F.log_softmax(rdrop_logits1, dim=-1)
                    lp2 = F.log_softmax(rdrop_logits2, dim=-1)
                    rdrop_kl = 0.5 * (
                        F.kl_div(lp1, lp2.exp(), reduction="none").sum(-1)
                        + F.kl_div(lp2, lp1.exp(), reduction="none").sum(-1)
                    )
                    loss_values = 0.5 * (loss_values + per_row_loss(logits2, labels, batch_idx)) + args.rdrop_alpha * rdrop_kl
                    total_kl += float(rdrop_kl.detach().mean().cpu()) * len(batch_idx)
                loss = (loss_values * weights_for_samples).sum() / torch.clamp(weights_for_samples.sum(), min=1.0)
                if action_margin_requested:
                    margin_student_logits = logits if kd_logits is None else kd_logits
                    margin_mask = action_margin_kd_coverage_mask(
                        teacher[1][batch_idx].to(device, non_blocking=True),
                        labels,
                        args.action_margin_kd_label_scope,
                    )
                    margin_rows = int(margin_mask.sum())
                    margin_loss = action_margin_kd_loss(
                        margin_student_logits,
                        teacher[0][batch_idx],
                        labels,
                        margin_mask,
                        topk=args.action_margin_kd_topk,
                        temperature=args.distill_temp,
                    )
                    if margin_rows > 0 and action_margin_weight is None:
                        action_margin_weight, calibration = calibrate_action_margin_weight(
                            loss,
                            margin_loss,
                            margin_student_logits,
                            args.action_margin_kd_target_grad_ratio,
                        )
                        calibration.update(
                            {
                                "epoch": epoch,
                                "step": step,
                                "matched_rows": margin_rows,
                            }
                        )
                        action_margin_meta["calibrated_weight"] = action_margin_weight
                        action_margin_meta["calibration"] = calibration
                        print(
                            "action-margin KD calibrated: "
                            f"weight={action_margin_weight:.8f} "
                            f"target={args.action_margin_kd_target_grad_ratio:.6f} "
                            f"achieved={calibration['achieved_grad_ratio']:.6f} "
                            f"base_grad={calibration['base_logit_grad_norm']:.8f} "
                            f"aux_grad={calibration['unweighted_margin_logit_grad_norm']:.8f}"
                        )
                    if action_margin_weight is not None:
                        loss = loss + action_margin_weight * margin_loss
                    if margin_rows > 0:
                        total_action_margin += float(margin_loss.detach().cpu())
                        action_margin_batches += 1
                        action_margin_rows += margin_rows
                if relational is not None:
                    relation_mask = relational["mask"][batch_idx].to(
                        device, non_blocking=True
                    )
                    relation_rows = int(relation_mask.sum())
                    relation_loss = centered_cosine_gram_loss(
                        student_hidden,
                        relational["hidden"][batch_idx],
                        mask=relation_mask,
                    )
                    loss = loss + args.relational_kd_weight * relation_loss
                    if relation_rows >= 2:
                        total_relational += float(relation_loss.detach().cpu())
                        relational_batches += 1
                        relational_rows += relation_rows
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}")
            scaler.scale(loss / accum).backward()
            if step % accum == 0:
                optimizer_step()
            total_loss += float(loss.detach().cpu()) * len(batch_idx)
            seen += len(batch_idx)
            if args.log_every and step % args.log_every == 0:
                kl_note = f" rdrop_kl={total_kl / max(1, seen):.5f}" if args.rdrop_alpha > 0 else ""
                relational_note = (
                    f" rel_gram={total_relational / max(1, relational_batches):.5f}"
                    f" rel_rows={relational_rows}"
                    if relational is not None
                    else ""
                )
                action_margin_note = (
                    f" am_kd={total_action_margin / max(1, action_margin_batches):.5f}"
                    f" am_rows={action_margin_rows}"
                    f" am_weight={float(action_margin_weight or 0.0):.8f}"
                    if action_margin_requested
                    else ""
                )
                print(
                    f"    step={step:04d} loss={total_loss / max(1, seen):.5f}"
                    f"{kl_note}{relational_note}{action_margin_note}"
                )
        if step % accum != 0:
            optimizer_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        kl_note = f" rdrop_kl={total_kl / max(1, seen):.5f}" if args.rdrop_alpha > 0 else ""
        relational_note = (
            f" rel_gram={total_relational / max(1, relational_batches):.5f}"
            f" rel_batches={relational_batches} rel_rows={relational_rows}"
            if relational is not None
            else ""
        )
        action_margin_note = (
            f" am_kd={total_action_margin / max(1, action_margin_batches):.5f}"
            f" am_batches={action_margin_batches} am_rows={action_margin_rows}"
            f" am_weight={float(action_margin_weight or 0.0):.8f}"
            if action_margin_requested
            else ""
        )
        if action_margin_requested:
            action_margin_meta["epochs"].append(
                {
                    "epoch": epoch,
                    "mean_unweighted_loss": total_action_margin
                    / max(1, action_margin_batches),
                    "matched_batches": action_margin_batches,
                    "matched_rows": action_margin_rows,
                }
            )
        print(
            f"  epoch={epoch:02d} train_loss={total_loss / max(1, seen):.5f}"
            f"{kl_note}{relational_note}{action_margin_note}"
        )
        if args.epoch_checkpoint_dir:
            try:
                save_epoch_checkpoint(
                    model,
                    tokenizer,
                    args.epoch_checkpoint_dir,
                    epoch,
                    snapshot_epoch=bool(args.snapshot_epoch_checkpoints),
                    training_state={
                        "action_margin_kd": copy.deepcopy(action_margin_meta)
                    }
                    if action_margin_requested
                    else None,
                )
            except Exception as exc:
                # checkpoint is insurance only — a Drive-mount hiccup must not kill the run
                print(f"  epoch checkpoint FAILED (continuing): {exc}")
    if action_margin_requested and action_margin_weight is None:
        raise ValueError("action-margin KD never found a teacher-covered training batch")
    amp_step_stats["final_scale"] = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    args.amp_step_stats = dict(amp_step_stats)
    print(
        "AMP optimizer steps: "
        f"attempted={amp_step_stats['attempted']} skipped={amp_step_stats['skipped']} "
        f"enabled={amp_step_stats['enabled']} init_scale={amp_step_stats['init_scale']:.1f} "
        f"final_scale={amp_step_stats['final_scale']:.1f} "
        f"growth_interval={amp_step_stats['growth_interval']}"
    )
    if args.require_zero_amp_skips and amp_step_stats["skipped"]:
        raise FloatingPointError(
            f"AMP skipped-step audit failed: skipped={amp_step_stats['skipped']} "
            f"of attempted={amp_step_stats['attempted']}"
        )
    return model


def save_epoch_checkpoint(
    model,
    tokenizer,
    ckpt_dir,
    epoch,
    snapshot_epoch=False,
    training_state=None,
):
    """Crash insurance for preemptible runtimes: overwrite ckpt_dir with an
    fp16 copy of the last completed epoch (point it at a Drive path)."""
    import copy

    start = time.perf_counter()
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "peft_config"):
        model.save_pretrained(ckpt_dir, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_dir)
    else:
        snapshot = copy.deepcopy(model).half().cpu()
        snapshot.save_pretrained(ckpt_dir, safe_serialization=True)
        del snapshot
        if epoch == 1:
            tokenizer.save_pretrained(ckpt_dir)
    checkpoint_state = {"last_completed_epoch": epoch}
    if training_state:
        checkpoint_state.update(training_state)
    (ckpt_dir / "checkpoint_state.json").write_text(
        json.dumps(checkpoint_state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if snapshot_epoch:
        snapshot_dir = ckpt_dir.with_name(f"{ckpt_dir.name}_ep{epoch}")
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        shutil.copytree(ckpt_dir, snapshot_dir)
        print(f"  epoch checkpoint snapshot -> {snapshot_dir}")
    print(f"  epoch checkpoint -> {ckpt_dir} (epoch={epoch}, {time.perf_counter() - start:.0f}s)")


def save_hf_artifact(model, tokenizer, output_dir, class_bias, args, metrics):
    output_dir = Path(output_dir)
    hf_dir = output_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    if args.save_fp16:
        model = model.half()
    model.save_pretrained(hf_dir, safe_serialization=True)
    tokenizer.save_pretrained(hf_dir)
    if int(getattr(args, "lora_r", 0) or 0) > 0:
        try:
            from importlib.metadata import PackageNotFoundError, version

            def package_version(name):
                try:
                    return version(name)
                except PackageNotFoundError:
                    return "missing"

            try:
                repo_dir = Path(__file__).resolve().parent
                git_sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repo_dir,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                git_status = subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=repo_dir,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).splitlines()
                worktree_digest = hashlib.sha256(
                    subprocess.check_output(
                        ["git", "diff", "--binary", "HEAD"],
                        cwd=repo_dir,
                        stderr=subprocess.DEVNULL,
                    )
                )
                untracked = []
                for line in git_status:
                    if not line.startswith("?? "):
                        continue
                    candidate = repo_dir / line[3:]
                    paths = sorted(candidate.rglob("*")) if candidate.is_dir() else [candidate]
                    untracked.extend(path for path in paths if path.is_file())
                for path in sorted(set(untracked)):
                    rel = str(path.relative_to(repo_dir)).encode("utf-8")
                    worktree_digest.update(len(rel).to_bytes(4, "big"))
                    worktree_digest.update(rel)
                    worktree_digest.update(path.read_bytes())
                worktree_sha = worktree_digest.hexdigest()
            except (OSError, subprocess.CalledProcessError):
                git_sha = "unknown"
                git_status = ["unknown"]
                worktree_sha = "unknown"
            cloud_manifest_path = repo_dir / "cloud_manifest.json"
            cloud_manifest = None
            if cloud_manifest_path.is_file():
                cloud_manifest = json.loads(cloud_manifest_path.read_text(encoding="utf-8"))
            provenance = {
                "git_sha": git_sha,
                "git_dirty": bool(git_status),
                "git_status": git_status,
                "working_tree_diff_sha256": worktree_sha,
                "cloud_manifest": cloud_manifest,
                "python": sys.version.split()[0],
                "packages": {
                    "peft": package_version("peft"),
                    "transformers": package_version("transformers"),
                    "torch": package_version("torch"),
                    "safetensors": package_version("safetensors"),
                },
                "train_command": " ".join(shlex.quote(part) for part in sys.argv),
                "resume_from": str(args.resume_from),
                "serializer_name": args.serializer,
                "terminal_token": safe_text(getattr(args, "terminal_token", "")),
                "train_label_filter": args.train_label_filter,
            }
            (hf_dir / "weak4_training_provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            raise RuntimeError(f"failed to record LoRA training provenance: {exc}") from exc
    meta = {
        "classes": ALL_CLASSES,
        "class_bias": [float(x) for x in class_bias.tolist()],
        "max_length": args.max_length,
        "batch_size": args.eval_batch_size,
        "validation_macro_f1": metrics["macro_f1"],
        "validation_split": args.split,
        "fold_id": args.fold_id if args.split == "session_oof" else None,
        "n_folds": args.n_folds if args.split == "session_oof" else None,
        "serializer_name": args.serializer,
        "terminal_token": safe_text(getattr(args, "terminal_token", "")),
        "replay_mode": args.replay_mode,
        "replay_meta_mode": getattr(args, "replay_meta_mode", "current"),
        "replay_sample_weight": args.replay_sample_weight,
        "replay_predecessor_audit": getattr(args, "replay_predecessor_audit", None),
        "replay_kd": getattr(args, "replay_kd_audit", None),
        "base_model": args.base_model,
        "model_class": getattr(args, "model_class", "auto"),
        "lora_r": int(getattr(args, "lora_r", 0) or 0),
        "train_label_filter": args.train_label_filter,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "train_batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "loss": args.loss,
        "focal_gamma": float(args.focal_gamma),
        "class_weight_power": float(args.class_weight_power),
        "optim": args.optim,
        "bf16": bool(args.bf16),
        "explorer4_loss_weight": float(args.explorer4_loss_weight),
        "explorer4_loss_balance": bool(args.explorer4_loss_balance),
        "sim_early_turn_loss": getattr(args, "sim_early_turn_loss_meta", None),
        "consensus_reliability": getattr(args, "consensus_reliability_meta", None),
        "relational_kd": getattr(args, "relational_kd_meta", None),
        "action_margin_kd": getattr(args, "action_margin_kd_meta", None),
        "trained_with_cuda": torch.cuda.is_available(),
        "final_refit": bool(args.final_model),
        "saved_fp16": bool(args.save_fp16),
    }
    if args.class_bias_artifact:
        meta["class_bias_source"] = str(args.class_bias_artifact)
    if args.rule_boosts_path:
        with Path(args.rule_boosts_path).open(encoding="utf-8") as f:
            rule_payload = json.load(f)
        meta["rule_boosts"] = rule_payload.get("rules", [])
        meta["rule_boosts_source"] = str(args.rule_boosts_path)
    with (output_dir / "hf_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def dir_size_mb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total / (1024 * 1024)


def summarize_weak_classes(metrics, count=5):
    return ";".join(f"{name}:{score:.4f}" for name, score in sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:count])


def explorer4_metrics(y_true, y_pred, full_metrics=None):
    if full_metrics is None:
        full_metrics = f1_metrics(y_true, y_pred)
    per_class = {
        label: float(full_metrics["per_class_f1"].get(label, 0.0))
        for label in EXPLORER4_CLASSES
    }
    confusion = [[0 for _ in EXPLORER4_CLASSES] for _ in EXPLORER4_CLASSES]
    true_explorer4 = 0
    true_explorer4_pred_non = 0
    true_non_explorer4 = 0
    true_non_pred_explorer4 = 0
    pair_confusions = Counter()
    for true_id, pred_id in zip(y_true, y_pred):
        true_is_e4 = true_id in EXPLORER4_CLASS_ID_SET
        pred_is_e4 = pred_id in EXPLORER4_CLASS_ID_SET
        if true_is_e4:
            true_explorer4 += 1
            if pred_is_e4:
                confusion[EXPLORER4_CLASS_IDS.index(true_id)][EXPLORER4_CLASS_IDS.index(pred_id)] += 1
                if true_id != pred_id:
                    pair_confusions[(ALL_CLASSES[true_id], ALL_CLASSES[pred_id])] += 1
            else:
                true_explorer4_pred_non += 1
        else:
            true_non_explorer4 += 1
            if pred_is_e4:
                true_non_pred_explorer4 += 1
    explorer4_sum = sum(per_class.values())
    return {
        "explorer4_classes": EXPLORER4_CLASSES,
        "explorer4_macro_f1": explorer4_sum / len(EXPLORER4_CLASSES),
        "explorer4_sum_f1": explorer4_sum,
        "explorer4_per_class": per_class,
        "explorer4_confusion_4x4": confusion,
        "true_explorer4_pred_non_explorer_rate": true_explorer4_pred_non / max(1, true_explorer4),
        "true_non_explorer_pred_explorer4_rate": true_non_pred_explorer4 / max(1, true_non_explorer4),
        "explorer4_pair_confusions": [
            {"count": count, "true": true_label, "pred": pred_label}
            for (true_label, pred_label), count in pair_confusions.most_common(12)
        ],
    }


def explorer4_metrics_from_logits(logits, y_true, bias, full_metrics=None):
    pred = predict_with_bias(logits, bias)
    return explorer4_metrics(y_true, pred, full_metrics=full_metrics)


def weak4_conditional_metrics(logits, y_true):
    weak_rows = [idx for idx, label in enumerate(y_true) if label in WEAK4_CLASS_ID_SET]
    if not weak_rows:
        raise ValueError("validation set has no true Weak4 rows")
    row_idx = torch.tensor(weak_rows, dtype=torch.long)
    weak_ids = torch.tensor(WEAK4_CLASS_IDS, dtype=torch.long)
    pred_local = torch.argmax(logits.float()[row_idx][:, weak_ids], dim=1).tolist()
    true_local = [int(y_true[idx]) for idx in weak_rows]
    confusion = [[0 for _ in WEAK4_CLASSES] for _ in WEAK4_CLASSES]
    for true_id, pred_id in zip(true_local, pred_local):
        confusion[true_id][pred_id] += 1
    per_class = {}
    for class_id, label in enumerate(WEAK4_CLASSES):
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(len(WEAK4_CLASSES))) - tp
        fn = sum(confusion[class_id]) - tp
        denom = 2 * tp + fp + fn
        per_class[label] = (2 * tp / denom) if denom else 0.0
    top_confusions = []
    for true_id, true_label in enumerate(WEAK4_CLASSES):
        for pred_id, pred_label in enumerate(WEAK4_CLASSES):
            if true_id != pred_id and confusion[true_id][pred_id]:
                top_confusions.append(
                    (confusion[true_id][pred_id], true_label, pred_label)
                )
    top_confusions.sort(reverse=True)
    return {
        "n_true_weak4": len(weak_rows),
        "macro_f1": sum(per_class.values()) / len(per_class),
        "per_class_f1": per_class,
        "confusion_4x4": confusion,
        "prediction_distribution": dict(Counter(WEAK4_CLASSES[pred] for pred in pred_local)),
        "top_confusions": top_confusions,
    }


def load_tokenizer_for_args(args):
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=not args.slow_tokenizer)
    except ImportError:
        if args.slow_tokenizer:
            raise
        print(f"fast tokenizer unavailable for {args.base_model}; falling back to slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        # decoder checkpoints (Qwen etc.) ship without a pad token; tokenizer.pad() needs one
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def ensure_model_pad_token(model, pad_token_id):
    """Set pad_token_id on top-level and nested text configs when present."""
    configs = [getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)]
    for config in configs:
        if config is None:
            continue
        if not hasattr(config, "pad_token_id") or getattr(config, "pad_token_id") is None:
            setattr(config, "pad_token_id", pad_token_id)


def load_class_bias_artifact(path):
    if not path:
        return None, None
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    raw_bias = payload.get("class_bias")
    if raw_bias is None:
        raise ValueError(f"{path} does not contain class_bias")
    if isinstance(raw_bias, dict):
        bias_values = [float(raw_bias.get(label, 0.0)) for label in ALL_CLASSES]
    else:
        bias_values = [float(value) for value in raw_bias]
    if len(bias_values) != len(ALL_CLASSES):
        raise ValueError(f"{path} class_bias has {len(bias_values)} values, expected {len(ALL_CLASSES)}")
    metrics = payload.get("metrics") or {}
    if "macro_f1" not in metrics:
        metrics = {"macro_f1": float(payload.get("validation_macro_f1", 0.0))}
    return torch.tensor(bias_values, dtype=torch.float32), metrics


def append_log(
    experiment_id,
    args,
    raw_metrics,
    metrics,
    decision,
    runtime,
    val_logits_path,
    artifact_size_mb,
    old_bias_metrics=None,
):
    weak = sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:5]
    strong = sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    quick_note = f", quick_val_size={args.quick_val_size}" if args.quick_val_size else ""
    fold_note = f", fold={args.fold_id}/{args.n_folds}" if args.split == "session_oof" else ""
    lines = [
        f"## {experiment_id}",
        "",
        f"- Date/time: {now}",
        "- Hypothesis: A cached multilingual transformer pipeline should make fixed-session screening faster without changing the model family.",
        f"- Code/config changes: `{args.base_model}`, serializer={args.serializer}, replay={args.replay_mode}, max_length={args.max_length}, epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, bucket_multiplier={args.bucket_multiplier}, explorer4_loss={args.explorer4_loss_weight}, explorer4_balance={args.explorer4_loss_balance}.",
        f"- Validation setup: {args.split}{fold_note}{quick_note}",
        f"- Raw Macro-F1: {raw_metrics['macro_f1']:.6f}",
        f"- Old bias-tuned Macro-F1: {old_bias_metrics['macro_f1']:.6f}" if old_bias_metrics else "- Old bias-tuned Macro-F1: not run",
        f"- Overall Macro-F1: {metrics['macro_f1']:.6f}",
        "- Per-class observations:",
        f"  - Weakest: {', '.join(f'{k}={v:.3f}' for k, v in weak)}",
        f"  - Strongest: {', '.join(f'{k}={v:.3f}' for k, v in strong)}",
        f"- Top confusions: {metrics['top_confusions'][:8]}",
        f"- Prediction distribution: {metrics['prediction_distribution']}",
        f"- Runtime or package-size concerns: runtime={runtime['total_sec']:.1f}s, tokenize={runtime['tokenize_sec']:.1f}s, train={runtime['train_sec']:.1f}s, eval={runtime['eval_sec']:.1f}s, artifact_size_mb={artifact_size_mb:.1f}.",
        f"- Validation logits: {val_logits_path or 'not saved'}",
        f"- Decision: {decision}",
        "- Next suggested experiment: quick-screen serializer/replay variants, then promote only broad fixed-session improvements to OOF.",
        "",
    ]
    with open("research_log.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_val_logits(
    experiment_id,
    logits,
    y_true,
    ordered_indices,
    samples,
    bias,
    raw_metrics,
    metrics,
    args,
    old_bias=None,
    old_bias_metrics=None,
    weak4_conditional=None,
):
    if not args.save_val_logits:
        return ""
    path = Path(args.logits_dir) / f"{experiment_id}_val_logits.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    zero_bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
    torch.save(
        {
            "logits": logits.float().cpu(),
            "y_true": y_true,
            "indices": ordered_indices,
            "ids": [samples[i].get("id", "") for i in ordered_indices],
            "classes": ALL_CLASSES,
            "class_bias": [float(x) for x in bias.tolist()],
            "old_class_bias": [float(x) for x in old_bias.tolist()] if old_bias is not None else None,
            "raw_metrics": raw_metrics,
            "old_bias_metrics": old_bias_metrics,
            "metrics": metrics,
            "raw_explorer4_metrics": explorer4_metrics_from_logits(logits, y_true, zero_bias, full_metrics=raw_metrics),
            "old_bias_explorer4_metrics": (
                explorer4_metrics_from_logits(logits, y_true, old_bias, full_metrics=old_bias_metrics)
                if old_bias is not None and old_bias_metrics is not None
                else None
            ),
            "explorer4_metrics": explorer4_metrics_from_logits(logits, y_true, bias, full_metrics=metrics),
            "base_model": args.base_model,
            "serializer_name": args.serializer,
            "terminal_token": safe_text(getattr(args, "terminal_token", "")),
            "max_length": args.max_length,
            "split": args.split,
            "fold_id": args.fold_id if args.split == "session_oof" else None,
            "n_folds": args.n_folds if args.split == "session_oof" else None,
            "seed": args.seed,
            "train_label_filter": args.train_label_filter,
            "weak4_conditional_metrics": weak4_conditional,
        },
        path,
    )
    return str(path)


def experiment_id_for(args):
    parts = [
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "gpu_transformer",
        args.split,
        safe_slug(args.serializer),
        f"len{args.max_length}",
    ]
    if args.split == "session_oof":
        parts.append(f"fold{args.fold_id}-of{args.n_folds}")
    if args.quick_val_size:
        parts.append(f"qv{args.quick_val_size}")
    if args.replay_mode != "none":
        parts.append(f"replay-{safe_slug(args.replay_mode)}")
    suffix = args.experiment_suffix or args.notes
    if suffix:
        parts.append(safe_slug(suffix)[:48])
    return "_".join(parts)


def run(args):
    run_start = time.perf_counter()
    if args.final_model and args.split == "session_oof":
        raise ValueError("Use --split session, not session_oof, for final refit.")
    if args.save_val_model and args.output_dir == "model":
        raise ValueError("--save-val-model needs an explicit --output-dir; refusing to overwrite the packaged model/")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    validate_specialist_warmstart(args)
    if (
        (args.train_label_filter != "none" or args.assert_val_ids)
        and args.resume_from
        and (Path(args.resume_from) / "checkpoint_state.json").exists()
    ):
        raise ValueError(
            "specialist warm-start --resume-from must be a completed full-weight checkpoint; "
            "checkpoint_state.json indicates an epoch-resume directory"
        )
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"device=cuda name={torch.cuda.get_device_name(0)}")
    else:
        print("device=cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    samples = load_jsonl(train_path)
    labels_by_id = load_labels(data_dir / "train_labels.csv")
    y = [CLASS_TO_ID[labels_by_id[sample["id"]]] for sample in samples]
    base_samples = samples
    base_y = y
    if args.final_only:
        if not args.final_model:
            raise ValueError("--final-only requires --final-model")
        if args.split == "session_oof":
            raise ValueError("Use --split session, not session_oof, for --final-only")
        tokenizer = load_tokenizer_for_args(args)
        if args.replay_mode == "none":
            final_samples = base_samples
            final_y = base_y
            final_idx = list(range(len(base_samples)))
            final_sample_weights = [1.0] * len(base_samples)
            final_replay_size = 0
        else:
            final_samples, final_y, final_idx, final_sample_weights, final_replay_size = add_replay_examples(
                base_samples,
                base_y,
                list(range(len(base_samples))),
                args,
            )
        final_idx = filter_train_indices(final_idx, final_y, args.train_label_filter)
        final_sample_weights = apply_sim_early_turn_loss_scale(
            final_samples,
            final_y,
            final_idx,
            final_sample_weights,
            args,
        )
        print(f"split=final_refit train={len(final_idx)} replay_size={final_replay_size}")
        token_start = time.perf_counter()
        final_texts, text_cache_path = build_serialized_texts(
            final_samples,
            args,
            train_path,
            cache_scope="final",
            tokenizer=tokenizer,
        )
        final_encoded_features, final_lengths, token_cache_path = tokenize_texts(
            tokenizer,
            final_texts,
            args,
            train_path,
            cache_scope="final",
        )
        token_sec = time.perf_counter() - token_start
        avg_len = sum(final_lengths) / max(1, len(final_lengths))
        print(f"token lengths avg={avg_len:.1f} max={max(final_lengths) if final_lengths else 0}")
        if args.cache_only:
            print("cache-only requested; skipping final refit")
            return

        train_start = time.perf_counter()
        final_teacher, final_consensus, final_relational = build_training_signals(
            final_samples, final_y, final_idx, args
        )
        final_model = train_model(
            tokenizer,
            final_encoded_features,
            final_lengths,
            final_y,
            final_sample_weights,
            final_idx,
            args,
            device,
            teacher=final_teacher,
            consensus=final_consensus,
            relational=final_relational,
        )
        train_sec = time.perf_counter() - train_start
        artifact_bias, source_metrics = load_class_bias_artifact(args.class_bias_artifact)
        if artifact_bias is None:
            artifact_bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
            source_metrics = {"macro_f1": 0.0}
        save_hf_artifact(final_model, tokenizer, args.output_dir, artifact_bias, args, source_metrics)
        artifact_size = dir_size_mb(args.output_dir)
        runtime = {
            "tokenize_sec": token_sec,
            "train_sec": train_sec,
            "eval_sec": 0.0,
            "total_sec": time.perf_counter() - run_start,
        }
        experiment_id = experiment_id_for(args)
        metric_value = source_metrics.get("macro_f1", "")
        weak = ""
        if source_metrics.get("per_class_f1"):
            weak = summarize_weak_classes(source_metrics)
        append_results_csv(
            Path("experiments/results.csv"),
            {
                "experiment_id": experiment_id,
                "model_family": (
                    "weak4_lora_specialist_final_refit"
                    if args.train_label_filter == "weak4"
                    else "torch_gpu_transformer_final_refit"
                ),
                "base_model": args.base_model,
                "features": "serialized prompt/action/workspace text",
                "serializer_name": args.serializer,
                "split_type": "final_refit",
                "seed": args.seed,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "class_weight_power": args.class_weight_power,
                "label_smoothing": args.label_smoothing,
                "replay_mode": args.replay_mode,
                "replay_size": final_replay_size,
                "macro_f1": f"{metric_value:.6f}" if isinstance(metric_value, (float, int)) else "",
                "weakest_classes": weak,
                "artifact_path": args.output_dir,
                "runtime_sec": f"{runtime['total_sec']:.3f}",
                "artifact_size_mb": f"{artifact_size:.3f}",
                "train_command": " ".join(shlex.quote(part) for part in sys.argv),
                "notes": args.notes or "final refit",
                "decision": (
                    "final Weak4 LoRA refit complete; sparse is forbidden; build with the screen tuner report"
                    if args.train_label_filter == "weak4"
                    else "final transformer refit complete; requires sparse artifact and package smoke test"
                ),
            },
        )
        metrics_path = Path("experiments/artifacts") / f"{experiment_id}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": experiment_id,
                    "metrics_source": args.class_bias_artifact,
                    "source_metrics": source_metrics,
                    "class_bias": dict(zip(ALL_CLASSES, [float(x) for x in artifact_bias.tolist()])),
                    "device": str(device),
                    "runtime": runtime,
                    "amp_optimizer_steps": getattr(args, "amp_step_stats", None),
                    "serializer_name": args.serializer,
                    "terminal_token": safe_text(getattr(args, "terminal_token", "")),
                    "text_cache_path": str(text_cache_path),
                    "token_cache_path": str(token_cache_path),
                    "replay_size": final_replay_size,
                    "replay_meta_mode": getattr(args, "replay_meta_mode", "current"),
                    "replay_predecessor_audit": getattr(
                        args, "replay_predecessor_audit", None
                    ),
                    "replay_kd": getattr(args, "replay_kd_audit", None),
                    "artifact_size_mb": artifact_size,
                    "rule_boosts_path": args.rule_boosts_path,
                    "explorer4_loss_weight": args.explorer4_loss_weight,
                    "explorer4_loss_balance": args.explorer4_loss_balance,
                    "consensus_reliability": getattr(
                        args, "consensus_reliability_meta", None
                    ),
                    "relational_kd": getattr(args, "relational_kd_meta", None),
                    "action_margin_kd": getattr(args, "action_margin_kd_meta", None),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        lines = [
            f"## {experiment_id}",
            "",
            f"- Date/time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "- Validation setup: final refit on all labeled rows; no validation rows used.",
            f"- Code/config changes: `{args.base_model}`, serializer={args.serializer}, replay={args.replay_mode}, max_length={args.max_length}, epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, explorer4_loss={args.explorer4_loss_weight}, explorer4_balance={args.explorer4_loss_balance}.",
            f"- OOF class-bias source: {args.class_bias_artifact or 'none'}",
            f"- Rule boosts source: {args.rule_boosts_path or 'none'}",
            f"- Runtime or package-size concerns: runtime={runtime['total_sec']:.1f}s, tokenize={runtime['tokenize_sec']:.1f}s, train={runtime['train_sec']:.1f}s, artifact_size_mb={artifact_size:.1f}.",
            (
                "- Decision: final Weak4 LoRA refit complete; use the fixed screen tuner config and no sparse stack."
                if args.train_label_filter == "weak4"
                else "- Decision: final transformer refit complete; next train final sparse SVC artifact and smoke-test offline submission package."
            ),
            "",
        ]
        if not args.no_research_log:
            with Path("research_log.md").open("a", encoding="utf-8") as f:
                f.write("\n".join(lines))
        print(f"saved final HF artifact: {args.output_dir} artifact_size_mb={artifact_size:.1f}")
        return

    train_idx, val_idx = split_for_args(samples, y, args)
    assert_validation_anchor(samples, y, train_idx, val_idx, args)
    full_val_count = len(val_idx)
    val_idx = select_balanced_subset(val_idx, y, args.quick_val_size, args.seed + 17)
    samples, y, train_idx, sample_weights, replay_size = add_replay_examples(samples, y, train_idx, args)
    train_idx = filter_train_indices(train_idx, y, args.train_label_filter)
    sample_weights = apply_sim_early_turn_loss_scale(
        samples,
        y,
        train_idx,
        sample_weights,
        args,
    )
    fold_text = f" fold={args.fold_id}/{args.n_folds}" if args.split == "session_oof" else ""
    print(f"split={args.split}{fold_text} train={len(train_idx)} val={len(val_idx)} full_val={full_val_count}")

    tokenizer = load_tokenizer_for_args(args)
    token_start = time.perf_counter()
    train_cache_scope = "train"
    if args.split == "session_oof":
        train_cache_scope = f"oof-fold{args.fold_id}-of{args.n_folds}"
    texts, text_cache_path = build_serialized_texts(
        samples,
        args,
        train_path,
        cache_scope=train_cache_scope,
        tokenizer=tokenizer,
    )
    encoded_features, lengths, token_cache_path = tokenize_texts(tokenizer, texts, args, train_path, cache_scope=train_cache_scope)
    token_sec = time.perf_counter() - token_start
    avg_len = sum(lengths) / max(1, len(lengths))
    print(f"token lengths avg={avg_len:.1f} max={max(lengths) if lengths else 0}")
    if args.cache_only:
        print("cache-only requested; skipping training")
        return

    train_start = time.perf_counter()
    teacher, consensus, relational = build_training_signals(
        samples, y, train_idx, args
    )
    model = train_model(
        tokenizer, encoded_features, lengths, y, sample_weights, train_idx, args, device,
        teacher=teacher,
        consensus=consensus,
        relational=relational,
    )
    train_sec = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    logits, y_val, raw_metrics, ordered_val_idx = evaluate(model, tokenizer, encoded_features, lengths, y, val_idx, args, device)
    eval_sec = time.perf_counter() - eval_start
    print(f"  raw_macro_f1={raw_metrics['macro_f1']:.6f}")

    weak4_conditional = None
    if args.train_label_filter == "weak4":
        weak4_conditional = weak4_conditional_metrics(logits, y_val)
        print(
            "  weak4_conditional "
            f"rows={weak4_conditional['n_true_weak4']} "
            f"macro_f1={weak4_conditional['macro_f1']:.6f}"
        )
        print("  weak4 conditional confusion (rows=true, cols=pred):")
        print(f"    {'':18s} " + " ".join(f"{label[:8]:>8s}" for label in WEAK4_CLASSES))
        for label, row in zip(WEAK4_CLASSES, weak4_conditional["confusion_4x4"]):
            print(f"    {label:18s} " + " ".join(f"{value:8d}" for value in row))

    bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
    old_bias = None
    old_bias_metrics = None
    metrics = raw_metrics
    raw_explorer4_metrics = explorer4_metrics_from_logits(logits, y_val, bias, full_metrics=raw_metrics)
    old_bias_explorer4_metrics = None
    if args.tune_bias:
        print("  tuning old class bias")
        old_bias, _ = tune_class_bias(logits, y_val, rounds=2)
        old_pred = predict_with_bias(logits, old_bias)
        old_bias_metrics = f1_metrics(y_val, old_pred)
        old_bias_explorer4_metrics = explorer4_metrics(y_val, old_pred, full_metrics=old_bias_metrics)
        print(f"  old_tuned_macro_f1={old_bias_metrics['macro_f1']:.6f}")
        print("  tuning 2-stage class bias")
        bias, _ = tune_class_bias_two_stage(
            logits,
            y_val,
            initial_bias=old_bias,
            initial_best=old_bias_metrics["macro_f1"],
            fine_rounds=2,
        )
        pred = predict_with_bias(logits, bias)
        metrics = f1_metrics(y_val, pred)
        print(f"  tuned_2stage_macro_f1={metrics['macro_f1']:.6f}")
    final_explorer4_metrics = explorer4_metrics_from_logits(logits, y_val, bias, full_metrics=metrics)
    print(
        "  explorer4 "
        f"raw={raw_explorer4_metrics['explorer4_macro_f1']:.6f} "
        f"final={final_explorer4_metrics['explorer4_macro_f1']:.6f} "
        f"sum={final_explorer4_metrics['explorer4_sum_f1']:.6f}"
    )
    report_raw_metrics = weak4_conditional if weak4_conditional is not None else raw_metrics
    report_metrics = weak4_conditional if weak4_conditional is not None else metrics

    experiment_id = experiment_id_for(args)
    val_logits_path = save_val_logits(
        experiment_id,
        logits,
        y_val,
        ordered_val_idx,
        samples,
        bias,
        raw_metrics,
        metrics,
        args,
        old_bias=old_bias,
        old_bias_metrics=old_bias_metrics,
        weak4_conditional=weak4_conditional,
    )
    runtime = {
        "tokenize_sec": token_sec,
        "train_sec": train_sec,
        "eval_sec": eval_sec,
        "total_sec": time.perf_counter() - run_start,
    }
    if args.quick_val_size:
        decision = "screening only; require full fixed-session validation"
    elif args.split == "session_oof":
        decision = "oof fold complete; aggregate before decision"
    else:
        decision = "keep as GPU candidate" if report_metrics["macro_f1"] >= args.keep_threshold else "discard or revisit"

    artifact_size = 0.0
    if args.final_model:
        print("training final transformer on all training rows")
        if args.replay_mode == "none":
            final_encoded_features = encoded_features
            final_lengths = lengths
            final_y = y
            final_sample_weights = [1.0] * len(samples)
            final_idx = list(range(len(samples)))
            final_train_samples = samples
        else:
            final_samples, final_y, final_idx, final_sample_weights, final_replay_size = add_replay_examples(
                base_samples,
                base_y,
                list(range(len(base_samples))),
                args,
            )
            final_texts, _ = build_serialized_texts(
                final_samples,
                args,
                train_path,
                cache_scope="final",
                tokenizer=tokenizer,
            )
            final_encoded_features, final_lengths, _ = tokenize_texts(tokenizer, final_texts, args, train_path, cache_scope="final")
            print(f"final replay_size={final_replay_size}")
            final_train_samples = final_samples
        final_idx = filter_train_indices(final_idx, final_y, args.train_label_filter)
        final_sample_weights = apply_sim_early_turn_loss_scale(
            final_train_samples,
            final_y,
            final_idx,
            final_sample_weights,
            args,
        )
        final_teacher, final_consensus, final_relational = build_training_signals(
            final_train_samples, final_y, final_idx, args
        )
        final_model = train_model(
            tokenizer,
            final_encoded_features,
            final_lengths,
            final_y,
            final_sample_weights,
            final_idx,
            args,
            device,
            teacher=final_teacher,
            consensus=final_consensus,
            relational=final_relational,
        )
        save_hf_artifact(final_model, tokenizer, args.output_dir, bias, args, report_metrics)
        artifact_size = dir_size_mb(args.output_dir)
        print(f"saved HF artifact: {args.output_dir}")
    elif args.save_val_model:
        save_hf_artifact(model, tokenizer, args.output_dir, bias, args, report_metrics)
        artifact_size = dir_size_mb(args.output_dir)
        print(f"saved val-split HF artifact: {args.output_dir}")
    elif Path(args.output_dir).exists():
        artifact_size = dir_size_mb(args.output_dir)

    split_type = args.split
    if args.split == "session_oof":
        split_type = f"session_oof_fold{args.fold_id}_of{args.n_folds}"
    if args.quick_val_size:
        split_type = f"{split_type}_quick{args.quick_val_size}"

    append_results_csv(
        Path("experiments/results.csv"),
        {
            "experiment_id": experiment_id,
            "model_family": "weak4_lora_specialist" if args.train_label_filter == "weak4" else "torch_gpu_transformer",
            "base_model": args.base_model,
            "features": "serialized prompt/action/workspace text",
            "serializer_name": args.serializer,
            "split_type": split_type,
            "seed": args.seed,
            "fold_id": f"{args.fold_id}/{args.n_folds}" if args.split == "session_oof" else "",
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "class_weight_power": args.class_weight_power,
            "label_smoothing": args.label_smoothing,
            "replay_mode": args.replay_mode,
            "replay_size": replay_size,
            "macro_f1_raw": f"{report_raw_metrics['macro_f1']:.6f}",
            "macro_f1_bias_tuned": f"{old_bias_metrics['macro_f1']:.6f}" if old_bias_metrics else "",
            "macro_f1_bias_tuned_2stage": f"{metrics['macro_f1']:.6f}" if args.tune_bias else "",
            "macro_f1": f"{report_metrics['macro_f1']:.6f}",
            "weakest_classes": summarize_weak_classes(report_metrics),
            "top_confusions": json.dumps(report_metrics["top_confusions"][:8], ensure_ascii=False),
            "prediction_distribution": json.dumps(report_metrics["prediction_distribution"], ensure_ascii=False, sort_keys=True),
            "artifact_path": args.output_dir if args.final_model else "",
            "val_logits_path": val_logits_path,
            "test_logits_path": "",
            "inference_time_sec": "",
            "runtime_sec": f"{runtime['total_sec']:.3f}",
            "artifact_size_mb": f"{artifact_size:.3f}",
            "train_command": " ".join(shlex.quote(part) for part in sys.argv),
            "notes": args.notes or args.base_model,
            "decision": decision,
        },
    )
    if not args.no_research_log:
        append_log(
            experiment_id,
            args,
            report_raw_metrics,
            report_metrics,
            decision,
            runtime,
            val_logits_path,
            artifact_size,
            old_bias_metrics if weak4_conditional is None else None,
        )

    metrics_path = Path("experiments/artifacts") / f"{experiment_id}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "raw_metrics": raw_metrics,
                "old_bias_metrics": old_bias_metrics,
                "metrics": metrics,
                "raw_explorer4_metrics": raw_explorer4_metrics,
                "old_bias_explorer4_metrics": old_bias_explorer4_metrics,
                "explorer4_metrics": final_explorer4_metrics,
                "weak4_conditional_metrics": weak4_conditional,
                "class_bias": dict(zip(ALL_CLASSES, [float(x) for x in bias.tolist()])),
                "old_class_bias": dict(zip(ALL_CLASSES, [float(x) for x in old_bias.tolist()])) if old_bias is not None else None,
                "device": str(device),
                "runtime": runtime,
                "serializer_name": args.serializer,
                "terminal_token": safe_text(getattr(args, "terminal_token", "")),
                "text_cache_path": str(text_cache_path),
                "token_cache_path": str(token_cache_path),
                "val_logits_path": val_logits_path,
                "validation_rows": len(val_idx),
                "full_validation_rows": full_val_count,
                "fold_id": args.fold_id if args.split == "session_oof" else None,
                "n_folds": args.n_folds if args.split == "session_oof" else None,
                "replay_size": replay_size,
                "replay_meta_mode": getattr(args, "replay_meta_mode", "current"),
                "replay_sample_weight": args.replay_sample_weight,
                "replay_predecessor_audit": getattr(
                    args, "replay_predecessor_audit", None
                ),
                "replay_kd": getattr(args, "replay_kd_audit", None),
                "train_label_filter": args.train_label_filter,
                "explorer4_loss_weight": args.explorer4_loss_weight,
                "explorer4_loss_balance": args.explorer4_loss_balance,
                "consensus_reliability": getattr(
                    args, "consensus_reliability_meta", None
                ),
                "relational_kd": getattr(args, "relational_kd_meta", None),
                "action_margin_kd": getattr(args, "action_margin_kd_meta", None),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("  weakest classes:")
    for label, score in sorted(report_metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:8]:
        print(f"    {label:18s} {score:.4f}")
    print("  top confusions:")
    for count, true_label, pred_label in report_metrics["top_confusions"][:10]:
        print(f"    {true_label:18s} -> {pred_label:18s} {count}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--base-model", default="distilbert-base-multilingual-cased")
    parser.add_argument("--serializer", choices=["current_v1", "chat_v1_contract", "weak_nav_v1", "weak_nav_paths_v1", "current_v2", "current_v5", "current_v6", "current_v6e", "current_v7", "current_v7r", "current_v7rl", "current_v7rm", "current_v7rd", "current_v7rb", "current_v7rc", "current_v7rw", "current_v7rg", "current_v7rcgw", "current_v8", "current_v8t", "current_v9o", "current_v9f", "current_v9h", "current_v10", "current_v11s", "state_v2", "recent_pairs_v1", "compact_events_v1", "hybrid_v1"], default="current_v1")
    parser.add_argument(
        "--terminal-token",
        default="",
        help=(
            "reserve the final sequence position and append this exact single, non-pad "
            "token after content truncation (for decoder last-token pooling)"
        ),
    )
    parser.add_argument("--split", choices=["random", "session", "session_oof"], default="session")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--distill-logits", default=None,
                        help="OOF teacher payload (.pt with ids + log-prob logits); rows matched by id, replay rows get no KD term")
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-alpha-weak", type=float, default=None,
                        help="KD alpha override for teacher-matched original rows whose true label is "
                             "Weak4 (list_directory/read_file/grep_search/glob_pattern); other matched "
                             "rows keep --distill-alpha (team condalpha option)")
    parser.add_argument(
        "--distill-alpha-weak-consensus-lambda",
        type=float,
        default=0.0,
        help=(
            "class-centered consensus reliability slope for matched original "
            "Weak4 rows; preserves both mean Weak4 alpha and per-class nominal "
            "hard-backbone mass after the existing consensus sieve"
        ),
    )
    parser.add_argument(
        "--distill-alpha-c0",
        type=float,
        default=None,
        help=(
            "KD alpha floor for teacher-matched rows whose ID-aligned consensus "
            "correct_count is zero; applied after the Weak4 override"
        ),
    )
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument(
        "--action-margin-kd-weight",
        type=float,
        default=0.0,
        help=(
            "fixed weight for teacher top-k true-vs-negative margin KD; 0 disables "
            "unless --action-margin-kd-target-grad-ratio is active"
        ),
    )
    parser.add_argument(
        "--action-margin-kd-target-grad-ratio",
        type=float,
        default=0.0,
        help=(
            "calibrate one fixed action-margin weight on the first covered batch "
            "to this auxiliary/base logit-gradient norm ratio"
        ),
    )
    parser.add_argument(
        "--action-margin-kd-topk",
        type=int,
        default=3,
        help="number of detached-teacher hard negatives used by action-margin KD",
    )
    parser.add_argument(
        "--action-margin-kd-label-scope",
        choices=("all", "weak4"),
        default="all",
        help=(
            "true-label scope for action-margin KD only; ordinary logit KD remains "
            "unchanged on every teacher-covered row"
        ),
    )
    parser.add_argument(
        "--relational-teacher-hidden",
        default="",
        help=(
            "fp16 pooled-classifier-hidden cache for training-only relational KD; "
            "must cover every original row exactly and excludes replay"
        ),
    )
    parser.add_argument(
        "--relational-teacher-base-model",
        default="",
        help="exact teacher base_model expected in --relational-teacher-hidden",
    )
    parser.add_argument(
        "--relational-teacher-max-length",
        type=int,
        default=0,
        help="exact teacher export max_length expected in the hidden cache",
    )
    parser.add_argument(
        "--relational-teacher-terminal-token",
        default="",
        help="exact terminal token expected in the teacher hidden cache (empty by default)",
    )
    parser.add_argument(
        "--relational-kd-weight",
        type=float,
        default=0.0,
        help=(
            "weight for one-minus centered cosine-Gram alignment on matched "
            "original rows (0 disables with no forward-path change)"
        ),
    )
    parser.add_argument(
        "--consensus-reliability",
        default="",
        help=(
            "OOF correctness-consensus artifact; keeps full hard-label gradient on "
            "the classifier head while gating only the backbone gradient"
        ),
    )
    parser.add_argument(
        "--consensus-backbone-weights",
        default="",
        help=(
            "optional comma-separated c=0..N backbone scales; default uses the "
            "weights embedded in --consensus-reliability"
        ),
    )
    parser.add_argument(
        "--no-consensus-class-normalize",
        dest="consensus_class_normalize",
        action="store_false",
        help=(
            "do not normalize raw consensus scales to mean 1 inside each true "
            "class on the training split"
        ),
    )
    parser.set_defaults(consensus_class_normalize=True)
    parser.add_argument(
        "--consensus-kd-weights",
        default=None,
        help=(
            "apply the consensus sieve to the KD branch too: per-c-bin KD "
            "backbone-gradient scales (e.g. '0,0.25,0.75,1'), class-normalized "
            "like the hard-branch weights; KD head gradient stays full. "
            "Requires --consensus-reliability and --distill-logits. Unset "
            "keeps the KD branch untouched (bit-identical to prior behavior)"
        ),
    )
    parser.add_argument("--train-label-filter", choices=["none", "weak4"], default="none",
                        help="restrict optimizer rows to canonical Weak4 labels and train with conditional 4-way loss")
    parser.add_argument("--assert-val-ids", default="",
                        help="validation-logits payload whose classes/split/seed/id set must match before training")
    parser.add_argument("--explorer4-loss-weight", type=float, default=0.0,
                        help="extra plain CE on true list/read/grep/glob rows, restricted to those four logits")
    parser.add_argument("--explorer4-loss-balance", action="store_true",
                        help="normalize existing main class weights inside the Explorer4 auxiliary CE")
    parser.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw",
                        help="adamw8bit (bitsandbytes) fits 0.6B training in 8GB VRAM")
    parser.add_argument("--bf16", action="store_true",
                        help="load weights and autocast in bfloat16, GradScaler off (no fp32 master "
                             "copy — required for 9B-class full FT on a single 80-96GB GPU; "
                             "default fp32+fp16-autocast path is unchanged without this flag)")
    parser.add_argument("--rdrop-alpha", type=float, default=0.0,
                        help="R-Drop: weight of the symmetric KL between two dropout-perturbed forward passes (0 disables; ~2x train time when on)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="override model dropout probs (attention_dropout etc.) at load; decoder configs default to 0.0, required for --rdrop-alpha to bite")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp-init-scale", type=float, default=65536.0)
    parser.add_argument("--amp-growth-interval", type=int, default=2000)
    parser.add_argument(
        "--require-zero-amp-skips",
        action="store_true",
        help="fail after training if CUDA GradScaler skipped any optimizer step",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--final-model", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--save-val-model", action="store_true",
                        help="save the val-split-trained model (no refit) as a submittable HF artifact")
    parser.add_argument("--epoch-checkpoint-dir", default="",
                        help="overwrite this dir with an fp16 snapshot after every epoch (crash insurance; use a Drive path on Colab)")
    parser.add_argument("--snapshot-epoch-checkpoints", action="store_true",
                        help="after each epoch checkpoint, also copy it to <epoch-checkpoint-dir>_epN")
    parser.add_argument("--resume-from", default="",
                        help="resume from an epoch-checkpoint dir; scheduler/RNG skip past completed epochs, optimizer state restarts")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="optimizer step every N batches (effective batch = batch-size x N); matches teammate recipe batch4 x accum4")
    parser.add_argument("--lora-r", type=int, default=0,
                        help="LoRA rank for teacher training (0=full fine-tune)")
    parser.add_argument("--model-class", choices=["auto", "qwen35text", "gemma3text", "gemma4custom"], default="auto",
                        help="text-only/custom sequence-classification class override")
    parser.add_argument("--save-fp16", action="store_true")
    parser.add_argument("--output-dir", default="model")
    parser.add_argument("--rule-boosts-path", default="")
    parser.add_argument("--class-bias-artifact", default="")
    parser.add_argument("--keep-threshold", type=float, default=0.60)
    parser.add_argument("--notes", default="")
    parser.add_argument("--experiment-suffix", default="")
    parser.add_argument("--no-research-log", action="store_true")
    parser.add_argument("--cache-dir", default="experiments/cache")
    parser.add_argument("--no-text-cache", action="store_true")
    parser.add_argument("--no-token-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--tokenize-batch-size", type=int, default=4096)
    parser.add_argument("--slow-tokenizer", action="store_true")
    parser.add_argument("--quick-val-size", type=int, default=0)
    parser.add_argument("--replay-mode", choices=["none", "last1", "last2"], default="none")
    parser.add_argument("--max-replay-samples", type=int, default=20000)
    parser.add_argument("--replay-sample-weight", type=float, default=0.5)
    parser.add_argument(
        "--replay-meta-mode",
        choices=["current", "predecessor"],
        default="current",
        help=(
            "metadata source for replay rows: current preserves the historical "
            "recipe; predecessor requires an exact same-session/step original "
            "row and drops unmatched candidates before the balanced cap"
        ),
    )
    parser.add_argument(
        "--replay-kd-source",
        choices=["none", "predecessor"],
        default="none",
        help=(
            "after the unchanged legacy replay cap, optionally inherit the exact "
            "predecessor original row's teacher target on trusted consensus rows; "
            "does not change replay metadata, text, IDs, or order"
        ),
    )
    parser.add_argument(
        "--replay-kd-min-consensus",
        type=int,
        default=2,
        help=(
            "minimum predecessor OOF correct_count for --replay-kd-source "
            "predecessor (checked against artifact model_count)"
        ),
    )
    parser.add_argument(
        "--replay-kd-expected-predecessors",
        type=int,
        default=-1,
        help="fail unless this many selected replay rows have exact predecessors; -1 disables",
    )
    parser.add_argument(
        "--replay-kd-expected-unmatched",
        type=int,
        default=-1,
        help="fail unless this many selected replay rows remain unmatched; -1 disables",
    )
    parser.add_argument(
        "--replay-kd-expected-active",
        type=int,
        default=-1,
        help="fail unless this many replay rows pass the consensus threshold; -1 disables",
    )
    parser.add_argument(
        "--replay-kd-expected-replay-id-sha256",
        default="",
        help="optional fail-closed SHA256 of the ordered selected replay ID list",
    )
    parser.add_argument(
        "--sim-early-turn-loss-scale",
        type=float,
        default=1.0,
        help=(
            "raw whole-loss multiplier for non-replay sess_sim rows at turn_index<=2; "
            "normalized to mean 1 inside each SIM true class (1 disables)"
        ),
    )
    parser.add_argument("--bucket-multiplier", type=int, default=8)
    parser.add_argument("--eval-bucket-multiplier", type=int, default=50)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--save-val-logits", dest="save_val_logits", action="store_true", default=True)
    parser.add_argument("--no-save-val-logits", dest="save_val_logits", action="store_false")
    parser.add_argument("--logits-dir", default="experiments/logits")
    args = parser.parse_args()
    if (
        not math.isfinite(args.sim_early_turn_loss_scale)
        or not 0.0 < args.sim_early_turn_loss_scale <= 1.0
    ):
        parser.error("--sim-early-turn-loss-scale must be finite and in (0, 1]")
    if args.replay_meta_mode == "predecessor" and args.replay_mode == "none":
        parser.error("--replay-meta-mode predecessor requires --replay-mode last1 or last2")
    replay_kd_requested = args.replay_kd_source != "none"
    if replay_kd_requested:
        if args.replay_mode == "none":
            parser.error("--replay-kd-source predecessor requires replay augmentation")
        if args.replay_meta_mode != "current":
            parser.error(
                "--replay-kd-source predecessor requires --replay-meta-mode current"
            )
        if args.serializer != "current_v1":
            parser.error(
                "--replay-kd-source predecessor is registered only for --serializer current_v1"
            )
        if not args.distill_logits:
            parser.error("--replay-kd-source predecessor requires --distill-logits")
        if not args.consensus_reliability:
            parser.error(
                "--replay-kd-source predecessor requires --consensus-reliability"
            )
        if not math.isfinite(args.distill_alpha) or not 0.0 < args.distill_alpha <= 1.0:
            parser.error(
                "--replay-kd-source predecessor requires --distill-alpha in (0, 1]"
            )
        if args.distill_alpha_weak is None:
            parser.error(
                "--replay-kd-source predecessor requires --distill-alpha-weak"
            )
        if args.replay_kd_min_consensus < 1:
            parser.error("--replay-kd-min-consensus must be >= 1")
        if args.relational_teacher_hidden or args.relational_kd_weight > 0.0:
            parser.error(
                "trusted replay KD cannot be combined with hidden-Gram relational KD"
            )
        if args.action_margin_kd_weight > 0.0 or args.action_margin_kd_target_grad_ratio > 0.0:
            parser.error(
                "trusted replay KD cannot be combined with action-margin KD in this card"
            )
    for name in (
        "replay_kd_expected_predecessors",
        "replay_kd_expected_unmatched",
        "replay_kd_expected_active",
    ):
        if int(getattr(args, name)) < -1:
            parser.error(f"--{name.replace('_', '-')} must be -1 or nonnegative")
    expected_replay_sha = safe_text(args.replay_kd_expected_replay_id_sha256).lower()
    if expected_replay_sha and not re.fullmatch(r"[0-9a-f]{64}", expected_replay_sha):
        parser.error(
            "--replay-kd-expected-replay-id-sha256 must be 64 lowercase/uppercase hex characters"
        )
    if not replay_kd_requested and (
        any(
            int(getattr(args, name)) >= 0
            for name in (
                "replay_kd_expected_predecessors",
                "replay_kd_expected_unmatched",
                "replay_kd_expected_active",
            )
        )
        or expected_replay_sha
    ):
        parser.error("replay KD expected-value assertions require --replay-kd-source")
    if args.rdrop_alpha > 0 and not args.dropout:
        parser.error(
            "--rdrop-alpha > 0 requires --dropout > 0: decoder configs default all dropout "
            "to 0.0, so both R-Drop passes would be identical (KL=0) at 2x train cost"
        )
    if args.explorer4_loss_weight < 0:
        parser.error("--explorer4-loss-weight must be >= 0")
    if not math.isfinite(args.action_margin_kd_weight) or args.action_margin_kd_weight < 0.0:
        parser.error("--action-margin-kd-weight must be finite and >= 0")
    if (
        not math.isfinite(args.action_margin_kd_target_grad_ratio)
        or args.action_margin_kd_target_grad_ratio < 0.0
    ):
        parser.error("--action-margin-kd-target-grad-ratio must be finite and >= 0")
    if args.action_margin_kd_weight > 0.0 and args.action_margin_kd_target_grad_ratio > 0.0:
        parser.error(
            "--action-margin-kd-weight and --action-margin-kd-target-grad-ratio "
            "are mutually exclusive"
        )
    if not 1 <= args.action_margin_kd_topk < len(ALL_CLASSES):
        parser.error(
            f"--action-margin-kd-topk must be in [1, {len(ALL_CLASSES) - 1}]"
        )
    action_margin_requested = (
        args.action_margin_kd_weight > 0.0
        or args.action_margin_kd_target_grad_ratio > 0.0
    )
    if action_margin_requested:
        if not args.distill_logits:
            parser.error("action-margin KD requires --distill-logits")
        if not math.isfinite(args.distill_temp) or args.distill_temp <= 0.0:
            parser.error("action-margin KD requires --distill-temp > 0")
        if args.relational_teacher_hidden or args.relational_kd_weight > 0.0:
            parser.error("action-margin KD cannot be combined with hidden-Gram relational KD")
        if args.rdrop_alpha > 0.0:
            parser.error("action-margin KD does not support --rdrop-alpha")
        if args.train_label_filter != "none":
            parser.error("action-margin KD does not support --train-label-filter")
    if not math.isfinite(args.relational_kd_weight) or args.relational_kd_weight < 0.0:
        parser.error("--relational-kd-weight must be finite and >= 0")
    relational_requested = bool(args.relational_teacher_hidden) or args.relational_kd_weight > 0.0
    if relational_requested:
        if args.relational_kd_weight <= 0.0:
            parser.error(
                "--relational-teacher-hidden requires --relational-kd-weight > 0"
            )
        if not args.relational_teacher_hidden:
            parser.error(
                "--relational-kd-weight > 0 requires --relational-teacher-hidden"
            )
        if not args.relational_teacher_base_model:
            parser.error(
                "relational KD requires --relational-teacher-base-model"
            )
        if args.relational_teacher_max_length <= 0:
            parser.error(
                "relational KD requires --relational-teacher-max-length > 0"
            )
        if not args.distill_logits:
            parser.error(
                "relational KD requires --distill-logits so replay/unmatched "
                "coverage can be checked against the ordinary KD mask"
            )
        if args.rdrop_alpha > 0:
            parser.error("relational KD does not support --rdrop-alpha")
        if args.train_label_filter != "none":
            parser.error("relational KD does not support --train-label-filter")
    if args.distill_alpha_weak is not None:
        if not args.distill_logits:
            parser.error("--distill-alpha-weak requires --distill-logits")
        if args.distill_alpha <= 0:
            parser.error("--distill-alpha-weak requires --distill-alpha > 0 (mask carries alpha_weak/alpha)")
    if (
        not math.isfinite(args.distill_alpha_weak_consensus_lambda)
        or args.distill_alpha_weak_consensus_lambda < 0.0
    ):
        parser.error(
            "--distill-alpha-weak-consensus-lambda must be finite and >= 0"
        )
    if args.distill_alpha_weak_consensus_lambda > 0.0:
        if not args.distill_logits:
            parser.error(
                "--distill-alpha-weak-consensus-lambda requires --distill-logits"
            )
        if args.distill_alpha_weak is None:
            parser.error(
                "--distill-alpha-weak-consensus-lambda requires --distill-alpha-weak"
            )
        if (
            not math.isfinite(args.distill_alpha_weak)
            or not 0.0 < args.distill_alpha_weak < 1.0
        ):
            parser.error(
                "--distill-alpha-weak-consensus-lambda requires a finite "
                "--distill-alpha-weak in (0, 1)"
            )
        if not args.consensus_reliability:
            parser.error(
                "--distill-alpha-weak-consensus-lambda requires "
                "--consensus-reliability"
            )
        if not args.consensus_class_normalize:
            parser.error(
                "--distill-alpha-weak-consensus-lambda requires class-normalized "
                "consensus scales"
            )
        if not math.isfinite(args.distill_alpha) or args.distill_alpha <= 0.0:
            parser.error(
                "--distill-alpha-weak-consensus-lambda requires --distill-alpha > 0"
            )
        if args.distill_alpha_c0 is not None:
            parser.error(
                "--distill-alpha-weak-consensus-lambda cannot be combined with "
                "--distill-alpha-c0"
            )
    if args.distill_alpha_c0 is not None:
        if not math.isfinite(args.distill_alpha_c0) or not 0.0 <= args.distill_alpha_c0 <= 1.0:
            parser.error("--distill-alpha-c0 must be a finite value in [0, 1]")
        if not args.distill_logits:
            parser.error("--distill-alpha-c0 requires --distill-logits")
        if not args.consensus_reliability:
            parser.error("--distill-alpha-c0 requires --consensus-reliability")
        if not math.isfinite(args.distill_alpha) or args.distill_alpha <= 0:
            parser.error("--distill-alpha-c0 requires --distill-alpha > 0")
    if args.consensus_backbone_weights and not args.consensus_reliability:
        parser.error("--consensus-backbone-weights requires --consensus-reliability")
    if not args.consensus_class_normalize and not args.consensus_reliability:
        parser.error("--no-consensus-class-normalize requires --consensus-reliability")
    if args.consensus_backbone_weights:
        try:
            parse_consensus_backbone_weights(args.consensus_backbone_weights)
        except ValueError as exc:
            parser.error(str(exc))
    if args.consensus_kd_weights:
        if not args.consensus_reliability:
            parser.error("--consensus-kd-weights requires --consensus-reliability")
        if not args.distill_logits:
            parser.error("--consensus-kd-weights requires --distill-logits")
        try:
            parse_consensus_backbone_weights(args.consensus_kd_weights)
        except ValueError as exc:
            parser.error(str(exc))
    if args.consensus_reliability and args.rdrop_alpha > 0:
        parser.error(
            "--consensus-reliability does not support --rdrop-alpha; the sieve "
            "is intentionally isolated to the hard-label branch"
        )
    if args.consensus_reliability and args.train_label_filter != "none":
        parser.error("--consensus-reliability does not support --train-label-filter")
    if args.train_label_filter != "none" and args.replay_mode != "none":
        parser.error("--train-label-filter requires --replay-mode none")
    if args.train_label_filter != "none" and args.distill_logits:
        parser.error("--train-label-filter does not support --distill-logits")
    if args.train_label_filter == "weak4":
        if args.lora_r != 16:
            parser.error("--train-label-filter weak4 requires --lora-r 16")
        if not args.save_fp16:
            parser.error("--train-label-filter weak4 requires --save-fp16")
        if args.tune_bias or args.class_bias_artifact or args.rule_boosts_path:
            parser.error("Weak4 specialist training forbids bias/rule post-processing inputs")
        if args.explorer4_loss_weight > 0 or args.explorer4_loss_balance:
            parser.error("Weak4 specialist training cannot be combined with Explorer4 auxiliary loss")
        if args.final_only:
            if args.assert_val_ids:
                parser.error("--final-only has no validation split; omit --assert-val-ids")
        else:
            if not args.assert_val_ids:
                parser.error("Weak4 screen training requires --assert-val-ids")
            if args.quick_val_size:
                parser.error("Weak4 screen validation must keep all 14,001 rows")
            if not args.save_val_model:
                parser.error("Weak4 screen training requires --save-val-model")
            if args.final_model:
                parser.error("run the Weak4 final refit separately with --final-model --final-only")
    return args


if __name__ == "__main__":
    run(parse_args())
