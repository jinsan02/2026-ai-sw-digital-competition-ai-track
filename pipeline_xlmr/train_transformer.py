import argparse
import json
import math
import random
import re
import shlex
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from script import ALL_CLASSES, load_jsonl, safe_text, serialize_transformer_sample
from train import CLASS_TO_ID, append_results_csv, f1_metrics, load_labels, predict_with_bias, session_id, split_indices, tune_class_bias


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
    weights = []
    for class_id in range(len(ALL_CLASSES)):
        weights.append((total / (len(ALL_CLASSES) * max(1, counts[class_id]))) ** power)
    mean = sum(weights) / len(weights)
    return torch.tensor([w / mean for w in weights], dtype=torch.float32, device=device)


def compute_loss(logits, labels, args, weights, sample_w):
    """손실 선택(ce/focal) + replay용 per-sample weight 보존. (노진산)"""
    logits = logits.float()
    if getattr(args, "loss", "ce") == "focal":
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        cw = weights[labels]
        per = -(cw * (1.0 - pt).pow(args.focal_gamma) * logpt)
    else:
        per = F.cross_entropy(
            logits, labels, weight=weights, label_smoothing=args.label_smoothing, reduction="none"
        )
    return (per * sample_w).sum() / torch.clamp(sample_w.sum(), min=1.0)


class FGM:
    """Fast Gradient Method — word embedding에 정규화 gradient 섭동 (노진산)."""

    def __init__(self, model, eps=1.0, emb_name="word_embeddings"):
        self.model = model
        self.eps = eps
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        applied = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)
                    applied += 1
        return applied

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


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
    # split-seed 분리: soup 멤버처럼 학습 seed만 바꾸고 split은 고정해야 할 때 사용
    split_seed = getattr(args, "split_seed", None)
    if split_seed is None:
        split_seed = args.seed
    if args.split == "session_oof":
        return session_oof_split_indices(samples, y, args.n_folds, args.fold_id, split_seed)
    return split_indices(samples, y, args.split, split_seed)


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


def replay_examples_for_sample(sample, pair_limit):
    history = sample.get("history") or []
    candidates = []
    for idx, event in enumerate(history[:-1]):
        if event.get("role") != "user":
            continue
        next_event = history[idx + 1]
        if next_event.get("role") == "assistant_action":
            label = safe_text(next_event.get("name"))
            if label in CLASS_TO_ID:
                candidates.append((idx, event, label))

    replay_samples = []
    for idx, user_event, label in candidates[-pair_limit:]:
        replay_samples.append(
            (
                CLASS_TO_ID[label],
                {
                    "id": f"{safe_text(sample.get('id'))}::replay_{idx}_{label}",
                    "session_meta": sample.get("session_meta") or {},
                    "history": history[:idx],
                    "current_prompt": safe_text(user_event.get("content")),
                },
            )
        )
    return replay_samples


def add_replay_examples(samples, y, train_idx, args):
    sample_weights = [1.0] * len(samples)
    if args.replay_mode == "none":
        return samples, y, train_idx, sample_weights, 0
    if args.split not in ("session", "session_oof"):
        raise ValueError("Replay augmentation is only enabled for session-aware splits to avoid session leakage.")

    pair_limit = {"last1": 1, "last2": 2}[args.replay_mode]
    replay_examples = []
    for sample_idx in train_idx:
        replay_examples.extend(replay_examples_for_sample(samples[sample_idx], pair_limit))
    replay_examples = balanced_cap_replay(replay_examples, args.max_replay_samples, args.seed + 101)

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
        f"cap={args.max_replay_samples} weight={args.replay_sample_weight}"
    )
    return new_samples, new_y, new_train_idx, sample_weights, len(replay_samples)


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
    replay = ""
    if getattr(args, "replay_mode", "none") != "none":
        replay = (
            f"_replay-{safe_slug(args.replay_mode)}-n{args.max_replay_samples}"
            f"-w{safe_slug(args.replay_sample_weight)}-scope-{safe_slug(cache_scope)}"
        )
    return (
        Path(args.cache_dir)
        / f"{kind}_{base_model}_{serializer}{replay}_{source_path.stem}_n{sample_count}_m{stamp}_len{args.max_length}.pt"
    )


def build_serialized_texts(samples, args, source_path, cache_scope="train"):
    path = cache_path(args, source_path, len(samples), "texts", cache_scope)
    if not args.no_text_cache and path.exists() and not args.rebuild_cache:
        payload = torch_load(path)
        if payload.get("serializer_name") == args.serializer and len(payload.get("texts", [])) == len(samples):
            print(f"loaded serialized text cache: {path}")
            return payload["texts"], path

    start = time.perf_counter()
    texts = [serialize_transformer_sample(sample, args.serializer) for sample in samples]
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
        encoded = tokenizer(chunk_texts, padding=False, truncation=True, max_length=args.max_length)
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
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.float16):
                logits = model(**encoded).logits.float()
            logits_parts.append(logits.detach().cpu())
            ordered_indices.extend(batch_idx)
    logits = torch.cat(logits_parts, dim=0)
    y_true = [y[i] for i in ordered_indices]
    pred = torch.argmax(logits, dim=1).tolist()
    metrics = f1_metrics(y_true, pred)
    return logits, y_true, metrics, ordered_indices


def train_model(tokenizer, encoded_features, lengths, y, sample_weights, train_idx, args, device, val_idx=None):
    # soup 멤버용: 분류 헤드 초기화를 init-seed로 고정(같은 basin), 학습 랜덤은 args.seed 유지
    init_seed = getattr(args, "init_seed", None)
    if init_seed is not None:
        torch.manual_seed(init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(init_seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(ALL_CLASSES),
        id2label={i: label for i, label in enumerate(ALL_CLASSES)},
        label2id={label: i for i, label in enumerate(ALL_CLASSES)},
    ).to(device)
    if init_seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    weights = class_weights([y[i] for i in train_idx], device, args.class_weight_power)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    accum = max(1, getattr(args, "grad_accum", 1))
    total_steps = math.ceil(math.ceil(len(train_idx) / args.batch_size) / accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = random.Random(args.seed)
    fgm = FGM(model, eps=args.fgm_eps) if getattr(args, "fgm", False) else None
    fgm_logged = False
    # best-epoch checkpoint 선택 (노진산, 팀 Lane A-3): 에폭별 val 평가 → 최고 에폭 가중치 복원
    select_best = getattr(args, "select_best_epoch", False) and val_idx is not None
    best_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        batches = list(make_batches(train_idx, args.batch_size, rng, lengths, args.bucket_multiplier))
        optimizer.zero_grad(set_to_none=True)
        for step, batch_idx in enumerate(batches, 1):
            labels = torch.tensor([y[i] for i in batch_idx], dtype=torch.long, device=device)
            weights_for_samples = torch.tensor([sample_weights[i] for i in batch_idx], dtype=torch.float32, device=device)
            encoded = make_encoded_batch(tokenizer, encoded_features, batch_idx, args, device)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.float16):
                logits = model(**encoded).logits
                loss = compute_loss(logits, labels, args, weights, weights_for_samples) / accum
            scaler.scale(loss).backward()
            if fgm is not None:
                applied = fgm.attack()
                if not fgm_logged:
                    print(f"    [fgm] perturbed embedding params={applied} eps={args.fgm_eps}")
                    fgm_logged = True
                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.float16):
                    logits_adv = model(**encoded).logits
                    loss_adv = compute_loss(logits_adv, labels, args, weights, weights_for_samples) / accum
                scaler.scale(loss_adv).backward()
                fgm.restore()
            if step % accum == 0 or step == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * accum * len(batch_idx)
            seen += len(batch_idx)
            if args.log_every and step % args.log_every == 0:
                print(f"    step={step:04d} loss={total_loss / max(1, seen):.5f}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"  epoch={epoch:02d} train_loss={total_loss / max(1, seen):.5f}")
        if select_best:
            _, _, epoch_metrics, _ = evaluate(model, tokenizer, encoded_features, lengths, y, val_idx, args, device)
            epoch_f1 = epoch_metrics["macro_f1"]
            print(f"  [epoch {epoch:02d}] val_macro_f1(raw)={epoch_f1:.6f}")
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"  [best-epoch] restored epoch={best_epoch} val_macro_f1={best_f1:.6f}")
    return model


def save_hf_artifact(model, tokenizer, output_dir, class_bias, args, metrics):
    output_dir = Path(output_dir)
    hf_dir = output_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_dir, safe_serialization=True)
    tokenizer.save_pretrained(hf_dir)
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
        "replay_mode": args.replay_mode,
        "replay_sample_weight": args.replay_sample_weight,
        "base_model": args.base_model,
        "trained_with_cuda": torch.cuda.is_available(),
    }
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


def append_log(experiment_id, args, raw_metrics, metrics, decision, runtime, val_logits_path, artifact_size_mb):
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
        f"- Code/config changes: `{args.base_model}`, serializer={args.serializer}, replay={args.replay_mode}, max_length={args.max_length}, epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, bucket_multiplier={args.bucket_multiplier}.",
        f"- Validation setup: {args.split}{fold_note}{quick_note}",
        f"- Raw Macro-F1: {raw_metrics['macro_f1']:.6f}",
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


def save_val_logits(experiment_id, logits, y_true, ordered_indices, samples, bias, raw_metrics, metrics, args):
    if not args.save_val_logits:
        return ""
    path = Path(args.logits_dir) / f"{experiment_id}_val_logits.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": logits.float().cpu(),
            "y_true": y_true,
            "indices": ordered_indices,
            "ids": [samples[i].get("id", "") for i in ordered_indices],
            "classes": ALL_CLASSES,
            "class_bias": [float(x) for x in bias.tolist()],
            "raw_metrics": raw_metrics,
            "metrics": metrics,
            "base_model": args.base_model,
            "serializer_name": args.serializer,
            "max_length": args.max_length,
            "split": args.split,
            "fold_id": args.fold_id if args.split == "session_oof" else None,
            "n_folds": args.n_folds if args.split == "session_oof" else None,
            "seed": args.seed,
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
    train_idx, val_idx = split_for_args(samples, y, args)
    full_val_count = len(val_idx)
    val_idx = select_balanced_subset(val_idx, y, args.quick_val_size, args.seed + 17)
    samples, y, train_idx, sample_weights, replay_size = add_replay_examples(samples, y, train_idx, args)
    # E1 specialist: 학습 샘플만 subset 클래스로 필터 (split/헤드/로짓은 14-way 그대로 → base fold와 정렬)
    subset_ids = None
    if getattr(args, "class_subset", ""):
        subset_names = [s.strip() for s in args.class_subset.split(",") if s.strip()]
        unknown = [n for n in subset_names if n not in CLASS_TO_ID]
        if unknown:
            raise ValueError(f"unknown class in --class-subset: {unknown}")
        subset_ids = {CLASS_TO_ID[n] for n in subset_names}
        before = len(train_idx)
        train_idx = [i for i in train_idx if y[i] in subset_ids]
        print(f"[class-subset] {sorted(subset_names)} train {before} -> {len(train_idx)}")
    fold_text = f" fold={args.fold_id}/{args.n_folds}" if args.split == "session_oof" else ""
    print(f"split={args.split}{fold_text} train={len(train_idx)} val={len(val_idx)} full_val={full_val_count}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    token_start = time.perf_counter()
    train_cache_scope = "train"
    if args.split == "session_oof":
        train_cache_scope = f"oof-fold{args.fold_id}-of{args.n_folds}"
    texts, text_cache_path = build_serialized_texts(samples, args, train_path, cache_scope=train_cache_scope)
    encoded_features, lengths, token_cache_path = tokenize_texts(tokenizer, texts, args, train_path, cache_scope=train_cache_scope)
    token_sec = time.perf_counter() - token_start
    avg_len = sum(lengths) / max(1, len(lengths))
    print(f"token lengths avg={avg_len:.1f} max={max(lengths) if lengths else 0}")
    if args.cache_only:
        print("cache-only requested; skipping training")
        return

    train_start = time.perf_counter()
    model = train_model(tokenizer, encoded_features, lengths, y, sample_weights, train_idx, args, device, val_idx=val_idx)
    train_sec = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    logits, y_val, raw_metrics, ordered_val_idx = evaluate(model, tokenizer, encoded_features, lengths, y, val_idx, args, device)
    eval_sec = time.perf_counter() - eval_start
    print(f"  raw_macro_f1={raw_metrics['macro_f1']:.6f}")
    if subset_ids:
        # specialist 평가: 정답이 subset인 val 행만, subset 클래스 F1 평균 (전체 14-way macro는 무의미)
        sub_pred = torch.argmax(logits, dim=1).tolist()
        pairs = [(t, p) for t, p in zip(y_val, sub_pred) if t in subset_ids]
        if pairs:
            sub_metrics = f1_metrics([t for t, _ in pairs], [p for _, p in pairs])
            sub_f1 = [sub_metrics["per_class_f1"][ALL_CLASSES[c]] for c in sorted(subset_ids)]
            print(f"  [subset] rows={len(pairs)} mean_f1={sum(sub_f1)/len(sub_f1):.6f} "
                  + " ".join(f"{ALL_CLASSES[c]}={sub_metrics['per_class_f1'][ALL_CLASSES[c]]:.4f}" for c in sorted(subset_ids)))

    bias = torch.zeros(len(ALL_CLASSES), dtype=torch.float32)
    metrics = raw_metrics
    if args.tune_bias:
        print("  tuning class bias")
        bias, _ = tune_class_bias(logits, y_val, rounds=3)
        pred = predict_with_bias(logits, bias)
        metrics = f1_metrics(y_val, pred)
        print(f"  tuned_macro_f1={metrics['macro_f1']:.6f}")

    experiment_id = experiment_id_for(args)
    val_logits_path = save_val_logits(experiment_id, logits, y_val, ordered_val_idx, samples, bias, raw_metrics, metrics, args)
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
        decision = "keep as GPU candidate" if metrics["macro_f1"] >= args.keep_threshold else "discard or revisit"

    artifact_size = 0.0
    if getattr(args, "save_split_model", False) and not args.final_model:
        save_hf_artifact(model, tokenizer, args.output_dir, bias, args, metrics)
        artifact_size = dir_size_mb(args.output_dir)
        print(f"saved split-model artifact (soup member): {args.output_dir}")
    if args.final_model:
        print("training final transformer on all training rows")
        if args.replay_mode == "none":
            final_encoded_features = encoded_features
            final_lengths = lengths
            final_y = y
            final_sample_weights = sample_weights
            final_idx = list(range(len(samples)))
        else:
            final_samples, final_y, final_idx, final_sample_weights, final_replay_size = add_replay_examples(
                base_samples,
                base_y,
                list(range(len(base_samples))),
                args,
            )
            final_texts, _ = build_serialized_texts(final_samples, args, train_path, cache_scope="final")
            final_encoded_features, final_lengths, _ = tokenize_texts(tokenizer, final_texts, args, train_path, cache_scope="final")
            print(f"final replay_size={final_replay_size}")
        final_model = train_model(
            tokenizer,
            final_encoded_features,
            final_lengths,
            final_y,
            final_sample_weights,
            final_idx,
            args,
            device,
        )
        save_hf_artifact(final_model, tokenizer, args.output_dir, bias, args, metrics)
        artifact_size = dir_size_mb(args.output_dir)
        print(f"saved HF artifact: {args.output_dir}")
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
            "model_family": "torch_gpu_transformer",
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
            "macro_f1_raw": f"{raw_metrics['macro_f1']:.6f}",
            "macro_f1_bias_tuned": f"{metrics['macro_f1']:.6f}" if args.tune_bias else "",
            "macro_f1": f"{metrics['macro_f1']:.6f}",
            "weakest_classes": summarize_weak_classes(metrics),
            "top_confusions": json.dumps(metrics["top_confusions"][:8], ensure_ascii=False),
            "prediction_distribution": json.dumps(metrics["prediction_distribution"], ensure_ascii=False, sort_keys=True),
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
    append_log(experiment_id, args, raw_metrics, metrics, decision, runtime, val_logits_path, artifact_size)

    metrics_path = Path("experiments/artifacts") / f"{experiment_id}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "raw_metrics": raw_metrics,
                "metrics": metrics,
                "class_bias": dict(zip(ALL_CLASSES, [float(x) for x in bias.tolist()])),
                "device": str(device),
                "runtime": runtime,
                "serializer_name": args.serializer,
                "text_cache_path": str(text_cache_path),
                "token_cache_path": str(token_cache_path),
                "val_logits_path": val_logits_path,
                "validation_rows": len(val_idx),
                "full_validation_rows": full_val_count,
                "fold_id": args.fold_id if args.split == "session_oof" else None,
                "n_folds": args.n_folds if args.split == "session_oof" else None,
                "replay_size": replay_size,
                "replay_sample_weight": args.replay_sample_weight,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("  weakest classes:")
    for label, score in sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1])[:8]:
        print(f"    {label:18s} {score:.4f}")
    print("  top confusions:")
    for count, true_label, pred_label in metrics["top_confusions"][:10]:
        print(f"    {true_label:18s} -> {pred_label:18s} {count}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--base-model", default="distilbert-base-multilingual-cased")
    parser.add_argument("--serializer", choices=["current_v1", "state_v2", "recent_pairs_v1", "compact_events_v1", "hybrid_v1"], default="current_v1")
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
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--final-model", action="store_true")
    parser.add_argument("--output-dir", default="model")
    parser.add_argument("--rule-boosts-path", default="")
    parser.add_argument("--keep-threshold", type=float, default=0.60)
    parser.add_argument("--notes", default="")
    parser.add_argument("--experiment-suffix", default="")
    parser.add_argument("--cache-dir", default="experiments/cache")
    parser.add_argument("--no-text-cache", action="store_true")
    parser.add_argument("--no-token-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--tokenize-batch-size", type=int, default=4096)
    parser.add_argument("--quick-val-size", type=int, default=0)
    parser.add_argument("--replay-mode", choices=["none", "last1", "last2"], default="none")
    parser.add_argument("--max-replay-samples", type=int, default=20000)
    parser.add_argument("--replay-sample-weight", type=float, default=0.5)
    parser.add_argument("--bucket-multiplier", type=int, default=8)
    parser.add_argument("--eval-bucket-multiplier", type=int, default=50)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--select-best-epoch", action="store_true",
                        help="에폭별 val 평가 후 최고 에폭 가중치로 복원 (final refit에는 미적용)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="gradient accumulation steps (len384 등 배치 축소 시 유효 배치 유지)")
    parser.add_argument("--save-split-model", action="store_true",
                        help="세션 split 학습 모델을 output-dir에 저장 (model soup 재료용, final refit 없이)")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="split 전용 seed (미지정 시 --seed 사용). soup 멤버는 --split-seed 42 고정")
    parser.add_argument("--init-seed", type=int, default=None,
                        help="분류 헤드 초기화 전용 seed (soup 멤버는 --init-seed 42 고정)")
    parser.add_argument("--class-subset", default="",
                        help="E1 specialist: 콤마 구분 클래스명 — 학습 샘플을 해당 정답 클래스로만 필터")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--fgm", action="store_true")
    parser.add_argument("--fgm-eps", type=float, default=1.0)
    parser.add_argument("--save-val-logits", dest="save_val_logits", action="store_true", default=True)
    parser.add_argument("--no-save-val-logits", dest="save_val_logits", action="store_false")
    parser.add_argument("--logits-dir", default="experiments/logits")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
