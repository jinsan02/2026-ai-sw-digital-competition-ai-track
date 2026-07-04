import csv
import json
import math
import os
import re
import zlib
from collections import Counter

import torch
import torch.nn as nn


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]+|[가-힣]+")
DEFAULT_BUCKETS = {
    "word": 262_144,
    "char": 262_144,
    "meta": 131_072,
    "last_user": 65_536,
}


def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return samples


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def bin_numeric(name, value, bins):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"{name}_missing"
    for label, upper in bins:
        if value <= upper:
            return f"{name}_{label}"
    return f"{name}_hi"


def path_tokens(path):
    path = safe_text(path).replace("\\", "/")
    if not path:
        return []
    parts = [p for p in path.split("/") if p]
    tokens = [f"path:{path}", f"base:{parts[-1]}"] if parts else [f"path:{path}"]
    for part in parts[:-1]:
        tokens.append(f"dir:{part}")
    if parts and "." in parts[-1]:
        tokens.append(f"ext:{parts[-1].rsplit('.', 1)[-1].lower()}")
    return tokens


def flatten_value(value, prefix="", depth=0):
    if depth > 2:
        return []
    tokens = []
    if isinstance(value, dict):
        for key in sorted(value):
            key_text = safe_text(key)
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            tokens.append(f"key:{next_prefix}")
            tokens.extend(flatten_value(value[key], next_prefix, depth + 1))
    elif isinstance(value, list):
        tokens.append(f"{prefix}_list_len_{min(len(value), 8)}")
        for item in value[:8]:
            tokens.extend(flatten_value(item, prefix, depth + 1))
    else:
        text = safe_text(value)
        if text:
            short = text[:160].replace("\n", " ")
            tokens.append(f"{prefix}={short}")
            if "/" in text or "\\" in text or "." in text:
                tokens.extend(path_tokens(text))
    return tokens


def prompt_intent_tokens(prompt):
    text = safe_text(prompt).lower()
    tokens = []
    patterns = [
        ("intent_run_tests", r"\b(test|tests|pytest|jest|vitest|happy path|specs?)\b|테스트|검증"),
        ("intent_lint", r"\b(lint|eslint|ruff|mypy|typecheck|type-check|tsc|typing)\b|타입|린트"),
        ("intent_run_bash", r"\b(run|execute|shell|command|terminal|build|install|pip|npm|yarn|pnpm|docker|server)\b|실행|빌드|터미널|명령"),
        ("intent_read", r"\b(open|show|read|inspect|look at|current impl|what'?s in)\b|열어|보여|읽어|확인"),
        ("intent_grep", r"\b(grep|search|find references|occurrences|look for|where is|찾아|검색)\b"),
        ("intent_glob", r"\b(glob|pattern|files matching|all .* files|\*\.[a-z0-9]+)\b"),
        ("intent_list", r"\b(list|ls|directory|folder|tree|what files)\b|목록"),
        ("intent_edit", r"\b(change|fix|update|modify|edit|rename|refactor|patch|touch)\b|수정|고쳐|바꿔|패치"),
        ("intent_write", r"\b(create|new file|write a|add a file|scaffold)\b|새 파일|작성"),
        ("intent_plan", r"\b(plan|steps|break down|outline|roadmap|approach)\b|계획|단계|쪼개"),
        ("intent_ask", r"\b(ask me|confirm|clarify|question)\b|물어|확인해줘"),
        ("intent_web", r"\b(web|internet|search online|latest|docs|documentation|browser)\b|웹|인터넷|검색해"),
        ("intent_respond", r"\b(explain|summarize|tell me|what do you think|answer)\b|설명|요약|답변"),
    ]
    for name, pattern in patterns:
        if re.search(pattern, text):
            tokens.append(name)
    tokens.append(bin_numeric("prompt_len", len(text), [("xs", 20), ("s", 60), ("m", 140), ("l", 320)]))
    if "?" in text or "어?" in text or "까" in text:
        tokens.append("prompt_has_question")
    if any(ch in text for ch in ("ㅎ", "ㅋㅋ", "lol", "thanks", "cheers")):
        tokens.append("prompt_casual")
    return tokens


def hash_index(channel, token, offsets, buckets):
    raw = f"{channel}\0{token}".encode("utf-8", errors="ignore")
    return offsets[channel] + (zlib.crc32(raw) % buckets[channel])


def tokenize(text):
    return TOKEN_RE.findall(safe_text(text).lower())


def word_ngram_tokens(text, max_tokens=96):
    tokens = tokenize(text)[:max_tokens]
    output = []
    for n in (1, 2, 3):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            output.append(f"{n}:{' '.join(tokens[i:i+n])}")
    return output


def char_ngram_tokens(text, max_chars=1200):
    text = safe_text(text).lower()[:max_chars]
    output = []
    for word in re.findall(r"\S+", text):
        padded = f" {word} "
        for n in (3, 4, 5):
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                output.append(f"{n}:{padded[i:i+n]}")
    return output


def add_word_ngrams(features, text, channel, offsets, buckets, max_tokens=96):
    for token in word_ngram_tokens(text, max_tokens=max_tokens):
        features.add(hash_index(channel, token, offsets, buckets))


def add_char_ngrams(features, text, offsets, buckets, max_chars=1200):
    for token in char_ngram_tokens(text, max_chars=max_chars):
        features.add(hash_index("char", token, offsets, buckets))


def build_offsets(buckets):
    offsets = {}
    cursor = 0
    for name in ("word", "char", "meta", "last_user"):
        offsets[name] = cursor
        cursor += buckets[name]
    return offsets, cursor


def build_sample_token_counts(sample):
    channel_counts = {
        "word": Counter(),
        "char": Counter(),
        "meta": Counter(),
        "last_user": Counter(),
    }
    prompt = safe_text(sample.get("current_prompt", ""))
    channel_counts["word"].update(word_ngram_tokens(prompt))
    channel_counts["char"].update(char_ngram_tokens(prompt))

    meta_tokens = []
    history = sample.get("history") or []
    action_names = []
    user_turns = []
    for event in history:
        role = event.get("role")
        if role == "user":
            user_turns.append(safe_text(event.get("content")))
        elif role == "assistant_action":
            name = safe_text(event.get("name"))
            action_names.append(name)
            meta_tokens.append(f"hist_action_{name}")
            meta_tokens.extend(flatten_value(event.get("args") or {}, f"args_{name}"))
            result = safe_text(event.get("result_summary"))
            if result:
                for token in tokenize(result)[:24]:
                    meta_tokens.append(f"result_{name}_{token}")

    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    meta_tokens.extend(prompt_intent_tokens(prompt))
    meta_tokens.append(f"tier_{safe_text(sm.get('user_tier', 'missing'))}")
    meta_tokens.append(f"lang_{safe_text(sm.get('language_pref', 'missing'))}")
    meta_tokens.append(f"dirty_{safe_text(ws.get('git_dirty', 'missing')).lower()}")
    meta_tokens.append(f"ci_{safe_text(ws.get('last_ci_status', 'missing'))}")
    meta_tokens.append(bin_numeric("turn", sm.get("turn_index"), [("01", 1), ("02", 2), ("04", 4), ("08", 8), ("12", 12)]))
    meta_tokens.append(bin_numeric("budget", sm.get("budget_tokens_remaining"), [("tiny", 6_000), ("low", 15_000), ("mid", 60_000), ("high", 130_000)]))
    meta_tokens.append(bin_numeric("elapsed", sm.get("elapsed_session_sec"), [("fresh", 120), ("short", 600), ("mid", 1800), ("long", 3600)]))
    meta_tokens.append(bin_numeric("loc", ws.get("loc"), [("tiny", 1_000), ("small", 5_000), ("mid", 20_000), ("large", 80_000)]))
    meta_tokens.append(f"history_len_{min(len(history), 12)}")

    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict):
        sorted_langs = sorted(language_mix.items(), key=lambda kv: (-float(kv[1]), kv[0]))
        for lang, ratio in sorted_langs[:4]:
            meta_tokens.append(f"code_lang_{safe_text(lang)}")
            try:
                meta_tokens.append(f"code_lang_{safe_text(lang)}_{int(round(float(ratio) * 10))}")
            except (TypeError, ValueError):
                pass

    open_files = ws.get("open_files") or []
    meta_tokens.append(f"open_files_{min(len(open_files), 6)}")
    for path in open_files[:8]:
        meta_tokens.extend(path_tokens(path))

    if action_names:
        meta_tokens.append(f"last_action_{action_names[-1]}")
        for name in action_names[-4:]:
            meta_tokens.append(f"recent_action_{name}")
        for a, b in zip(action_names[-5:], action_names[-4:]):
            meta_tokens.append(f"action_bigram_{a}>{b}")
        counts = Counter(action_names)
        for name, count in counts.items():
            meta_tokens.append(f"action_count_{name}_{min(count, 4)}")
    else:
        meta_tokens.append("no_history")

    for token in meta_tokens[:420]:
        channel_counts["meta"][token.lower()] += 1

    if user_turns:
        channel_counts["last_user"].update(word_ngram_tokens(user_turns[-1], max_tokens=72))

    if not any(channel_counts[channel] for channel in channel_counts):
        channel_counts["meta"]["empty_sample"] = 1
    return channel_counts


def extract_feature_indices(sample, config=None):
    config = config or {}
    buckets = config.get("buckets") or DEFAULT_BUCKETS
    offsets, _ = build_offsets(buckets)
    features = set()
    channel_counts = build_sample_token_counts(sample)
    for channel, counts in channel_counts.items():
        for token in counts:
            features.add(hash_index(channel, token, offsets, buckets))
    return sorted(features)


def extract_feature_items(sample, config=None):
    config = config or {}
    if config.get("feature_mode") != "vocab":
        return [(idx, 1.0) for idx in extract_feature_indices(sample, config)]

    channel_counts = build_sample_token_counts(sample)
    vocab_maps = config["vocab_maps"]
    idf = config["idf"]
    offsets = config["vocab_offsets"]
    items = []
    norm_sq = 0.0
    for channel, counts in channel_counts.items():
        vocab = vocab_maps.get(channel, {})
        channel_idf = idf.get(channel, [])
        offset = offsets.get(channel, 0)
        for token, count in counts.items():
            local_idx = vocab.get(token)
            if local_idx is None:
                continue
            weight = (1.0 + math.log(float(count))) * float(channel_idf[local_idx])
            items.append((offset + local_idx, weight))
            norm_sq += weight * weight
    if not items:
        items = [(0, 1.0)]
        norm_sq = 1.0
    norm = math.sqrt(max(norm_sq, 1e-12))
    return [(idx, weight / norm) for idx, weight in items]


class ActionDecisionModel(nn.Module):
    def __init__(self, num_features, num_classes, model_type="linear", hidden_dim=64):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        if model_type == "linear":
            self.embedding = nn.EmbeddingBag(num_features, num_classes, mode="sum", include_last_offset=False)
            self.bias = nn.Parameter(torch.zeros(num_classes))
        elif model_type == "mlp":
            self.embedding = nn.EmbeddingBag(num_features, hidden_dim, mode="sum", include_last_offset=False)
            self.head = nn.Linear(hidden_dim, num_classes)
        else:
            raise ValueError(f"unknown model_type: {model_type}")

    def forward(self, indices, offsets, lengths, weights=None):
        if self.model_type == "linear":
            logits = self.embedding(indices, offsets, per_sample_weights=weights)
            if weights is None:
                scale = torch.sqrt(torch.clamp(lengths.float(), min=1.0)).unsqueeze(1)
                logits = logits / scale
            return logits + self.bias
        hidden = torch.relu(self.embedding(indices, offsets, per_sample_weights=weights))
        return self.head(hidden)


def make_batch(feature_lists, device):
    lengths = [max(1, len(features)) for features in feature_lists]
    total = sum(lengths)
    flat = torch.empty(total, dtype=torch.long)
    weights = torch.empty(total, dtype=torch.float32)
    offsets = torch.empty(len(feature_lists), dtype=torch.long)
    cursor = 0
    for i, features in enumerate(feature_lists):
        offsets[i] = cursor
        if features:
            if isinstance(features[0], tuple):
                flat[cursor:cursor + len(features)] = torch.tensor([idx for idx, _ in features], dtype=torch.long)
                weights[cursor:cursor + len(features)] = torch.tensor([weight for _, weight in features], dtype=torch.float32)
            else:
                flat[cursor:cursor + len(features)] = torch.tensor(features, dtype=torch.long)
                weights[cursor:cursor + len(features)] = 1.0
            cursor += len(features)
        else:
            flat[cursor] = 0
            weights[cursor] = 1.0
            cursor += 1
    return (
        flat.to(device, non_blocking=True),
        offsets.to(device, non_blocking=True),
        torch.tensor(lengths, device=device),
        weights.to(device, non_blocking=True),
    )


def tensor_from_file(path, shape, device):
    numel = math.prod(shape)
    tensor = torch.from_file(path, dtype=torch.float32, size=numel).reshape(shape)
    return tensor.to(device)


def load_model_artifact(model_dir, device):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if config.get("feature_mode") == "vocab":
        with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
            vocab_data = json.load(f)
        config["vocab_tokens"] = vocab_data["tokens"]
        config["idf"] = vocab_data["idf"]
        config["vocab_maps"] = {
            channel: {token: idx for idx, token in enumerate(tokens)}
            for channel, tokens in vocab_data["tokens"].items()
        }
    model = ActionDecisionModel(
        num_features=config["num_features"],
        num_classes=len(config["classes"]),
        model_type=config.get("model_type", "linear"),
        hidden_dim=config.get("hidden_dim", 64),
    ).to(device)
    with torch.no_grad():
        if config.get("model_type", "linear") == "linear":
            model.embedding.weight.copy_(tensor_from_file(
                os.path.join(model_dir, "embedding.bin"),
                [config["num_features"], len(config["classes"])],
                device,
            ))
            model.bias.copy_(tensor_from_file(os.path.join(model_dir, "bias.bin"), [len(config["classes"])], device))
        else:
            model.embedding.weight.copy_(tensor_from_file(
                os.path.join(model_dir, "embedding.bin"),
                [config["num_features"], config["hidden_dim"]],
                device,
            ))
            model.head.weight.copy_(tensor_from_file(
                os.path.join(model_dir, "head_weight.bin"),
                [len(config["classes"]), config["hidden_dim"]],
                device,
            ))
            model.head.bias.copy_(tensor_from_file(os.path.join(model_dir, "head_bias.bin"), [len(config["classes"])], device))
    class_bias = torch.tensor(config.get("class_bias", [0.0] * len(config["classes"])), dtype=torch.float32, device=device)
    model.eval()
    return model, config, class_bias


def load_sample_submission(path, ids):
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if fieldnames is None or fieldnames[:2] != ["id", "action"]:
            raise ValueError(f"sample_submission columns must start with id,action: {fieldnames}")
        return fieldnames, rows
    return ["id", "action"], [{"id": sample_id, "action": ALL_CLASSES[0]} for sample_id in ids]


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serialize_transformer_sample_current(sample):
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    action_names = []
    last_user = ""
    result_bits = []
    arg_bits = []
    for event in history:
        if event.get("role") == "user":
            last_user = safe_text(event.get("content", ""))
        elif event.get("role") == "assistant_action":
            name = safe_text(event.get("name"))
            action_names.append(name)
            result = safe_text(event.get("result_summary"))
            if result:
                result_bits.append(f"{name}:{result[:120]}")
            args = event.get("args") or {}
            if isinstance(args, dict):
                for key, value in list(args.items())[:4]:
                    arg_bits.append(f"{name}.{safe_text(key)}={safe_text(value)[:80]}")

    open_files = ws.get("open_files") or []
    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict):
        langs = " ".join(f"{safe_text(k)}={float(v):.2f}" for k, v in list(language_mix.items())[:5])
    else:
        langs = ""

    parts = [
        f"current: {prompt}",
        f"meta: tier={safe_text(sm.get('user_tier'))} lang={safe_text(sm.get('language_pref'))} turn={safe_text(sm.get('turn_index'))} budget={safe_text(sm.get('budget_tokens_remaining'))} elapsed={safe_text(sm.get('elapsed_session_sec'))}",
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} loc={safe_text(ws.get('loc'))} langs={langs} open={' | '.join(safe_text(x) for x in open_files[:6])}",
        f"actions: {' > '.join(action_names[-8:]) if action_names else 'none'}",
    ]
    if last_user:
        parts.append(f"last_user: {last_user}")
    if arg_bits:
        parts.append(f"args: {' | '.join(arg_bits[-10:])}")
    if result_bits:
        parts.append(f"results: {' | '.join(result_bits[-8:])}")
    return "\n".join(parts)


def history_user_action_pairs(history):
    pairs = []
    for idx, event in enumerate(history):
        if event.get("role") != "user":
            continue
        next_event = history[idx + 1] if idx + 1 < len(history) else {}
        if next_event.get("role") == "assistant_action":
            pairs.append((safe_text(event.get("content")), next_event))
    return pairs


def summarize_action_event(event):
    name = safe_text(event.get("name"))
    bits = [f"action={name}"]
    args = event.get("args") or {}
    if isinstance(args, dict):
        for key, value in list(args.items())[:4]:
            bits.append(f"{name}.{safe_text(key)}={safe_text(value)[:80]}")
    result = safe_text(event.get("result_summary"))
    if result:
        bits.append(f"result={result[:120]}")
    return " ".join(bits)


def compact_event_tokens(sample):
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    tokens = []

    tokens.extend(prompt_intent_tokens(prompt))
    tokens.extend(f"prompt_{token}" for token in tokenize(prompt)[:40])
    action_names = []
    for event in history:
        if event.get("role") != "assistant_action":
            continue
        name = safe_text(event.get("name"))
        if not name:
            continue
        action_names.append(name)
        tokens.append(f"hist_action_{name}")
        args = event.get("args") or {}
        if isinstance(args, dict):
            for key, value in list(args.items())[:4]:
                key_text = safe_text(key)
                value_text = safe_text(value)
                tokens.append(f"arg_{name}_{key_text}")
                for path_token in path_tokens(value_text)[:8]:
                    tokens.append(f"arg_{name}_{path_token}")
        result = safe_text(event.get("result_summary"))
        for result_token in tokenize(result)[:16]:
            tokens.append(f"result_{name}_{result_token}")

    for name in action_names[-10:]:
        tokens.append(f"recent_action_{name}")
    for a, b in zip(action_names[-6:], action_names[-5:]):
        tokens.append(f"action_bigram_{a}>{b}")

    open_files = ws.get("open_files") or []
    for path in open_files[:8]:
        tokens.extend(path_tokens(path))
    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict):
        for lang, ratio in sorted(language_mix.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:4]:
            tokens.append(f"code_lang_{safe_text(lang)}")
            try:
                tokens.append(f"code_lang_{safe_text(lang)}_{int(round(float(ratio) * 10))}")
            except (TypeError, ValueError):
                pass

    tokens.append(f"tier_{safe_text(sm.get('user_tier', 'missing'))}")
    tokens.append(f"lang_{safe_text(sm.get('language_pref', 'missing'))}")
    tokens.append(f"dirty_{safe_text(ws.get('git_dirty', 'missing')).lower()}")
    tokens.append(f"ci_{safe_text(ws.get('last_ci_status', 'missing'))}")
    tokens.append(bin_numeric("turn", sm.get("turn_index"), [("01", 1), ("02", 2), ("04", 4), ("08", 8), ("12", 12)]))
    tokens.append(bin_numeric("budget", sm.get("budget_tokens_remaining"), [("tiny", 6_000), ("low", 15_000), ("mid", 60_000), ("high", 130_000)]))
    tokens.append(bin_numeric("elapsed", sm.get("elapsed_session_sec"), [("fresh", 120), ("short", 600), ("mid", 1800), ("long", 3600)]))
    tokens.append(bin_numeric("loc", ws.get("loc"), [("tiny", 1_000), ("small", 5_000), ("mid", 20_000), ("large", 80_000)]))
    return " ".join(tokens[:700])


def serialize_transformer_sample_recent_pairs(sample, pair_count=3):
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    pairs = history_user_action_pairs(history)[-pair_count:]
    parts = [f"current: {prompt}"]
    for idx, (user_text, action_event) in enumerate(pairs, 1):
        parts.append(f"pair_{idx}_user: {user_text}")
        parts.append(f"pair_{idx}_assistant: {summarize_action_event(action_event)}")
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    open_files = ws.get("open_files") or []
    parts.append(
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} "
        f"open={' | '.join(safe_text(x) for x in open_files[:6])}"
    )
    return "\n".join(parts)


def serialize_transformer_sample_state_v2(sample):
    """state_v2 (팀 Lane A-1 스펙): 최신 user/action/result/args를 앞쪽 배치(len192 절단에도 생존),
    저신호 메타(budget/tier/elapsed/loc/lang_pref) 제거, turn/dirty/ci/open_files/top_lang만 유지."""
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    last_user = ""
    action_names = []
    for event in history:
        if event.get("role") == "user":
            last_user = safe_text(event.get("content", ""))
        elif event.get("role") == "assistant_action":
            action_names.append(safe_text(event.get("name")))
    recent_bits = []  # 최신 우선 — 절단 시 오래된 것부터 잘리게
    for event in reversed(history):
        if event.get("role") != "assistant_action":
            continue
        if len(recent_bits) >= 4:
            break
        name = safe_text(event.get("name"))
        args = event.get("args") or {}
        piece = name
        if isinstance(args, dict) and args:
            piece += "(" + " ".join(
                f"{safe_text(k)}={safe_text(v)[:60]}" for k, v in list(args.items())[:3]
            ) + ")"
        result = safe_text(event.get("result_summary"))[:100]
        if result:
            piece += f" -> {result}"
        recent_bits.append(piece)
    language_mix = ws.get("language_mix") or {}
    top_lang = ""
    if isinstance(language_mix, dict) and language_mix:
        top_lang = safe_text(sorted(language_mix.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0])
    open_files = ws.get("open_files") or []
    parts = [f"current: {prompt}"]
    if last_user:
        parts.append(f"last_user: {last_user}")
    if recent_bits:
        parts.append("recent: " + " | ".join(recent_bits))
    parts.append(f"actions: {' > '.join(action_names[-8:]) if action_names else 'none'}")
    parts.append(
        f"state: turn={safe_text(sm.get('turn_index'))} dirty={safe_text(ws.get('git_dirty'))} "
        f"ci={safe_text(ws.get('last_ci_status'))} top_lang={top_lang} "
        f"open={' | '.join(safe_text(x) for x in open_files[:4])}"
    )
    return "\n".join(parts)


def serialize_transformer_sample(sample, serializer_name="current_v1"):
    if serializer_name in ("current", "current_v1"):
        return serialize_transformer_sample_current(sample)
    if serializer_name == "state_v2":
        return serialize_transformer_sample_state_v2(sample)
    if serializer_name == "recent_pairs_v1":
        return serialize_transformer_sample_recent_pairs(sample, pair_count=3)
    if serializer_name == "compact_events_v1":
        return compact_event_tokens(sample)
    if serializer_name == "hybrid_v1":
        return "\n".join([
            f"current: {safe_text(sample.get('current_prompt', ''))}",
            f"events: {compact_event_tokens(sample)}",
        ])
    raise ValueError(f"unknown serializer_name: {serializer_name}")


def rule_text_has(pattern, text):
    return bool(re.search(pattern, text))


def rule_path_like_tokens(text):
    text = safe_text(text).replace("\\", "/").lower()
    tokens = []
    if not text:
        return tokens
    if "/" in text or re.search(r"\.[a-z0-9]{1,6}\b", text):
        tokens.append("arg_has_path")
    for ext in ("py", "js", "ts", "tsx", "json", "md", "yml", "yaml", "toml", "sh", "txt", "csv"):
        if re.search(rf"\.{ext}\b", text):
            tokens.append(f"arg_ext:{ext}")
    if "*" in text:
        tokens.append("arg_has_glob")
    return tokens


def rule_base_feature_flags(sample):
    prompt = safe_text(sample.get("current_prompt", ""))
    prompt_l = prompt.lower()
    flags = set()
    for token in prompt_intent_tokens(prompt):
        flags.add(f"prompt_{token}")
    if "?" in prompt:
        flags.add("prompt_has_question")
    if rule_text_has(r"\b(open|show|read|inspect|look at|cat|view)\b|열어|보여|읽어", prompt_l):
        flags.add("prompt_read_words")
    if rule_text_has(r"\b(grep|search|find|lookup|occurrence|references?)\b|검색|찾아", prompt_l):
        flags.add("prompt_search_words")
    if rule_text_has(r"\b(list|ls|tree|directory|folder|files?)\b|목록", prompt_l):
        flags.add("prompt_list_words")
    if rule_text_has(r"\b(glob|pattern|\*\.[a-z0-9]+|\*)\b", prompt_l):
        flags.add("prompt_glob_words")
    if rule_text_has(r"\b(web|internet|browser|online|latest|docs?|documentation)\b|웹|인터넷", prompt_l):
        flags.add("prompt_web_words")
    if rule_text_has(r"\b(lint|typecheck|type-check|mypy|pyright|tsc|ruff|eslint)\b", prompt_l):
        flags.add("prompt_lint_words")
    if rule_text_has(r"\b(test|tests|pytest|jest|vitest|spec)\b", prompt_l):
        flags.add("prompt_test_words")
    if rule_text_has(r"\b(run|execute|shell|terminal|build|install|npm|pip|docker)\b", prompt_l):
        flags.add("prompt_run_words")
    for token in rule_path_like_tokens(prompt_l):
        flags.add(f"prompt_{token}")

    history = sample.get("history") or []
    action_names = []
    result_tokens = []
    arg_tokens = []
    last_user_text = ""
    for event in history:
        if event.get("role") == "user":
            last_user_text = safe_text(event.get("content", ""))
        if event.get("role") != "assistant_action":
            continue
        name = safe_text(event.get("name"))
        if not name:
            continue
        action_names.append(name)
        args = event.get("args") or {}
        if isinstance(args, dict):
            for value in list(args.values())[:6]:
                arg_tokens.extend(rule_path_like_tokens(value))
                if rule_text_has(r"(^|/)tests?(/|_)|test_|_test\.|\.spec\.|\.test\.", safe_text(value).replace("\\", "/").lower()):
                    arg_tokens.append("arg_test_path")
        result = safe_text(event.get("result_summary", "")).lower()
        if result:
            if rule_text_has(r"\b(found|match|matches|occurrences?)\b", result):
                result_tokens.append("result_found")
            if rule_text_has(r"\b(no matches|not found|empty|0 matches)\b", result):
                result_tokens.append("result_none")
            if rule_text_has(r"\b(error|failed|traceback|exception)\b", result):
                result_tokens.append("result_failed")
            if rule_text_has(r"\b(pass|passed|success|ok)\b", result):
                result_tokens.append("result_passed")
            if rule_text_has(r"\b(exit code|return code)\b", result):
                result_tokens.append("result_exit_code")

    # E2 확장 피처 (ask↔plan / bash↔tests 혼동 겨냥, 노진산 07-03)
    if rule_text_has(r"(^|/)tests?(/|_)|test_|_test\.|\.spec\.|\.test\.", prompt_l.replace("\\", "/")):
        flags.add("prompt_test_path")
    if "?" in prompt and len(prompt) < 60:
        flags.add("prompt_short_question")
    if last_user_text:
        lu = last_user_text.lower()
        if "?" in last_user_text:
            flags.add("last_user_question")
        if rule_text_has(r"\b(plan|steps|roadmap|approach|어떻게|계획|단계)\b", lu):
            flags.add("last_user_plan_words")
        if rule_text_has(r"\b(test|pytest|jest|spec)\b|테스트", lu):
            flags.add("last_user_test_words")

    flags.add(f"hist_len:{min(len(history), 12)}")
    if action_names:
        flags.add(f"last_action:{action_names[-1]}")
        for name in action_names[-4:]:
            flags.add(f"recent_action:{name}")
        if len(action_names) >= 2:
            flags.add(f"last_pair:{action_names[-2]}>{action_names[-1]}")
    for token in set(arg_tokens):
        flags.add(token)
    for token in set(result_tokens):
        flags.add(token)

    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    flags.add(bin_numeric("turn", sm.get("turn_index"), [("01", 1), ("02", 2), ("04", 4), ("08", 8), ("12", 12)]).replace("_", ":", 1))
    flags.add(f"dirty:{safe_text(ws.get('git_dirty', 'missing')).lower()}")
    flags.add(f"ci:{safe_text(ws.get('last_ci_status', 'missing')).lower()}")
    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict) and language_mix:
        top_lang = sorted(language_mix.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
        flags.add(f"top_lang:{safe_text(top_lang).lower()}")
    open_files = ws.get("open_files") or []
    if open_files:
        flags.add("has_open_files")
        for path in open_files[:6]:
            for token in rule_path_like_tokens(path):
                flags.add(f"open_{token}")
    return flags


def rule_feature_flags(sample, base_scores_row, class_names=None):
    class_names = class_names or ALL_CLASSES
    flags = rule_base_feature_flags(sample)
    top_values, top_indices = torch.topk(base_scores_row.detach().cpu(), k=2)
    top = class_names[int(top_indices[0])]
    second = class_names[int(top_indices[1])]
    margin = float(top_values[0] - top_values[1])
    logit_flags = {
        f"base_top:{top}",
        f"base_second:{second}",
        f"base_pair:{top}>{second}",
    }
    for threshold in (0.25, 0.50, 1.00):
        if margin <= threshold:
            logit_flags.add(f"margin_le:{threshold:.2f}")
    composite_sources = [
        flag for flag in flags
        if flag.startswith(("prompt_", "last_action:", "recent_action:", "last_pair:", "arg_", "open_arg_", "top_lang:", "result_"))
    ]
    composite_flags = set()
    for flag in composite_sources:
        composite_flags.add(f"{flag}|top:{top}")
        composite_flags.add(f"{flag}|pair:{top}>{second}")
    return flags | logit_flags | composite_flags


def apply_rule_boosts_to_logits(logits, samples, rules, class_names):
    if not rules:
        return logits
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    base_logits = logits.detach().cpu()
    boosted = logits.clone()
    for row_idx, sample in enumerate(samples):
        flags = rule_feature_flags(sample, base_logits[row_idx], class_names)
        for rule in rules:
            if rule.get("feature") not in flags:
                continue
            target_id = class_to_idx.get(rule.get("target"))
            if target_id is not None:
                boosted[row_idx, target_id] += float(rule.get("boost", 0.0))
    return boosted


def run_hf_inference(model_dir, data_dir, output_path, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_dir = os.path.join(model_dir, "hf_model")
    with open(os.path.join(model_dir, "hf_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_dir, local_files_only=True).to(device)
    if device.type == "cuda":
        model.half()
    model.eval()

    test_path = os.path.join(data_dir, "test.jsonl")
    sample_submission_path = os.path.join(data_dir, "sample_submission.csv")
    samples = load_jsonl(test_path)
    ids = [safe_text(sample.get("id", "")) for sample in samples]
    serializer_name = meta.get("serializer_name", "current_v1")
    texts = [serialize_transformer_sample(sample, serializer_name) for sample in samples]
    class_bias = torch.tensor(meta.get("class_bias", [0.0] * len(meta["classes"])), dtype=torch.float32, device=device)
    rule_boosts = meta.get("rule_boosts") or []

    preds = []
    batch_size = int(meta.get("batch_size", 32))
    max_length = int(meta.get("max_length", 192))
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float() + class_bias
            logits = apply_rule_boosts_to_logits(logits, samples[start:start + batch_size], rule_boosts, meta["classes"])
            pred_ids = torch.argmax(logits, dim=1).detach().cpu().tolist()
            preds.extend(meta["classes"][i] for i in pred_ids)

    fieldnames, rows = load_sample_submission(sample_submission_path, ids)
    pred_map = dict(zip(ids, preds))
    for row in rows:
        if row["id"] in pred_map:
            row["action"] = pred_map[row["id"]]
    save_submission(output_path, fieldnames, rows)
    print(f"Saved {output_path} rows={len(rows)}")


def main():
    data_dir = first_existing(["./data", "./open/data"])
    model_dir = "./model"
    output_path = "./output/submission.csv"
    test_path = os.path.join(data_dir, "test.jsonl")
    sample_submission_path = os.path.join(data_dir, "sample_submission.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Required ./model directory is missing. Place the trained model artifacts at ./model before inference.")

    if os.path.exists(os.path.join(model_dir, "hf_model", "config.json")):
        print(f"Load transformer GPU model from {model_dir}; device={device}")
        run_hf_inference(model_dir, data_dir, output_path, device)
        return

    print(f"Load GPU model from {model_dir}; device={device}")
    model, config, class_bias = load_model_artifact(model_dir, device)
    samples = load_jsonl(test_path)
    ids = [safe_text(sample.get("id", "")) for sample in samples]
    feature_lists = [extract_feature_items(sample, config) for sample in samples]

    preds = []
    batch_size = int(config.get("inference_batch_size", 512))
    with torch.inference_mode():
        for start in range(0, len(feature_lists), batch_size):
            batch = feature_lists[start:start + batch_size]
            indices, offsets, lengths, weights = make_batch(batch, device)
            logits = model(indices, offsets, lengths, weights) + class_bias
            pred_ids = torch.argmax(logits, dim=1).detach().cpu().tolist()
            preds.extend(config["classes"][i] for i in pred_ids)

    valid = set(ALL_CLASSES)
    bad = sorted(set(preds) - valid)
    if bad:
        raise ValueError(f"invalid predicted labels: {bad}")

    fieldnames, rows = load_sample_submission(sample_submission_path, ids)
    pred_map = dict(zip(ids, preds))
    missing = 0
    for row in rows:
        pred = pred_map.get(row["id"])
        if pred is None:
            missing += 1
        else:
            row["action"] = pred
    if missing:
        print(f"Warning: missing predictions for {missing} sample_submission ids")
    save_submission(output_path, fieldnames, rows)
    print(f"Saved {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
