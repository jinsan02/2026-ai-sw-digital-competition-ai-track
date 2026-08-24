import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import time
import traceback

import torch


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


def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def tokenize_texts_with_terminal(tokenizer, texts, max_length, terminal_token=""):
    """Tokenize text while reserving the final non-pad position for one token.

    Decoder sequence classifiers pool the last non-pad hidden state.  Appending
    the token *after* truncating content to ``max_length - 1`` guarantees that
    the pooling position is stable even for an input that reaches the length
    cap.  The empty-token path is the historical tokenizer call byte-for-byte.
    """
    terminal_token = safe_text(terminal_token)
    if not terminal_token:
        return tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
    if max_length < 2:
        raise ValueError("terminal-token encoding requires max_length >= 2")

    terminal_encoded = tokenizer(
        terminal_token,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    terminal_ids = terminal_encoded.get("input_ids")
    if not isinstance(terminal_ids, list) or len(terminal_ids) != 1:
        raise ValueError(
            f"terminal token must encode to exactly one id: {terminal_token!r} -> {terminal_ids!r}"
        )
    terminal_id = int(terminal_ids[0])
    if terminal_id == getattr(tokenizer, "pad_token_id", None):
        raise ValueError(
            f"terminal token {terminal_token!r} resolves to pad_token_id={terminal_id}; "
            "it would not become the last non-pad pooling position"
        )

    encoded = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length - 1,
    )
    supported = {"input_ids", "attention_mask", "token_type_ids"}
    unexpected = sorted(set(encoded.keys()) - supported)
    if unexpected:
        raise ValueError(
            "terminal-token encoding does not support tokenizer outputs: "
            + ", ".join(unexpected)
        )
    for row, input_ids in enumerate(encoded["input_ids"]):
        input_ids.append(terminal_id)
        if "attention_mask" in encoded:
            encoded["attention_mask"][row].append(1)
        if "token_type_ids" in encoded:
            token_types = encoded["token_type_ids"][row]
            token_types.append(token_types[-1] if token_types else 0)
    return encoded


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


def tokenize(text):
    return TOKEN_RE.findall(safe_text(text).lower())


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


def serialize_transformer_sample_current_parts(sample):
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
    return parts


def serialize_transformer_sample_current(sample):
    return "\n".join(serialize_transformer_sample_current_parts(sample))


CHAT_V1_CONTRACT_TOOL_LIST = " ".join(ALL_CLASSES)


def chat_v1_contract_messages(sample):
    """Build the HCX ChatML roles without dropping any current_v1 fields.

    The current request occupies the final user block, while all remaining
    current_v1 lines stay byte-identical inside the system block.  Keeping the
    message construction separate from rendering makes the contract directly
    testable and lets callers use the checkpoint's own chat template.
    """
    parts = serialize_transformer_sample_current_parts(sample)
    return [
        {"role": "tool_list", "content": CHAT_V1_CONTRACT_TOOL_LIST},
        {"role": "system", "content": "\n".join(parts[1:])},
        {"role": "user", "content": safe_text(sample.get("current_prompt", ""))},
    ]


def render_chatml_generation_prompt(messages):
    """Render the simple HCX ChatML contract when no tokenizer is available."""
    blocks = [
        f"<|im_start|>{message['role']}\n{safe_text(message.get('content'))}<|im_end|>\n"
        for message in messages
    ]
    blocks.append("<|im_start|>assistant\n")
    return "".join(blocks)


def serialize_transformer_sample_chat_v1_contract(sample, tokenizer=None):
    """Render current_v1 through the instruction checkpoint's chat contract.

    A loaded tokenizer is authoritative because a checkpoint may revise its
    template.  The deterministic fallback matches the HCX text-instruct
    template and supports cache/audit code that serializes before loading a
    tokenizer.
    """
    messages = chat_v1_contract_messages(sample)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template) and getattr(tokenizer, "chat_template", None):
        rendered = apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(rendered, str):
            raise TypeError(
                "chat_v1_contract tokenizer.apply_chat_template must return text; "
                f"got {type(rendered).__name__}"
            )
        return rendered
    return render_chatml_generation_prompt(messages)


TURN_BIN_EDGES = (1, 2, 4, 6)
TURN_BIN_NAMES = ("start", "early", "mid", "late", "long")


def turn_bin_token(turn_value):
    try:
        turn = int(float(turn_value))
    except (TypeError, ValueError):
        return "na"
    return TURN_BIN_NAMES[sum(1 for edge in TURN_BIN_EDGES if turn > edge)]


def turn_exact_token(turn_value):
    try:
        turn = int(float(turn_value))
    except (TypeError, ValueError):
        return "na"
    if turn < 0:
        return "na"
    if turn >= 14:
        return "14+"
    return f"{turn:02d}"


def turn_v6_token(turn_value):
    return f"{turn_exact_token(turn_value)}/{turn_bin_token(turn_value)}"


def top_language_pair(ws):
    language_mix = ws.get("language_mix") or {}
    if not (isinstance(language_mix, dict) and language_mix):
        return "na"
    ranked = sorted(language_mix.items(), key=lambda kv: (-float(kv[1]), safe_text(kv[0])))
    names = [safe_text(k).lower() for k, _ in ranked[:2] if safe_text(k)]
    return "+".join(names) if names else "na"


def top_language_dominance_pair(ws):
    language_mix = ws.get("language_mix") or {}
    if not (isinstance(language_mix, dict) and language_mix):
        return "na"
    ranked = []
    for key, value in language_mix.items():
        name = safe_text(key).lower()
        if not name:
            continue
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            ratio = 0.0
        ranked.append((name, ratio))
    ranked.sort(key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return "na"
    top1, ratio = ranked[0]
    top2 = ranked[1][0] if len(ranked) > 1 else "na"
    sep = "!" if ratio >= 0.7 else "~"
    return f"{top1}{sep}{top2}"


PATH_MENTION_RE = re.compile(
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|"
    r"\*\.[A-Za-z0-9]{1,8}|"
    r"\b[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}\b"
)

COMMON_CODE_DIRS = {
    "app", "api", "cmd", "components", "config", "configs", "docs", "internal",
    "k8s", "lib", "models", "pages", "plugins", "routes", "scripts", "server",
    "src", "utils",
}

OVERLAP_STOPWORDS = {
    "about", "after", "again", "also", "and", "before", "check", "code", "file",
    "files", "from", "into", "just", "look", "open", "please", "read", "show",
    "that", "the", "this", "with", "좀", "한번", "그", "그거", "파일", "보고",
    "열어", "확인",
}


def path_specificity_token(prompt):
    text = safe_text(prompt)
    lower = text.lower()
    if "*" in lower or re.search(r"\b(glob|pattern|matching)\b", lower):
        return "glob"
    mentions = PATH_MENTION_RE.findall(text)
    if any("/" in mention.replace("\\", "/") for mention in mentions):
        return "exact"
    if mentions:
        return "basename"
    if re.search(r"\b(that file|this file|same file|first one|second one|there)\b", lower):
        return "pronoun"
    if re.search(r"\b(open|show|read|inspect|pull up)\s+it\b", lower):
        return "pronoun"
    if re.search(r"그\s*파일|그거|거기|방금|첫\s*번째", lower):
        return "pronoun"
    tokens = TOKEN_RE.findall(text)
    normalized = [token.strip(".,;:()[]{}'\"").lower() for token in tokens]
    if any(
        "_" in token or re.search(r"[a-z][A-Z]", token) or re.search(r"\w+\(\)", token)
        for token in tokens
    ):
        return "symbol"
    if any(token in COMMON_CODE_DIRS for token in normalized):
        return "dir"
    return "none"


def path_overlap_terms(value):
    text = safe_text(value)
    terms = set()
    for mention in PATH_MENTION_RE.findall(text):
        cleaned = mention.replace("\\", "/").strip(".,;:()[]{}'\"").lower()
        if not cleaned:
            continue
        base = cleaned.rsplit("/", 1)[-1]
        terms.add(cleaned)
        terms.add(base)
        if "." in base:
            terms.add(base.rsplit(".", 1)[0])
    for token in TOKEN_RE.findall(text):
        stripped = token.strip(".,;:()[]{}'\"")
        lower = stripped.lower()
        if not lower or lower in OVERLAP_STOPWORDS:
            continue
        if (
            "/" in lower
            or "." in lower
            or "_" in lower
            or "-" in lower
            or re.search(r"[a-z][A-Z]", stripped)
            or lower in COMMON_CODE_DIRS
        ):
            terms.add(lower)
    return {term for term in terms if len(term) >= 2}


def last_action_event(history):
    for event in reversed(history or []):
        if event.get("role") == "assistant_action":
            return event
    return None


def struct_overlap_token(prompt, ws, last_event):
    prompt_terms = path_overlap_terms(prompt)
    if not prompt_terms:
        return "none"
    hits = []
    open_terms = set()
    for path in (ws.get("open_files") or [])[:6]:
        open_terms |= path_overlap_terms(path)
    if prompt_terms & open_terms:
        hits.append("open")

    arg_terms = set()
    if last_event:
        args = last_event.get("args") or {}
        if isinstance(args, dict):
            for value in args.values():
                arg_terms |= path_overlap_terms(value)
    if prompt_terms & arg_terms:
        hits.append("last_arg")

    result_terms = path_overlap_terms(last_event.get("result_summary", "")) if last_event else set()
    if prompt_terms & result_terms:
        hits.append("result")
    return "+".join(hits) if hits else "none"


def numeric_count_from_text(text):
    match = re.search(r"\b(\d{1,4})\b", safe_text(text))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def count_bucket(value):
    if value is None:
        return "unknown"
    if value <= 0:
        return "none"
    if value == 1:
        return "single"
    if value <= 3:
        return "few"
    return "many"


def candidate_state_token(last_event):
    if not last_event:
        return "none"
    name = safe_text(last_event.get("name"))
    result = safe_text(last_event.get("result_summary"))
    lower = result.lower()
    if re.search(r"\b(no matches?|not found|0 matches?|0 files?|empty|no results?)\b", lower):
        return "none"
    if re.search(r"\b(error|failed|failure|traceback|exception|timed out|timeout)\b", lower):
        return "unknown"
    if name == "read_file":
        return "single"
    if name in ("grep_search", "glob_pattern", "list_directory"):
        return count_bucket(numeric_count_from_text(lower))
    return "unknown"


def explorer_struct_line(prompt, ws, history):
    last_event = last_action_event(history)
    return (
        f"struct: path={path_specificity_token(prompt)} "
        f"overlap={struct_overlap_token(prompt, ws, last_event)} "
        f"cand={candidate_state_token(last_event)}"
    )


def serialize_transformer_sample_current_v5(sample):
    """current_v1 with denoised meta/workspace lines (fe_current_v5_spec.md):
    tier/lang_pref/budget/elapsed/loc dropped, turn_index binned to regime
    tokens (edges fixed from train quantile-free regime analysis 2026-07-05),
    language_mix floats replaced by top-2 language names. All other lines are
    byte-identical to current_v1."""
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

    parts = [
        f"current: {prompt}",
        f"meta: turn={turn_bin_token(sm.get('turn_index'))}",
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} "
        f"lang={top_language_pair(ws)} open={' | '.join(safe_text(x) for x in open_files[:6])}",
        f"actions: {' > '.join(action_names[-8:]) if action_names else 'none'}",
    ]
    if last_user:
        parts.append(f"last_user: {last_user}")
    if arg_bits:
        parts.append(f"args: {' | '.join(arg_bits[-10:])}")
    if result_bits:
        parts.append(f"results: {' | '.join(result_bits[-8:])}")
    return "\n".join(parts)


def serialize_transformer_sample_current_v6(sample):
    """current_v5 plus exact turn and language dominance markers
    (fe_current_v6_spec.md). No struct line."""
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

    parts = [
        f"current: {prompt}",
        f"meta: turn={turn_v6_token(sm.get('turn_index'))}",
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} "
        f"lang={top_language_dominance_pair(ws)} open={' | '.join(safe_text(x) for x in open_files[:6])}",
        f"actions: {' > '.join(action_names[-8:]) if action_names else 'none'}",
    ]
    if last_user:
        parts.append(f"last_user: {last_user}")
    if arg_bits:
        parts.append(f"args: {' | '.join(arg_bits[-10:])}")
    if result_bits:
        parts.append(f"results: {' | '.join(result_bits[-8:])}")
    return "\n".join(parts)


def serialize_transformer_sample_current_v6e(sample):
    """current_v5 plus exact turn and explorer evidence tokens.

    This intentionally keeps v5's top-2 language names instead of v6's
    dominance marker; the added evidence targets read/list/grep/glob
    candidate-state ambiguity."""
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
            last_user = safe_text(event.get("content"))
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

    parts = [
        f"current: {prompt}",
        f"meta: turn={turn_v6_token(sm.get('turn_index'))}",
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} "
        f"lang={top_language_pair(ws)} open={' | '.join(safe_text(x) for x in open_files[:6])}",
        f"actions: {' > '.join(action_names[-8:]) if action_names else 'none'}",
    ]
    if last_user:
        parts.append(f"last_user: {last_user}")
    if arg_bits:
        parts.append(f"args: {' | '.join(arg_bits[-10:])}")
    if result_bits:
        parts.append(f"results: {' | '.join(result_bits[-8:])}")
    parts.append(explorer_struct_line(prompt, ws, history))
    return "\n".join(parts)


# NOTE: no `retry`/`재시도` here — in this corpus those overwhelmingly mean
# "implement retry logic" (code change), not "run it again" (measured 2026-07-08).
V7_RERUN_RE = re.compile(
    r"\b(again|rerun|re-run|once more|one more time)\b"
    r"|다시|한번 더|한 번 더|한번만 더|한 번만 더|방금 그|아까 그|재실행|또 돌려|또 실행"
)
V7_NUM_RE = re.compile(r"\d+")


def v7_result_bucket(result):
    text = safe_text(result)
    if not text:
        return "na"
    low = text.lower()
    if "exit=" in low:
        return "exit0" if "exit=0" in low else "exitN"
    match = V7_NUM_RE.search(low)
    if match:
        count = int(match.group())
        if count == 0:
            return "zero"
        if count == 1:
            return "one"
        if count <= 5:
            return "few"
        return "many"
    if any(word in low for word in ("ok", "pass", "clean", "no issues")):
        return "ok"
    if any(word in low for word in ("fail", "error", "conflict")):
        return "fail"
    return "other"


def v7_target_anchor(prompt, ptype):
    """First concrete target mention for the given specificity class, so the
    state line carries a redundant anchor of the load-bearing prompt token."""
    text = safe_text(prompt)
    if ptype in ("exact", "basename", "glob"):
        match = PATH_MENTION_RE.search(text)
        if match:
            return match.group(0)[:40]
    if ptype == "symbol":
        for token in TOKEN_RE.findall(text):
            if "_" in token or re.search(r"[a-z][A-Z]", token) or re.search(r"\w+\(\)", token):
                return token[:40]
    return ""


def serialize_transformer_sample_current_v7(sample):
    """current_v1 plus a derived-state line right after the current line and a
    compact echo of it as the final line. Rationale (attention-sink probe,
    diag_serializer_headroom_20260707.json): the early copy gets causal
    exposure to every later token and survives truncation; the echo sits in
    the classification token's local window (last-40 tokens receive 0.263 of
    its non-sink mass). Normalized last/prev action:result-bucket, prompt
    target-specificity with a literal anchor, rerun marker. Every current_v1
    line is preserved byte-identical."""
    base = serialize_transformer_sample_current(sample)
    history = sample.get("history") or []
    action_names = []
    result_summaries = []
    for event in history:
        if event.get("role") == "assistant_action":
            action_names.append(safe_text(event.get("name")))
            result_summaries.append(safe_text(event.get("result_summary")))
    last_action = action_names[-1] if action_names else "none"
    prev_action = action_names[-2] if len(action_names) > 1 else "none"
    last_bucket = v7_result_bucket(result_summaries[-1]) if result_summaries else "na"
    prev_bucket = v7_result_bucket(result_summaries[-2]) if len(result_summaries) > 1 else "na"
    prompt = safe_text(sample.get("current_prompt", ""))
    rerun = "y" if V7_RERUN_RE.search(prompt.lower()) else "n"
    ptype = path_specificity_token(prompt)
    anchor = v7_target_anchor(prompt, ptype)
    ptype_bit = f"ptype={ptype}:{anchor}" if anchor else f"ptype={ptype}"
    state = (
        f"state: last={last_action}:{last_bucket} prev={prev_action}:{prev_bucket} "
        f"{ptype_bit} rerun={rerun}"
    )
    echo = f"state2: last={last_action}:{last_bucket} {ptype_bit} rerun={rerun}"
    lines = base.split("\n")
    return "\n".join([lines[0], state] + lines[1:] + [echo])


# 16 identical '$$$$' tokens on the HCX tokenizer: constant-K/V register slots
# (dedicated attention dump sites; ViT-register analogue). Placed early so
# every later query can use them and truncation can never drop them.
V7R_REGISTER_LINE = "reg: " + "$" * 64


def serialize_transformer_sample_current_v7r(sample):
    """current_v7 plus a constant register line right after the state line.

    Register rationale (filler_substitution_probe_20260708): a $-wall
    mechanically absorbs the same attention mass as natural low-info tokens
    but with CONSTANT key/value vectors, so heads that dump attention there
    inject a learnable constant bias instead of row-varying noise. Whether
    fine-tuning learns to exploit this is the open question this variant
    screens; single-variable increment over current_v7."""
    base = serialize_transformer_sample_current_v7(sample)
    lines = base.split("\n")
    return "\n".join(lines[:2] + [V7R_REGISTER_LINE] + lines[2:])


# Register-design iteration over current_v7r (screen 20260707_184650 confirmed
# the register mechanism: +0.004429 vs v7, plan_task recovered; open regression
# grep_search -0.0101). One design variable per variant, all vs the v7r anchor.
# Measured HCX merge widths: '$'/'^' 4 chars/token, '&' 2, '░' 1.
V7RL_REGISTER_LINE = "reg: " + "$" * 16
V7RM_REGISTER_LINE = "reg: " + "$" * 16 + "^" * 16 + "&" * 8 + "░" * 4
V7RD_EARLY_LINE = "reg: " + "$" * 32
V7RD_LATE_LINE = "reg2: " + "$" * 32


def serialize_transformer_sample_current_v7rl(sample):
    """current_v7r with register capacity cut 16 -> 4 slots. Tests whether the
    grep_search regression is over-absorption (registers stealing attention
    that discriminative lexical tokens need); ViT-register literature found 4
    slots sufficient. 12 tokens cheaper than v7r."""
    base = serialize_transformer_sample_current_v7(sample)
    lines = base.split("\n")
    return "\n".join(lines[:2] + [V7RL_REGISTER_LINE] + lines[2:])


def serialize_transformer_sample_current_v7rm(sample):
    """current_v7r at equal capacity (16 slots) but 4 distinct K/V types
    ($$$$/^^^^/&&/░ x4 each) instead of one. Identical tokens differ only by
    RoPE phase; distinct embeddings let heads address slot groups separately,
    matching how ViT registers are distinct learned vectors."""
    base = serialize_transformer_sample_current_v7(sample)
    lines = base.split("\n")
    return "\n".join(lines[:2] + [V7RM_REGISTER_LINE] + lines[2:])


def serialize_transformer_sample_current_v7rd(sample):
    """current_v7r capacity split 8 early + 8 immediately before the tail echo.
    The classification token's non-sink attention is tail-local (last 40 tokens
    take 0.263), so trained late registers can serve as its aggregation buffer
    — placed before the echo so overflow still clips echo tokens first and the
    classification position stays on signal, never on a register."""
    base = serialize_transformer_sample_current_v7(sample)
    lines = base.split("\n")
    return "\n".join(
        lines[:2] + [V7RD_EARLY_LINE] + lines[2:-1] + [V7RD_LATE_LINE, lines[-1]]
    )


# --- current_v8: composite redesign (2026-07-08, user-approved single shot) ---
# Deadline call: verified pieces composed at once instead of one-variable
# screens. Content -> bins: v6 turn (exact/regime) and language-dominance
# tokens ride the early v7 state line; only the phase bit is echoed in the
# tail — the classification window (~40 tokens, 0.263 of non-sink mass) is
# shared with the newest result_summary, so a fat echo would evict the most
# action-predictive raw content. Structure -> walls: zero-content numerals
# AND their field words (xmeta recovery +0.0002) are replaced IN PLACE by
# constant runs — the C1 substitution probe showed this form is mechanically
# equivalent to v1's natural filler — with a distinct char per site so heads
# can address the meta-position and workspace-position slot groups
# separately. The dedicated reg: line is dropped: the in-place walls carry
# the register mass (~29 slots), avoiding over-absorption from stacking
# walls on top of natural noise (v7r's grep_search regression). dirty=/ci=
# stay: never dropped in any variant, and they are the only workspace signal
# tied to run_tests/lint.
V8_META_WALL = "$" * 64  # 16 slots; replaces the whole meta line (tier/lang/turn/budget/elapsed)
V8_WS_WALL = "^" * 48  # ~13 slots; replaces loc=/langs= inside the workspace line


def serialize_transformer_sample_current_v8(sample):
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    action_names = []
    last_user = ""
    result_bits = []
    arg_bits = []
    result_summaries = []
    for event in history:
        if event.get("role") == "user":
            last_user = safe_text(event.get("content", ""))
        elif event.get("role") == "assistant_action":
            name = safe_text(event.get("name"))
            action_names.append(name)
            result = safe_text(event.get("result_summary"))
            result_summaries.append(result)
            if result:
                result_bits.append(f"{name}:{result[:120]}")
            args = event.get("args") or {}
            if isinstance(args, dict):
                for key, value in list(args.items())[:4]:
                    arg_bits.append(f"{name}.{safe_text(key)}={safe_text(value)[:80]}")

    open_files = ws.get("open_files") or []

    last_action = action_names[-1] if action_names else "none"
    prev_action = action_names[-2] if len(action_names) > 1 else "none"
    last_bucket = v7_result_bucket(result_summaries[-1]) if result_summaries else "na"
    prev_bucket = v7_result_bucket(result_summaries[-2]) if len(result_summaries) > 1 else "na"
    rerun = "y" if V7_RERUN_RE.search(prompt.lower()) else "n"
    ptype = path_specificity_token(prompt)
    anchor = v7_target_anchor(prompt, ptype)
    ptype_bit = f"ptype={ptype}:{anchor}" if anchor else f"ptype={ptype}"
    turn_bit = turn_v6_token(sm.get("turn_index"))
    lang_bit = top_language_dominance_pair(ws)

    parts = [
        f"current: {prompt}",
        f"state: last={last_action}:{last_bucket} prev={prev_action}:{prev_bucket} "
        f"{ptype_bit} rerun={rerun} turn={turn_bit} lang={lang_bit}",
        V8_META_WALL,
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} "
        f"{V8_WS_WALL} open={' | '.join(safe_text(x) for x in open_files[:6])}",
        f"actions: {' > '.join(action_names[-8:]) if action_names else 'none'}",
    ]
    if last_user:
        parts.append(f"last_user: {last_user}")
    if arg_bits:
        parts.append(f"args: {' | '.join(arg_bits[-10:])}")
    if result_bits:
        parts.append(f"results: {' | '.join(result_bits[-8:])}")
    parts.append(
        f"state2: last={last_action}:{last_bucket} {ptype_bit} rerun={rerun} turn={turn_bit}"
    )
    return "\n".join(parts)


# Satellite cell: does a trained constant buffer INSIDE the classification
# window help (aggregation-buffer hypothesis) or hurt (untrained tail junk
# hijacked 0.250 of classification attention pre-training)? 8 slots of '&'
# (distinct from both wall sites), inserted right before the echo so the
# final position always stays on signal.
V8T_TAIL_WALL = "&" * 16  # 8 slots ('&' merges 2 chars/token)


def serialize_transformer_sample_current_v8t(sample):
    lines = serialize_transformer_sample_current_v8(sample).split("\n")
    return "\n".join(lines[:-1] + [V8T_TAIL_WALL, lines[-1]])


# current_v7rb: v7r + v6 phase/language bins appended to the early state line.
# Functional substitution (2026-07-08 design law): raw turn/langs floats stay
# untouched as structural substrate; single-token derived copies land at the
# proven early landing site so attention can migrate on its own. budget/
# elapsed bins deliberately excluded — they are correlated re-encodings of
# session progress that turn already carries. Echo and all other lines are
# v7r byte-identical.
def serialize_transformer_sample_current_v7rb(sample):
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    lines = serialize_transformer_sample_current_v7r(sample).split("\n")
    lines[1] += f" turn={turn_v6_token(sm.get('turn_index'))} lang={top_language_dominance_pair(ws)}"
    return "\n".join(lines)


# current_v7rc: v7r with the meta line (tier/lang_pref/turn/budget/elapsed)
# removed entirely -- no wall in its place. Isolates a question v8 could not
# answer cleanly: v8 bundled "delete/replace this content" with "drop the
# dedicated reg line", so its failure couldn't be attributed to either alone.
# Here reg stays untouched and adjacent (state, reg, then straight to
# workspace) -- testing whether an existing nearby register already gives a
# downstream natural field's removal a safe dump site, without any new wall.
def serialize_transformer_sample_current_v7rc(sample):
    lines = serialize_transformer_sample_current_v7r(sample).split("\n")
    return "\n".join(l for l in lines if not l.startswith("meta:"))


def _v7r_workspace_pieces(sample):
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict):
        langs = " ".join(f"{safe_text(k)}={float(v):.2f}" for k, v in list(language_mix.items())[:5])
    else:
        langs = ""
    open_str = " | ".join(safe_text(x) for x in (ws.get("open_files") or [])[:6])
    return ws, langs, open_str


# current_v7rw / current_v7rg: decoupled single-variable siblings of v7rc's
# question, applied to workspace's loc/langs instead of the whole meta line.
# Reconstructed directly from `sample` (not regex on flattened text -- langs
# values contain internal '=' chars, e.g. "py=0.92 yaml=0.05", which broke an
# earlier throwaway regex-based estimate into a silent no-op).
V7RW_LOC_WALL = "^" * 16  # 4 slots (^ merges 4 chars/token) -- ViT-register "4 slots suffice" precedent


def serialize_transformer_sample_current_v7rw(sample):
    """v7r with the loc= field replaced by a bare wall run (no 'loc=' label --
    the key is meaningless once its value is constant), langs/dirty/ci
    untouched. loc is a pure scalar with no established content value (same
    xmeta-dropped family as budget/elapsed); reg stays adjacent and
    untouched -- isolates the wall-substitution question from v8's bundled
    reg removal. 4 slots kept minimal since this is pure addition over v1
    (no field removed elsewhere to offset it)."""
    ws, langs, open_str = _v7r_workspace_pieces(sample)
    new_line = (
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} "
        f"{V7RW_LOC_WALL} langs={langs} open={open_str}"
    )
    lines = serialize_transformer_sample_current_v7r(sample).split("\n")
    return "\n".join(new_line if l.startswith("workspace:") else l for l in lines)


def serialize_transformer_sample_current_v7rg(sample):
    """v7r with the langs= float list replaced by v6's compact dominance-pair
    marker (e.g. go!yaml) instead of a wall -- langs carries real
    distributional signal (unlike loc), and v6 already validated this
    non-destructive compact form; loc stays a raw numeral, untouched."""
    ws, langs, open_str = _v7r_workspace_pieces(sample)
    new_line = (
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} "
        f"loc={safe_text(ws.get('loc'))} langs={top_language_dominance_pair(ws)} open={open_str}"
    )
    lines = serialize_transformer_sample_current_v7r(sample).split("\n")
    return "\n".join(new_line if l.startswith("workspace:") else l for l in lines)


# current_v7rcgw: v7rc (meta deleted) + langs v6-compact, PLUS loc fully
# removed (no wall in its own spot, matching meta's treatment) with its wall
# relocated to a new bare line between the actions/last_user block and the
# args/results block. Caveat (unlike reg's fixed-early, always-safe position):
# this boundary's distance from the sequence end varies with how much
# args/results content exists -- short rows can land it close to the
# classification window (the v7rd/v8t tail-hijack failure mode), long rows
# put it safely mid-sequence. Untested placement, not validated-safe like reg.
def serialize_transformer_sample_current_v7rcgw(sample):
    ws, langs, open_str = _v7r_workspace_pieces(sample)
    ws_line = (
        f"workspace: dirty={safe_text(ws.get('git_dirty'))} ci={safe_text(ws.get('last_ci_status'))} "
        f"langs={top_language_dominance_pair(ws)} open={open_str}"
    )
    lines = serialize_transformer_sample_current_v7rc(sample).split("\n")
    lines = [ws_line if l.startswith("workspace:") else l for l in lines]

    if any(l.startswith("results:") for l in lines):
        insert_at = next(i for i, l in enumerate(lines) if l.startswith("results:"))
    elif any(l.startswith("args:") for l in lines):
        insert_at = next(i for i, l in enumerate(lines) if l.startswith("args:")) + 1
    elif any(l.startswith("last_user:") for l in lines):
        insert_at = next(i for i, l in enumerate(lines) if l.startswith("last_user:")) + 1
    else:
        insert_at = next(i for i, l in enumerate(lines) if l.startswith("actions:")) + 1
    lines.insert(insert_at, V7RW_LOC_WALL)
    return "\n".join(lines)


# --- current_v9o / current_v9f: tag-schema variants of current_v7r ---
# Marker-layer-only change (2026-07-08): natural tokens, content bytes, and
# state/reg/echo placement are v7r-identical; only our invented field markers
# change from "field: " to XML-style tags. Rationale: HCX pretraining is
# tag-segmented (StarCoder-lineage added_tokens like <pr_diff>), and '<'/'</'
# are clean single tokens. v9o uses opening tags only — measured cost 0
# ("field: " and "<field>" are both 3 tokens, content keeps its leading-space
# tokenization). v9f adds closing tags (+~28 tokens/row) for explicit segment
# ends. Lines with an unrecognized prefix (e.g. multi-line prompts in unseen
# test data) pass through untouched.
V9_FIELDS = (
    "current", "state", "reg", "meta", "workspace",
    "actions", "last_user", "args", "results", "state2",
)


def _v9_tagged(sample, close):
    out = []
    for line in serialize_transformer_sample_current_v7r(sample).split("\n"):
        name, sep, rest = line.partition(": ")
        if sep and name in V9_FIELDS:
            if close:
                out.append(f"<{name}> {rest} </{name}>")
            else:
                out.append(f"<{name}> {rest}")
        else:
            out.append(line)
    return "\n".join(out)


def serialize_transformer_sample_current_v9o(sample):
    return _v9_tagged(sample, close=False)


def serialize_transformer_sample_current_v9f(sample):
    return _v9_tagged(sample, close=True)


# current_v9h: hybrid closing. v9o (open-only, all fields) collapsed (-0.017,
# broken-markup penalty) and v9f (open+close, all fields) lost twice more
# (len384 -0.0027, len416 -0.0067 despite removing the truncation confound) --
# 3/3 against tag-wrapping every field. The one place closing tags carry a
# real signal is boundary ambiguity: current/last_user are the only free-text
# fields where arbitrary user text could contain schema-like substrings, so
# "where does this field end" is a genuine question there and nowhere else
# (every other line is single-line, newline-terminated, self-delimiting).
# v9h wraps only those two with open+close tags; every other field keeps
# v1's "field: " prefix untouched -- 2 fields tagged instead of 9, so far
# cheaper than v9f regardless of outcome.
def serialize_transformer_sample_current_v9h(sample):
    lines = serialize_transformer_sample_current_v7r(sample).split("\n")
    out = []
    for line in lines:
        if line.startswith("current: "):
            out.append(f"<current> {line[len('current: '):]} </current>")
        elif line.startswith("last_user: "):
            out.append(f"<last_user> {line[len('last_user: '):]} </last_user>")
        else:
            out.append(line)
    return "\n".join(out)


ROUTE_USAGE_RE = re.compile(
    r"\b(grep|search|find|lookup|reference|references|called|calls?|uses?|used|"
    r"import|imports|hardcoded|occurrence|occurrences|where\s+.*\b(live|used|called|defined))\b|"
    r"검색|찾아|어디서|쓰는지|부르는지|호출|참조|정의|남아|흩어|훑|긁",
    re.IGNORECASE,
)
ROUTE_INVENTORY_RE = re.compile(
    r"\b(all|every|recursive|full\s+list|which\s+files|what\s+files|files?\s+under|"
    r"files?\s+matching|glob|pattern)\b|"
    r"\*\*?[/.\w-]*|\*\.[A-Za-z0-9]{1,8}|"
    r"전체|전부|몇\s*개|목록|어디어디|파일들|흩어져|패턴",
    re.IGNORECASE,
)
ROUTE_STRONG_INVENTORY_RE = re.compile(
    r"\b(recursive|full\s+list|which\s+files|what\s+files|files?\s+under|"
    r"files?\s+matching|all\s+[\w\s.-]{0,32}files?|every\s+[\w\s.-]{0,32}files?|"
    r"glob|pattern)\b|"
    r"\*\*?[/.\w-]*|\*\.[A-Za-z0-9]{1,8}|"
    r"전체|전부|몇\s*개|목록|어디어디|파일들|흩어져|패턴",
    re.IGNORECASE,
)
ROUTE_DIR_RE = re.compile(
    r"\b(list|ls|tree|directory|folder|top[- ]level|layout|what'?s\s+in|"
    r"what\s+lives\s+under|contents?)\b|"
    r"폴더|디렉토리|디렉터리|뭐뭐|들어있|구조|레이아웃",
    re.IGNORECASE,
)
ROUTE_READ_RE = re.compile(
    r"\b(open|show|read|inspect|look\s+at|pull\s+up|cat|view|current\s+impl|body)\b|"
    r"열어|보여|읽어|본문|통째로|펼쳐|다시\s*봐|직접\s*보고|내용",
    re.IGNORECASE,
)


def route_short(value, limit=48):
    text = safe_text(value).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def route_target(prompt):
    text = safe_text(prompt)
    lower = text.lower()
    mention = PATH_MENTION_RE.search(text)
    if mention:
        value = mention.group(0).replace("\\", "/")
        if "*" in value or re.search(r"\b(glob|pattern|matching)\b", lower):
            return "glob", route_short(value, 40)
        if "/" in value:
            return "exact_path", route_short(value, 40)
        return "basename", route_short(value, 40)
    if re.search(r"\b(that file|this file|same file|first one|second one|there|it)\b", lower):
        return "pronoun", "it"
    if re.search(r"그\s*파일|그거|거기|방금|첫\s*번째", lower):
        return "pronoun", route_short(re.search(r"그\s*파일|그거|거기|방금|첫\s*번째", lower).group(0), 20)
    tokens = TOKEN_RE.findall(text)
    for token in tokens:
        if "_" in token or re.search(r"[a-z][A-Z]", token) or re.search(r"\w+\(\)", token):
            return "symbol", route_short(token, 40)
    normalized = [token.strip(".,;:()[]{}'\"").lower() for token in tokens]
    for token in normalized:
        if token in COMMON_CODE_DIRS:
            return "dir", token
    return "none", "na"


def route_first_cue(pattern, prompt):
    match = pattern.search(safe_text(prompt))
    return route_short(match.group(0), 36) if match else "na"


def route_scope(op, target_kind, overlap):
    if "open" in overlap.split("+"):
        return "open"
    if target_kind == "dir":
        return "dir"
    if target_kind in ("pronoun", "none") and op != "unknown":
        return "candidate"
    if op in ("content_search", "file_set"):
        return "repo"
    if op == "single_file_read":
        return "repo"
    return "unknown"


def route_fields(sample):
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    last_event = last_action_event(history)
    target_kind, target_value = route_target(prompt)
    overlap = struct_overlap_token(prompt, ws, last_event)

    has_usage = bool(ROUTE_USAGE_RE.search(prompt))
    has_inventory = bool(ROUTE_INVENTORY_RE.search(prompt))
    has_strong_inventory = target_kind == "glob" or bool(ROUTE_STRONG_INVENTORY_RE.search(prompt))
    has_dir = bool(ROUTE_DIR_RE.search(prompt))
    has_read = bool(ROUTE_READ_RE.search(prompt))

    op = "unknown"
    cue = "na"
    if has_usage and not has_strong_inventory:
        op = "content_search"
        cue = route_first_cue(ROUTE_USAGE_RE, prompt)
    elif has_inventory or has_strong_inventory:
        op = "file_set"
        cue = route_first_cue(ROUTE_STRONG_INVENTORY_RE if has_strong_inventory else ROUTE_INVENTORY_RE, prompt)
    elif has_read and target_kind in ("exact_path", "basename", "pronoun", "symbol"):
        op = "single_file_read"
        cue = route_first_cue(ROUTE_READ_RE, prompt)
    elif has_dir or target_kind == "dir":
        op = "dir_children"
        cue = route_first_cue(ROUTE_DIR_RE, prompt)

    multiplicity = {
        "single_file_read": "single",
        "content_search": "occurrences",
        "dir_children": "children",
        "file_set": "all_files",
    }.get(op, "unknown")
    family = "file_nav" if op != "unknown" or target_kind != "none" else "unknown"
    scope = route_scope(op, target_kind, overlap)
    return family, op, target_kind, target_value, multiplicity, scope, cue


def route_count_bucket_from_result(result):
    lower = safe_text(result).lower()
    if not lower:
        return "na"
    if re.search(r"\b(error|failed|failure|traceback|exception|permission denied|timed out|timeout|conflict)\b", lower):
        return "error"
    if re.search(r"\b(no matches?|not found|0 matches?|0 files?|0 entries|empty|no results?)\b", lower):
        return "zero"
    count = numeric_count_from_text(lower)
    if count is not None and re.search(r"\b(matches?|occurrences?|files matched|entries|results?)\b", lower):
        if count <= 0:
            return "zero"
        if count == 1:
            return "one"
        if count <= 5:
            return "few"
        return "many"
    if re.search(r"\b(pass(?:ed)?|success(?:ful)?|succeeded|ok|clean|no errors?|read)\b", lower):
        return "ok"
    return "na"


def route_arg_kind(last_event):
    if not last_event:
        return "none"
    args = last_event.get("args") or {}
    if not isinstance(args, dict) or not args:
        return "none"
    for key, value in args.items():
        key_l = safe_text(key).lower()
        value_l = safe_text(value).lower()
        if key_l in ("cmd", "command"):
            return "cmd"
        if "*" in value_l or key_l == "pattern":
            return "glob"
        if key_l in ("scope", "cwd"):
            return "dir"
        if "/" in value_l or re.search(r"\b[\w.-]+\.[a-z0-9]{1,8}\b", value_l):
            return "exact_path"
        if key_l in ("symbol", "target_symbol") or "_" in value_l or re.search(r"[a-z][A-Z]", safe_text(value)):
            return "symbol"
    return "none"


def weak_nav_repeat_token(action_names):
    if not action_names:
        return "0"
    last = action_names[-1]
    count = 0
    for name in reversed(action_names):
        if name != last:
            break
        count += 1
    return "3+" if count >= 3 else str(count)


def weak_nav_line(sample):
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    action_events = [event for event in history if event.get("role") == "assistant_action"]
    action_names = [safe_text(event.get("name")) for event in action_events]
    last_event = action_events[-1] if action_events else None
    prev_event = action_events[-2] if len(action_events) > 1 else None
    last_action = action_names[-1] if action_names else "none"
    prev_action = action_names[-2] if len(action_names) > 1 else "none"
    last_bucket = route_count_bucket_from_result(last_event.get("result_summary") if last_event else "")
    prev_bucket = route_count_bucket_from_result(prev_event.get("result_summary") if prev_event else "")
    return (
        f"nav: prev={prev_action}:{prev_bucket} last={last_action}:{last_bucket} "
        f"repeat={weak_nav_repeat_token(action_names)} "
        f"turn={turn_v6_token(sm.get('turn_index'))} arg={route_arg_kind(last_event)}"
    )


def weak_nav_path_values(sample, limit=4, char_limit=60):
    history = sample.get("history") or []
    path_keys = {
        "path", "paths", "scope", "cwd", "directory", "dir",
        "file", "filename", "target",
    }
    values = []
    seen = set()
    for event in reversed(history):
        if event.get("role") != "assistant_action":
            continue
        args = event.get("args") or {}
        if not isinstance(args, dict):
            continue
        for key, raw_value in args.items():
            value = re.sub(r"\s+", " ", safe_text(raw_value)).strip()
            if not value:
                continue
            key_l = safe_text(key).lower()
            looks_path_like = bool(
                "/" in value
                or "\\" in value
                or "*" in value
                or re.search(r"(?:^|[/\\])[\w.-]+\.[A-Za-z0-9]{1,12}$", value)
            )
            if key_l not in path_keys and not (key_l == "pattern" and looks_path_like):
                continue
            if value in seen:
                continue
            seen.add(value)
            values.append(value[:char_limit])
            if len(values) >= limit:
                return values
    return values


def serialize_transformer_sample_weak_nav_v1(sample):
    parts = serialize_transformer_sample_current_parts(sample)
    parts.insert(1, weak_nav_line(sample))
    return "\n".join(parts)


def serialize_transformer_sample_weak_nav_paths_v1(sample):
    parts = serialize_transformer_sample_current_parts(sample)
    paths = weak_nav_path_values(sample)
    parts[1:1] = [weak_nav_line(sample), f"last_paths: {' | '.join(paths) if paths else 'none'}"]
    return "\n".join(parts)


def route_candidate_pool(last_event):
    if not last_event:
        return "none"
    name = safe_text(last_event.get("name"))
    result = safe_text(last_event.get("result_summary"))
    bucket = route_count_bucket_from_result(result)
    if bucket in ("zero", "error"):
        return "diagnostic" if bucket == "error" else "none"
    if name == "list_directory":
        return "dir_entries"
    if name == "glob_pattern":
        return "file_set"
    if name == "grep_search":
        return "content_hits"
    if name == "read_file" and bucket == "ok":
        return "single_file"
    if name in ("run_bash", "run_tests", "lint_or_typecheck") or bucket == "ok":
        return "diagnostic"
    return "none"


def route_open_relation(prompt, ws, target_kind, target_value):
    open_files = [safe_text(path).replace("\\", "/") for path in (ws.get("open_files") or [])[:6]]
    n_open = min(len(open_files), 6)
    if target_kind == "none":
        target_open = "unknown"
    elif not open_files:
        target_open = "no"
    else:
        target_terms = path_overlap_terms(target_value)
        open_terms = set()
        for path in open_files:
            open_terms |= path_overlap_terms(path)
        if target_kind in ("exact_path", "basename") and target_terms and target_terms <= open_terms:
            target_open = "same"
        elif target_terms & open_terms:
            target_open = "overlap"
        else:
            prompt_terms = path_overlap_terms(prompt)
            target_open = "overlap" if prompt_terms & open_terms else "no"

    prompt_terms = path_overlap_terms(prompt)
    open_dirs = set()
    open_exts = set()
    for path in open_files:
        parts = [part for part in path.lower().split("/") if part]
        open_dirs.update(parts[:-1])
        if parts and "." in parts[-1]:
            open_exts.add(parts[-1].rsplit(".", 1)[-1])
    dir_overlap = "yes" if prompt_terms & open_dirs else "no"
    target_ext = ""
    if "." in target_value:
        target_ext = target_value.lower().rsplit(".", 1)[-1].strip(".,;:()[]{}'\"")
    ext_overlap = "yes" if target_ext and target_ext in open_exts else "no"
    return n_open, target_open, dir_overlap, ext_overlap


def serialize_transformer_sample_current_v10(sample):
    """current_v1 plus typed route/trail/open_rel lines derived from weak-class
    qualitative analysis. Keeps every current_v1 line intact while surfacing
    file-navigation state that raw prompt/history text made hard to separate."""
    base = serialize_transformer_sample_current(sample)
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    action_events = [event for event in history if event.get("role") == "assistant_action"]
    action_names = [safe_text(event.get("name")) for event in action_events if safe_text(event.get("name"))]
    last_event = action_events[-1] if action_events else None
    last_action = action_names[-1] if action_names else "none"
    prev_action = action_names[-2] if len(action_names) > 1 else "none"

    family, op, target_kind, target_value, multiplicity, scope, cue = route_fields(sample)
    n_open, target_open, dir_overlap, ext_overlap = route_open_relation(prompt, ws, target_kind, target_value)
    trail = (
        f"trail: pair={prev_action}>{last_action} "
        f"last_result={route_count_bucket_from_result(last_event.get('result_summary') if last_event else '')} "
        f"last_arg={route_arg_kind(last_event)} candidate_pool={route_candidate_pool(last_event)}"
    )
    route = (
        f"route: family={family} op={op} target={target_kind}:{route_short(target_value, 40)} "
        f"multiplicity={multiplicity} scope={scope} cue={cue}"
    )
    open_rel = (
        f"open_rel: n={n_open} target_open={target_open} "
        f"dir_overlap={dir_overlap} ext_overlap={ext_overlap}"
    )
    lines = base.split("\n")
    return "\n".join([lines[0], route, trail, open_rel] + lines[1:])


V11S_REGISTER = "$" * 16
V11S_HINT_ALL_RE = re.compile(
    r"\*\.[A-Za-z0-9]{1,8}|\*\*|"
    r"\b(glob|pattern|recursive|full\s+list|all\s+(?:the\s+)?files?|"
    r"every\s+file|which\s+files|what\s+files|files?\s+under|files?\s+matching)\b|"
    r"전체|전부|파일\s*다|파일들|목록|노트북\s*파일\s*다",
    re.IGNORECASE,
)
V11S_HINT_CHILD_RE = re.compile(
    r"\b(list|ls|tree|directory|folder|layout|structure|what'?s\s+in|"
    r"what\s+lives\s+under|contents?)\b|"
    r"구조|폴더|디렉토리|디렉터리|뭐뭐|들어있|구성",
    re.IGNORECASE,
)
V11S_HINT_OCC_RE = re.compile(
    r"\b(grep|search|reference|references|occurrence|occurrences|uses?|used|"
    r"calls?|called|where\s+.*\b(used|called|defined|referenced))\b|"
    r"어디서|어디에|어디\s+.*(쓰|사용|호출|참조)|검색|문자열|참조|호출|쓰는지|사용처|레퍼런스",
    re.IGNORECASE,
)
V11S_LITERAL_RE = re.compile(r"`([^`\n]{1,32})`|\"([^\"\n]{1,32})\"|'([^'\n]{1,32})'")


def v11s_num_bin(value, cuts, names):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "u"
    for high, name in zip(cuts, names):
        if number <= high:
            return name
    return names[-1]


def v11s_turn_token(turn_value):
    exact = turn_exact_token(turn_value)
    phase = turn_bin_token(turn_value)
    if exact == "na":
        exact = "xx"
    if phase == "na":
        phase = "u"
    return f"{exact}/{phase}"


def v11s_budget_bin(value):
    return v11s_num_bin(value, (25000, 100000, float("inf")), ("low", "mid", "high"))


def v11s_elapsed_bin(value):
    return v11s_num_bin(value, (300, 1200, float("inf")), ("short", "mid", "long"))


def v11s_bool_token(value):
    if isinstance(value, bool):
        return "T" if value else "F"
    text = safe_text(value).lower()
    if text in ("true", "t", "1", "yes"):
        return "T"
    if text in ("false", "f", "0", "no"):
        return "F"
    return "u"


def v11s_ci_token(value):
    text = safe_text(value).lower()
    if text in ("passed", "pass", "ok", "clean", "success"):
        return "pass"
    if text in ("failed", "fail", "error", "errored", "red"):
        return "fail"
    if text in ("none", "na", ""):
        return "none"
    return "u"


def v11s_basename(value):
    text = safe_text(value).replace("\\", "/").strip()
    if not text:
        return ""
    return route_short(text.rsplit("/", 1)[-1], 32)


def v11s_open_token(ws):
    open_files = ws.get("open_files") or []
    count = min(len(open_files), 6)
    bases = [v11s_basename(path) for path in open_files[:2]]
    bases = [base for base in bases if base]
    return f"{count}:{'|'.join(bases) if bases else 'na'}"


def v11s_hint_token(prompt):
    text = safe_text(prompt)
    if V11S_HINT_ALL_RE.search(text):
        return "a"
    if V11S_HINT_OCC_RE.search(text):
        return "o"
    if V11S_HINT_CHILD_RE.search(text):
        return "c"
    return ""


def v11s_literal_token(prompt):
    match = V11S_LITERAL_RE.search(safe_text(prompt))
    if not match:
        return ""
    value = next((group for group in match.groups() if group), "")
    return route_short(value, 32)


def v11s_qp_tokens(prompt):
    target_kind, target_value = route_target(prompt)
    q_kind, q_value = "none", "na"
    p_kind, p_value = "none", "na"
    if target_kind == "symbol":
        q_kind, q_value = "sym", route_short(target_value, 32)
    elif target_kind == "exact_path":
        p_kind, p_value = "exact", route_short(target_value, 32)
    elif target_kind == "basename":
        p_kind, p_value = "base", route_short(target_value, 32)
    elif target_kind == "glob":
        p_kind, p_value = "glob", route_short(target_value, 32)
    elif target_kind == "dir":
        p_kind, p_value = "dir", route_short(target_value, 32)
    elif target_kind == "pronoun":
        p_kind, p_value = "pro", "it"
    literal = v11s_literal_token(prompt)
    if q_kind == "none" and literal:
        q_kind, q_value = "lit", literal
    return q_kind, q_value, p_kind, p_value, target_kind, target_value


def v11s_arg_token(arg_kind):
    return {
        "exact_path": "path",
        "glob": "glob",
        "symbol": "sym",
        "none": "none",
    }.get(arg_kind, arg_kind)


def v11s_nav_line(sample):
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    action_events = [event for event in history if event.get("role") == "assistant_action"]
    action_names = [safe_text(event.get("name")) for event in action_events if safe_text(event.get("name"))]
    last_event = action_events[-1] if action_events else None
    last_action = action_names[-1] if action_names else "none"
    prev_action = action_names[-2] if len(action_names) > 1 else "none"

    q_kind, q_value, p_kind, p_value, target_kind, target_value = v11s_qp_tokens(prompt)
    hint = v11s_hint_token(prompt)
    n_open, target_open, dir_overlap, ext_overlap = route_open_relation(prompt, ws, target_kind, target_value)
    open_token = "0" if n_open <= 0 else f"{n_open}/{target_open}/{dir_overlap}/{ext_overlap}"
    pieces = [
        f"q={q_kind}:{q_value}",
        f"p={p_kind}:{p_value}",
    ]
    if hint:
        pieces.append(f"h={hint}")
    pieces.extend([
        f"c={prev_action}>{last_action}",
        f"r={route_count_bucket_from_result(last_event.get('result_summary') if last_event else '')}@{route_candidate_pool(last_event)}",
        f"a={v11s_arg_token(route_arg_kind(last_event))}",
        f"o={open_token}",
    ])
    return f"nav: {' '.join(pieces)}"


def serialize_transformer_sample_current_v11s(sample):
    """Scaffold-preserving v1 variant: constant register + compact navigation
    evidence + masked high-entropy meta/workspace values."""
    base = serialize_transformer_sample_current(sample)
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    meta = (
        f"meta: tier=x lang=x turn={v11s_turn_token(sm.get('turn_index'))} "
        f"budget={v11s_budget_bin(sm.get('budget_tokens_remaining'))} "
        f"elapsed={v11s_elapsed_bin(sm.get('elapsed_session_sec'))}"
    )
    workspace = (
        f"workspace: dirty={v11s_bool_token(ws.get('git_dirty'))} "
        f"ci={v11s_ci_token(ws.get('last_ci_status'))} loc=x "
        f"langs={top_language_pair(ws)} open={v11s_open_token(ws)}"
    )
    out = []
    for line in base.split("\n"):
        if line.startswith("current: "):
            out.append(line)
            out.append(f"reg: {V11S_REGISTER}")
            out.append(v11s_nav_line(sample))
        elif line.startswith("meta: "):
            out.append(meta)
        elif line.startswith("workspace: "):
            out.append(workspace)
        else:
            out.append(line)
    return "\n".join(out)


def serialize_transformer_sample_current_v2(sample):
    """Priority-ordered rewrite of current_v1: highest-signal fields first so
    right-truncation drops the oldest history pairs instead of args/results,
    and every user utterance is kept (current_v1 kept only the last one),
    newest first as full user->action pairs."""
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}

    action_events = [e for e in history if e.get("role") == "assistant_action"]
    action_names = [safe_text(e.get("name")) for e in action_events if safe_text(e.get("name"))]
    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict):
        langs = " ".join(f"{safe_text(k)}={float(v):.2f}" for k, v in list(language_mix.items())[:5])
    else:
        langs = ""
    open_files = " | ".join(compact_text(x, 48) for x in (ws.get("open_files") or [])[:6])

    parts = [
        f"current: {prompt}",
        f"state: turn={safe_text(sm.get('turn_index'))} tier={safe_text(sm.get('user_tier'))} "
        f"lang={safe_text(sm.get('language_pref'))} budget={safe_text(sm.get('budget_tokens_remaining'))} "
        f"elapsed={safe_text(sm.get('elapsed_session_sec'))} dirty={safe_text(ws.get('git_dirty'))} "
        f"ci={safe_text(ws.get('last_ci_status'))} loc={safe_text(ws.get('loc'))} "
        f"langs={langs} open={open_files}",
        f"acts: {' > '.join(action_names) if action_names else 'none'}",
    ]
    if action_events:
        last = action_events[-1]
        parts.append(
            f"last: {safe_text(last.get('name'))} "
            f"args={compact_action_args(last, max_items=4, value_limit=80)} "
            f"res={compact_text(last.get('result_summary'), 120) or 'na'}"
        )
    pairs = history_user_action_pairs(history)
    for idx, (user_text, event) in enumerate(reversed(pairs), 1):
        parts.append(
            f"p{idx}: u={compact_text(user_text, 200)} -> {safe_text(event.get('name'))} "
            f"args={compact_action_args(event, max_items=2, value_limit=60)} "
            f"res={compact_text(event.get('result_summary'), 100) or 'na'}"
        )
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


def compact_text(value, limit=120):
    text = re.sub(r"\s+", " ", safe_text(value)).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def compact_arg_value(value, limit=80):
    if isinstance(value, dict):
        pieces = [f"{safe_text(k)}:{compact_text(v, 40)}" for k, v in list(value.items())[:4]]
        text = " ".join(pieces)
    elif isinstance(value, (list, tuple)):
        text = " ".join(compact_text(item, 40) for item in value[:4])
    else:
        text = safe_text(value)
    return compact_text(text, limit)


ARG_KEY_PRIORITY = (
    "path",
    "file",
    "pattern",
    "glob",
    "query",
    "regex",
    "cmd",
    "command",
    "url",
    "cwd",
)


def compact_action_args(event, max_items=4, value_limit=80):
    args = event.get("args") or {}
    if not isinstance(args, dict) or not args:
        return "na"

    def priority(item):
        order, key, _ = item
        key_l = safe_text(key).lower()
        for idx, token in enumerate(ARG_KEY_PRIORITY):
            if token in key_l:
                return idx, order
        return len(ARG_KEY_PRIORITY), order

    selected = sorted(
        [(order, key, value) for order, (key, value) in enumerate(args.items())],
        key=priority,
    )[:max_items]
    bits = []
    for _, key, value in selected:
        key_text = compact_text(key, 24)
        value_text = compact_arg_value(value, value_limit)
        if value_text:
            bits.append(f"{key_text}={value_text}")
    return " ".join(bits) if bits else "na"


def result_semantic(result_summary):
    text = safe_text(result_summary).lower()
    if not text:
        return "na"
    if re.search(r"\b(exit code|return code)\s*[:=]?\s*0\b", text):
        return "pass"
    if re.search(r"\b(exit code|return code)\s*[:=]?\s*[1-9][0-9]*\b", text):
        return "fail"
    if re.search(r"\b(no matches?|not found|0 matches?|empty|no results?|none)\b", text):
        return "none"
    if re.search(r"\b(error|failed|failure|fail|traceback|exception|permission denied|timed out|timeout)\b", text):
        return "fail"
    if re.search(r"\b(pass(?:ed)?|success(?:ful)?|succeeded|ok|clean|no errors?)\b", text):
        return "pass"
    if re.search(r"\b(found|matches?|occurrences?|results?)\b", text):
        return "found"
    if re.search(r"\b(exit code|return code)\b", text):
        return "exit"
    return "na"


def workspace_language_signal(ws):
    language_mix = ws.get("language_mix") or {}
    if isinstance(language_mix, dict) and language_mix:
        top_lang, _ = sorted(language_mix.items(), key=lambda kv: (-float(kv[1]), safe_text(kv[0])))[0]
        return safe_text(top_lang).lower()
    counts = {}
    for path in ws.get("open_files") or []:
        base = safe_text(path).replace("\\", "/").rsplit("/", 1)[-1]
        if "." in base:
            ext = base.rsplit(".", 1)[-1].lower()
            counts[ext] = counts.get(ext, 0) + 1
    if counts:
        return "ext:" + sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return "na"


def serialize_transformer_sample_state_v2(sample):
    prompt = safe_text(sample.get("current_prompt", ""))
    history = sample.get("history") or []
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    action_events = [event for event in history if event.get("role") == "assistant_action"]
    action_names = [safe_text(event.get("name")) for event in action_events if safe_text(event.get("name"))]

    parts = [f"cur: {prompt}"]
    if action_events:
        last = action_events[-1]
        parts.append(
            f"last: a={safe_text(last.get('name')) or 'na'} "
            f"args={compact_action_args(last, max_items=2, value_limit=36)} "
            f"res={result_semantic(last.get('result_summary'))}"
        )

    pairs = list(reversed(history_user_action_pairs(history)[-3:]))
    for idx, (user_text, action_event) in enumerate(pairs, 1):
        result = safe_text(action_event.get("result_summary"))
        summary = compact_text(result, 24)
        line = (
            f"h{idx}: u={compact_text(user_text, 52)} "
            f"a={safe_text(action_event.get('name')) or 'na'} "
            f"args={compact_action_args(action_event, max_items=1, value_limit=32)} "
            f"res={result_semantic(result)}"
        )
        if summary:
            line += f" \"{summary}\""
        parts.append(line)

    parts.append(f"acts: {' > '.join(action_names[-8:]) if action_names else 'none'}")
    open_files = [compact_text(path, 48) for path in (ws.get("open_files") or [])[:4]]
    parts.append(
        f"ws: turn={safe_text(sm.get('turn_index'))} "
        f"dirty={safe_text(ws.get('git_dirty'))} "
        f"ci={safe_text(ws.get('last_ci_status'))} "
        f"open={' | '.join(open_files) if open_files else 'none'} "
        f"lang={workspace_language_signal(ws)}"
    )
    return "\n".join(parts)


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
        semantic = result_semantic(result)
        if semantic != "na":
            tokens.append(f"result_{name}_{semantic}")
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


def serialize_transformer_sample(sample, serializer_name="current_v1", tokenizer=None):
    if serializer_name in ("current", "current_v1"):
        return serialize_transformer_sample_current(sample)
    if serializer_name == "chat_v1_contract":
        return serialize_transformer_sample_chat_v1_contract(sample, tokenizer=tokenizer)
    if serializer_name == "weak_nav_v1":
        return serialize_transformer_sample_weak_nav_v1(sample)
    if serializer_name == "weak_nav_paths_v1":
        return serialize_transformer_sample_weak_nav_paths_v1(sample)
    if serializer_name == "current_v2":
        return serialize_transformer_sample_current_v2(sample)
    if serializer_name == "current_v5":
        return serialize_transformer_sample_current_v5(sample)
    if serializer_name == "current_v6":
        return serialize_transformer_sample_current_v6(sample)
    if serializer_name == "current_v6e":
        return serialize_transformer_sample_current_v6e(sample)
    if serializer_name == "current_v7":
        return serialize_transformer_sample_current_v7(sample)
    if serializer_name == "current_v7r":
        return serialize_transformer_sample_current_v7r(sample)
    if serializer_name == "current_v7rl":
        return serialize_transformer_sample_current_v7rl(sample)
    if serializer_name == "current_v7rm":
        return serialize_transformer_sample_current_v7rm(sample)
    if serializer_name == "current_v7rd":
        return serialize_transformer_sample_current_v7rd(sample)
    if serializer_name == "current_v8":
        return serialize_transformer_sample_current_v8(sample)
    if serializer_name == "current_v8t":
        return serialize_transformer_sample_current_v8t(sample)
    if serializer_name == "current_v7rb":
        return serialize_transformer_sample_current_v7rb(sample)
    if serializer_name == "current_v7rc":
        return serialize_transformer_sample_current_v7rc(sample)
    if serializer_name == "current_v7rw":
        return serialize_transformer_sample_current_v7rw(sample)
    if serializer_name == "current_v7rg":
        return serialize_transformer_sample_current_v7rg(sample)
    if serializer_name == "current_v7rcgw":
        return serialize_transformer_sample_current_v7rcgw(sample)
    if serializer_name == "current_v9o":
        return serialize_transformer_sample_current_v9o(sample)
    if serializer_name == "current_v9f":
        return serialize_transformer_sample_current_v9f(sample)
    if serializer_name == "current_v9h":
        return serialize_transformer_sample_current_v9h(sample)
    if serializer_name == "current_v10":
        return serialize_transformer_sample_current_v10(sample)
    if serializer_name == "current_v11s":
        return serialize_transformer_sample_current_v11s(sample)
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
    for event in history:
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
        result = safe_text(event.get("result_summary", "")).lower()
        if result:
            semantic = result_semantic(result)
            if semantic != "na":
                result_tokens.append(f"result_{semantic}")
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


def load_sparse_ensemble(model_dir, class_names):
    model_path = os.path.join(model_dir, "sparse_svc.pkl")
    meta_path = os.path.join(model_dir, "sparse_meta.json")
    if not os.path.exists(model_path):
        return None
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Found sparse_svc.pkl but missing sparse_meta.json")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    sparse_classes = meta.get("classes", class_names)
    if list(sparse_classes) != list(class_names):
        raise ValueError("sparse ensemble class order does not match transformer class order")
    return {
        "vectorizer": payload["vectorizer"],
        "model": payload["model"],
        "meta": meta,
    }


def sparse_ensemble_scores(sparse_ensemble, samples):
    serializer_name = sparse_ensemble.get("meta", {}).get("text_serializer", "current_v1")
    texts = [serialize_transformer_sample(sample, serializer_name) for sample in samples]
    features = sparse_ensemble["vectorizer"].transform(texts)
    scores = sparse_ensemble["model"].decision_function(features)
    if getattr(scores, "ndim", 1) != 2:
        raise ValueError("expected multiclass sparse SVC decision scores")
    model_classes = [int(value) for value in getattr(sparse_ensemble["model"], "classes_", range(len(ALL_CLASSES)))]
    expected = list(range(len(ALL_CLASSES)))
    if model_classes != expected:
        order = [model_classes.index(class_id) for class_id in expected]
        scores = scores[:, order]
    return torch.tensor(scores, dtype=torch.float32)


def apply_sparse_controls(sparse_scores, base_logits, sparse_meta, class_names):
    class_mask = sparse_meta.get("sparse_class_mask") or sparse_meta.get("class_mask") or []
    gate_margin = sparse_meta.get("sparse_gate_margin", sparse_meta.get("gate_margin"))
    gate_topk = int(sparse_meta.get("sparse_gate_topk", sparse_meta.get("gate_topk", 0)) or 0)
    if not class_mask and gate_margin in (None, "", "none") and gate_topk <= 0:
        return sparse_scores

    adjusted = sparse_scores.clone()
    mask_idx = []
    if class_mask:
        class_to_idx = {label: idx for idx, label in enumerate(class_names)}
        mask_idx = [class_to_idx[label] for label in class_mask if label in class_to_idx]
        if mask_idx:
            keep = torch.zeros(adjusted.shape[1], dtype=torch.bool, device=adjusted.device)
            keep[torch.tensor(mask_idx, dtype=torch.long, device=adjusted.device)] = True
            adjusted[:, ~keep] = 0.0

    gate = None
    if gate_margin not in (None, "", "none"):
        top2 = base_logits.topk(2, dim=1).values
        gate = (top2[:, 0] - top2[:, 1]) <= float(gate_margin)
    if gate_topk > 0 and mask_idx:
        top_idx = base_logits.topk(min(gate_topk, base_logits.shape[1]), dim=1).indices
        weak = torch.zeros(base_logits.shape[1], dtype=torch.bool, device=base_logits.device)
        weak[torch.tensor(mask_idx, dtype=torch.long, device=base_logits.device)] = True
        top_gate = weak[top_idx].any(dim=1)
        gate = top_gate if gate is None else (gate & top_gate)
    if gate is not None:
        adjusted[~gate.to(adjusted.device)] = 0.0
    return adjusted


def normalize_prior(values, floor=1e-4):
    prior = torch.tensor(values, dtype=torch.float32)
    if prior.ndim != 1:
        raise ValueError("prior must be a 1D vector")
    prior = torch.clamp(prior, min=float(floor))
    return prior / prior.sum().clamp_min(1e-12)


def batch_prior_calibration_bias(base_scores, calibration, class_names, device):
    """Estimate a small transductive class bias from the whole test batch.

    The packaged calibration matrix is P(model-prob-class | true-class) from a
    held-out run.  At inference we match the test batch's mean softmax vector to
    that matrix, shrink the solved prior toward the reference validation prior,
    then add a capped log-prior-ratio bias.  This is intentionally conservative:
    it changes global posterior mass, not individual labels.
    """
    if not calibration or not calibration.get("enabled", False):
        return torch.zeros(len(class_names), dtype=torch.float32, device=device)
    cal_classes = calibration.get("classes", class_names)
    if list(cal_classes) != list(class_names):
        raise ValueError("prior calibration class order does not match model class order")

    n_classes = len(class_names)
    floor = float(calibration.get("prior_floor", 1e-4))
    ref_prior = normalize_prior(calibration["reference_prior"], floor=floor)
    confusion = torch.tensor(
        calibration.get("soft_confusion") or calibration.get("confusion"),
        dtype=torch.float32,
    )
    if confusion.shape != (n_classes, n_classes):
        raise ValueError(f"prior calibration confusion shape {tuple(confusion.shape)} != {(n_classes, n_classes)}")

    probs = torch.softmax(base_scores.float().cpu(), dim=-1)
    q_test = torch.clamp(probs.mean(dim=0), min=floor)
    q_test = q_test / q_test.sum().clamp_min(1e-12)

    ridge = float(calibration.get("ridge", 0.05))
    eye = torch.eye(n_classes, dtype=torch.float32)
    lhs = confusion.T @ confusion + ridge * eye
    rhs = confusion.T @ q_test + ridge * ref_prior
    try:
        solved = torch.linalg.solve(lhs, rhs)
    except Exception:
        solved = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution.squeeze(1)
    solved = torch.clamp(solved, min=floor)
    solved = solved / solved.sum().clamp_min(1e-12)

    prior_blend = float(calibration.get("prior_blend", 0.25))
    prior_blend = max(0.0, min(1.0, prior_blend))
    target_prior = (1.0 - prior_blend) * ref_prior + prior_blend * solved
    target_prior = torch.clamp(target_prior, min=floor)
    target_prior = target_prior / target_prior.sum().clamp_min(1e-12)

    bias_scale = float(calibration.get("bias_scale", 0.35))
    cap = float(calibration.get("bias_cap", 0.18))
    bias = bias_scale * torch.log(target_prior / ref_prior.clamp_min(floor))
    bias = torch.clamp(bias, min=-cap, max=cap)

    protected = set(calibration.get("protected_classes", []))
    protected_cap = calibration.get("protected_cap")
    if protected and protected_cap is not None:
        protected_cap = float(protected_cap)
        for idx, label in enumerate(class_names):
            if label in protected:
                if protected_cap <= 0:
                    bias[idx] = 0.0
                else:
                    bias[idx] = torch.clamp(bias[idx], min=-protected_cap, max=protected_cap)

    top_changes = sorted(
        ((class_names[i], float(bias[i]), float(ref_prior[i]), float(target_prior[i]), float(q_test[i])) for i in range(n_classes)),
        key=lambda row: abs(row[1]),
        reverse=True,
    )[:5]
    print(
        "Prior calibration: "
        f"blend={prior_blend:.2f} scale={bias_scale:.2f} cap={cap:.2f} "
        f"top_bias={[(name, round(delta, 4)) for name, delta, _, _, _ in top_changes]}"
    )
    return bias.to(device)


LEAK_LOOKUP_FILENAME = "leak_lookup.json.gz"
LEAK_LOOKUP_FORMAT = "leak-lookup-v2"
LEAK_STEP_RE = re.compile(r"^(?P<sess>.+)-step_(?P<step>\d+)$")


def leak_text_key(text):
    return hashlib.sha1(safe_text(text).encode("utf-8")).hexdigest()


def history_pair_labels(history):
    """(user_content, action_name) pairs; history is strictly user/action alternating."""
    pairs = []
    for content, event in history_user_action_pairs(history or []):
        name = safe_text(event.get("name"))
        if name:
            pairs.append((content, name))
    return pairs


def leak_hashed_pairs(sample):
    return [(leak_text_key(content), action) for content, action in history_pair_labels(sample.get("history"))]


def leak_last_action(sample):
    pairs = history_pair_labels(sample.get("history"))
    return pairs[-1][1] if pairs else "NONE"


def load_leak_lookup(model_dir):
    path = os.path.join(model_dir, LEAK_LOOKUP_FILENAME)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != LEAK_LOOKUP_FORMAT:
        raise ValueError(f"unknown leak lookup format: {payload.get('format')}")
    return payload


def compute_leak_overrides(samples, train_lookup=None, valid_classes=None):
    """Recover labels for rows whose outcome is embedded in other rows' histories.

    Tier "positional": the last m user/action pairs of row (sess, step) are exactly
    steps step-m..step-1 of the same session, so a later row pins the label of an
    earlier one via step arithmetic alone (train: 60,553 recovered, 0 wrong).
    Tier "aligned": id-free variant for anonymized ids — row X matches pair j of
    row Y only if X's entire (user, action) history equals Y's pairs[j-m:j], and
    every such candidate agrees on the action.
    Tier "train_prompt_last" / "train_prompt": conflict-free lookups built from
    training data (model/leak_lookup.json.gz), keyed by (prompt, last action) and
    by prompt alone (holdout precision 0.978 / 0.918 vs model ~0.74). Bare
    cross-session prompt matching inside the test set is deliberately absent:
    it measured 0.427 on train.
    """
    valid = set(valid_classes or ALL_CLASSES)
    overrides = {}
    stats = {"positional": 0, "aligned": 0, "train_prompt_last": 0, "train_prompt": 0}

    def record(sample_id, action, tier):
        if sample_id and action in valid and sample_id not in overrides:
            overrides[sample_id] = action
            stats[tier] += 1

    parsed = [LEAK_STEP_RE.match(safe_text(sample.get("id", ""))) for sample in samples]
    if parsed and all(match is not None for match in parsed):
        by_session = {}
        for sample, match in zip(samples, parsed):
            by_session.setdefault(match.group("sess"), {})[int(match.group("step"))] = sample
        for steps in by_session.values():
            for step, sample in steps.items():
                pairs = history_pair_labels(sample.get("history"))
                for offset, (_, action) in enumerate(pairs):
                    source = steps.get(step - len(pairs) + offset)
                    if source is not None:
                        record(safe_text(source.get("id")), action, "positional")

    remaining = [sample for sample in samples if safe_text(sample.get("id")) not in overrides]
    if remaining:
        pair_index = {}
        for sample in samples:
            hashed = leak_hashed_pairs(sample)
            for j, (content_key, _) in enumerate(hashed):
                pair_index.setdefault(content_key, []).append((hashed, j))
        for sample in remaining:
            own = leak_hashed_pairs(sample)
            if not own:
                continue  # empty history matches on prompt alone, which is unreliable
            prompt_key = leak_text_key(sample.get("current_prompt"))
            candidates = set()
            for hashed, j in pair_index.get(prompt_key, ()):
                m = len(own)
                if j - m >= 0 and hashed[j - m:j] == own:
                    candidates.add(hashed[j][1])
            if len(candidates) == 1:
                record(safe_text(sample.get("id")), candidates.pop(), "aligned")

    if train_lookup:
        by_prompt_last = train_lookup.get("by_prompt_last", {})
        by_prompt = train_lookup.get("by_prompt", {})
        for sample in samples:
            sample_id = safe_text(sample.get("id"))
            if sample_id in overrides:
                continue
            prompt_key = leak_text_key(sample.get("current_prompt"))
            record(sample_id, by_prompt_last.get(f"{prompt_key}|{leak_last_action(sample)}"), "train_prompt_last")
            record(sample_id, by_prompt.get(prompt_key), "train_prompt")

    return overrides, stats


def compute_test_batch_graph_overrides(samples, valid_classes=None, use_positional=True, use_aligned=True):
    """Recover only labels implied by other rows in the same test batch.

    This intentionally does not use train-derived prompt lookups.  A row is
    overridden only when same-batch histories point to a single valid action,
    and positional matches also verify the source row's current prompt against
    the history content before trusting step arithmetic.
    """
    valid = set(valid_classes or ALL_CLASSES)
    candidates = {}
    candidate_tiers = {}

    def add_candidate(sample_id, action, tier):
        if not sample_id or action not in valid:
            return
        candidates.setdefault(sample_id, set()).add(action)
        candidate_tiers.setdefault(sample_id, {}).setdefault(action, set()).add(tier)

    if use_positional:
        parsed = [LEAK_STEP_RE.match(safe_text(sample.get("id", ""))) for sample in samples]
        by_session = {}
        for sample, match in zip(samples, parsed):
            if match is None:
                continue
            by_session.setdefault(match.group("sess"), {})[int(match.group("step"))] = sample
        for steps in by_session.values():
            for step, sample in steps.items():
                pairs = history_pair_labels(sample.get("history"))
                for offset, (content, action) in enumerate(pairs):
                    source = steps.get(step - len(pairs) + offset)
                    if source is None:
                        continue
                    if leak_text_key(source.get("current_prompt")) != leak_text_key(content):
                        continue
                    add_candidate(safe_text(source.get("id")), action, "positional")

    if use_aligned:
        pair_index = {}
        for sample in samples:
            owner_id = safe_text(sample.get("id"))
            hashed = leak_hashed_pairs(sample)
            for j, (content_key, action) in enumerate(hashed):
                pair_index.setdefault(content_key, []).append((owner_id, hashed, j, action))
        for sample in samples:
            sample_id = safe_text(sample.get("id"))
            own = leak_hashed_pairs(sample)
            if not own:
                continue
            prompt_key = leak_text_key(sample.get("current_prompt"))
            for owner_id, hashed, j, action in pair_index.get(prompt_key, ()):
                if owner_id == sample_id:
                    continue
                m = len(own)
                if j - m >= 0 and hashed[j - m:j] == own:
                    add_candidate(sample_id, action, "aligned")

    overrides = {}
    stats = {"positional": 0, "aligned": 0, "conflict": 0}
    for sample in samples:
        sample_id = safe_text(sample.get("id"))
        actions = candidates.get(sample_id, set())
        if not actions:
            continue
        if len(actions) != 1:
            stats["conflict"] += 1
            continue
        action = next(iter(actions))
        tiers = candidate_tiers.get(sample_id, {}).get(action, set())
        tier = "positional" if "positional" in tiers else "aligned"
        overrides[sample_id] = action
        stats[tier] += 1
    return overrides, stats


INT8_FORMAT_VERSION = "int8-rowwise-v1"
INT8_SCALE_SUFFIX = ".__scale__"
INT8_PATCH_ROWS_SUFFIX = ".__patch_rows__"
INT8_PATCH_IDX_SUFFIX = ".__patch_idx__"
INT8_AUX_SUFFIXES = (INT8_SCALE_SUFFIX, INT8_PATCH_ROWS_SUFFIX, INT8_PATCH_IDX_SUFFIX)

INT4_SCALE_SUFFIX = ".__scale__"
INT4_BASE_ROW_IDX_SUFFIX = ".__base_row_idx__"
INT4_INT8_ROW_IDX_SUFFIX = ".__int8_row_idx__"
INT4_INT8_ROWS_SUFFIX = ".__int8_rows__"
INT4_INT8_ROW_SCALE_SUFFIX = ".__int8_row_scale__"
INT4_ROW_AUX_SUFFIXES = (
    INT4_BASE_ROW_IDX_SUFFIX,
    INT4_INT8_ROW_IDX_SUFFIX,
    INT4_INT8_ROWS_SUFFIX,
    INT4_INT8_ROW_SCALE_SUFFIX,
)


def load_int8_state_dict(path, dtype=torch.float32, shared_from=None):
    """Reconstruct an fp state_dict from a quantize_checkpoint.py int8 codec file.

    Tensors listed in the sidecar meta's `shared_tensors` are stored as a donor
    reference plus a sparse int8 row patch (rows that differ from the donor) and
    this leg's own scales — reconstruction is bit-exact, so multi-model packs
    can store near-identical large tensors (e.g. embeddings) once."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    packed = load_file(path)
    with open(path + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["format"] != INT8_FORMAT_VERSION:
        raise ValueError(f"unknown int8 codec format {meta['format']}")
    quantized = set(meta["quantized"])
    state = {}
    for name, tensor in packed.items():
        if name.endswith(INT8_AUX_SUFFIXES):
            continue
        if name in quantized:
            scale = packed[name + INT8_SCALE_SUFFIX]
            shaped = scale.view(-1, *([1] * (tensor.ndim - 1)))
            state[name] = (tensor.float() * shaped).to(dtype)
        elif tensor.is_floating_point():
            state[name] = tensor.to(dtype=dtype, copy=True)
        else:
            state[name] = tensor.clone()
    shared_names = meta.get("shared_tensors") or []
    if shared_names:
        if not shared_from:
            raise ValueError(f"{path} declares shared_tensors {shared_names} but no donor dir was given")
        donor_path = os.path.join(shared_from, "model.int8.safetensors")
        with safe_open(donor_path, framework="pt") as donor:
            for name in shared_names:
                q = donor.get_tensor(name).clone()
                idx = packed.get(name + INT8_PATCH_IDX_SUFFIX)
                rows = packed.get(name + INT8_PATCH_ROWS_SUFFIX)
                if idx is not None and rows is not None:
                    q[idx.long()] = rows
                scale = packed[name + INT8_SCALE_SUFFIX]
                shaped = scale.view(-1, *([1] * (q.ndim - 1)))
                state[name] = (q.float() * shaped).to(dtype)
        print(f"Reconstructed shared tensors from {donor_path}: {shared_names}")
    return state


def load_int4_state_dict(path, dtype=torch.float16):
    """Reconstruct fp weights from the local int4 group/mixed storage codec.

    ``int4-mixed-v1`` can keep selected tensors in fp16, store selected full
    tensors rowwise-int8, and split hot embedding rows into rowwise-int8 while
    the remaining rows stay grouped int4. This is a storage codec only: the
    returned model still runs with ordinary fp16/fp32 kernels.
    """
    from safetensors.torch import load_file

    path = os.fspath(path)
    packed = load_file(path)
    with open(path + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    expected_formats = {f"int4-group{meta['group_size']}-v1", "int4-mixed-v1"}
    if meta.get("format") not in expected_formats:
        raise ValueError(f"unknown int4 codec format {meta.get('format')}")

    quantized = set(meta["quantized"])
    tensor_group_sizes = meta.get("tensor_group_sizes") or {}
    split_rowwise_int8 = set(meta.get("split_rowwise_int8") or [])
    rowwise_int8 = set(meta.get("rowwise_int8") or [])
    state = {}
    for name, tensor in packed.items():
        if name.endswith(INT4_SCALE_SUFFIX) or name.endswith(INT4_ROW_AUX_SUFFIXES):
            continue
        if name in rowwise_int8:
            scale = packed[name + INT4_SCALE_SUFFIX].float()
            shaped = scale.view(-1, *([1] * (tensor.ndim - 1)))
            state[name] = (tensor.float() * shaped).to(dtype)
        elif name in quantized:
            shape = meta["shapes"][name]
            rows = tensor.shape[0]
            lo = (tensor & 0x0F).to(torch.int8) - 8
            hi = (tensor >> 4).to(torch.int8) - 8
            q = torch.stack((lo, hi), dim=2).reshape(rows, -1)
            scale = packed[name + INT4_SCALE_SUFFIX].float()
            cols = 1
            for dim in shape[1:]:
                cols *= dim
            group_size = int(tensor_group_sizes.get(name, meta["group_size"]))
            padded_cols = ((cols + group_size - 1) // group_size) * group_size
            q = q[:, :padded_cols]
            w = (
                q.float().reshape(rows, -1, group_size) * scale.unsqueeze(2)
            ).reshape(rows, -1)
            base_w = w[:, :cols]
            if name in split_rowwise_int8:
                restored = torch.empty(shape, dtype=dtype)
                base_idx = packed[name + INT4_BASE_ROW_IDX_SUFFIX].long()
                int8_idx = packed[name + INT4_INT8_ROW_IDX_SUFFIX].long()
                restored[base_idx] = base_w.to(dtype)
                int8_scale = packed[name + INT4_INT8_ROW_SCALE_SUFFIX].float().unsqueeze(1)
                int8_w = packed[name + INT4_INT8_ROWS_SUFFIX].float() * int8_scale
                restored[int8_idx] = int8_w.to(dtype)
                state[name] = restored
            else:
                state[name] = base_w.reshape(shape).to(dtype)
        elif tensor.is_floating_point():
            state[name] = tensor.to(dtype=dtype, copy=True)
        else:
            state[name] = tensor.clone()
    return state


def disable_decoder_cache(model):
    """Sequence classification never reuses KV/cache; force it off for decoder
    configs whose library version may otherwise default to config.use_cache."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    base = getattr(model, "model", None)
    base_config = getattr(base, "config", None)
    if base_config is not None and hasattr(base_config, "use_cache"):
        base_config.use_cache = False


def load_hf_model(hf_dir, device, shared_from=None):
    from transformers import AutoConfig, AutoModelForSequenceClassification

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    int4_path = os.path.join(hf_dir, "model.int4.safetensors")
    int8_path = os.path.join(hf_dir, "model.int8.safetensors")
    if os.path.exists(int4_path):
        # Decode before allocating the full model so the largest temporary
        # dequantization tensor never overlaps a second 1.5B parameter copy.
        state = load_int4_state_dict(int4_path, dtype=dtype)
        config = AutoConfig.from_pretrained(hf_dir, local_files_only=True)
        config.torch_dtype = dtype
        model = AutoModelForSequenceClassification.from_config(config, torch_dtype=dtype)
        if dtype == torch.float16:
            model.half()
        else:
            model.float()
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"int4 checkpoint mismatch: missing={missing} unexpected={unexpected}")
        del state
        disable_decoder_cache(model)
        print(f"Loaded int4-codec checkpoint {os.path.basename(int4_path)}")
        return model.to(device)
    if os.path.exists(int8_path):
        config = AutoConfig.from_pretrained(hf_dir, local_files_only=True)
        if dtype == torch.float16:
            config.torch_dtype = torch.float16
        model = AutoModelForSequenceClassification.from_config(config)
        if dtype == torch.float16:
            model.half()
        else:
            model.float()
        state = load_int8_state_dict(int8_path, dtype=dtype, shared_from=shared_from)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"int8 checkpoint mismatch: missing={missing} unexpected={unexpected}")
        disable_decoder_cache(model)
        print(f"Loaded int8-codec checkpoint {os.path.basename(int8_path)}")
        return model.to(device)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_dir, local_files_only=True, torch_dtype=dtype
    )
    disable_decoder_cache(model)
    return model.to(device)


def model_logits_sorted(
    model,
    tokenizer,
    texts,
    max_length,
    batch_size,
    device,
    terminal_token="",
):
    """Tokenize once, infer in length-sorted batches (cuts padding waste ~30-40%
    on the T4/10-min budget), return logits in the original order [N, C]."""
    encoded_all = tokenize_texts_with_terminal(
        tokenizer,
        texts,
        max_length,
        terminal_token,
    )
    keys = list(encoded_all.keys())
    feats = [{key: encoded_all[key][i] for key in keys} for i in range(len(texts))]
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    out = [None] * len(feats)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            batch = tokenizer.pad([feats[i] for i in chunk], padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.float().cpu()
            for row, i in enumerate(chunk):
                out[i] = logits[row]
    return torch.stack(out, dim=0)


def validate_weak4_specialist_config(config, class_names=ALL_CLASSES):
    if not isinstance(config, dict):
        raise TypeError("weak4_specialist must be an object")
    classes = [int(value) for value in config.get("classes", [])]
    if classes != list(range(4)):
        raise ValueError(f"weak4_specialist classes must be canonical [0, 1, 2, 3], got {classes}")
    if list(class_names[:4]) != ALL_CLASSES[:4]:
        raise ValueError("model class order does not preserve the canonical Weak4 prefix")
    alpha = float(config.get("alpha", -1.0))
    route_fraction = float(config.get("route_fraction", -1.0))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"weak4_specialist alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= route_fraction <= 1.0:
        raise ValueError(
            f"weak4_specialist route_fraction must be in [0, 1], got {route_fraction}"
        )
    if not safe_text(config.get("lora_dir")).strip():
        raise ValueError("weak4_specialist lora_dir is required")
    serializer_name = config.get("serializer_name", "weak_nav_v1")
    if serializer_name not in {"current_v1", "weak_nav_v1", "weak_nav_paths_v1"}:
        raise ValueError(f"unsupported weak4 specialist serializer: {serializer_name}")
    if int(config.get("max_length", 0)) <= 0 or int(config.get("batch_size", 0)) <= 0:
        raise ValueError("weak4_specialist max_length and batch_size must be positive")
    return classes, alpha, route_fraction


def select_weak4_routes(main_logits, classes, route_fraction):
    """Route raw-Weak4 predictions, capped by lowest conditional Weak4 margin."""
    if main_logits.ndim != 2:
        raise ValueError(f"main_logits must be rank 2, got shape={tuple(main_logits.shape)}")
    classes = [int(value) for value in classes]
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise ValueError(f"classes must contain distinct class ids, got {classes}")
    if min(classes) < 0 or max(classes) >= main_logits.shape[1]:
        raise ValueError(f"classes out of range for logits shape {tuple(main_logits.shape)}: {classes}")
    route_fraction = float(route_fraction)
    if not 0.0 <= route_fraction <= 1.0:
        raise ValueError(f"route_fraction must be in [0, 1], got {route_fraction}")

    main_cpu = main_logits.detach().float().cpu()
    main_pred = torch.argmax(main_cpu, dim=1)
    class_set = set(classes)
    candidates = [idx for idx, pred in enumerate(main_pred.tolist()) if pred in class_set]
    cap = min(len(candidates), int(route_fraction * len(main_cpu)))
    if cap <= 0:
        return torch.empty(0, dtype=torch.long)

    weak_probs = torch.softmax(main_cpu[:, classes], dim=1)
    top2 = torch.topk(weak_probs, k=2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    routed = sorted(candidates, key=lambda idx: (float(margins[idx]), idx))[:cap]
    return torch.tensor(routed, dtype=torch.long)


def weak4_family_locked_predictions(main_logits, specialist_logits, routed_indices, classes, alpha):
    """Replace only routed Weak4 choices using a conditional four-way blend."""
    if main_logits.ndim != 2 or specialist_logits.ndim != 2:
        raise ValueError("main and specialist logits must both be rank 2")
    classes = [int(value) for value in classes]
    routed = torch.as_tensor(routed_indices, dtype=torch.long).cpu()
    if specialist_logits.shape[0] != len(routed):
        raise ValueError(
            "specialist row count does not match routed indices: "
            f"{specialist_logits.shape[0]} != {len(routed)}"
        )
    if len(routed) and (int(routed.min()) < 0 or int(routed.max()) >= main_logits.shape[0]):
        raise ValueError("routed index is outside main_logits")
    if len(routed) != len(set(routed.tolist())):
        raise ValueError("routed indices contain duplicates")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    main_cpu = main_logits.detach().float().cpu()
    spec_cpu = specialist_logits.detach().float().cpu()
    pred = torch.argmax(main_cpu, dim=1)
    if not len(routed) or alpha == 0.0:
        return pred
    if spec_cpu.shape[1] == len(classes):
        spec4_logits = spec_cpu
    elif spec_cpu.shape[1] == main_cpu.shape[1]:
        spec4_logits = spec_cpu[:, classes]
    else:
        raise ValueError(
            f"specialist logits width must be {len(classes)} or {main_cpu.shape[1]}, "
            f"got {spec_cpu.shape[1]}"
        )
    main4 = torch.softmax(main_cpu[routed][:, classes], dim=1)
    spec4 = torch.softmax(spec4_logits, dim=1)
    mixed4 = (1.0 - alpha) * main4 + alpha * spec4
    local_pred = torch.argmax(mixed4, dim=1)
    class_tensor = torch.tensor(classes, dtype=torch.long)
    pred[routed] = class_tensor[local_pred]

    routed_mask = torch.zeros(len(pred), dtype=torch.bool)
    routed_mask[routed] = True
    original = torch.argmax(main_cpu, dim=1)
    if not torch.equal(pred[~routed_mask], original[~routed_mask]):
        raise AssertionError("weak4 specialist changed a non-routed prediction")
    if any(value not in set(classes) for value in pred[routed].tolist()):
        raise AssertionError("weak4 specialist emitted a non-Weak4 label on a routed row")
    return pred


def _lora_target_module_names(model, target_modules):
    if not isinstance(target_modules, (list, tuple, set)) or not target_modules:
        raise ValueError(f"adapter target_modules must be a non-empty list, got {target_modules!r}")
    targets = [safe_text(value) for value in target_modules]
    modules = {}
    matched_targets = {target: 0 for target in targets}
    for name, module in model.named_modules():
        for target in targets:
            if name == target or name.endswith("." + target):
                weight = getattr(module, "weight", None)
                if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                    raise ValueError(f"LoRA target {name} does not expose a rank-2 weight")
                modules[name] = module
                matched_targets[target] += 1
                break
    missing_targets = [target for target, count in matched_targets.items() if count == 0]
    if missing_targets:
        raise ValueError(f"adapter targets matched no model modules: {missing_targets}")
    return modules


def _match_adapter_module_key(key, module_names, marker):
    marker_text = f".{marker}."
    if marker_text not in f".{key}":
        return None
    matches = [
        name
        for name in module_names
        if f".{name}.{marker}." in f".{key}"
    ]
    if len(matches) != 1:
        raise ValueError(f"adapter key must match exactly one target module: key={key} matches={matches}")
    return matches[0]


def merge_lora_inplace(model, lora_dir):
    """Merge a PEFT LoRA adapter without importing PEFT at evaluation time."""
    from safetensors.torch import load_file

    lora_dir = os.path.abspath(lora_dir)
    if getattr(model, "_weak4_lora_merged", False):
        raise RuntimeError("refusing to merge a Weak4 LoRA adapter more than once")
    config_path = os.path.join(lora_dir, "adapter_config.json")
    weights_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"LoRA directory must contain adapter_config.json and adapter_model.safetensors: {lora_dir}"
        )
    with open(config_path, encoding="utf-8") as f:
        adapter_config = json.load(f)
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError(f"unsupported PEFT adapter type: {adapter_config.get('peft_type')!r}")
    if bool(adapter_config.get("fan_in_fan_out", False)):
        raise ValueError("fan_in_fan_out LoRA adapters are unsupported")
    if bool(adapter_config.get("use_rslora", False)):
        raise ValueError("use_rslora adapters are unsupported")
    if bool(adapter_config.get("use_dora", False)):
        raise ValueError("DoRA adapters are unsupported")
    if adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern"):
        raise ValueError("per-module LoRA rank/alpha patterns are unsupported")
    if safe_text(adapter_config.get("bias", "none")).lower() not in {"", "none"}:
        raise ValueError("LoRA bias tensors are unsupported")
    active_variant_fields = {
        key: adapter_config.get(key)
        for key in (
            "alora_invocation_tokens",
            "arrow_config",
            "corda_config",
            "eva_config",
            "exclude_modules",
            "layer_replication",
            "layers_pattern",
            "layers_to_transform",
            "lora_ga_config",
            "megatron_config",
            "target_parameters",
            "trainable_token_indices",
        )
        if adapter_config.get(key) not in (None, False, "", [], {})
    }
    for key in ("ensure_weight_tying", "lora_bias", "use_bdlora", "use_qalora"):
        if bool(adapter_config.get(key, False)):
            active_variant_fields[key] = adapter_config.get(key)
    if adapter_config.get("loftq_config") not in (None, {}):
        active_variant_fields["loftq_config"] = adapter_config.get("loftq_config")
    if active_variant_fields:
        raise ValueError(f"unsupported active LoRA variants: {sorted(active_variant_fields)}")

    rank = int(adapter_config.get("r", 0))
    lora_alpha = float(adapter_config.get("lora_alpha", 0.0))
    if rank <= 0 or lora_alpha <= 0:
        raise ValueError(f"invalid LoRA rank/alpha: r={rank} alpha={lora_alpha}")
    modules = _lora_target_module_names(model, adapter_config.get("target_modules"))
    state = load_file(weights_path, device="cpu")
    pairs = {name: {} for name in modules}
    unmatched = []
    score_state = {}
    module_names = sorted(modules, key=len, reverse=True)

    for key, tensor in state.items():
        marker = None
        if ".lora_A." in f".{key}":
            marker = "lora_A"
        elif ".lora_B." in f".{key}":
            marker = "lora_B"
        if marker:
            module_name = _match_adapter_module_key(key, module_names, marker)
            if module_name is None or marker in pairs[module_name]:
                raise ValueError(f"duplicate or unmatched LoRA tensor: {key}")
            pairs[module_name][marker] = tensor
            continue
        if re.search(r"(?:^|\.)score(?:\.|$)", key):
            parameter_name = key.rsplit(".", 1)[-1]
            if parameter_name not in {"weight", "bias"} or parameter_name in score_state:
                raise ValueError(f"unsupported or duplicate score tensor: {key}")
            score_state[parameter_name] = tensor
            continue
        unmatched.append(key)

    incomplete = [name for name, pair in pairs.items() if set(pair) != {"lora_A", "lora_B"}]
    if incomplete:
        details = {name: sorted(pairs[name]) for name in incomplete[:5]}
        raise ValueError(f"adapter A/B tensors do not exactly cover target modules: {details}")
    if unmatched:
        raise ValueError(f"unmatched adapter tensors: {unmatched[:10]}")

    score = getattr(model, "score", None)
    if score is None or not isinstance(getattr(score, "weight", None), torch.Tensor):
        raise ValueError("model has no score.weight to replace")
    required_score = {"weight"} | ({"bias"} if score.bias is not None else set())
    if set(score_state) != required_score:
        raise ValueError(
            f"adapter score tensors mismatch: expected={sorted(required_score)} actual={sorted(score_state)}"
        )
    if tuple(score_state["weight"].shape) != tuple(score.weight.shape):
        raise ValueError(
            f"adapter score.weight shape mismatch: {tuple(score_state['weight'].shape)} "
            f"!= {tuple(score.weight.shape)}"
        )
    if score.bias is not None and tuple(score_state["bias"].shape) != tuple(score.bias.shape):
        raise ValueError("adapter score.bias shape mismatch")

    scale = lora_alpha / rank
    for name, pair in pairs.items():
        if tuple(pair["lora_A"].shape) != (rank, modules[name].weight.shape[1]):
            raise ValueError(f"LoRA A shape mismatch for {name}: {tuple(pair['lora_A'].shape)}")
        if tuple(pair["lora_B"].shape) != (modules[name].weight.shape[0], rank):
            raise ValueError(f"LoRA B shape mismatch for {name}: {tuple(pair['lora_B'].shape)}")

    merge_start = time.perf_counter()
    with torch.no_grad():
        for name, pair in pairs.items():
            weight = modules[name].weight
            a = pair["lora_A"].to(device=weight.device, dtype=torch.float32)
            b = pair["lora_B"].to(device=weight.device, dtype=torch.float32)
            delta = (b @ a).mul_(scale)
            # Match PEFT's in-place merge: the FP32 delta participates in the
            # addition before the result is stored back in the base dtype.
            weight.add_(delta)
            del a, b, delta
        score.weight.copy_(score_state["weight"].to(device=score.weight.device, dtype=score.weight.dtype))
        if score.bias is not None:
            score.bias.copy_(score_state["bias"].to(device=score.bias.device, dtype=score.bias.dtype))
    model._weak4_lora_merged = True
    model._weak4_lora_merged_from = lora_dir
    model._weak4_lora_merge_seconds = time.perf_counter() - merge_start
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"Merged LoRA adapter in place: modules={len(modules)} scale={scale:g} "
        f"seconds={model._weak4_lora_merge_seconds:.2f} dir={lora_dir}"
    )
    return model


def model_logits_compiled(
    model,
    tokenizer,
    texts,
    compile_meta,
    model_dir,
    device,
    terminal_token="",
):
    """torch.compile(mode=reduce-overhead) + bucket-padded fixed-size batches.

    Opt-in via hf_meta.json {"compile": {"buckets": [...], "batch_size": N}} —
    only used for architectures whose eager fallback is too slow for the server
    budget (Qwen3.5 DeltaNet). Shipped inductor/triton caches under
    model/compile_cache cut the cold compile; the first batch of each bucket
    shape compiles (or cache-hits) inline. Any failure raises so the caller
    falls back to eager sorted batching.
    """
    cache_root = os.path.join(model_dir, compile_meta.get("cache_dir", "compile_cache"))
    for env_key, sub in (("TORCHINDUCTOR_CACHE_DIR", "venv311_inductor_cache"),
                         ("TRITON_CACHE_DIR", "venv311_triton_cache")):
        cand = os.path.join(cache_root, sub)
        if os.path.isdir(cand):
            os.environ.setdefault(env_key, os.path.abspath(cand))
    mega = os.path.join(cache_root, "megacache.bin")
    if os.path.exists(mega):
        try:
            with open(mega, "rb") as f:
                torch.compiler.load_cache_artifacts(f.read())
            print("Loaded compile megacache")
        except Exception:
            traceback.print_exc()
    torch._dynamo.config.cache_size_limit = 64
    compiled = torch.compile(model, mode=compile_meta.get("mode", "reduce-overhead"))

    buckets = sorted(int(b) for b in compile_meta["buckets"])
    batch_size = int(compile_meta.get("batch_size", 128))
    max_length = buckets[-1]
    encoded_all = tokenize_texts_with_terminal(
        tokenizer,
        texts,
        max_length,
        terminal_token,
    )
    keys = list(encoded_all.keys())
    feats = [{key: encoded_all[key][i] for key in keys} for i in range(len(texts))]
    lengths = [len(feats[i]["input_ids"]) for i in range(len(feats))]
    order = sorted(range(len(feats)), key=lambda i: lengths[i])
    out = [None] * len(feats)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            need = max(lengths[i] for i in chunk)
            bucket = next((b for b in buckets if b >= need), buckets[-1])
            feats_batch = [feats[i] for i in chunk]
            fill = batch_size - len(feats_batch)
            if fill:
                feats_batch = feats_batch + [feats_batch[-1]] * fill
            batch = tokenizer.pad(feats_batch, padding="max_length", max_length=bucket,
                                  return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = compiled(**batch).logits.float().cpu()
            if fill:
                logits = logits[:len(chunk)]
            for row, i in enumerate(chunk):
                out[i] = logits[row]
    print(f"Compiled inference done (buckets={buckets}, batch={batch_size})")
    return torch.stack(out, dim=0)


def encoder_probs(model_dir, spec, texts, device):
    """Sequentially run one ensemble encoder; return CPU softmax probs [N, C]."""
    from transformers import AutoTokenizer

    hf_dir = os.path.join(model_dir, spec["hf_dir"])
    tokenizer_dir = os.path.join(model_dir, spec.get("tokenizer_dir", spec["hf_dir"]))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    model = load_hf_model(
        hf_dir,
        device,
        shared_from=os.path.join(model_dir, spec["shared_from"]) if spec.get("shared_from") else None,
    )
    if device.type == "cuda":
        model.half()
    else:
        model.float()
    model.eval()
    logits = model_logits_sorted(model, tokenizer, texts,
                                 int(spec.get("max_length", 192)),
                                 int(spec.get("batch_size", 32)), device,
                                 safe_text(spec.get("terminal_token")))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Encoder {spec['hf_dir']} done ({len(texts)} rows)")
    return torch.softmax(logits, dim=-1)


def cascade_scores(model_dir, cascade, samples, base_texts, device):
    """Two-leg routed cascade: the base leg scores every row, then the
    lowest-margin route_fraction of rows are re-scored by the secondary leg
    (with its own serializer) and prob-blended. Routing is fraction-based
    (sort by margin, take bottom k) so the inference budget is deterministic
    regardless of the test margin distribution. The tuned chain (bias/rules)
    downstream was fit on OOF logits built with exactly this routing."""
    base_probs = encoder_probs(model_dir, cascade["base"], base_texts, device)
    top2 = base_probs.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    k = int(float(cascade["route_fraction"]) * len(samples))
    if k > 0:
        routed = margin.argsort()[:k]
        secondary = cascade["secondary"]
        sec_texts = [
            serialize_transformer_sample(samples[i], secondary.get("serializer_name", "current_v1"))
            for i in routed.tolist()
        ]
        sec_probs = encoder_probs(model_dir, secondary, sec_texts, device)
        w_base, w_sec = (float(w) for w in cascade.get("blend", [0.5, 0.5]))
        base_probs[routed] = w_base * base_probs[routed] + w_sec * sec_probs
        print(f"Cascade: re-scored {k}/{len(samples)} lowest-margin rows via {secondary['hf_dir']}")
    return torch.log(base_probs.clamp_min(1e-12))


def run_hf_inference(model_dir, data_dir, output_path, device):
    from transformers import AutoTokenizer

    inference_start = time.perf_counter()
    hf_dir = os.path.join(model_dir, "hf_model")
    with open(os.path.join(model_dir, "hf_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    encoders = meta.get("encoders")
    cascade = meta.get("cascade")
    weak4_specialist = meta.get("weak4_specialist") or {}
    specialist_enabled = bool(weak4_specialist.get("enabled", False))
    if specialist_enabled:
        validate_weak4_specialist_config(weak4_specialist, meta.get("classes") or [])
        if encoders or cascade:
            raise ValueError("weak4_specialist requires the dedicated single-model inference path")
        if meta.get("compile"):
            raise ValueError("weak4_specialist does not support the compiled inference path")
        if os.path.exists(os.path.join(model_dir, LEAK_LOOKUP_FILENAME)):
            raise ValueError("weak4_specialist forbids train-derived leak overrides")
        if (meta.get("test_batch_graph_backfill") or {}).get("enabled", False):
            raise ValueError("weak4_specialist forbids test-batch graph backfill")
    tokenizer = None
    model = None
    if not encoders and not cascade:
        tokenizer = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
        model = load_hf_model(hf_dir, device)
        if device.type == "cuda":
            model.half()
        else:
            model.float()
        model.eval()

    test_path = os.path.join(data_dir, "test.jsonl")
    sample_submission_path = os.path.join(data_dir, "sample_submission.csv")
    samples = load_jsonl(test_path)
    ids = [safe_text(sample.get("id", "")) for sample in samples]

    # Legacy train lookup remains fully opt-in.  Same-batch graph backfill is a
    # separate opt-in meta flag so it can be probed without train->test lookups.
    leak_overrides = {}
    try:
        train_lookup = load_leak_lookup(model_dir)
        if train_lookup is not None:
            leak_overrides, leak_stats = compute_leak_overrides(
                samples, train_lookup=train_lookup, valid_classes=meta["classes"]
            )
            tier_text = " ".join(f"{tier}={count}" for tier, count in leak_stats.items())
            print(f"Leak overrides: total={len(leak_overrides)}/{len(samples)} {tier_text}")
        elif (meta.get("test_batch_graph_backfill") or {}).get("enabled", False):
            graph_cfg = meta.get("test_batch_graph_backfill") or {}
            leak_overrides, leak_stats = compute_test_batch_graph_overrides(
                samples,
                valid_classes=meta["classes"],
                use_positional=graph_cfg.get("positional", True),
                use_aligned=graph_cfg.get("aligned", True),
            )
            tier_text = " ".join(f"{tier}={count}" for tier, count in leak_stats.items())
            print(f"Test-batch graph backfill: total={len(leak_overrides)}/{len(samples)} {tier_text}")
        else:
            print("Leak overrides disabled")
    except Exception:
        print("Leak override computation failed; falling back to model-only predictions")
        traceback.print_exc()
        leak_overrides = {}
    if specialist_enabled and leak_overrides:
        raise AssertionError("weak4 specialist must not have prediction overrides")

    serializer_name = meta.get("serializer_name", "current_v1")
    texts = [
        serialize_transformer_sample(sample, serializer_name, tokenizer=tokenizer)
        for sample in samples
    ]
    class_bias = torch.tensor(meta.get("class_bias", [0.0] * len(meta["classes"])), dtype=torch.float32, device=device)
    rule_boosts = meta.get("rule_boosts") or []
    sparse_ensemble = load_sparse_ensemble(model_dir, meta["classes"])
    sparse_scores = None
    sparse_weight = 0.0
    sparse_bias = None
    if sparse_ensemble is not None:
        sparse_meta = sparse_ensemble["meta"]
        sparse_scores = sparse_ensemble_scores(sparse_ensemble, samples)
        sparse_weight = float(sparse_meta.get("sparse_weight", 0.0))
        sparse_bias = torch.tensor(
            sparse_meta.get("class_bias", [0.0] * len(meta["classes"])),
            dtype=torch.float32,
            device=device,
        )
        print(f"Loaded sparse SVC ensemble weight={sparse_weight}")

    batch_size = int(meta.get("batch_size", 32))
    max_length = int(meta.get("max_length", 192))
    terminal_token = safe_text(meta.get("terminal_token"))
    if cascade:
        # meta serializer_name must be the base leg's serializer, so `texts`
        # above already holds the base-leg serialization for all rows
        base_scores = cascade_scores(model_dir, cascade, samples, texts, device)
    elif encoders:
        # sequential encoders -> softmax average; the tuned chain (bias/rules/
        # sparse) operates on log-average-probs, matching how it was tuned
        probs = None
        for spec in encoders:
            enc = encoder_probs(model_dir, spec, texts, device)
            probs = enc if probs is None else probs + enc
        base_scores = torch.log((probs / len(encoders)).clamp_min(1e-12))
    else:
        compile_meta = meta.get("compile") if device.type == "cuda" else None
        base_scores = None
        if compile_meta:
            try:
                base_scores = model_logits_compiled(
                    model,
                    tokenizer,
                    texts,
                    compile_meta,
                    model_dir,
                    device,
                    terminal_token,
                )
            except Exception:
                print("Compiled inference failed; falling back to eager sorted batching")
                traceback.print_exc()
                base_scores = None
        if base_scores is None:
            base_scores = model_logits_sorted(
                model,
                tokenizer,
                texts,
                max_length,
                batch_size,
                device,
                terminal_token,
            )

    class_bias = class_bias + batch_prior_calibration_bias(
        base_scores,
        meta.get("prior_calibration"),
        meta["classes"],
        device,
    )

    main_logits = torch.empty_like(base_scores, dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            logits = base_scores[start:start + len(batch_texts)].to(device) + class_bias
            logits = apply_rule_boosts_to_logits(logits, samples[start:start + batch_size], rule_boosts, meta["classes"])
            if sparse_scores is not None:
                sparse_batch = sparse_scores[start:start + len(batch_texts)].to(device)
                sparse_batch = apply_sparse_controls(sparse_batch, logits, sparse_meta, meta["classes"])
                logits = logits + sparse_weight * sparse_batch + sparse_bias
            main_logits[start:start + len(batch_texts)] = logits.detach().float().cpu()

    main_pred_ids = torch.argmax(main_logits, dim=1)
    pred_ids = main_pred_ids
    if specialist_enabled:
        classes, alpha, route_fraction = validate_weak4_specialist_config(
            weak4_specialist, meta["classes"]
        )
        routed = select_weak4_routes(main_logits, classes, route_fraction)
        if len(routed):
            lora_rel = safe_text(weak4_specialist["lora_dir"])
            lora_dir = os.path.abspath(os.path.join(model_dir, lora_rel))
            model_root = os.path.abspath(model_dir)
            if os.path.commonpath([model_root, lora_dir]) != model_root:
                raise ValueError(f"weak4_specialist lora_dir escapes model directory: {lora_rel}")
            merge_lora_inplace(model, lora_dir)
            model.eval()
            serializer = weak4_specialist.get("serializer_name", "weak_nav_v1")
            routed_texts = [
                serialize_transformer_sample(samples[idx], serializer)
                for idx in routed.tolist()
            ]
            specialist_logits = model_logits_sorted(
                model,
                tokenizer,
                routed_texts,
                int(weak4_specialist["max_length"]),
                int(weak4_specialist["batch_size"]),
                device,
                safe_text(weak4_specialist.get("terminal_token", terminal_token)),
            )
            pred_ids = weak4_family_locked_predictions(
                main_logits,
                specialist_logits,
                routed,
                classes,
                alpha,
            )
        routed_mask = torch.zeros(len(main_pred_ids), dtype=torch.bool)
        routed_mask[routed] = True
        if not torch.equal(pred_ids[~routed_mask], main_pred_ids[~routed_mask]):
            raise AssertionError("weak4 specialist violated non-routed prediction identity")
        print(
            "Weak4 specialist: "
            f"routed={len(routed)}/{len(samples)} cap={route_fraction:.3f} alpha={alpha:.3f}"
        )
    preds = [meta["classes"][int(class_id)] for class_id in pred_ids.tolist()]

    fieldnames, rows = load_sample_submission(sample_submission_path, ids)
    pred_map = dict(zip(ids, preds))
    pred_map.update(leak_overrides)
    for row in rows:
        if row["id"] in pred_map:
            row["action"] = pred_map[row["id"]]
    save_submission(output_path, fieldnames, rows)
    print(f"Saved {output_path} rows={len(rows)}")
    return {
        "rows": len(rows),
        "wall_seconds": time.perf_counter() - inference_start,
        "specialist_enabled": specialist_enabled,
        "routed_rows": len(routed) if specialist_enabled else 0,
        "route_fraction": float(weak4_specialist.get("route_fraction", 0.0)) if specialist_enabled else 0.0,
        "lora_merge_seconds": float(getattr(model, "_weak4_lora_merge_seconds", 0.0)) if model is not None else 0.0,
    }


def main():
    data_dir = first_existing(["./data", "./open/data"])
    model_dir = "./model"
    output_path = "./output/submission.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Required ./model directory is missing. Place the trained model artifacts at ./model before inference.")

    hf_config_path = os.path.join(model_dir, "hf_model", "config.json")
    if not os.path.exists(hf_config_path):
        raise FileNotFoundError("Expected HuggingFace model artifacts at ./model/hf_model/config.json.")
    weight_paths = [os.path.join(model_dir, "hf_model", name)
                    for name in ("model.int4.safetensors", "model.int8.safetensors", "model.safetensors")]
    if not any(os.path.exists(path) for path in weight_paths):
        raise FileNotFoundError(
            "Expected model weights at ./model/hf_model/ "
            "(model.int4.safetensors, model.int8.safetensors, or model.safetensors)."
        )

    print(f"Load transformer model from {model_dir}; device={device}")
    run_hf_inference(model_dir, data_dir, output_path, device)


if __name__ == "__main__":
    main()
