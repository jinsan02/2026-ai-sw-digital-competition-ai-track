import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from script import (
    ALL_CLASSES,
    disable_decoder_cache,
    load_hf_model,
    load_int8_state_dict,
    load_jsonl,
    safe_text,
    serialize_transformer_sample,
    tokenize_texts_with_terminal,
)
from train_transformer import (
    RELATIONAL_HIDDEN_KIND,
    RELATIONAL_HIDDEN_POOLING,
    RELATIONAL_HIDDEN_SCHEMA_VERSION,
    RELATIONAL_HIDDEN_USAGE_SCOPE,
    find_terminal_classifier_head,
    forward_with_classifier_input,
    pool_classifier_hidden,
)


def load_labels(labels_path):
    if not labels_path or not labels_path.exists():
        return {}
    with labels_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or reader.fieldnames[:2] != ["id", "action"]:
            raise ValueError(f"labels file must start with id,action: {labels_path}")
        return {row["id"]: row["action"] for row in reader}


def resolve_input_jsonl(args):
    if args.input_jsonl:
        return Path(args.input_jsonl)
    return Path(args.data_dir) / f"{args.split}.jsonl"


def load_source_ids(args):
    input_jsonl = resolve_input_jsonl(args)
    samples = load_jsonl(input_jsonl)
    if args.limit:
        samples = samples[: args.limit]
    return input_jsonl, samples, [safe_text(sample.get("id", "")) for sample in samples]


def dtype_for_save(name):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_weights_sha256(hf_dir, prefer_fp16=False):
    """Hash the exact weight file set selected for export."""
    hf_dir = Path(hf_dir)
    if prefer_fp16:
        candidates = sorted(hf_dir.glob("model*.safetensors"))
        candidates = [
            path
            for path in candidates
            if ".int8." not in path.name and ".int4." not in path.name
        ]
    else:
        adapter_weights = hf_dir / "adapter_model.safetensors"
        adapter_config = hf_dir / "adapter_config.json"
        if adapter_weights.is_file() and adapter_config.is_file():
            # Adapter-only custom teachers (notably Gemma4) intentionally have
            # no model*.safetensors in this directory. Their loader reconstructs
            # the base and merges this exact PEFT payload.
            candidates = [adapter_weights]
        else:
            candidates = []
            for pattern in (
                "model.int4.safetensors",
                "model.int8.safetensors",
                "model*.safetensors",
            ):
                matches = sorted(hf_dir.glob(pattern))
                if matches:
                    candidates = matches
                    break
    if not candidates:
        raise FileNotFoundError(f"no selected model weights found under {hf_dir}")
    digest = hashlib.sha256()
    for path in candidates:
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), [path.name for path in candidates]


def tokenized_features(tokenizer, texts, max_length, chunk_size, terminal_token=""):
    features = []
    total = (len(texts) + chunk_size - 1) // chunk_size if texts else 0
    for chunk_no, start in enumerate(range(0, len(texts), chunk_size), 1):
        chunk = texts[start:start + chunk_size]
        encoded = tokenize_texts_with_terminal(
            tokenizer,
            chunk,
            max_length,
            terminal_token,
        )
        keys = list(encoded.keys())
        features.extend({key: encoded[key][i] for key in keys} for i in range(len(chunk)))
        if total > 1 and (chunk_no == 1 or chunk_no == total or chunk_no % 10 == 0):
            print(f"tokenized chunk {chunk_no}/{total} rows={len(features)}", flush=True)
    lengths = [len(feature["input_ids"]) for feature in features]
    return features, lengths


def load_export_tokenizer(hf_dir):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise
    config_path = Path(hf_dir) / "tokenizer_config.json"
    tokenizer_path = Path(hf_dir) / "tokenizer.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    special_tokens = {
        key: config[key]
        for key in ("bos_token", "eos_token", "unk_token", "pad_token")
        if config.get(key)
    }
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **special_tokens)
    if config.get("model_max_length"):
        tokenizer.model_max_length = int(config["model_max_length"])
    print("loaded tokenizer.json via PreTrainedTokenizerFast fallback", flush=True)
    return tokenizer


def load_gemma4_export_model(hf_dir, device, base_model):
    from gemma4_seqcls import build_gemma4_seqcls

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    label_kwargs = {
        "num_labels": len(ALL_CLASSES),
        "id2label": {i: label for i, label in enumerate(ALL_CLASSES)},
        "label2id": {label: i for i, label in enumerate(ALL_CLASSES)},
    }
    adapter_config = Path(hf_dir) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        cfg = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_source = base_model or cfg.get("base_model_name_or_path")
        if not base_source:
            raise ValueError(f"{adapter_config} has no base_model_name_or_path; pass --base-model")
        base = build_gemma4_seqcls(base_source, **label_kwargs, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, hf_dir)
        model = model.merge_and_unload()
        print(f"Loaded Gemma4 custom seq-cls + LoRA adapter from {hf_dir}", flush=True)
    else:
        model = build_gemma4_seqcls(hf_dir, **label_kwargs, torch_dtype=dtype)
        print(f"Loaded Gemma4 custom seq-cls full checkpoint from {hf_dir}", flush=True)
    disable_decoder_cache(model)
    return model.to(device)


def load_export_model(
    hf_dir, device, model_class="auto", base_model="", prefer_fp16=False
):
    if model_class == "gemma4custom":
        return load_gemma4_export_model(hf_dir, device, base_model)
    try:
        if prefer_fp16:
            from transformers import AutoModelForSequenceClassification

            dtype = torch.float16 if device.type == "cuda" else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_dir,
                local_files_only=True,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            disable_decoder_cache(model)
            print("Loaded standard fp16/fp32 checkpoint; custom low-bit files ignored", flush=True)
            return model.to(device)
        return load_hf_model(str(hf_dir), device)
    except ValueError as exc:
        if "qwen3_5_text" not in str(exc):
            raise

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextConfig,
        Qwen3_5TextForSequenceClassification,
    )

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    config = Qwen3_5TextConfig.from_pretrained(hf_dir, local_files_only=True)
    config.dtype = "float16" if dtype == torch.float16 else "float32"
    if dtype == torch.float16:
        config.torch_dtype = torch.float16
    model = Qwen3_5TextForSequenceClassification(config)
    model = model.half() if dtype == torch.float16 else model.float()

    int8_path = Path(hf_dir) / "model.int8.safetensors"
    if int8_path.exists() and not prefer_fp16:
        state = load_int8_state_dict(str(int8_path), dtype=dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [key for key in missing if not key.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"qwen3_5_text int8 mismatch: missing={missing} unexpected={unexpected}")
        print("Loaded qwen3_5_text int8 checkpoint via qwen3_5 direct class", flush=True)
    else:
        model = Qwen3_5TextForSequenceClassification.from_pretrained(
            hf_dir,
            local_files_only=True,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    disable_decoder_cache(model)
    return model.to(device)


def maybe_rebind_fp16_deltanet(model):
    try:
        from colab.m8_fp16_deltanet_probe import make_fp16_rule, rebind_rule

        rule = make_fp16_rule(state_in_fp32=False)
        rebound = rebind_rule(model, rule)
        print(f"fp16 DeltaNet rebound_layers={rebound}", flush=True)
        return rebound
    except Exception as exc:
        print(f"fp16 DeltaNet rebind skipped: {exc!r}", flush=True)
        return 0


def ensure_model_pad_token(model, pad_token_id):
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if config is not None and (not hasattr(config, "pad_token_id") or getattr(config, "pad_token_id") is None):
            setattr(config, "pad_token_id", pad_token_id)


def infer_logits(args, samples):
    if args.hf_model:
        hf_dir = Path(args.hf_model)
        model_dir = hf_dir.parent if hf_dir.name == "hf_model" else hf_dir
    else:
        model_dir = Path(args.model_dir)
        hf_dir = model_dir / "hf_model"
    meta_path = model_dir / "hf_meta.json"
    if not meta_path.exists() and not args.hf_model:
        raise FileNotFoundError(f"missing hf_meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    serializer = args.serializer or meta.get("serializer_name", "current_v1")
    terminal_token = (
        args.terminal_token
        if args.terminal_token is not None
        else safe_text(meta.get("terminal_token"))
    )
    max_length = args.max_length or int(meta.get("max_length", 192))
    batch_size = args.batch_size or int(meta.get("batch_size", 16))
    model_class = args.model_class or meta.get("model_class", "auto")
    base_model = args.base_model or meta.get("base_model", "")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"device=cuda name={torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("device=cpu", flush=True)

    tokenizer = load_export_tokenizer(hf_dir)
    weights_sha256, weight_files = model_weights_sha256(
        hf_dir, prefer_fp16=bool(args.prefer_fp16_weights)
    )
    model = load_export_model(
        hf_dir,
        device,
        model_class=model_class,
        base_model=base_model,
        prefer_fp16=bool(args.prefer_fp16_weights),
    )
    ensure_model_pad_token(model, tokenizer.pad_token_id)
    if device.type == "cuda":
        model.half()
    else:
        model.float()
    model.eval()
    if args.fp16_deltanet:
        export_meta_rebound = maybe_rebind_fp16_deltanet(model)
    else:
        export_meta_rebound = 0

    texts = [serialize_transformer_sample(sample, serializer) for sample in samples]
    features, lengths = tokenized_features(
        tokenizer,
        texts,
        max_length,
        args.tokenize_batch_size,
        terminal_token,
    )
    order = sorted(range(len(features)), key=lambda i: lengths[i])
    logits = torch.empty((len(features), len(ALL_CLASSES)), dtype=torch.float32)
    pooled_hidden = None
    classifier_head = None
    if args.hidden_output:
        classifier_head, classifier_head_name = find_terminal_classifier_head(model)
        print(f"capturing pooled classifier input from {classifier_head_name}", flush=True)
    else:
        classifier_head_name = ""
    total_batches = (len(order) + batch_size - 1) // batch_size if order else 0
    with torch.inference_mode():
        for batch_no, start in enumerate(range(0, len(order), batch_size), 1):
            chunk = order[start:start + batch_size]
            batch = tokenizer.pad(
                [features[i] for i in chunk],
                padding=True,
                pad_to_multiple_of=args.pad_to_multiple_of if args.pad_to_multiple_of > 1 else None,
                return_tensors="pt",
            )
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.float16):
                if classifier_head is None:
                    outputs = model(**batch)
                    batch_hidden = None
                else:
                    outputs, classifier_input = forward_with_classifier_input(
                        model, batch, classifier_head=classifier_head
                    )
                    batch_hidden = pool_classifier_hidden(
                        classifier_input, outputs.logits, batch
                    )
                batch_logits = outputs.logits.float().cpu()
            logits[chunk] = batch_logits
            if batch_hidden is not None:
                batch_hidden = batch_hidden.detach().float().cpu()
                if batch_hidden.ndim != 2 or batch_hidden.shape[0] != len(chunk):
                    raise ValueError(
                        "unexpected pooled hidden shape: "
                        f"hidden={tuple(batch_hidden.shape)} batch={len(chunk)}"
                    )
                if pooled_hidden is None:
                    pooled_hidden = torch.empty(
                        (len(features), batch_hidden.shape[1]), dtype=torch.float32
                    )
                if pooled_hidden.shape[1] != batch_hidden.shape[1]:
                    raise ValueError(
                        "pooled hidden size changed across batches: "
                        f"expected={pooled_hidden.shape[1]} actual={batch_hidden.shape[1]}"
                    )
                pooled_hidden[chunk] = batch_hidden
            if batch_no == 1 or batch_no == total_batches or batch_no % args.progress_every == 0:
                print(f"infer batch {batch_no}/{total_batches} rows={min(start + batch_size, len(order))}", flush=True)

    export_meta = {
        "source": "hf_model_export",
        "model_dir": str(model_dir),
        "base_model": base_model or meta.get("base_model"),
        "model_class": model_class,
        "serializer_name": serializer,
        "terminal_token": terminal_token,
        "max_length": max_length,
        "batch_size": batch_size,
        "fp16_deltanet_rebound_layers": export_meta_rebound,
        "source_note": args.source_note,
        "prefer_fp16_weights": bool(args.prefer_fp16_weights),
        "model_weights_sha256": weights_sha256,
        "model_weight_files": weight_files,
        "classifier_head": classifier_head_name,
    }
    if args.hidden_output and pooled_hidden is None:
        raise ValueError("hidden export requested but no pooled hidden rows were captured")
    return logits, export_meta, pooled_hidden


def load_payload_logits(args, expected_ids):
    payload = torch.load(args.input_payload, map_location="cpu", weights_only=False)
    logits = payload["logits"].float().cpu()
    ids = [safe_text(x) for x in payload["ids"]]
    if args.limit:
        ids = ids[: args.limit]
        logits = logits[: args.limit]
    if len(ids) != logits.shape[0]:
        raise ValueError("input payload ids/logits row count mismatch")
    if expected_ids is not None:
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        if missing or extra:
            raise ValueError(
                f"payload ids do not match source ids: missing={missing[:5]} extra={extra[:5]}"
            )
    export_meta = {
        "source": "payload_repackage",
        "input_payload": args.input_payload,
        "input_payload_keys": sorted(payload.keys()),
    }
    return logits, ids, export_meta, None


def attach_labels(payload, ids, labels_path):
    labels = load_labels(labels_path)
    if not labels:
        return
    class_to_id = {label: idx for idx, label in enumerate(ALL_CLASSES)}
    missing = [sample_id for sample_id in ids if sample_id not in labels]
    if missing:
        raise ValueError(f"labels missing for ids: {missing[:5]}")
    label_names = [labels[sample_id] for sample_id in ids]
    payload["labels"] = label_names
    payload["y_true"] = torch.tensor([class_to_id[label] for label in label_names], dtype=torch.long)


def reorder(ids, logits, payload, sort_by_id):
    if not sort_by_id:
        return ids, logits, payload
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ids = [ids[i] for i in order]
    logits = logits[order]
    for key in ("labels", "y_true"):
        if key in payload:
            value = payload[key]
            if torch.is_tensor(value):
                payload[key] = value[order]
            else:
                payload[key] = [value[i] for i in order]
    return ids, logits, payload


def assert_reference_logits(payload, reference_path, min_agreement, max_abs):
    """Fail closed unless the new forward matches the established KD surface."""
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if not isinstance(reference, dict):
        raise ValueError("asserted logits payload must be a dict")
    if list(reference.get("classes") or []) != list(payload.get("classes") or []):
        raise ValueError("asserted logits class order mismatch")
    reference_ids = [safe_text(value) for value in reference.get("ids") or []]
    current_ids = [safe_text(value) for value in payload.get("ids") or []]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("asserted logits payload contains duplicate ids")
    if len(current_ids) != len(set(current_ids)):
        raise ValueError("current logits payload contains duplicate ids")
    if set(reference_ids) != set(current_ids):
        missing = sorted(set(current_ids) - set(reference_ids))[:5]
        extra = sorted(set(reference_ids) - set(current_ids))[:5]
        raise ValueError(
            "asserted logits id coverage mismatch: "
            f"missing={missing} extra={extra}"
        )
    reference_logits = reference.get("logits")
    current_logits = payload.get("logits")
    if not isinstance(reference_logits, torch.Tensor) or reference_logits.ndim != 2:
        raise ValueError("asserted logits tensor must be rank 2")
    if not isinstance(current_logits, torch.Tensor) or current_logits.ndim != 2:
        raise ValueError("current logits tensor must be rank 2")
    reference_pos = {sample_id: idx for idx, sample_id in enumerate(reference_ids)}
    order = torch.tensor([reference_pos[sample_id] for sample_id in current_ids])
    reference_logits = reference_logits[order].float()
    current_logits = current_logits.float()
    if reference_logits.shape != current_logits.shape:
        raise ValueError(
            "asserted logits shape mismatch: "
            f"reference={tuple(reference_logits.shape)} current={tuple(current_logits.shape)}"
        )
    if not bool(torch.isfinite(reference_logits).all()) or not bool(
        torch.isfinite(current_logits).all()
    ):
        raise ValueError("asserted/current logits contain non-finite values")

    reference_meta = reference.get("metadata") or {}
    current_meta = payload.get("metadata") or {}
    for key in ("base_model", "serializer_name", "max_length"):
        if reference_meta.get(key) != current_meta.get(key):
            raise ValueError(
                f"asserted logits metadata mismatch for {key}: "
                f"reference={reference_meta.get(key)!r} current={current_meta.get(key)!r}"
            )
    if "y_true" in reference and "y_true" in payload:
        reference_y = torch.as_tensor(reference["y_true"], dtype=torch.long)[order]
        current_y = torch.as_tensor(payload["y_true"], dtype=torch.long)
        if not torch.equal(reference_y, current_y):
            raise ValueError("asserted logits y_true mismatch")

    diff = (current_logits - reference_logits).abs()
    agreement = float(
        (current_logits.argmax(dim=1) == reference_logits.argmax(dim=1))
        .float()
        .mean()
    )
    observed_max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    print(
        "asserted reference logits observed: "
        f"rows={len(current_ids)} agreement={agreement:.9f} "
        f"max_abs={observed_max_abs:.6f} mean_abs={mean_abs:.6f}",
        flush=True,
    )
    if agreement < min_agreement:
        raise ValueError(
            "asserted logits argmax agreement below threshold: "
            f"observed={agreement:.9f} minimum={min_agreement:.9f}"
        )
    if observed_max_abs > max_abs:
        raise ValueError(
            "asserted logits max_abs above threshold: "
            f"observed={observed_max_abs:.9f} maximum={max_abs:.9f}"
        )
    metrics = {
        "path": str(reference_path),
        "sha256": sha256_file(reference_path),
        "rows": len(current_ids),
        "argmax_agreement": agreement,
        "max_abs": observed_max_abs,
        "mean_abs": mean_abs,
        "min_argmax_agreement": float(min_agreement),
        "max_abs_threshold": float(max_abs),
    }
    print("asserted reference logits passed", flush=True)
    return metrics


def save_npz(path, payload):
    import numpy as np

    arrays = {
        "ids": np.asarray(payload["ids"]),
        "logits": payload["logits"].cpu().numpy(),
        "classes": np.asarray(payload["classes"]),
    }
    if "y_true" in payload:
        arrays["y_true"] = payload["y_true"].cpu().numpy()
    if "labels" in payload:
        arrays["labels"] = np.asarray(payload["labels"])
    arrays["metadata_json"] = np.asarray(json.dumps(payload["metadata"], ensure_ascii=False, sort_keys=True))
    np.savez_compressed(path, **arrays)


def save_output(path, payload, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pt":
        torch.save(payload, path)
    elif fmt == "npz":
        save_npz(path, payload)
    else:
        raise ValueError(f"unsupported output format: {fmt}")
    print(f"saved {path} rows={len(payload['ids'])} dtype={payload['logits'].dtype}", flush=True)


def save_hidden_output(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(
        f"saved {path} rows={len(payload['ids'])} "
        f"hidden={tuple(payload['hidden'].shape)} dtype={payload['hidden'].dtype}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export or repackage teacher logits for KD.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", help="HF artifact dir containing hf_model/ and hf_meta.json")
    source.add_argument("--hf-model", help="Direct hf_model/ directory; useful for adapter-only teacher exports")
    source.add_argument("--input-payload", help="Existing .pt payload with ids/logits to repackage")
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--labels-csv", default="", help="Defaults to <data-dir>/train_labels.csv for train split")
    parser.add_argument(
        "--output",
        "--out",
        default="",
        help="logit payload output; optional when --hidden-output is supplied",
    )
    parser.add_argument("--also-npz", default="", help="Optional second output path in .npz format")
    parser.add_argument(
        "--hidden-output",
        default="",
        help="optional fp16 pooled-classifier-hidden cache for relational KD",
    )
    parser.add_argument(
        "--assert-logits-payload",
        default="",
        help="existing teacher-logit payload that the export must faithfully reproduce",
    )
    parser.add_argument(
        "--assert-logits-min-argmax-agreement",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--assert-logits-max-abs",
        type=float,
        default=0.25,
    )
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--serializer", default="")
    parser.add_argument(
        "--terminal-token",
        default=None,
        help="single non-pad token appended after reserving one position; defaults to model metadata",
    )
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--tokenize-batch-size", type=int, default=4096)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--fp16-deltanet", action="store_true")
    parser.add_argument(
        "--prefer-fp16-weights",
        action="store_true",
        help="load standard model.safetensors even when a custom int8/int4 file is present",
    )
    parser.add_argument("--model-class", choices=["auto", "gemma4custom"], default="")
    parser.add_argument("--base-model", default="", help="Base model id/path for adapter-only custom exports")
    parser.add_argument("--source-note", default="")
    parser.add_argument("--limit", type=int, default=0, help="Debug export on the first N source rows")
    parser.add_argument("--preserve-input-order", dest="sort_by_id", action="store_false")
    parser.set_defaults(sort_by_id=True)
    args = parser.parse_args()
    if not args.output and not args.hidden_output:
        parser.error("at least one of --output or --hidden-output is required")
    if args.also_npz and not args.output:
        parser.error("--also-npz requires --output")
    if args.hidden_output and args.input_payload:
        parser.error("--hidden-output requires --model-dir or --hf-model")
    if args.hidden_output and args.split != "train":
        parser.error("--hidden-output requires --split train")
    if args.hidden_output and args.limit:
        parser.error("--hidden-output requires the complete training split; do not use --limit")
    if args.hidden_output and args.dtype != "fp16":
        parser.error("--hidden-output requires --dtype fp16")
    if args.hidden_output and not args.prefer_fp16_weights:
        parser.error("--hidden-output requires --prefer-fp16-weights")
    if args.hidden_output and not args.assert_logits_payload:
        parser.error("--hidden-output requires --assert-logits-payload")
    if args.prefer_fp16_weights and args.input_payload:
        parser.error("--prefer-fp16-weights requires --model-dir or --hf-model")
    if not 0.0 <= args.assert_logits_min_argmax_agreement <= 1.0:
        parser.error("--assert-logits-min-argmax-agreement must be in [0, 1]")
    if args.assert_logits_max_abs < 0.0:
        parser.error("--assert-logits-max-abs must be >= 0")
    return args


def main():
    args = parse_args()
    input_jsonl, samples, source_ids = load_source_ids(args)
    if args.model_dir or args.hf_model:
        logits, export_meta, pooled_hidden = infer_logits(args, samples)
        ids = source_ids
    else:
        logits, ids, export_meta, pooled_hidden = load_payload_logits(args, source_ids)

    original_ids = list(ids)
    data_sha256 = sha256_file(input_jsonl)
    labels_path = Path(args.labels_csv) if args.labels_csv else Path(args.data_dir) / "train_labels.csv"
    payload = {
        "ids": ids,
        "logits": logits.to(dtype_for_save(args.dtype)).cpu(),
        "classes": list(ALL_CLASSES),
        "metadata": {
            **export_meta,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_path": str(input_jsonl),
            "row_count": len(ids),
            "dtype": args.dtype,
            "id_order": "lexicographic_id" if args.sort_by_id else "input_order",
            "data_sha256": data_sha256,
        },
    }
    if args.split == "train":
        attach_labels(payload, ids, labels_path)
    ids, payload["logits"], payload = reorder(ids, payload["logits"], payload, args.sort_by_id)
    payload["ids"] = ids
    if "y_true" in payload:
        y_true = payload["y_true"]
        pred = torch.argmax(payload["logits"].float(), dim=1)
        acc = float((pred == y_true).float().mean())
        payload["metadata"]["train_argmax_acc"] = acc
        print(f"teacher train-set argmax acc={acc:.4f}", flush=True)
    if args.assert_logits_payload:
        payload["metadata"]["asserted_reference_logits"] = assert_reference_logits(
            payload,
            Path(args.assert_logits_payload),
            args.assert_logits_min_argmax_agreement,
            args.assert_logits_max_abs,
        )

    if args.output:
        output_path = Path(args.output)
        fmt = "npz" if output_path.suffix == ".npz" else "pt"
        save_output(output_path, payload, fmt)
        if args.also_npz:
            save_output(Path(args.also_npz), payload, "npz")

    if args.hidden_output:
        if pooled_hidden is None:
            raise ValueError("--hidden-output requested but inference returned no hidden cache")
        if len(original_ids) != len(set(original_ids)):
            raise ValueError("source data contains duplicate ids; cannot align hidden cache")
        source_pos = {sample_id: idx for idx, sample_id in enumerate(original_ids)}
        hidden_order = torch.tensor(
            [source_pos[sample_id] for sample_id in ids], dtype=torch.long
        )
        hidden = pooled_hidden[hidden_order].to(dtype_for_save(args.dtype)).cpu()
        hidden_metadata = {
            **payload["metadata"],
            "pooling": RELATIONAL_HIDDEN_POOLING,
            "hidden_size": int(hidden.shape[1]),
            "row_count": len(ids),
        }
        hidden_payload = {
            "schema_version": RELATIONAL_HIDDEN_SCHEMA_VERSION,
            "kind": RELATIONAL_HIDDEN_KIND,
            "usage_scope": RELATIONAL_HIDDEN_USAGE_SCOPE,
            "ids": list(ids),
            "hidden": hidden,
            "classes": list(ALL_CLASSES),
            "metadata": hidden_metadata,
        }
        if "labels" in payload:
            hidden_payload["labels"] = list(payload["labels"])
        if "y_true" in payload:
            hidden_payload["y_true"] = payload["y_true"].clone()
        save_hidden_output(Path(args.hidden_output), hidden_payload)


if __name__ == "__main__":
    main()
