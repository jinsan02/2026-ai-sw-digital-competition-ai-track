"""Export deployment-faithful trio-ensemble teacher logits over the train set.

Reuses the received champion pack's own script module (codec loaders,
serializer, sorted-batch inference) so the teacher is bit-faithful to the
deployed predictor: main-model raw logits everywhere; rows whose main margin
is under the routing threshold get the mean of row-centered member logits,
exactly as in the deployed ensemble block. Rules and the ask-user boost are
deliberately excluded — they stay inference-side in any pack.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


def load_pack_module(pack_dir: Path):
    spec = importlib.util.spec_from_file_location("trio_pack_script", pack_dir / "script.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trio_pack_script"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", required=True, help="dir with script.py, model/, model_b/, model_c/")
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--margin", type=float, default=1.25)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--serializer", default="current_v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir)
    pack = load_pack_module(pack_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = []
    with (Path(args.data_dir) / "train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            samples.append(json.loads(line))
    samples.sort(key=lambda s: str(s["id"]))
    ids = [str(s["id"]) for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate train ids")
    texts = [pack.serialize_transformer_sample(s, args.serializer) for s in samples]
    print(f"serialized {len(texts)} rows with {args.serializer}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(pack_dir / "model" / "hf_model"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def forward(member: str, member_texts):
        model = pack.load_hf_model(str(pack_dir / member / "hf_model"), device)
        started = time.time()
        scores = pack.model_logits_sorted(
            model, tokenizer, member_texts, args.max_length, args.batch_size, device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{member}: {len(member_texts)} rows in {time.time()-started:.1f}s", flush=True)
        return scores.float().cpu()

    base_scores = forward("model", texts)
    srt = torch.sort(base_scores, dim=1).values
    low_margin = (srt[:, -1] - srt[:, -2]) < args.margin
    route_idx = torch.nonzero(low_margin, as_tuple=False).flatten().tolist()
    print(f"routing {len(route_idx)}/{len(texts)} rows below margin {args.margin}", flush=True)

    routed_texts = [texts[i] for i in route_idx]
    scores_b = forward("model_b", routed_texts)
    scores_c = forward("model_c", routed_texts)
    sub_a = base_scores[route_idx]
    cent_list = [
        sub_a - sub_a.mean(dim=1, keepdim=True),
        scores_b - scores_b.mean(dim=1, keepdim=True),
        scores_c - scores_c.mean(dim=1, keepdim=True),
    ]
    teacher = base_scores.clone()
    teacher[route_idx] = sum(cent_list) / float(len(cent_list))

    agree = (teacher.argmax(1) == base_scores.argmax(1)).float().mean().item()
    payload = {
        "ids": ids,
        "logits": teacher.to(torch.float16),
        "classes": list(pack.ALL_CLASSES),
        "metadata": {
            "source": "trio_ensemble_teacher_export",
            "pack_dir": str(pack_dir),
            "serializer_name": args.serializer,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "routing_margin": args.margin,
            "routed_rows": len(route_idx),
            "argmax_agreement_vs_main": agree,
            "blend": "mean of row-centered member logits on routed rows; raw main logits elsewhere",
            "row_count": len(ids),
            "dtype": "fp16",
            "id_order": "lexicographic_id",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved teacher payload: {out} rows={len(ids)} routed={len(route_idx)} "
          f"main_agreement={agree:.4f}", flush=True)


if __name__ == "__main__":
    main()
