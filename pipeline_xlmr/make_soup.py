"""Model soup: 같은 레시피·다른 seed로 학습한 HF 모델들의 가중치 평균. (노진산, S2-1)

사용:
  python make_soup.py --model-dirs model_s42 model_s43 model_s44 --output-dir model_soup
  (각 model_dir는 hf_model/ 하위에 safetensors 저장된 학습 산출물)

평균 후 검증은 train_transformer.py의 evaluate 경로 재사용 대신,
eval_soup.py 또는 기존 세션 split 재추론으로 fixed F1을 확인할 것.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dirs", nargs="+", required=True,
                        help="학습 산출 디렉토리들 (각각 hf_model/ 포함)")
    parser.add_argument("--output-dir", default="model_soup")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="선택: 모델별 가중치 (기본 균등)")
    args = parser.parse_args()

    dirs = [Path(d) / "hf_model" for d in args.model_dirs]
    for d in dirs:
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"hf_model not found: {d}")
    n = len(dirs)
    w = args.weights or [1.0 / n] * n
    if len(w) != n:
        raise ValueError("weights 개수가 model-dirs와 다름")
    total = sum(w)
    w = [x / total for x in w]

    print(f"souping {n} models, weights={['%.3f' % x for x in w]}")
    models = [AutoModelForSequenceClassification.from_pretrained(d, torch_dtype=torch.float32) for d in dirs]
    base = models[0]
    soup_state = {k: v.clone() * w[0] for k, v in base.state_dict().items()}
    for wi, m in zip(w[1:], models[1:]):
        sd = m.state_dict()
        if set(sd.keys()) != set(soup_state.keys()):
            raise ValueError("state_dict 키 불일치 — 같은 아키텍처/레시피인지 확인")
        for k in soup_state:
            if soup_state[k].dtype.is_floating_point:
                soup_state[k] += sd[k] * wi
            # int 버퍼(position_ids 등)는 첫 모델 값 유지

    base.load_state_dict(soup_state)
    out = Path(args.output_dir)
    hf_out = out / "hf_model"
    hf_out.mkdir(parents=True, exist_ok=True)
    base.half().save_pretrained(hf_out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(dirs[0])
    tokenizer.save_pretrained(hf_out)

    # hf_meta는 첫 모델 것을 복사 (class_bias는 soup 후 val 로짓으로 재튜닝 권장)
    src_meta = Path(args.model_dirs[0]) / "hf_meta.json"
    if src_meta.exists():
        meta = json.loads(src_meta.read_text(encoding="utf-8"))
        meta["soup_members"] = [str(d) for d in dirs]
        meta["soup_weights"] = w
        (out / "hf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("hf_meta.json copied (class_bias는 soup 검증 후 재튜닝할 것)")

    size_mb = sum(f.stat().st_size for f in hf_out.rglob("*") if f.is_file()) / 1e6
    print(f"saved soup → {hf_out} ({size_mb:.0f} MB fp16)")


if __name__ == "__main__":
    main()
