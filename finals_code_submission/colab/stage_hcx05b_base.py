"""Stage the exact HCX-0.5B base snapshot into the local HF cache.

Colab's anonymous Xet signed URL can fail even though the public Hub resolve
URL is healthy.  The asset is uploaded once through the local control plane,
verified here by the official safetensors SHA256, and installed under the
canonical model-id/revision cache path so training keeps the original
``--base-model`` value and initialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


MODEL_CACHE_NAME = (
    "models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-0.5B"
)
REVISION = "3da5046fb0195d14f2497de198136987d35fd644"
MODEL_SHA256 = "f40f63fa6a118eb9b59e8910c3ab5a6e6499805966284bb2cd41e197719d156b"
MODEL_SIZE = 1_132_587_360
SMALL_FILES = {
    "config.json": (
        760,
        "2f7b047eafde467097fddbfa7281da5579c6ad9920af06e52c8666c87729b745",
    ),
    "special_tokens_map.json": (
        1_929,
        "ec0481f56b50b66a4723b905c89345c52d45eed6571dd87a023285b99a929cc9",
    ),
    "tokenizer.json": (
        8_029_694,
        "1ce80c377be4ae8a4e57758da0d6ee3dc6da3f25857d72c07a52a156d7de7fab",
    ),
    "tokenizer_config.json": (
        11_905,
        "64982c20eb3c1ebd5469934a1d576afc17a96e02df0d90c8e1c81233d27047e2",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".aadp-part")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    exchange = os.environ.get("AADP_EXCHANGE_DIR", "AADP_exchange")
    parser.add_argument(
        "--asset-dir",
        default=f"/content/drive/MyDrive/{exchange}/assets/hcx05b_base",
    )
    parser.add_argument(
        "--hub-cache",
        default=str(Path.home() / ".cache/huggingface/hub"),
    )
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    source_model = asset_dir / "model.safetensors"
    if not source_model.is_file():
        raise FileNotFoundError(f"missing HCX base asset: {source_model}")
    if source_model.stat().st_size != MODEL_SIZE:
        raise ValueError(
            f"HCX base size mismatch: {source_model.stat().st_size} != {MODEL_SIZE}"
        )
    source_sha = sha256_file(source_model)
    if source_sha != MODEL_SHA256:
        raise ValueError(f"HCX base SHA256 mismatch: {source_sha} != {MODEL_SHA256}")

    repo_cache = Path(args.hub_cache) / MODEL_CACHE_NAME
    blob = repo_cache / "blobs" / MODEL_SHA256
    if not blob.is_file() or blob.stat().st_size != MODEL_SIZE:
        copy_atomic(source_model, blob)
    if sha256_file(blob) != MODEL_SHA256:
        raise ValueError(f"staged HCX blob failed SHA256 verification: {blob}")

    snapshot = repo_cache / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, (expected_size, expected_sha) in SMALL_FILES.items():
        source = asset_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing HCX base asset: {source}")
        if source.stat().st_size != expected_size or sha256_file(source) != expected_sha:
            raise ValueError(f"HCX base asset failed size/SHA256 verification: {source}")
        target = snapshot / name
        copy_atomic(source, target)
        if target.stat().st_size != expected_size or sha256_file(target) != expected_sha:
            raise ValueError(f"staged HCX asset failed size/SHA256 verification: {target}")

    model_link = snapshot / "model.safetensors"
    if model_link.is_symlink() or model_link.exists():
        model_link.unlink()
    model_link.symlink_to(Path("../../blobs") / MODEL_SHA256)
    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(REVISION, encoding="utf-8")

    print(
        "AADP_RESULT="
        + json.dumps(
            {
                "ok": True,
                "op": "stage_hcx05b_base",
                "revision": REVISION,
                "sha256": MODEL_SHA256,
                "size": MODEL_SIZE,
                "companion_files": {
                    name: {"size": size, "sha256": digest}
                    for name, (size, digest) in SMALL_FILES.items()
                },
                "snapshot": str(snapshot),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
