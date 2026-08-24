"""Bootstrap an AADP workspace inside a persistent Colab CLI session.

This file is sent as one kernel cell by ``colab exec -f``.  The local wrapper
prefixes the lane environment variables to the source before execution, so it
must not rely on ``__file__`` or local argv.
"""

import csv
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


RESULT_PREFIX = "AADP_RESULT="
DEFAULT_PACKAGES = ["transformers==4.46.3", "sentencepiece", "safetensors"]


def _requested_packages():
    raw = os.environ.get("AADP_BOOTSTRAP_PACKAGES_JSON")
    packages = json.loads(raw) if raw else DEFAULT_PACKAGES
    if not isinstance(packages, list) or not packages or not all(
        isinstance(item, str) and item for item in packages
    ):
        raise ValueError("AADP_BOOTSTRAP_PACKAGES_JSON must be a non-empty JSON string list")
    return packages


def _distribution_name(requirement):
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        requirement = requirement.split(marker, 1)[0]
    return requirement.strip()


def _packages_ready(packages):
    for requirement in packages:
        name = _distribution_name(requirement)
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return False
        if "==" in requirement:
            expected = requirement.split("==", 1)[1].split(";", 1)[0].strip()
            if installed != expected:
                return False
    return True


def _extract(archive, destination):
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(destination, filter="data")
        except TypeError:  # pragma: no cover - Colab currently runs Python 3.12
            tf.extractall(destination)


def bootstrap():
    exchange_name = os.environ.get("AADP_EXCHANGE_DIR", "AADP_exchange")
    drive_root = Path(os.environ.get("AADP_DRIVE_ROOT", "/content/drive/MyDrive"))
    exchange = drive_root / exchange_name
    work = Path(os.environ.get("AADP_WORK_DIR", "/content/AADP"))
    state_path = Path(os.environ.get("AADP_STATE_PATH", "/content/aadp_state.json"))
    keep = {"open", "experiments", "logs"}

    if not exchange.is_dir():
        raise FileNotFoundError(
            f"Drive exchange is not mounted: {exchange}; run `aadp_colab.py mount` first"
        )
    bundles = sorted((exchange / "code").glob("code_*.tar.gz"))
    if not bundles:
        raise FileNotFoundError(
            f"no code bundle under {exchange / 'code'}; run `aadp_colab.py push <lane>`"
        )
    bundle = bundles[-1]

    work.mkdir(parents=True, exist_ok=True)
    for item in work.iterdir():
        if item.name in keep:
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    _extract(bundle, work)

    manifest_path = work / "cloud_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("bundle:", bundle.name)
    print("commit:", manifest["commit"], "| dirty_files:", manifest["dirty_file_count"])

    train_path = work / "open/data/train.jsonl"
    if not train_path.exists():
        data_bundle = exchange / "data/open_data.tar.gz"
        if not data_bundle.is_file():
            raise FileNotFoundError(
                f"dataset bundle missing: {data_bundle}; run `aadp_colab.py push <lane> --data`"
            )
        _extract(data_bundle, work)
    print("train_jsonl_mb:", round(train_path.stat().st_size / 1e6, 1))

    packages = _requested_packages()
    if _packages_ready(packages):
        print("dependencies: already satisfied")
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *packages],
            check=True,
        )

    import torch
    import transformers

    print(
        "torch:",
        torch.__version__,
        "| transformers:",
        transformers.__version__,
        "| cuda:",
        torch.cuda.is_available(),
    )

    results_path = work / "experiments/results.csv"
    with results_path.open(newline="", encoding="utf-8") as handle:
        baseline_ids = [row["experiment_id"] for row in csv.DictReader(handle)]
    state = {
        "baseline_experiment_ids": baseline_ids,
        "bootstrap_ts": time.time(),
        "commit": manifest["commit"],
        "bundle": bundle.name,
        "exchange": exchange_name,
        "control_mode": "cli",
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    (work / "logs").mkdir(exist_ok=True)
    print("baseline_rows:", len(baseline_ids))
    return {
        "ok": True,
        "op": "bootstrap",
        "bundle": bundle.name,
        "commit": manifest["commit"],
        "baseline_rows": len(baseline_ids),
        "exchange": exchange_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }


def main():
    try:
        result = bootstrap()
    except Exception as exc:
        traceback.print_exc()
        result = {"ok": False, "op": "bootstrap", "error": repr(exc)}
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
