"""Provision the isolated Qwen3.5 training/export environment on a Colab VM."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


VENV = Path("/content/venv311")
PYTHON = VENV / "bin/python"


def run(command):
    print("running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def pip_is_usable():
    if not PYTHON.exists():
        return False
    return (
        subprocess.run(
            [PYTHON, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def ensure_venv():
    if pip_is_usable():
        return

    # Some Colab images ship Python without a working stdlib ensurepip. Try the
    # cheap native path first, then fall back to virtualenv from the base pip.
    shutil.rmtree(VENV, ignore_errors=True)
    native = subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", VENV]
    )
    if native.returncode == 0 and pip_is_usable():
        return

    print("stdlib venv unavailable; falling back to virtualenv", flush=True)
    shutil.rmtree(VENV, ignore_errors=True)
    run([sys.executable, "-m", "pip", "install", "--quiet", "virtualenv"])
    run(
        [
            sys.executable,
            "-m",
            "virtualenv",
            "--system-site-packages",
            VENV,
        ]
    )
    if not pip_is_usable():
        raise RuntimeError(f"created environment has no usable pip: {VENV}")


def main():
    ensure_venv()
    run(
        [
            PYTHON,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "transformers>=5.13,<5.14",
            "safetensors==0.8.0",
        ]
    )
    probe = subprocess.run(
        [
            str(PYTHON),
            "-c",
            (
                "import json, torch, transformers, safetensors; "
                "from transformers import Qwen3_5TextForSequenceClassification; "
                "print(json.dumps({'python': __import__('sys').version.split()[0], "
                "'torch': torch.__version__, 'transformers': transformers.__version__, "
                "'safetensors': safetensors.__version__}))"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    versions = json.loads(probe.stdout.strip().splitlines()[-1])
    if not str(versions["transformers"]).startswith("5.13."):
        raise RuntimeError(f"expected transformers 5.13.x, got {versions}")
    if str(versions["safetensors"]) != "0.8.0":
        raise RuntimeError(f"expected safetensors 0.8.0, got {versions}")
    print("qwen35 venv ready:", json.dumps(versions, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
