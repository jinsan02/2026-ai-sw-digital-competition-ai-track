"""Run a JSON training plan sequentially inside one detached Colab job.

Unlike ``chain_runs.py`` this wrapper runs on the VM.  ``vm_agent.py`` sees the
whole plan as one live process, so the second arm starts even when the local
Codex turn or websocket has ended.  Any failed arm stops the plan immediately.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--start-arm", type=int, default=1, help="1-based first arm to run")
    parser.add_argument(
        "--drive-exchange",
        default="",
        help="replace /MyDrive/AADP_exchange/ paths in plan args with this exchange",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a non-empty JSON list")
    if not 1 <= args.start_arm <= len(plan):
        raise ValueError(f"--start-arm must be between 1 and {len(plan)}")

    print(
        f"REMOTE PLAN start path={plan_path} arms={len(plan)} start_arm={args.start_arm}",
        flush=True,
    )
    for index, spec in enumerate(plan, 1):
        if index < args.start_arm:
            continue
        script = spec.get("script")
        run_args = spec.get("args")
        if not isinstance(script, str) or not isinstance(run_args, list):
            raise ValueError(f"invalid arm {index}: expected script string and args list")
        python = spec.get("python", sys.executable)
        if not isinstance(python, str) or not python:
            raise ValueError(f"invalid arm {index}: python must be a non-empty string")
        arm_env = spec.get("env", {})
        if (
            not isinstance(arm_env, dict)
            or any(not isinstance(key, str) or not key for key in arm_env)
            or any(not isinstance(value, str) for value in arm_env.values())
        ):
            raise ValueError(f"invalid arm {index}: env must map non-empty strings to strings")
        if args.drive_exchange:
            source = "/content/drive/MyDrive/AADP_exchange/"
            target = f"/content/drive/MyDrive/{args.drive_exchange}/"
            run_args = [str(value).replace(source, target) for value in run_args]
        cmd = [python, "-u", script, *map(str, run_args)]
        started = time.time()
        print(
            f"REMOTE PLAN arm={index}/{len(plan)} python={python} script={script} START",
            flush=True,
        )
        # The training stack is PyTorch-only.  Explicitly disable discovery of
        # installed TensorFlow/Flax backends before the child imports
        # transformers; some Colab images otherwise hang during TensorFlow
        # initialization after tokenization and before model loading.
        child_env = os.environ.copy()
        child_env.setdefault("USE_TORCH", "1")
        child_env.setdefault("USE_TF", "0")
        child_env.setdefault("USE_FLAX", "0")
        # Colab's Xet-backed Hugging Face download path can deadlock with an
        # incomplete blob held open and the HTTPS socket in CLOSE_WAIT.  The
        # ordinary HTTP downloader is slower only during the one-time cache
        # fill and is identical once from_pretrained reads the cached files.
        child_env.setdefault("HF_HUB_DISABLE_XET", "1")
        child_env.update(arm_env)
        result = subprocess.run(cmd, env=child_env)
        elapsed = time.time() - started
        print(
            f"REMOTE PLAN arm={index}/{len(plan)} rc={result.returncode} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if result.returncode:
            raise SystemExit(result.returncode)
    print("REMOTE PLAN ALL ARMS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
