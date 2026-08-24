"""Chain arbitrary training runs through the Colab command channel (one lane).

Generalizes chain_oof_folds.py: reads a JSON plan of launch specs and runs them
sequentially, waiting for each run's auto-collect before launching the next.
Waits for any already-live run first. Lane selection via AADP_EXCHANGE_DIR:

    .venv/bin/python colab/chain_runs.py --plan planA.json
    AADP_EXCHANGE_DIR=AADP_exchange_b .venv/bin/python colab/chain_runs.py --plan planB.json

Plan format (list, executed in order):
    [{"script": "train_transformer.py", "args": ["--device", "cuda", ...]}, ...]

Local process — it dies with the machine; safe to restart (already-collected
runs are skipped only if you edit the plan; check `cloud_sync.py list` first).
Exit codes: 0 = all collected, 2 = heartbeat stale, 3 = deadline,
4 = collect problem, 5 = launch failed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv/bin/python")
REMOTE = (f"{os.environ.get('AADP_RCLONE_REMOTE', 'gdrive')}:"
          f"{os.environ.get('AADP_EXCHANGE_DIR', 'AADP_exchange')}/cmd/heartbeat.json")


def read_hb(retries=0):
    """One heartbeat read; with retries > 0, survive transient rclone hangs (e.g. token refresh)."""
    for attempt in range(retries + 1):
        try:
            out = subprocess.run(["rclone", "cat", REMOTE], capture_output=True, text=True, timeout=90)
            return json.loads(out.stdout)
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"hb read failed ({exc}); retrying", flush=True)
            time.sleep(20)


def wait_collect(floor, deadline, poll_seconds, stale_note=""):
    while time.time() < deadline:
        time.sleep(poll_seconds)
        try:
            hb = read_hb()
        except Exception as exc:
            print("hb read failed:", exc, flush=True)
            continue
        age = time.time() - hb["ts"]
        note = hb.get("collect_note", "")
        run = hb.get("run") or {}
        tail = (run.get("tail") or [""])[-1]
        print(f"age={age:.0f}s alive={run.get('alive')} note={note[:55]} tail={tail.strip()[:70]}",
              flush=True)
        if note.startswith("collected:") and note[10:25] > floor:
            print("COLLECTED:", note[10:], flush=True)
            return
        # collect notes persist in the heartbeat across runs; only a NEW error/skip is fatal
        if note.startswith(("collect_error", "collect_skipped")) and note != stale_note:
            sys.exit(4)
        if age > 400:
            print(f"HEARTBEAT STALE ({age:.0f}s)", flush=True)
            sys.exit(2)
    print("deadline reached", flush=True)
    sys.exit(3)


def launch(spec, index):
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    args = ([PY, "colab/cloud_sync.py", "launch", spec["script"], "--"] + spec["args"])
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=400,
                       env=os.environ.copy())
    print(f"launch run{index}: rc={r.returncode}\n{r.stdout.strip()}", flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        sys.exit(5)
    return stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON file: [{script, args}, ...]")
    parser.add_argument("--max-minutes", type=int, default=420)
    parser.add_argument("--poll-seconds", type=int, default=60,
                        help="heartbeat polling interval between chained runs (default: 60)")
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    plan = json.loads(Path(args.plan).read_text())
    deadline = time.time() + args.max_minutes * 60

    hb0 = read_hb(retries=4)
    stale_note = hb0.get("collect_note", "")
    run = hb0.get("run") or {}
    match = re.match(r"run_(\d{8}_\d{6})\.log$", run.get("log") or "")
    if run.get("alive") and match:
        print(f"waiting on live run {run['log']}", flush=True)
        wait_collect(match.group(1), deadline, args.poll_seconds, stale_note)
    for index, spec in enumerate(plan):
        wait_collect(launch(spec, index), deadline, args.poll_seconds, stale_note)
    print("ALL RUNS COLLECTED", flush=True)


if __name__ == "__main__":
    main()
