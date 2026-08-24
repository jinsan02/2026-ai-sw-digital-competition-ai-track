"""Local <-> Colab exchange over a Google Drive rclone remote.

Usage:
    python colab/cloud_sync.py push [--data]   # code bundle (+ one-time data tarball) -> Drive
    python colab/cloud_sync.py push-anchor FILE # fixed validation payload -> Drive anchors/
    python colab/cloud_sync.py list            # list collected runs on Drive
    python colab/cloud_sync.py pull RUN_NAME   # download a run and merge results/artifacts
    python colab/cloud_sync.py pull-model NAME # download saved weights from Drive models/
                                               # (list names: rclone lsd gdrive:AADP_exchange/models)
    python colab/cloud_sync.py cmd "SHELL"     # run a shell command on the VM via the daemon
    python colab/cloud_sync.py launch SCRIPT -- ARGS...   # start a training run, no hand quoting
    python colab/cloud_sync.py hb              # read the VM daemon heartbeat (cheap poll)
    python colab/cloud_sync.py unassign        # legacy synchronous-notebook lanes only

cmd/hb need the daemon started once per runtime by `aadp_colab.py daemon` or the
legacy notebook [agent] cell. CLI lanes release through `aadp_colab.py down`;
their daemon rejects this module's legacy `unassign` operation.

Requires an rclone remote for Google Drive (default name: gdrive,
override with AADP_RCLONE_REMOTE). See colab/COLAB.md for setup.

Multi-lane: AADP_EXCHANGE_DIR selects the Drive exchange folder (default
AADP_exchange). Each concurrent Colab runtime gets its own lane, e.g.
    AADP_EXCHANGE_DIR=AADP_exchange_b python colab/cloud_sync.py hb
paired with colab/colab_runner_b.ipynb on the VM side.
"""

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMOTE = os.environ.get("AADP_RCLONE_REMOTE", "gdrive")
EXCHANGE = f"{REMOTE}:{os.environ.get('AADP_EXCHANGE_DIR', 'AADP_exchange')}"


def sh(cmd, cwd=None):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, cwd=cwd).stdout


def require_rclone():
    if shutil.which("rclone") is None:
        sys.exit("rclone not found; install it and run `rclone config` (see colab/COLAB.md)")


def push(include_data):
    commit = sh(["git", "rev-parse", "HEAD"], cwd=REPO).strip()
    dirty = [line for line in sh(["git", "status", "--porcelain"], cwd=REPO).splitlines() if line.strip()]
    def ls_files(*flags):
        # open/ holds the 100MB dataset (tracked); it ships separately via push --data
        out = sh(["git", "ls-files", "-z", *flags], cwd=REPO)
        return [f for f in out.split("\0") if f and not f.startswith("open/")]

    tracked = ls_files()
    untracked = ls_files("--others", "--exclude-standard")
    files = tracked + untracked
    fingerprint = hashlib.sha256()
    for rel in sorted(files):
        path = REPO / rel
        if not path.is_file():
            continue
        rel_bytes = rel.encode("utf-8")
        fingerprint.update(len(rel_bytes).to_bytes(4, "big"))
        fingerprint.update(rel_bytes)
        fingerprint.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                fingerprint.update(chunk)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    tag = f"{stamp}_{commit[:8]}" + ("_dirty" if dirty else "")
    manifest = {
        "created_utc": stamp,
        "commit": commit,
        "dirty_file_count": len(dirty),
        "dirty_files": dirty[:100],
        "tracked_file_count": len(tracked),
        "untracked_file_count": len(untracked),
        "working_tree_fingerprint_sha256": fingerprint.hexdigest(),
    }
    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / f"code_{tag}.tar.gz"
        manifest_path = Path(td) / "cloud_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        with tarfile.open(bundle, "w:gz") as tf:
            for rel in files:
                path = REPO / rel
                if path.is_file():
                    tf.add(path, arcname=rel)
            tf.add(manifest_path, arcname="cloud_manifest.json")
        sh(["rclone", "copy", str(bundle), f"{EXCHANGE}/code/"])
        print(f"pushed {bundle.name} tracked_files={len(files)} dirty_files={len(dirty)}")
    if include_data:
        with tempfile.TemporaryDirectory() as td:
            data_bundle = Path(td) / "open_data.tar.gz"
            with tarfile.open(data_bundle, "w:gz") as tf:
                tf.add(REPO / "open/data", arcname="open/data")
            sh(["rclone", "copy", str(data_bundle), f"{EXCHANGE}/data/"])
            print("pushed open_data.tar.gz")


def push_anchor(path, remote_name):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"anchor payload not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    sh(["rclone", "copyto", str(path), f"{EXCHANGE}/anchors/{remote_name}"])
    print(
        f"pushed anchor {path} -> {EXCHANGE}/anchors/{remote_name} "
        f"size_mb={path.stat().st_size / 1e6:.2f} sha256={digest.hexdigest()}"
    )


def list_runs():
    out = subprocess.run(["rclone", "lsd", f"{EXCHANGE}/runs"], text=True, capture_output=True)
    print(out.stdout.strip() or "no runs collected yet")


def merge_results(rows_path):
    local = REPO / "experiments/results.csv"
    with local.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = {row["experiment_id"] for row in reader}
    with rows_path.open(newline="", encoding="utf-8") as f:
        new_rows = [row for row in csv.DictReader(f) if row["experiment_id"] not in existing]
    if not new_rows:
        print("no new results rows (all experiment_ids already present)")
        return 0
    raw = local.read_bytes()
    if raw and not raw.endswith(b"\n"):
        with local.open("ab") as f:
            f.write(b"\r\n")
    with local.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in new_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            print("merged:", row["experiment_id"])
    return len(new_rows)


def pull(run_name):
    dest = REPO / "experiments/incoming" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    sh(["rclone", "copy", f"{EXCHANGE}/runs/{run_name}", str(dest)])
    rows_path = dest / "results_rows.csv"
    added = merge_results(rows_path) if rows_path.exists() else 0
    placed = 0
    for sub in ("logits", "artifacts"):
        src = dest / sub
        if not src.exists():
            continue
        target_dir = REPO / "experiments" / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in src.iterdir():
            if not path.is_file():
                continue
            target = target_dir / path.name
            if target.exists():
                print("skip (exists):", target.name)
            else:
                shutil.copy2(path, target)
                placed += 1
    print(f"pulled {run_name}: +{added} results rows, +{placed} artifact files")
    print(f"raw copy kept at {dest} (extra/ contents such as model dirs stay there for manual placement)")


def pull_model(name):
    dest = REPO / "experiments/incoming/models" / name
    dest.mkdir(parents=True, exist_ok=True)
    sh(["rclone", "copy", f"{EXCHANGE}/models/{name}", str(dest)])
    files = [f for f in dest.rglob("*") if f.is_file()]
    if not files:
        sys.exit(f"nothing at {EXCHANGE}/models/{name} -- check `rclone lsd {EXCHANGE}/models`")
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"pulled model {name} -> {dest} ({len(files)} files, {size_mb:.0f} MB)")


def build_launch_cmd(script, run_args):
    """argv list -> the exact `vm_agent.py launch` shell string, quoting handled here.

    vm_agent.launch shlex.split()s its args string, so shlex.join() round-trips the
    argv exactly -- spaces in --notes, leading-dash values, etc. survive untouched.
    """
    if run_args[:1] == ["--"]:  # argparse.REMAINDER keeps the separator on some versions
        run_args = run_args[1:]
    if not run_args:
        sys.exit("no run args; usage: cloud_sync.py launch SCRIPT -- --device cuda ...")
    return (f"python colab/vm_agent.py launch {shlex.quote(script)} "
            f"{shlex.quote(shlex.join(run_args))}")


def send_cmd(command, timeout, wait):
    cmd_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + os.urandom(3).hex()
    spec = json.dumps({"id": cmd_id, "cmd": command, "timeout": timeout})
    subprocess.run(["rclone", "rcat", f"{EXCHANGE}/cmd/queue/{cmd_id}.json"],
                   input=spec, text=True, check=True)
    print(f"queued {cmd_id}: {command}")
    if wait <= 0:
        return
    deadline = time.time() + wait
    claimed = False
    while time.time() < deadline:
        out = subprocess.run(["rclone", "cat", f"{EXCHANGE}/cmd/done/{cmd_id}.json"],
                             text=True, capture_output=True)
        if out.returncode == 0 and out.stdout.strip():
            result = json.loads(out.stdout)
            print(f"rc={result['rc']} duration_s={result['duration_s']}")
            print(result["output"])
            sys.exit(0 if result["rc"] == 0 else 1)
        if not claimed:
            q = subprocess.run(["rclone", "lsf", f"{EXCHANGE}/cmd/queue/{cmd_id}.json"],
                               text=True, capture_output=True)
            if (q.returncode == 0 and not q.stdout.strip()) or "not found" in q.stderr:
                claimed = True
                print("claimed by daemon -- executing, or result still propagating")
        time.sleep(10)
    age = hb_age()
    age_note = f"heartbeat age {age:.0f}s" if age is not None else "no heartbeat"
    if claimed:
        sys.exit(f"no result after {wait}s, but the daemon CLAIMED the command ({age_note}) -- "
                 f"likely still executing or Drive lag; re-check later:\n"
                 f"  rclone cat {EXCHANGE}/cmd/done/{cmd_id}.json")
    sys.exit(f"no result after {wait}s and the command is STILL QUEUED ({age_note}) -- "
             f"the daemon has not seen it; check: python colab/cloud_sync.py hb")


def hb_age():
    out = subprocess.run(["rclone", "cat", f"{EXCHANGE}/cmd/heartbeat.json"],
                         text=True, capture_output=True)
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return time.time() - json.loads(out.stdout)["ts"]
    except (json.JSONDecodeError, KeyError):
        return None


def heartbeat():
    out = subprocess.run(["rclone", "cat", f"{EXCHANGE}/cmd/heartbeat.json"],
                         text=True, capture_output=True)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit("no heartbeat on Drive -- daemon not started this runtime? (run the [agent] cell)")
    hb = json.loads(out.stdout)
    age = time.time() - hb["ts"]
    if age > 300:
        note = " (STALE -- daemon or VM gone: recycled, [agent] cell stopped, or unassigned)"
    elif age > 90:
        note = " (stale-ish -- could be Drive propagation lag; retry in ~60s before concluding)"
    else:
        note = ""
    print(f"heartbeat age: {age:.0f}s{note}")
    print(out.stdout.strip())


def refuse_cli_unassign():
    """Fail closed when the CLI wrapper, not the notebook cell, owns VM release."""
    out = subprocess.run(["rclone", "cat", f"{EXCHANGE}/cmd/heartbeat.json"],
                         text=True, capture_output=True)
    if out.returncode != 0 or not out.stdout.strip():
        return
    try:
        hb = json.loads(out.stdout)
    except json.JSONDecodeError:
        return
    if hb.get("control_mode") == "cli":
        sys.exit("CLI lane detected: `cloud_sync.py unassign` cannot release this VM. "
                 "Use `python colab/aadp_colab.py down <lane>`.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_push = sub.add_parser("push", help="upload code bundle (and optionally data) to Drive")
    p_push.add_argument("--data", action="store_true", help="also upload open/data tarball (one-time)")
    p_anchor = sub.add_parser("push-anchor", help="upload the fixed validation anchor payload")
    p_anchor.add_argument("path")
    p_anchor.add_argument("--name", default="anchor_val_logits.pt")
    sub.add_parser("list", help="list collected runs on Drive")
    p_pull = sub.add_parser("pull", help="download a collected run and merge results")
    p_pull.add_argument("run_name")
    p_pm = sub.add_parser("pull-model", help="download saved model weights from Drive models/")
    p_pm.add_argument("model_name")
    p_cmd = sub.add_parser("cmd", help="run a shell command on the VM via the daemon")
    p_cmd.add_argument("shell_command")
    p_cmd.add_argument("--timeout", type=int, default=600, help="VM-side exec timeout (s)")
    p_cmd.add_argument("--wait", type=int, default=300, help="how long to poll for the result (s)")
    p_cmd.add_argument("--no-wait", action="store_true", help="queue and return immediately")
    p_vml = sub.add_parser("launch", help="start a training run on the VM; put run args after "
                                          "-- so nothing needs hand quoting")
    p_vml.add_argument("--timeout", type=int, default=60, help="VM-side exec timeout (s)")
    p_vml.add_argument("--wait", type=int, default=300, help="how long to poll for the result (s)")
    p_vml.add_argument("script")
    p_vml.add_argument("run_args", nargs=argparse.REMAINDER,
                       help="training args, after a -- separator")
    sub.add_parser("hb", help="read the VM daemon heartbeat")
    sub.add_parser("unassign", help="release a legacy synchronous-notebook runtime")
    args = parser.parse_args()
    require_rclone()
    if args.command == "push":
        push(args.data)
    elif args.command == "push-anchor":
        push_anchor(args.path, args.name)
    elif args.command == "list":
        list_runs()
    elif args.command == "pull":
        pull(args.run_name)
    elif args.command == "pull-model":
        pull_model(args.model_name)
    elif args.command == "cmd":
        send_cmd(args.shell_command, args.timeout, 0 if args.no_wait else args.wait)
    elif args.command == "launch":
        send_cmd(build_launch_cmd(args.script, args.run_args), args.timeout, args.wait)
    elif args.command == "hb":
        heartbeat()
    else:
        refuse_cli_unassign()
        send_cmd("@unassign", 60, 120)


if __name__ == "__main__":
    main()
