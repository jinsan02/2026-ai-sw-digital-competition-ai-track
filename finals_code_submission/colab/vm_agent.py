"""VM-side agent for the Colab cloud lane (runs on the Colab VM, not locally).

Subcommands:
    daemon     poll the Drive command queue, heartbeat, auto-collect finished
               runs, and auto-unassign the runtime when idle (CU safety)
    launch     background-start a training run (same bookkeeping as [launch])
    status     training process state + log tail + GPU (scriptable [poll])
    collect    ship new results rows + fresh artifacts to Drive ([collect])

The daemon is started once per runtime by the notebook [agent] cell (the only
Quick Pick of the session). After that the local side drives everything through
`python colab/cloud_sync.py cmd/hb/unassign` over the Drive folder
`AADP_exchange/cmd/`. See colab/COLAB.md.

Env knobs (set before starting the daemon):
    AADP_CMD_POLL_S       queue/heartbeat interval, default 15
    AADP_IDLE_MAX_MIN     idle minutes before auto-unassign, default 45
    AADP_AUTO_COLLECT     "0" disables auto-collect of finished runs, default on
    AADP_CMD_MAX_AGE_MIN  queued commands older than this expire unexecuted, default 30
    AADP_EXCHANGE_DIR     Drive exchange folder name, default AADP_exchange
                          (one folder per concurrent runtime = one lane)
    AADP_CLI_MODE         "1" when started by aadp_colab.py. In this mode the
                          daemon never pretends it can unassign the VM; the local
                          wrapper owns release via `colab stop`.
"""

import argparse
import calendar
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/content/AADP")
EXCHANGE = Path("/content/drive/MyDrive") / os.environ.get("AADP_EXCHANGE_DIR", "AADP_exchange")
CMD = EXCHANGE / "cmd"
STATE_PATH = Path("/content/aadp_state.json")
OUTPUT_CAP = 20000


def atomic_write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def read_last_run():
    path = WORK / "logs/last_run.json"
    return json.loads(path.read_text()) if path.exists() else None


def proc_state(pid):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return "gone"
    return stat.read_text().rsplit(")", 1)[1].split()[0]


def training_status():
    run = read_last_run()
    if run is None:
        return None, False
    try:
        os.waitpid(run["pid"], os.WNOHANG)  # reap zombies when we are the parent
    except (ChildProcessError, PermissionError):
        pass
    return run, proc_state(run["pid"]) not in ("Z", "gone")


def gpu_line():
    if shutil.which("nvidia-smi") is None:
        return "none"
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True)
    return out.stdout.strip() or "unavailable"


def log_tail(run, lines=5):
    if run is None or not Path(run["log"]).exists():
        return []
    out = subprocess.run(["tail", "-n", str(lines), run["log"]], capture_output=True, text=True)
    return out.stdout.splitlines()


def launch(script, args):
    previous, alive = training_status()
    if alive:
        raise RuntimeError(
            f"refusing concurrent launch: pid {previous['pid']} is still alive "
            f"({Path(previous['log']).name})"
        )
    logs = WORK / "logs"
    logs.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_path = logs / f"run_{stamp}.log"
    cmd = [sys.executable, "-u", script] + shlex.split(args)
    proc = subprocess.Popen(cmd, cwd=WORK, stdout=log_path.open("w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    (logs / "last_run.json").write_text(json.dumps(
        {"pid": proc.pid, "log": str(log_path), "script": script, "args": args,
         "started_utc": stamp}, indent=2))
    print("pid:", proc.pid, "| log:", log_path.name)


def status():
    run, alive = training_status()
    if run is None:
        print("no run launched on this VM yet")
        return
    state = proc_state(run["pid"])
    print("pid:", run["pid"], "| state:", state, "| alive:", alive, "| log:", run["log"])
    print("\n".join(log_tail(run, 15)))
    print("gpu:", gpu_line())


def collect(extra_paths):
    """Returns the run name; raises RuntimeError when there is nothing new."""
    state = json.loads(STATE_PATH.read_text())
    base_ids = set(state["baseline_experiment_ids"])
    with (WORK / "experiments/results.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        new_rows = [r for r in reader if r["experiment_id"] not in base_ids]
    if not new_rows:
        raise RuntimeError("no new results rows since bootstrap/last collect")

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = f"{stamp}_{new_rows[-1]['experiment_id'][:60]}"
    out = EXCHANGE / "runs" / run_name
    for sub in ("logits", "artifacts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with (out / "results_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    copied = 0
    for sub in ("logits", "artifacts"):
        src = WORK / "experiments" / sub
        if src.exists():
            for p in src.iterdir():
                if p.is_file() and p.stat().st_mtime >= state["bootstrap_ts"]:
                    shutil.copy2(p, out / sub / p.name)
                    copied += 1
    for extra in extra_paths:
        src = WORK / extra
        shutil.copytree(src, out / "extra" / src.name, dirs_exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(
        {"commit": state["commit"], "rows": len(new_rows), "files": copied,
         "created_utc": stamp}, indent=2))
    # fold collected ids into the baseline so a re-collect ships nothing twice
    state["baseline_experiment_ids"] += [r["experiment_id"] for r in new_rows]
    STATE_PATH.write_text(json.dumps(state))
    print("run:", run_name, "| rows:", len(new_rows), "| files:", copied)
    print("local next: .venv/bin/python colab/cloud_sync.py pull", run_name)
    return run_name


# google.colab.runtime.unassign() needs the IPython kernel and the daemon is a plain
# subprocess ('NoneType' object has no attribute 'kernel'), so the daemon cannot release
# the runtime itself. It exits with UNASSIGN_EXIT instead and the [agent] cell -- which
# runs in the kernel -- performs the actual unassign when it sees that exit code.
UNASSIGN_EXIT = 86


def cli_mode():
    return os.environ.get("AADP_CLI_MODE", "0") == "1"


def release_exit_code():
    """Preserve the notebook cell's rc=86 contract; CLI release is local-only."""
    return 0 if cli_mode() else UNASSIGN_EXIT


def spec_age_min(spec):
    """Age of a queued command in minutes, from the utc stamp in its id; None if unparseable."""
    try:
        ts = calendar.timegm(time.strptime(str(spec.get("id", ""))[:15], "%Y%m%d_%H%M%S"))
    except ValueError:
        return None
    return (time.time() - ts) / 60


def run_command(spec):
    started = time.time()
    cmd = spec.get("cmd", "")
    if cmd == "@unassign" and cli_mode():
        rc = 126
        output = "disabled in CLI mode; run locally: python colab/aadp_colab.py down <lane>"
    elif cmd in ("@unassign", "@stop"):
        rc, output = 0, f"builtin {cmd} acknowledged"
    else:
        try:
            proc = subprocess.run(cmd, shell=True, cwd=WORK, text=True,
                                  capture_output=True, timeout=spec.get("timeout", 600))
            rc = proc.returncode
            output = (proc.stdout + proc.stderr)[-OUTPUT_CAP:]
        except subprocess.TimeoutExpired:
            rc, output = 124, f"timeout after {spec.get('timeout', 600)}s"
    return {"id": spec["id"], "cmd": cmd, "rc": rc, "output": output,
            "duration_s": round(time.time() - started, 1),
            "finished_utc": time.strftime("%Y%m%d_%H%M%S", time.gmtime())}


def daemon():
    poll_s = int(os.environ.get("AADP_CMD_POLL_S", "15"))
    idle_max_min = float(os.environ.get("AADP_IDLE_MAX_MIN", "45"))
    cmd_max_age_min = float(os.environ.get("AADP_CMD_MAX_AGE_MIN", "30"))
    auto_collect = os.environ.get("AADP_AUTO_COLLECT", "1") != "0"
    (CMD / "queue").mkdir(parents=True, exist_ok=True)
    (CMD / "done").mkdir(parents=True, exist_ok=True)
    started = time.time()
    last_activity = started
    collected_for = None  # log path of the run already auto-collected
    collect_note = ""
    mode = "cli" if cli_mode() else "notebook"
    cli_idle_warned = False
    print(f"daemon up: poll={poll_s}s idle_max={idle_max_min}min "
          f"cmd_max_age={cmd_max_age_min}min auto_collect={auto_collect} mode={mode}", flush=True)

    while True:
        stop = False
        release = False
        try:
            queue_files = sorted((CMD / "queue").glob("*.json"))
        except OSError as exc:  # Drive hiccup; keep looping
            print("queue scan failed:", exc, flush=True)
            queue_files = []
        for qf in queue_files:
            try:
                spec = json.loads(qf.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # likely still uploading; retry next loop
            try:
                qf.unlink(missing_ok=True)
            except OSError:
                continue  # could not claim it; retry next loop
            last_activity = time.time()
            cli_idle_warned = False
            age_min = spec_age_min(spec)
            if age_min is not None and age_min > cmd_max_age_min:
                # a command queued at a dead daemon must not fire on (re)start --
                # a stale @unassign would release a fresh runtime before collect
                result = {"id": spec["id"], "cmd": spec.get("cmd", ""), "rc": 125,
                          "output": f"expired: queued {age_min:.0f} min ago (max {cmd_max_age_min:.0f})",
                          "duration_s": 0.0,
                          "finished_utc": time.strftime("%Y%m%d_%H%M%S", time.gmtime())}
            else:
                result = run_command(spec)
                if spec.get("cmd") == "@unassign" and not cli_mode():
                    result["output"] = "unassign scheduled: daemon exiting; [agent] cell releases the runtime"
                    release = True
                if spec.get("cmd") == "@stop":
                    stop = True
            try:
                atomic_write(CMD / "done" / f"{spec['id']}.json", json.dumps(result, indent=2))
            except OSError as exc:  # Drive hiccup; keep looping
                print("done write failed:", exc, flush=True)
            print(f"cmd {spec['id']} rc={result['rc']}: {spec.get('cmd', '')[:120]}", flush=True)
        if release:
            print("daemon exiting for kernel-side unassign", flush=True)
            return release_exit_code()
        if stop:
            print("daemon stopping on @stop", flush=True)
            return 0

        run, alive = training_status()
        if alive:
            last_activity = time.time()
            cli_idle_warned = False
        elif auto_collect and run is not None and collected_for != run["log"] and STATE_PATH.exists():
            try:
                collect_note = "collected:" + collect([])
            except RuntimeError as exc:
                collect_note = f"collect_skipped: {exc}"
                collected_for = run["log"]
            except Exception as exc:
                collect_note = f"collect_error: {exc!r}"
            else:
                collected_for = run["log"]
            print(collect_note, flush=True)

        idle_min = (time.time() - last_activity) / 60
        safe_to_release = run is None or (not alive and collected_for == run["log"])
        cli_idle_expired = cli_mode() and idle_min > idle_max_min and safe_to_release
        hb = {"ts": time.time(), "utc": time.strftime("%Y%m%d_%H%M%S", time.gmtime()),
              "daemon_pid": os.getpid(), "uptime_min": round((time.time() - started) / 60, 1),
              "gpu": gpu_line(), "idle_min": round(idle_min, 1),
              "idle_max_min": idle_max_min, "collect_note": collect_note,
              "control_mode": mode, "exchange": EXCHANGE.name,
              "release_safe": safe_to_release, "release_required": cli_idle_expired,
              "run": None if run is None else {
                  "pid": run["pid"], "state": proc_state(run["pid"]), "alive": alive,
                  "log": Path(run["log"]).name, "args": run.get("args", ""),
                  "tail": log_tail(run, 5)}}
        if cli_idle_expired:
            hb["status"] = "idle_expired_cli_stop_required"
        try:
            atomic_write(CMD / "heartbeat.json", json.dumps(hb, indent=2))
        except OSError as exc:  # Drive hiccup; keep looping
            print("heartbeat write failed:", exc, flush=True)

        if idle_min > idle_max_min and safe_to_release:
            if cli_mode():
                if not cli_idle_warned:
                    print("idle limit reached; CLI mode requires local aadp_colab.py down", flush=True)
                    cli_idle_warned = True
                time.sleep(poll_s)
                continue
            hb["status"] = "idle limit reached, unassigning"
            try:
                atomic_write(CMD / "heartbeat.json", json.dumps(hb, indent=2))
            except OSError:
                pass
            print("idle limit reached; daemon exiting for kernel-side unassign", flush=True)
            return release_exit_code()

        time.sleep(poll_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("daemon", help="command-channel daemon (started by the [agent] cell)")
    p_launch = sub.add_parser("launch", help="background-start a training run")
    p_launch.add_argument("script")
    p_launch.add_argument("args")
    sub.add_parser("status", help="training process state + log tail + GPU")
    p_collect = sub.add_parser("collect", help="ship new results to Drive")
    p_collect.add_argument("--extra", action="append", default=[],
                           help="extra WORK-relative dir to ship (e.g. model_out_final)")
    args = parser.parse_args()
    if args.command == "daemon":
        sys.exit(daemon())
    elif args.command == "launch":
        launch(args.script, args.args)
    elif args.command == "status":
        status()
    else:
        try:
            collect(args.extra)
        except RuntimeError as exc:
            sys.exit(str(exc))


if __name__ == "__main__":
    main()
