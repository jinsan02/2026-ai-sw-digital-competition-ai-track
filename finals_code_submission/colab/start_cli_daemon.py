"""Start the existing Drive command daemon as a detached Colab process."""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


RESULT_PREFIX = "AADP_RESULT="


def _proc_state(pid):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return "gone"
    return stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]


def _is_live_daemon(state):
    try:
        pid = int(state["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if _proc_state(pid) in ("Z", "gone"):
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "vm_agent.py daemon" in command


def start_daemon():
    work = Path(os.environ.get("AADP_WORK_DIR", "/content/AADP"))
    script = work / "colab/vm_agent.py"
    state_path = work / "logs/agent_daemon.json"
    log_path = work / "logs/agent_daemon.log"
    if not script.is_file():
        raise FileNotFoundError(f"bootstrap first; missing {script}")
    if not Path(os.environ.get("AADP_STATE_PATH", "/content/aadp_state.json")).is_file():
        raise FileNotFoundError("bootstrap first; /content/aadp_state.json is missing")

    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if _is_live_daemon(previous):
            return {
                "ok": True,
                "op": "start_daemon",
                "already_running": True,
                "pid": previous["pid"],
                "exchange": previous.get("exchange"),
            }

    env = os.environ.copy()
    env["AADP_CLI_MODE"] = "1"
    env.setdefault("AADP_EXCHANGE_DIR", "AADP_exchange")
    env.setdefault("AADP_IDLE_MAX_MIN", "45")
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script), "daemon"],
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state = {
        "pid": process.pid,
        "log": str(log_path),
        "started_utc": time.strftime("%Y%m%d_%H%M%S", time.gmtime()),
        "exchange": env["AADP_EXCHANGE_DIR"],
        "control_mode": "cli",
    }
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(state_path)
    time.sleep(0.25)
    state["state"] = _proc_state(process.pid)
    if state["state"] in ("Z", "gone"):
        raise RuntimeError(f"daemon exited immediately; inspect {log_path}")
    return {"ok": True, "op": "start_daemon", "already_running": False, **state}


def main():
    try:
        result = start_daemon()
    except Exception as exc:
        traceback.print_exc()
        result = {"ok": False, "op": "start_daemon", "error": repr(exc)}
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
