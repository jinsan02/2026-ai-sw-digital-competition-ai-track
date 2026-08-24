"""Fast Colab CLI control plane for the existing AADP Drive training lane.

Colab CLI owns VM lifecycle and keep-alive.  ``cloud_sync.py`` and
``vm_agent.py`` keep owning code/data transport, background training, heartbeat,
and collection.  Run ``python colab/aadp_colab.py --help`` for commands.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO = Path(__file__).resolve().parents[1]
LANES_FILE = REPO / "colab/lanes.json"
CLOUD_SYNC = REPO / "colab/cloud_sync.py"
BOOTSTRAP = REPO / "colab/bootstrap_cli.py"
START_DAEMON = REPO / "colab/start_cli_daemon.py"
CLI_VERSION = "0.6.0"
RESULT_PREFIX = "AADP_RESULT="
VALID_GPUS = frozenset({"T4", "L4", "G4", "H100", "A100"})
GPU_NAME_MARKERS = {
    "T4": ("T4",),
    "L4": ("L4",),
    "G4": ("G4", "RTX PRO 6000", "BLACKWELL"),
    "H100": ("H100",),
    "A100": ("A100",),
}


class AADPColabError(RuntimeError):
    pass


@dataclass(frozen=True)
class Lane:
    name: str
    session: str
    state_file: Path
    exchange: str
    gpu: str
    packages: tuple[str, ...]


def load_lanes(path: Path = LANES_FILE) -> dict[str, Lane]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AADPColabError(f"cannot read lane config {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise AADPColabError(f"lane config must be a non-empty object: {path}")

    lanes = {}
    private_root = (REPO / ".colab").resolve()
    for name, item in raw.items():
        if not isinstance(item, dict):
            raise AADPColabError(f"lane {name!r} must be an object")
        try:
            session = item["session"]
            exchange = item["exchange"]
            gpu = item["gpu"].upper()
            packages = tuple(item["packages"])
            state_file = (REPO / item["state_file"]).resolve()
        except (KeyError, TypeError, AttributeError) as exc:
            raise AADPColabError(f"invalid lane {name!r}: {exc}") from exc
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", session):
            raise AADPColabError(f"invalid session name for lane {name}: {session!r}")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", exchange):
            raise AADPColabError(f"invalid exchange folder for lane {name}: {exchange!r}")
        if gpu not in VALID_GPUS:
            raise AADPColabError(
                f"invalid GPU {gpu!r} for lane {name}; choose {', '.join(sorted(VALID_GPUS))}"
            )
        if not packages or not all(isinstance(package, str) and package for package in packages):
            raise AADPColabError(f"lane {name} packages must be a non-empty string list")
        if not state_file.is_relative_to(private_root):
            raise AADPColabError(
                f"lane {name} state_file must stay under gitignored {private_root}: {state_file}"
            )
        lanes[name] = Lane(name, session, state_file, exchange, gpu, packages)
    return lanes


def get_lane(name: str, path: Path = LANES_FILE) -> Lane:
    lanes = load_lanes(path)
    if name not in lanes:
        raise AADPColabError(f"unknown lane {name!r}; choose {', '.join(sorted(lanes))}")
    return lanes[name]


def _run_capture(
    argv: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    echo_output: bool = True,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(argv),
            cwd=REPO,
            env=env,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AADPColabError(f"command failed to start: {argv[0]}: {exc}") from exc
    if echo_output and result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if check and result.returncode != 0:
        raise AADPColabError(
            f"command failed (rc={result.returncode}): {' '.join(argv)}"
        )
    return result


def _run_live(
    argv: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(list(argv), cwd=REPO, env=env, text=True)
    except OSError as exc:
        raise AADPColabError(f"command failed to start: {argv[0]}: {exc}") from exc
    if check and result.returncode != 0:
        raise AADPColabError(
            f"command failed (rc={result.returncode}): {' '.join(argv)}"
        )
    return result


def _require_program(name: str, remediation: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise AADPColabError(f"{name} not found; {remediation}")
    return executable


def colab_argv(lane: Lane, *parts: str) -> list[str]:
    return [
        "colab",
        "--auth=adc",
        "--config",
        str(lane.state_file),
        *parts,
    ]


def _read_local_session(lane: Lane) -> dict | None:
    try:
        data = json.loads(lane.state_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise AADPColabError(f"cannot read Colab state {lane.state_file}: {exc}") from exc
    session = data.get(lane.session)
    return session if isinstance(session, dict) else None


def _refresh_local_session(lane: Lane) -> dict | None:
    if _read_local_session(lane) is None:
        return None
    _run_capture(
        colab_argv(lane, "status", "-s", lane.session),
        check=False,
        echo_output=False,
    )
    return _read_local_session(lane)


def _parse_assignments(output: str) -> dict[str, str]:
    """Parse v0.6.0 session lines into endpoint -> local name."""
    assignments = {}
    pattern = re.compile(r"^\[([^]]+)]\s+(\S+)\s+\|")
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match:
            assignments[match.group(2)] = match.group(1)
    return assignments


def _account_assignments(lane: Lane) -> dict[str, str]:
    result = _run_capture(
        colab_argv(lane, "sessions"),
        check=False,
        echo_output=False,
    )
    if result.returncode != 0:
        raise AADPColabError(
            f"cannot verify account-wide Colab assignments (sessions rc={result.returncode})"
        )
    return _parse_assignments(result.stdout)


def _known_lane_endpoints() -> set[str]:
    endpoints = set()
    for configured_lane in load_lanes().values():
        state = _read_local_session(configured_lane)
        if state and state.get("endpoint"):
            endpoints.add(state["endpoint"])
    return endpoints


def _unknown_assignments(lane: Lane) -> set[str]:
    return set(_account_assignments(lane)) - _known_lane_endpoints()


def _parse_remote_result(output: str, expected_op: str) -> dict:
    matches = [line[len(RESULT_PREFIX) :] for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
    if not matches:
        raise AADPColabError(
            f"remote {expected_op} returned no {RESULT_PREFIX}<json> sentinel; inspect output above"
        )
    try:
        result = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        raise AADPColabError(f"remote {expected_op} returned invalid result JSON") from exc
    if result.get("op") != expected_op:
        raise AADPColabError(
            f"remote result op mismatch: expected {expected_op!r}, got {result.get('op')!r}"
        )
    if result.get("ok") is not True:
        raise AADPColabError(f"remote {expected_op} failed: {result.get('error', result)}")
    return result


def _source_with_env(source: str, remote_env: dict[str, str]) -> str:
    assignments = ["import os"]
    assignments.extend(
        f"os.environ[{json.dumps(key)}] = {json.dumps(value)}"
        for key, value in sorted(remote_env.items())
    )
    return "\n".join(assignments) + "\n" + source


def _exec_file(
    lane: Lane,
    local_path: Path,
    *,
    remote_env: dict[str, str],
    timeout: int,
    expected_op: str,
) -> dict:
    source = _source_with_env(local_path.read_text(encoding="utf-8"), remote_env)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", prefix="aadp_colab_", encoding="utf-8", delete=False
        ) as handle:
            handle.write(source)
            temp_path = Path(handle.name)
        result = _run_capture(
            colab_argv(
                lane,
                "exec",
                "-s",
                lane.session,
                "-f",
                str(temp_path),
                "--timeout",
                str(timeout),
            )
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return _parse_remote_result(result.stdout, expected_op)


def _exec_source(lane: Lane, source: str, *, timeout: int, expected_op: str) -> dict:
    result = _run_capture(
        colab_argv(
            lane,
            "exec",
            "-s",
            lane.session,
            "--timeout",
            str(timeout),
        ),
        input_text=source,
    )
    return _parse_remote_result(result.stdout, expected_op)


def _remote_env(lane: Lane) -> dict[str, str]:
    return {
        "AADP_CLI_MODE": "1",
        "AADP_EXCHANGE_DIR": lane.exchange,
        "AADP_BOOTSTRAP_PACKAGES_JSON": json.dumps(list(lane.packages)),
    }


def _probe_source() -> str:
    return r'''
import json
import shutil
import subprocess
import sys
import traceback

try:
    import torch
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    memory_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else 0
    nvidia_smi = ""
    if shutil.which("nvidia-smi"):
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True
        ).stdout.strip()
    result = {
        "ok": True, "op": "probe", "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda": cuda, "gpu": gpu,
        "gpu_memory_gb": memory_gb, "nvidia_smi": nvidia_smi,
        "disk_free_gb": round(shutil.disk_usage("/").free / 1e9, 1),
    }
except Exception as exc:
    traceback.print_exc()
    result = {"ok": False, "op": "probe", "error": repr(exc)}
print("AADP_RESULT=" + json.dumps(result, separators=(",", ":")), flush=True)
'''


def probe(lane: Lane, expected_gpu: str | None = None, allow_mismatch: bool = False) -> dict:
    result = _exec_source(lane, _probe_source(), timeout=120, expected_op="probe")
    expected_gpu = (expected_gpu or lane.gpu).upper()
    if expected_gpu not in VALID_GPUS:
        raise AADPColabError(
            f"invalid GPU {expected_gpu!r}; choose {', '.join(sorted(VALID_GPUS))}"
        )
    if not result.get("cuda"):
        raise AADPColabError("allocated runtime has no CUDA GPU")
    actual = str(result.get("gpu", "")).upper()
    markers = GPU_NAME_MARKERS[expected_gpu]
    if not any(marker in actual for marker in markers):
        message = f"requested {expected_gpu}, but remote probe reports {result.get('gpu')!r}"
        if not allow_mismatch:
            raise AADPColabError(message)
        print("WARNING:", message)
    return result


def mount(lane: Lane) -> None:
    print("Drive OAuth may open a URL and wait for Enter in this terminal.")
    _run_live(colab_argv(lane, "drivemount", "-s", lane.session, "/content/drive"))


def _guard_workspace_mutation(lane: Lane) -> None:
    heartbeat = _read_heartbeat(lane)
    if heartbeat is None:
        raise AADPColabError(
            "cannot prove the existing workspace is idle: heartbeat missing; "
            "use `bootstrap <lane> --force` only after checking the VM"
        )
    age = time.time() - float(heartbeat.get("ts", 0))
    if age > 300:
        raise AADPColabError(
            f"cannot prove the existing workspace is idle: heartbeat stale ({age:.0f}s); "
            "use `bootstrap <lane> --force` only after checking the VM"
        )
    if heartbeat.get("control_mode") != "cli" or heartbeat.get("exchange") != lane.exchange:
        raise AADPColabError("cannot prove workspace ownership from this heartbeat")
    run = heartbeat.get("run") or {}
    if run.get("alive"):
        raise AADPColabError(
            f"refusing to replace the workspace while training pid {run.get('pid')} is alive"
        )
    if heartbeat.get("release_safe") is not True:
        raise AADPColabError("refusing to replace the workspace before the current run is collected")


def bootstrap(lane: Lane, *, force: bool = False) -> dict:
    if not force:
        _guard_workspace_mutation(lane)
    return _exec_file(
        lane,
        BOOTSTRAP,
        remote_env=_remote_env(lane),
        timeout=900,
        expected_op="bootstrap",
    )


def start_daemon(lane: Lane) -> dict:
    result = _exec_file(
        lane,
        START_DAEMON,
        remote_env=_remote_env(lane),
        timeout=120,
        expected_op="start_daemon",
    )
    daemon_pid = result.get("pid")
    if not isinstance(daemon_pid, int):
        raise AADPColabError(f"remote daemon returned invalid pid: {daemon_pid!r}")
    heartbeat = _wait_for_heartbeat(lane, timeout=120, daemon_pid=daemon_pid)
    if heartbeat.get("control_mode") != "cli":
        raise AADPColabError(f"daemon heartbeat is not in CLI mode: {heartbeat.get('control_mode')!r}")
    return result


def _cloud_env(lane: Lane) -> dict[str, str]:
    env = os.environ.copy()
    env["AADP_EXCHANGE_DIR"] = lane.exchange
    return env


def _remote_root(lane: Lane) -> str:
    remote = os.environ.get("AADP_RCLONE_REMOTE", "gdrive")
    return f"{remote}:{lane.exchange}"


def _read_heartbeat(lane: Lane) -> dict | None:
    result = _run_capture(
        ["rclone", "cat", f"{_remote_root(lane)}/cmd/heartbeat.json"],
        check=False,
        echo_output=False,
        timeout=90,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AADPColabError("Drive heartbeat is not valid JSON") from exc


def _wait_for_heartbeat(lane: Lane, timeout: int, daemon_pid: int | None = None) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = _read_heartbeat(lane)
        if (
            last is not None
            and time.time() - float(last.get("ts", 0)) < 180
            and last.get("control_mode") == "cli"
            and last.get("exchange") == lane.exchange
            and (daemon_pid is None or last.get("daemon_pid") == daemon_pid)
        ):
            return last
        time.sleep(5)
    raise AADPColabError(
        f"no fresh daemon heartbeat after {timeout}s for {lane.exchange}; inspect agent_daemon.log"
    )


def _cloud_call(lane: Lane, *parts: str) -> None:
    _run_live([sys.executable, str(CLOUD_SYNC), *parts], env=_cloud_env(lane))


def push(lane: Lane, include_data: bool) -> None:
    args = ["push"]
    if include_data:
        args.append("--data")
    _cloud_call(lane, *args)


def launch(lane: Lane, script: str, run_args: list[str]) -> None:
    if run_args[:1] == ["--"]:
        run_args = run_args[1:]
    if not run_args:
        raise AADPColabError("no run args; put training arguments after `--`")
    heartbeat = _read_heartbeat(lane)
    if heartbeat is None:
        raise AADPColabError("no daemon heartbeat; run `aadp_colab.py daemon <lane>`")
    if heartbeat.get("control_mode") != "cli" or heartbeat.get("exchange") != lane.exchange:
        raise AADPColabError("heartbeat belongs to a different/legacy control plane; restart the CLI daemon")
    age = time.time() - float(heartbeat.get("ts", 0))
    if age > 300:
        raise AADPColabError(f"daemon heartbeat is stale ({age:.0f}s); do not queue a launch")
    run = heartbeat.get("run") or {}
    if run.get("alive"):
        raise AADPColabError(
            f"lane {lane.name} already has live pid {run.get('pid')} ({run.get('log')}); launch refused"
        )
    _cloud_call(lane, "launch", script, "--", *run_args)


def status(lane: Lane) -> None:
    _run_capture(colab_argv(lane, "status", "-s", lane.session), check=False)
    heartbeat = _read_heartbeat(lane)
    if heartbeat is None:
        print("Drive heartbeat: missing")
        return
    age = time.time() - float(heartbeat.get("ts", 0))
    print(f"Drive heartbeat age: {age:.0f}s")
    print(json.dumps(heartbeat, indent=2, ensure_ascii=False))
    if age > 300:
        print("WARNING: heartbeat is stale; CLI status IDLE does not describe the detached training job")


def _latest_run(lane: Lane) -> str:
    result = _run_capture(
        ["rclone", "lsf", "--dirs-only", f"{_remote_root(lane)}/runs"],
        echo_output=False,
    )
    names = sorted(line.rstrip("/") for line in result.stdout.splitlines() if line.strip())
    if not names:
        raise AADPColabError(f"no collected runs under {_remote_root(lane)}/runs")
    return names[-1]


def pull(lane: Lane, run_name: str | None) -> None:
    _cloud_call(lane, "pull", run_name or _latest_run(lane))


def pull_model(lane: Lane, model_name: str) -> None:
    _cloud_call(lane, "pull-model", model_name)


def list_runs(lane: Lane) -> None:
    _cloud_call(lane, "list")


def _verify_stopped(lane: Lane, endpoint: str, attempts: int = 5) -> bool:
    for attempt in range(attempts):
        sessions = _run_capture(colab_argv(lane, "sessions"), check=False)
        local_gone = _read_local_session(lane) is None
        backend_gone = sessions.returncode == 0 and endpoint not in sessions.stdout
        if local_gone and backend_gone:
            return True
        if attempt + 1 < attempts:
            time.sleep(2)
    return False


def down(lane: Lane, *, force: bool, pull_latest: bool) -> None:
    session = _refresh_local_session(lane)
    if session is None:
        unknown = _unknown_assignments(lane)
        if unknown:
            raise AADPColabError(
                "no local state for this lane, but untracked account assignment(s) exist: "
                + ", ".join(sorted(unknown))
                + "; inspect the Colab UI before creating another VM"
            )
        print(f"lane {lane.name} is already down (no local session state)")
        return
    endpoint = session.get("endpoint")
    if not endpoint:
        raise AADPColabError(f"session state for {lane.session} has no endpoint; refusing blind stop")

    heartbeat = _read_heartbeat(lane)
    run = (heartbeat or {}).get("run") or {}
    blockers = []
    if heartbeat is None:
        blockers.append("Drive heartbeat is missing")
    else:
        age = time.time() - float(heartbeat.get("ts", 0))
        if age > 300:
            blockers.append(f"Drive heartbeat is stale ({age:.0f}s)")
        if heartbeat.get("control_mode") != "cli":
            blockers.append(f"heartbeat control_mode={heartbeat.get('control_mode')!r}")
        if heartbeat.get("exchange") != lane.exchange:
            blockers.append(f"heartbeat exchange={heartbeat.get('exchange')!r}")
        if str(heartbeat.get("collect_note", "")).startswith("collect_error"):
            blockers.append(str(heartbeat["collect_note"]))
        if run.get("alive"):
            blockers.append(f"training pid {run.get('pid')} is still alive")
        if heartbeat.get("release_safe") is not True:
            blockers.append("current run has not been safely collected")
    if blockers and not force:
        raise AADPColabError(
            "; ".join(blockers) + f"; resolve them or rerun `down {lane.name} --force`"
        )
    if heartbeat and heartbeat.get("collect_note"):
        print("last collect:", heartbeat["collect_note"])
    if pull_latest:
        pull(lane, None)

    _run_capture(colab_argv(lane, "stop", "-s", lane.session))
    if not _verify_stopped(lane, endpoint):
        raise AADPColabError(
            f"stop was issued but endpoint {endpoint} is still present; check `aadp_colab.py status {lane.name}` and the Colab UI"
        )
    print(f"lane {lane.name} stopped and backend assignment removed")


def _check_cli_version() -> None:
    result = _run_capture(["colab", "version"], echo_output=False)
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", result.stdout)
    actual = match.group(1) if match else "unknown"
    if actual != CLI_VERSION:
        raise AADPColabError(
            f"google-colab-cli {CLI_VERSION} required, found {actual}; "
            f"run `uv tool install --force 'google-colab-cli=={CLI_VERSION}'`"
        )


def _check_adc(lane: Lane) -> None:
    """Verify identity/scopes through the same auth path used by Colab CLI."""
    result = _run_capture(
        colab_argv(lane, "whoami"),
        check=False,
        echo_output=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise AADPColabError("ADC is unavailable; follow colab-operator authentication setup")


def doctor(lane: Lane) -> None:
    if sys.version_info < (3, 12):
        raise AADPColabError("google-colab-cli 0.6.0 requires Python 3.12+")
    _require_program("colab", f"run `uv tool install 'google-colab-cli=={CLI_VERSION}'`")
    _require_program("rclone", "install rclone and configure the Drive remote (see colab/COLAB.md)")
    _check_cli_version()
    _check_adc(lane)

    remote = os.environ.get("AADP_RCLONE_REMOTE", "gdrive")
    remotes = _run_capture(["rclone", "listremotes"], echo_output=False)
    if f"{remote}:" not in remotes.stdout.splitlines():
        raise AADPColabError(f"rclone remote {remote!r} is not configured")

    lane.state_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"colab CLI: {CLI_VERSION} | auth: adc | lane: {lane.name} | state: {lane.state_file}")
    _run_capture(colab_argv(lane, "sessions"), check=False)
    print("doctor passed; the first `up` is the definitive Colab allocation/auth canary")


def up(
    lane: Lane,
    *,
    gpu: str | None,
    reuse: bool,
    skip_mount: bool,
    skip_bootstrap: bool,
    skip_daemon: bool,
    allow_gpu_mismatch: bool,
    allow_account_orphans: bool,
) -> None:
    if gpu is not None and gpu.upper() not in VALID_GPUS:
        raise AADPColabError(
            f"invalid GPU {gpu.upper()!r}; choose {', '.join(sorted(VALID_GPUS))}"
        )
    _require_program("colab", f"run `uv tool install 'google-colab-cli=={CLI_VERSION}'`")
    _check_cli_version()
    lane.state_file.parent.mkdir(parents=True, exist_ok=True)

    existing = _refresh_local_session(lane)
    if existing is not None and not reuse:
        raise AADPColabError(
            f"session {lane.session!r} is already active; use `up {lane.name} --reuse` or stop it first"
        )
    if existing is None and reuse:
        raise AADPColabError(f"cannot reuse {lane.session!r}: no active local session")

    expected_gpu = (
        gpu.upper()
        if gpu is not None
        else str((existing or {}).get("accelerator") or lane.gpu).upper()
    )
    if expected_gpu not in VALID_GPUS:
        raise AADPColabError(
            f"session reports unsupported accelerator {expected_gpu!r}; choose an explicit --gpu"
        )

    account_before = _account_assignments(lane)
    unknown_before = set(account_before) - _known_lane_endpoints()
    if existing is None and unknown_before and not allow_account_orphans:
        raise AADPColabError(
            "untracked account assignment(s) already exist: "
            + ", ".join(sorted(unknown_before))
            + "; inspect the Colab UI, or use --allow-account-orphans if they belong to another project"
        )

    provisioning_started = False
    try:
        if existing is None:
            provisioning_started = True
            _run_capture(colab_argv(lane, "new", "-s", lane.session, "--gpu", expected_gpu))
            session = _read_local_session(lane)
            if session is None or not session.get("endpoint"):
                raise AADPColabError("Colab reported success but wrote no usable session state")
        probe(lane, expected_gpu=expected_gpu, allow_mismatch=allow_gpu_mismatch)
        if not skip_mount:
            mount(lane)
        if not skip_bootstrap:
            bootstrap(lane, force=existing is None)
        if not skip_daemon:
            start_daemon(lane)
    except BaseException:
        if provisioning_started:
            print("up failed; stopping the newly created session to avoid CU leakage", file=sys.stderr)
            try:
                created_state = _read_local_session(lane)
            except Exception as cleanup_exc:
                created_state = None
                print(f"CRITICAL: cannot read state during cleanup: {cleanup_exc}", file=sys.stderr)
            if created_state and created_state.get("endpoint"):
                endpoint = created_state["endpoint"]
                try:
                    _run_capture(colab_argv(lane, "stop", "-s", lane.session), check=False)
                    if not _verify_stopped(lane, endpoint, attempts=3):
                        print(
                            f"CRITICAL: cleanup could not verify removal of endpoint {endpoint}",
                            file=sys.stderr,
                        )
                except Exception as cleanup_exc:
                    print(f"CRITICAL: cleanup failed: {cleanup_exc}", file=sys.stderr)
            else:
                try:
                    leaked = set(_account_assignments(lane)) - set(account_before)
                except Exception as cleanup_exc:
                    print(f"CRITICAL: cannot audit failed allocation: {cleanup_exc}", file=sys.stderr)
                else:
                    if leaked:
                        print(
                            "CRITICAL: new account assignment lacks local state; release in Colab UI: "
                            + ", ".join(sorted(leaked)),
                            file=sys.stderr,
                        )
        raise
    print(
        f"lane {lane.name} ready: session={lane.session} "
        f"exchange={lane.exchange} gpu={expected_gpu}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="check pinned CLI, ADC, rclone, and lane state")
    p_doctor.add_argument("lane", nargs="?", default="a")

    p_up = sub.add_parser("up", help="create/reuse, probe, mount, bootstrap, and start a lane")
    p_up.add_argument("lane")
    p_up.add_argument("--gpu", help="override the lane default")
    p_up.add_argument("--reuse", action="store_true", help="reuse an existing named session")
    p_up.add_argument("--skip-mount", action="store_true")
    p_up.add_argument("--skip-bootstrap", action="store_true")
    p_up.add_argument("--skip-daemon", action="store_true")
    p_up.add_argument("--allow-gpu-mismatch", action="store_true")
    p_up.add_argument(
        "--allow-account-orphans",
        action="store_true",
        help="provision even when account assignments not tracked by this repo exist",
    )

    for name, help_text in (
        ("probe", "inspect the actual remote GPU/runtime"),
        ("mount", "mount Drive (one interactive OAuth approval per VM)"),
        ("bootstrap", "extract the latest bundle/data and install guarded dependencies"),
        ("daemon", "start/reuse the detached Drive command daemon"),
        ("status", "show CLI session plus authoritative Drive job heartbeat"),
        ("list", "list collected Drive runs"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("lane")
    p_probe = sub.choices["probe"]
    p_probe.add_argument("--expected-gpu")
    p_probe.add_argument("--allow-mismatch", action="store_true")
    p_bootstrap = sub.choices["bootstrap"]
    p_bootstrap.add_argument(
        "--force",
        action="store_true",
        help="replace an existing workspace without a fresh safe-release heartbeat",
    )

    p_push = sub.add_parser("push", help="delegate code/data bundle upload to cloud_sync.py")
    p_push.add_argument("lane")
    p_push.add_argument("--data", action="store_true")

    p_launch = sub.add_parser("launch", help="background-launch through the existing Drive daemon")
    p_launch.add_argument("lane")
    p_launch.add_argument("script")
    p_launch.add_argument("run_args", nargs=argparse.REMAINDER)

    p_pull = sub.add_parser("pull", help="pull a named run, or the latest when omitted")
    p_pull.add_argument("lane")
    p_pull.add_argument("run_name", nargs="?")

    p_model = sub.add_parser("pull-model", help="pull a saved Drive model directory")
    p_model.add_argument("lane")
    p_model.add_argument("model_name")

    p_down = sub.add_parser("down", help="guard live jobs, stop the VM, and verify unassignment")
    p_down.add_argument("lane")
    p_down.add_argument("--force", action="store_true", help="stop even when heartbeat says job alive")
    p_down.add_argument("--pull-latest", action="store_true", help="pull latest collected run first")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lane = get_lane(args.lane)
        if args.command == "doctor":
            doctor(lane)
        elif args.command == "up":
            up(
                lane,
                gpu=args.gpu,
                reuse=args.reuse,
                skip_mount=args.skip_mount,
                skip_bootstrap=args.skip_bootstrap,
                skip_daemon=args.skip_daemon,
                allow_gpu_mismatch=args.allow_gpu_mismatch,
                allow_account_orphans=args.allow_account_orphans,
            )
        elif args.command == "probe":
            probe(lane, args.expected_gpu, args.allow_mismatch)
        elif args.command == "mount":
            mount(lane)
        elif args.command == "bootstrap":
            bootstrap(lane, force=args.force)
        elif args.command == "daemon":
            start_daemon(lane)
        elif args.command == "push":
            push(lane, args.data)
        elif args.command == "launch":
            launch(lane, args.script, args.run_args)
        elif args.command == "status":
            status(lane)
        elif args.command == "list":
            list_runs(lane)
        elif args.command == "pull":
            pull(lane, args.run_name)
        elif args.command == "pull-model":
            pull_model(lane, args.model_name)
        elif args.command == "down":
            down(lane, force=args.force, pull_latest=args.pull_latest)
        else:  # pragma: no cover - argparse enforces this
            raise AssertionError(args.command)
    except AADPColabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
