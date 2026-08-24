"""VM-agent wrapper for long teacher-logit exports.

vm_agent.py launch always runs scripts with the system Python. The M8 export
needs /content/venv311 (torch 2.7.1 + transformers 5.13 qwen3_5 module), so
this wrapper lets the daemon track the long-running job while delegating the
actual export to the venv interpreter.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PY = Path("/content/venv311/bin/python")


def main():
    if not VENV_PY.exists():
        raise FileNotFoundError(f"missing venv python: {VENV_PY}")
    cmd = [str(VENV_PY), "-u", "export_teacher_logits.py", *sys.argv[1:]]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
