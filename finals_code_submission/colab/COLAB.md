# AADP Colab lane

This file is the repository-specific overlay for Colab operation. Generic Colab
CLI behavior, authentication, session inspection, recovery, and compute-unit
safety live in the installed `colab-operator` skill:

- installed: `$CODEX_HOME/skills/colab-operator/SKILL.md`
- upstream: <https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md>

When the skill and this file overlap, use the skill for generic `colab` behavior
and this file for AADP paths, wrappers, persistence, and promotion workflow. The
skill tracks CLI `main`; this repository currently pins
`google-colab-cli==0.6.0`, and `aadp_colab.py` always supplies the required
global auth/config flags explicitly.

## AADP control split

```text
aadp_colab.py
├── google-colab-cli       VM lifecycle, keep-alive, kernel calls, stop
├── cloud_sync.py          dirty code/data/model/result transport via Drive
└── vm_agent.py            detached training, heartbeat, auto-collect
```

Keep these boundaries:

- Colab CLI controls the VM, not large artifact transport.
- Drive/rclone remains the data and artifact plane.
- Training always runs as a detached `vm_agent.py` child, never as a long
  synchronous `colab exec` or `colab run` job.
- The existing `colab_runner*.ipynb` files are emergency fallback only.
- Evaluation-server `requirements.txt` must never contain Colab CLI packages.

## Human interaction boundary

`aadp_colab.py up` includes `colab drivemount`, which requires a real terminal
and human approval on a new VM. Following the operator skill, an agent must not
invoke that interactive step unattended.

Normal ownership is:

1. Agent: push code/data and prepare the run.
2. Human terminal: run `up` and complete Drive approval.
3. Agent: launch, monitor, collect, and stop the ready lane.

If Drive is already mounted in an existing session, the agent may use recovery
commands with `--skip-mount` instead of invoking the interactive mount again.

Run repository wrappers with `.venv/bin/python` locally. This workspace does
not assume a separate `python` executable is on `PATH`; VM-side commands still
use the bootstrap environment's `python`.

## Fast path

One-time for this repository:

```bash
uv tool install --force 'google-colab-cli==0.6.0'
.venv/bin/python colab/aadp_colab.py doctor a
```

Prepare the lane:

```bash
# Agent; add --data only on the first use of this exchange folder.
.venv/bin/python colab/aadp_colab.py push a --data

# Human terminal; complete Drive approval when prompted.
.venv/bin/python colab/aadp_colab.py up a --gpu A100
```

Operate the ready lane:

```bash
.venv/bin/python colab/aadp_colab.py launch a train_transformer.py -- \
  --device cuda \
  --quick-val-size 600 \
  --epochs 1 \
  --experiment-suffix cli_canary

.venv/bin/python colab/aadp_colab.py status a
.venv/bin/python colab/aadp_colab.py down a
.venv/bin/python colab/aadp_colab.py list a
.venv/bin/python colab/aadp_colab.py pull a <run-name>
```

On a long-lived exchange, always use `list` and pass the collected run name
explicitly. A bare `pull a` can select a lexically last legacy/manual directory
(observed with `maxpack_mirror`) instead of the newest run by wall time.

Once heartbeat reports `run.alive=false`, successful auto-collect, and
`release_safe=true`, prefer `down` before `pull` or `pull-model`. Results and
models are already persisted on Drive, so releasing first avoids paying for an
idle VM during a large local download. `down a --pull-latest` remains available
for a clean exchange, while `down a --force` is only for intentionally
terminating a live or unverifiable job.

## Lane map

Tracked, non-secret configuration lives in `colab/lanes.json`:

| Lane | CLI session | Drive exchange | Default GPU | Bootstrap packages |
| --- | --- | --- | --- | --- |
| `a` | `aadp-a` | `AADP_exchange` | A100 | standard |
| `b` | `aadp-b` | `AADP_exchange_b` | L4 | standard |
| `c` | `aadp-c` | `AADP_exchange_c` | G4 | safetensors 0.8 + bitsandbytes |
| `d` | `aadp-d` | `AADP_exchange_d` | T4 | standard |

CLI token/endpoint state is separate under gitignored `.colab/`. Never commit,
copy, or include that directory in a code bundle. One exchange folder maps to
exactly one runtime; paired comparisons stay on the same lane/GPU class.

## Repository wrapper commands

```bash
.venv/bin/python colab/aadp_colab.py doctor [lane]
.venv/bin/python colab/aadp_colab.py push <lane> [--data]
.venv/bin/python colab/aadp_colab.py up <lane> [--gpu GPU] [--reuse]
.venv/bin/python colab/aadp_colab.py probe <lane>
.venv/bin/python colab/aadp_colab.py mount <lane>       # human terminal only
.venv/bin/python colab/aadp_colab.py bootstrap <lane> [--force]
.venv/bin/python colab/aadp_colab.py daemon <lane>
.venv/bin/python colab/aadp_colab.py launch <lane> SCRIPT -- ARGS...
.venv/bin/python colab/aadp_colab.py status <lane>
.venv/bin/python colab/aadp_colab.py list <lane>
.venv/bin/python colab/aadp_colab.py pull <lane> [RUN_NAME]
.venv/bin/python colab/aadp_colab.py pull-model <lane> MODEL_NAME
.venv/bin/python colab/aadp_colab.py down <lane> [--pull-latest] [--force]
```

Recovery of an existing session should run only the required stage. `up
--reuse` never allocates another session with the same name. Bootstrap refuses
to replace a live or uncollected workspace; use `bootstrap --force` only after
manually confirming the VM is idle.

### Emergency `up` cleanup bypass

If `up` is wedged only after the lane is demonstrably ready, SIGKILLing the
local `up` wrapper can preserve the already detached CLI keep-alive, remote
kernel, Drive mount, and VM daemon. This is a recovery-only path, not the
normal way to finish `up`:

1. Record the tracked endpoint and prove it still appears account-wide.
2. Prove the keep-alive is a separate local PID/process group from the `up`
   wrapper; never signal the keep-alive.
3. Require a fresh Drive heartbeat with the expected `control_mode=cli`, exact
   exchange name, daemon PID, and no ownership mismatch. Confirm the Drive
   mount and AADP workspace through the existing kernel when possible.
4. SIGKILL only the local `up` wrapper. If it leaves a local `colab exec`
   child holding the lane lock, remove that child only after repeating the
   remote heartbeat/mount checks.
5. Immediately re-run account-wide `sessions`, lane `status`, the GPU probe,
   and heartbeat checks. The assignment remains billable and must retain a
   live keep-alive until a later verified `down`.

Never use this when assignment, endpoint ownership, keep-alive separation,
mount, or daemon readiness is uncertain. A transient CLI 404/401 can prune
only the local state while the assignment and daemon remain alive; in that
case do not run `new` and do not hand-edit the token JSON. Reattach only after
proving ownership from local history plus the lane-specific Drive heartbeat,
using the CLI's locked `StateStore` API and a freshly listed runtime-proxy
credential. `google-colab-cli==0.6.0` has no public orphan-adopt command.

## AADP persistence contract

`cloud_sync.py push` deliberately sends the current tracked and non-ignored
untracked working tree, including dirty provenance and a content fingerprint.
The Drive layout remains:

```text
<exchange>/
├── code/      code_<utc>_<sha8>[_dirty].tar.gz
├── data/      open_data.tar.gz
├── cmd/       queue/ done/ heartbeat.json
├── models/    saved fp16 checkpoints
└── runs/      <utc>_<experiment_id>/
               ├── results_rows.csv
               ├── logits/ artifacts/ extra/
               └── manifest.json
```

Screens must write submittable weights directly to Drive:

```text
--save-val-model --save-fp16
--output-dir /content/drive/MyDrive/<exchange>/models/<experiment-suffix>
```

After verified auto-collect, release the lane and fetch them with `pull-model`,
then use the normal `.venv/bin/python package_submission.py --no-sparse` path.
Cloud experiment rows enter the local ledger only through `cloud_sync.py pull`
or the wrapper; never hand-copy them.

Multi-run plans continue through `chain_runs.py`; its default heartbeat poll is
60 seconds and is configurable with `--poll-seconds`.

## Project-specific safety invariants

- In a CLI lane, release only with `aadp_colab.py down <lane>`. The legacy
  `cloud_sync.py unassign` route cannot release a detached CLI session and is
  rejected in both local and VM-side code.
- `heartbeat.run.alive`, not CLI `status` IDLE/BUSY, is authoritative for the
  detached trainer.
- The VM-side launch guard is the atomic boundary preventing two trainers from
  overwriting `last_run.json`.
- Missing/stale heartbeat, a live PID, an uncollected run, or collect failure
  blocks normal `down`; only explicit `--force` bypasses those checks.
- `down` verifies both local state removal and account-wide endpoint removal.
- `up` audits untracked account assignments before provisioning and cleans up a
  partially created session on failure or interruption.
- A CLI-mode daemon marks `idle_expired_cli_stop_required` after the project
  idle limit. It stays observable; run `down` promptly.
- `/content` is ephemeral. Models, checkpoints, and irreplaceable artifacts must
  be written to Drive before release.

## End-to-end live validation

Last verified on 2026-07-11:

- `doctor` passed through CLI `whoami` plus account-wide `sessions` for lanes A
  and B. Human-approved `up` mounted Drive, bootstrapped the latest clean bundle
  at commit `44d0b5c`, and started fresh CLI-mode daemons on two A100 sessions.
- A real `--quick-val-size 600 --epochs 1` canary completed on lane A. Detached
  launch, authoritative heartbeat PID/GPU state, auto-collect, explicit result
  pull, and manifest commit verification all passed.
- Lanes A and B then ran parallel three-epoch full refits for about 61 minutes
  each. All epoch checkpoints reached Drive, both final fp16 models were saved,
  auto-collect produced one result row per run, and the manifests recorded
  commit `44d0b5c`.
- Both lanes reached `run.alive=false`, successful `collect_note`, and
  `release_safe=true`. They were released with normal `down`; account-wide
  verification ended with no active server assignment or keep-alive state.
  Results and 1.1 GB models were successfully pulled from Drive after release.
- The run confirmed that CLI `status` may show the kernel as `IDLE` while the
  detached trainer saturates the GPU; `heartbeat.run.alive` remains the source
  of truth. Occasional empty rclone heartbeat reads were transient and
  succeeded on immediate retry, so retry once before diagnosing a stale daemon.

The complete AADP CLI path is therefore production-validated: `doctor`, `push`,
human `up`/Drive approval, bootstrap, daemon, launch, status/heartbeat,
checkpoint persistence, auto-collect, explicit pull, pull-model, and verified
down. The lane B default can also be overridden with `--gpu A100` for a matched
parallel comparison.

Do not repeat the full lifecycle canary before every experiment. Repeat this
short validation after a CLI/auth, wrapper, bootstrap, daemon, or Drive-contract
change:

1. `doctor a`.
2. Push the current bundle; human runs `up a --gpu T4` and approves Drive.
3. Check the fresh CLI heartbeat and run one real
   `--quick-val-size 600 --epochs 1` screen.
4. Confirm the trainer PID/GPU, auto-collect, and `release_safe=true`.
5. Run normal `down`, confirm no server assignment, then pull the explicit run
   name from Drive.

Local unit tests mock Colab/rclone and do not allocate a VM.

## Notebook fallback

If the CLI path is blocked, use the existing lane notebook unchanged:

1. Attach `colab_runner.ipynb` (or `_b`, `_c`, `_d`) to a Colab GPU runtime.
2. Run `[probe] -> [mount] -> [bootstrap] -> [agent]`.
3. Keep `[agent]` executing synchronously as the legacy keep-alive.
4. Use `cloud_sync.py` for launch, heartbeat, and pull.
5. Only in this synchronous-notebook mode, `cloud_sync.py unassign` exits with
   86 so the kernel cell can call `runtime.unassign()`.

Weak4 specialist and isolated teacher-export dependency procedures remain
unchanged. Continue with the Public-gated promotion rules in `AGENTS.md` after
collection.
