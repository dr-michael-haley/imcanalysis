# Local execution and the backend boundary

The local backend is a sequential Linux process runner. The scientific scripts
continue to accept the same config and reporting environment as on SLURM.
[The EC2 guide](../getting_started/ec2.md) covers normal use.

## Shared contract

- `RunPlan` resolves stage aliases, built-in modes, dependencies, assets, and
  environment overrides before a backend starts work. Its `execution_backend`
  is persisted with the workflow. Backend choice is included in preview tokens.
- `commands.stage_commands` describes scientific Python modules and their
  environment keys. It explicitly handles modules that switch environments.
  It contains no scheduler or process-launch logic.
- `runtime.sbt_environment` exports the immutable config snapshot, execution
  identity, output paths, and reporting context. SLURM and local use this same
  function. The old import from `pipeline.slurm` remains available.
- `create_run_record` owns config snapshots, workflow IDs and numbered execution
  allocation. Backends update the existing execution index and reporting layer.
- Status, summaries, refresh, logs, and cancellation dispatch using the recorded
  backend. Setting `SBT_BACKEND` affects new plans, not historical runs.

SLURM continues to submit the maintained wrappers. Local command definitions are
derived from the registry's modules and environment keys, with explicit mapping
for Cellpose, denoising QC, and CellVision Full. Shell-only utilities `debug`,
`zipqc`, and `scport`, plus the SLURM-log utility `slogs`, are rejected in local
plans. Scientific stage modules otherwise use the existing aliases. Moving the
external scPortrait converter into a registered module can extend that support
without changing the runner.

A future Nextflow backend can translate these scientific command descriptions
and dependencies into processes, supply the shared reporting environment, and
map workflow/task handles into status and logs. It will need its own adapter and
workflow definitions; this change does not implement Nextflow or promise that
its scheduling semantics match local execution. Project IDs, config snapshots,
stage entry points, and report locations can remain the same.

## Worker lifecycle

`local_run.yaml` contains typed local handles, ordered commands, environment
selection, per-stage logs, timestamps, and exit codes. Local PIDs never appear
in `slurm_job_id`. No complete shell environment or credentials are persisted.

The controller takes a project `flock` before allocation and passes its open
descriptor to the worker. The worker keeps it until termination. This closes
the race between two launch commands and avoids stale lock-file deletion.
The worker starts in its own session with stdin disconnected and output saved
to `worker.log`. Foreground mode follows this same worker; detach changes only
whether the controller waits.

The worker acknowledges startup after loading its config, holding the lock, and
installing signal handlers. It launches each command in a separate process group,
records its PID plus boot/host/start-time identity, and writes a combined log.
Cancellation uses a durable request plus a verified worker signal where Linux
pidfds are available. The worker owns termination of the active group, including
ordinary multiprocessing children. It retains the child leader until group
cleanup finishes, preventing PID reuse during cleanup.

There is no daemon, queue, database, automatic retry, failed-stage rerun, custom
workflow storage, or reboot recovery. A dead worker yields `unknown` for unfinished
stages. The overlap check also looks for recorded orphan process groups on the
same host/boot. Processes that deliberately create a separate session, external
services, and abrupt machine loss are outside this small runner's supervision.

## Verification

`tests/test_local_backend.py` covers planning, environment selection, previews,
status dispatch, logs, reports, failure propagation, and cancellation scope. It
also includes Linux integration tests that launch detached workers and verify
process-group cancellation and project locking using tiny Python subprocesses.
These integration tests skip on Windows; they do not run scientific datasets,
require a GPU, or contact SLURM.
