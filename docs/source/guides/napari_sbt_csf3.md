# Running NapariSBT on a CSF3 interactive node

NapariSBT must run on a compute node, not a CSF3 login node. The viewer uses an
X11 display forwarded to your computer. Expensive full-cohort feature extraction
remains a normal `cellfeat` batch job so it can continue after the graphical
session ends.

## Install the dedicated environment

From the toolkit checkout, create or refresh the fixed Linux environment:

```bash
cd "$HOME/imcanalysis"
bash install/bootstrap_napari_sbt_csf3.sh
```

This uses `HPC_env_files/sbt-napari/environment.yml`, installs the current
checkout as an editable no-dependency overlay, and runs import smoke checks. It
does not modify `sbt-cli`, `sbt-gui`, or the scientific stage environments.

The environment is centrally registered as `napari` with the fixed Conda name
`sbt-napari`. Its Linux intent specification is committed, but installation is
bootstrap-managed until a reviewed `linux-64` lock has been generated and
validated on CSF3.

## Connect with X11

On Windows, use MobaXterm or another SSH client with an X server. On macOS, run
XQuartz before connecting. Linux desktops normally already provide an X server.
Connect to CSF3 with X11 forwarding enabled and verify it on the login node:

```bash
echo "$DISPLAY"
```

For a direct SSH connection, the command should print a network-forwarded value
such as `localhost:10.0`. A value such as `:1`, `:2`, or `unix:1` is a
local-only desktop display. It is not sufficient for `srun-x11` unless that VNC
desktop was explicitly started with TCP X access. Do not launch Napari on the
login node.

On Windows, the simplest supported route is a direct MobaXterm SSH session to
`csf3.itservices.manchester.ac.uk` with **X11-forwarding** enabled under the
session's advanced SSH settings. If working off campus, connect to the
University VPN first.

## Request the interactive allocation

For ordinary exploration and classification, start with four CPUs:

```bash
srun-x11 -p interactive -t 60 -c 4
```

The CSF3 `interactive` partition has a maximum 60-minute wallclock. The current
CSF3 training material describes 8 GB of memory per requested core, so four
cores ordinarily provide about 32 GB. If the local `srun-x11` wrapper rejects
`-c`, use the CPU option documented by `srun-x11 --help`; NapariSBT's preflight
reports the CPU affinity actually received.

Inside the allocated shell, launch from either environment:

```bash
conda activate sbt-cli
cd "$HOME/scratch/HyperionProject"
sbt gui napari --check
sbt gui napari
```

The launcher discovers the initialized project from the current directory or
its parents. From elsewhere, `--project` accepts a registered project name,
project ID, or explicit path.

When `sbt-cli` does not contain Napari, the launcher automatically re-executes
the application in the registered `sbt-napari` environment. Alternatively:

```bash
conda activate sbt-napari
cd "$HOME/scratch/HyperionProject"
sbt gui napari
```

## Preflight checks

The `--check` command is side-effect free. It does not import Qt, open AnnData,
load images, or create experiment files. It checks:

- the Napari, Qt, AnnData, Parquet, image, and classifier modules;
- the X11 `DISPLAY` value;
- the Slurm job and compute-node context;
- project, AnnData, mask, image, and experiment paths;
- experiment-output write access;
- requested feature workers against the process CPU allocation.

Machine-readable output is available for support diagnostics:

```bash
sbt gui napari --check --check-format json --project /path/to/project
```

## Resource profiles

| Work | Suggested interactive allocation | Execution path |
|---|---:|---|
| Explore images and label cells | 2–4 CPUs | Napari interactive node |
| Train and score existing features | 2–4 CPUs | Napari interactive node |
| Feature-discovery trial on 1–3 ROIs | 8 CPUs | Napari if it fits within 60 minutes |
| Full multi-ROI feature extraction | 8 CPUs, 64 GB, 24 hours | `sbt run cellfeat` |

The Feature Building tab clamps its subprocess worker count to the CPUs visible
to the current process. The worker also enforces the limit, so a stale saved
worker setting cannot oversubscribe the allocation. The progress log records
when a request was reduced.

For a full build, save the experiment and set `napari_sbt.active_experiment` in
`config.yaml`. Then leave the GUI session and submit:

```bash
conda activate sbt-cli
cd /path/to/project
sbt run cellfeat --dry-run
sbt run cellfeat
```

After completion, start another interactive Napari session and reload the
experiment. Valid per-ROI fragments and the canonical feature assets are reused.

## Session lifetime and scientific state

There is no general Napari workspace persistence. Confirmed/proposed labels,
experiment revisions, feature fragments, model files, scores, population review
recipes, and viewed-ROI sets are explicit experiment assets. Save deliberate
interface changes before the allocation expires and quit Napari normally to
release the interactive resource.

The original images, masks, and source AnnData remain unchanged.

## Troubleshooting

### `DISPLAY is unset`

Reconnect with X11 enabled and request a new `srun-x11` session. Do not work
around this by launching on the login node.

### `Cannot use current desktop display for remote access`

The current terminal is normally inside a VNC desktop whose display looks like
`:1` and accepts only local Unix-socket connections. The compute node cannot
send Napari's windows back to it.

Preferred recovery on Windows:

1. Leave the current CSF terminal; the VNC desktop itself need not be deleted.
2. Open a direct MobaXterm SSH session to
   `csf3.itservices.manchester.ac.uk` with X11 forwarding enabled.
3. Confirm that `echo "$DISPLAY"` resembles `localhost:10.0` rather than `:1`.
4. Run `srun-x11 -p interactive -t 60 -c 4` again.

If the VNC route is required, restart it using the University-supported launch
method with TCP X access enabled, as suggested by `srun-x11`. Do not expose a
VNC/X server manually without following Research IT guidance.

### The environment cannot be found

Return to the toolkit checkout and rerun:

```bash
bash install/bootstrap_napari_sbt_csf3.sh
```

### OpenGL or blank-canvas errors

First try indirect OpenGL in a new allocation:

```bash
export LIBGL_ALWAYS_INDIRECT=1
sbt gui napari --project /path/to/project
```

If the viewer remains slow or unstable, the University Research Virtual
Desktop Service may provide better compression and OpenGL compatibility, but
its VNC display must be started with TCP X access before `srun-x11` can reuse
it. Otherwise use direct MobaXterm X11 forwarding.

### The allocation is too short for feature building

Cancel from the Feature Building tab and wait for the running ROI workers to
finish cleanly. Completed fingerprinted fragments remain valid. Submit the
full build with `sbt run cellfeat` rather than requesting more work inside the
short interactive session.
