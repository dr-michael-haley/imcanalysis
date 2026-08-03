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

The command must print a display value. Do not launch Napari on the login node.

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
sbt gui napari --check --project "$HOME/scratch/HyperionProject"
sbt gui napari --project "$HOME/scratch/HyperionProject"
```

When `sbt-cli` does not contain Napari, the launcher automatically re-executes
the application in the registered `sbt-napari` environment. Alternatively:

```bash
conda activate sbt-napari
sbt gui napari --project "$HOME/scratch/HyperionProject"
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

If the viewer remains slow or unstable, use the University Research Virtual
Desktop Service so the remote graphical session has better compression and
OpenGL compatibility.

### The allocation is too short for feature building

Cancel from the Feature Building tab and wait for the running ROI workers to
finish cleanly. Completed fingerprinted fragments remain valid. Submit the
full build with `sbt run cellfeat` rather than requesting more work inside the
short interactive session.
