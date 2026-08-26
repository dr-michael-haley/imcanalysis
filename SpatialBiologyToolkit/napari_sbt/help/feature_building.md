# Feature Building

Feature Building creates the numerical table used by the classifier. In a normal
experiment it covers the complete frozen cohort; in a Feature Discovery Trial it
covers only the selected trial ROIs. Full segmentation masks remain the authority
for boundaries, neighbours, and background exclusion.

## Feature readiness

The readiness card answers whether a durable, canonical feature table is available
for the current experiment revision. Unlike the live progress panel, it is rebuilt
from the saved feature table, dictionary, provenance manifest, active feature-set
ID, frozen cohort, and experiment revision. It therefore remains informative after
NapariSBT is closed and reopened.

**Ready for classification** means the required assets exist, their provenance
matches the current experiment and frozen cohort, and the table covers the intended
cells and ROIs. A Feature Discovery Trial instead reports **Ready for refinement**.
Amber states distinguish partial coverage, recorded build warnings, and controls
that have been changed since the active table was built. Existing built features
remain usable when controls have unapplied changes; rebuild only if those displayed
changes should replace the active table. Red states mean the assets are incomplete
or stale and should not be used for training.

The coverage bar reports cells in the canonical table, not merely completed
fragments. During a build it temporarily shows processed ROIs and does not declare
the table ready until final outputs and provenance have been written. Use **Refresh
saved feature status** after an externally managed or HPC build completes.

## Imported sources

CSV, Parquet, AnnData `X`/`obs`/`obsm`, and CellVision embeddings are joined by
`(ROI, ObjectNumber)`. Validate sources before building to see identity coverage,
missing cells, and usable numeric columns. Imported rows outside the current
feature-build scope are ignored.

Enter one source per line using the examples shown in the fields. Click **Validate
identities, cohort coverage, and numeric features** before building. A source with
partial coverage can still be useful, but the validation table should make that
decision explicit. Cells with no usable values across the selected sources remain
visible but cannot be scored.

## IMC channels

Click **Refresh available channels** after loading or changing an experiment.
Select the channels needed for the biological question; Ctrl/Command-click selects
several. **Select all** deliberately requests the widest table, whereas **Clear
selection** delegates to every consistently discovered channel. The read-only
summary below the list records the effective selection sent to the worker.

Start with a broad panel in a Feature Discovery Trial when feature refinement is
planned. For a full experiment, limiting channels can substantially reduce I/O and
feature-extraction time. A channel must be discoverable for the relevant ROIs and
must match the image/mask dimensions.

**Variable list order** changes the shared order used by every marker/channel
selector in NapariSBT. AnnData order is the default. Alphabetical order is
case-insensitive, while expression similarity is calculated from live `adata.X`
with the same clustering helper used by matrix plots and then cached. Reordering
preserves the channel selection and does not change the feature recipe.

**Select feature markers** restores the channel set represented in the
active built feature table. This is useful after reopening a workspace or changing
the visible channel selection. Before the first build it falls back to the current
recipe; when that recipe leaves the channel list blank, all currently available
channels are selected because the worker interprets blank as “use every discovered
channel.” Imported features and mask/context features do not count as markers.

## Synthetic features

Select IMC channels and feature families. Distribution statistics summarize
pixel intensities; region features describe cores, borders, centroids, local
background, and contrast; gradients capture texture and edges; morphology uses
the original mask; context uses the full segmentation; cohort-relative ranks
compare eligible cells within the same ROI.

A positive offset normally uses full-mask `expand_labels`, preventing expansion
through neighbouring cells. Enable overlap explicitly if measurements should
extend into neighbouring masks. Negative offsets erode measurement regions.
Offsets change intensity regions but never the original shape features.

Use the family checkboxes for broad inclusion and the feature tree for precise
control. The description column explains each measurement. Set the local worker
count conservatively on shared systems; each worker processes one ROI and loads
its mask and selected images. Review the selection summary before starting.

## Execution and progress

Local builds run in a subprocess with spawn-safe ROI workers. The progress panel
shows the orchestrator PID, heartbeats, completed/resumed/failed ROIs, recent
worker PIDs, and finalization. Cancellation preserves valid completed fragments.
Resume reuses fragments only when experiment, cohort, recipe, and input
fingerprints match. HPC builds use the same manifest through `sbt run cellfeat`.

Click **Build/resume features locally** once the sources, channels, and recipe are
ready. During a healthy build, the process state, heartbeat age, completed-ROI
counts, current ROI, and log should continue changing. A live process with a stale
heartbeat warrants inspection; **Cancel build** requests a clean stop and keeps
valid fragments for a later resume.
