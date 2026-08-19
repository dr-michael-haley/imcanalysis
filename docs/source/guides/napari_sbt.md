# `napari_sbt`: cohort-first IMC exploration and classification

`napari_sbt` combines the reusable parts of the IMC explorer and CellPose
active-learning viewer in one experiment-driven Napari dock. Setup first asks
whether the session is for data exploration, Population QC, classification,
manual labeling, population curation, dataset maintenance, or the full workspace,
then hides tabs that are irrelevant to that task.

Normal exploration and classification treat the original images, masks, and
AnnData as read-only. Dataset Maintenance creates separate derived image/mask
assets and changes AnnData in memory; replacing an existing AnnData file requires
an explicit replacement option. An experiment freezes the eligible
`(ROI, ObjectNumber)` identities before feature calculation, annotation, training,
scoring, or export.

## Launch

From an initialized SBT project, change into the project (or any directory
inside it) and launch without repeating the path:

```bash
cd /path/to/project
sbt gui napari
```

You can instead select a project explicitly by registered name, project ID, or
filesystem path:

```bash
sbt gui napari --project "Registered project name"
sbt gui napari --project /path/to/project
```

An explicit existing initialized path takes precedence over a registry lookup.
Outside an initialized project, omitting `--project` retains the blank Setup
workflow for manually supplied inputs.

Run a side-effect-free launch check first when using a new environment or an
HPC allocation:

```bash
sbt gui napari --check
```

On CSF3, install the dedicated Linux environment and launch only from an
X11-enabled interactive node. See [Running NapariSBT on a CSF3 interactive
node](napari_sbt_csf3.md).

Direct launch accepts explicit paths:

```bash
napari-sbt \
  --anndata project/anndata.h5ad \
  --masks project/masks \
  --images project/denoised_images
```

The Python factory returns the viewer, controller, and dock without entering
the event loop:

```python
from SpatialBiologyToolkit.napari_sbt import launch

viewer, controller, dock = launch(
    anndata_path="anndata.h5ad",
    masks_folder="masks",
    images_folders=["denoised_images"],
)
```

### Launch from a Jupyter Notebook with an in-memory AnnData

Use the Notebook launcher when `adata` is already loaded or has unsaved
in-memory changes:

```python
from SpatialBiologyToolkit.napari_sbt import launch_notebook

viewer, controller, dock = launch_notebook(
    adata=adata,
    project_root=project,
    masks_folder=project / "masks",
    images_folders=[project / "images"],
)
```

`launch_notebook` enables IPython's Qt event-loop integration when available
and returns immediately, so do not call `napari.run()` in the Notebook. The
general launcher also accepts either `anndata=adata` or the compatible
`anndata_path=adata` form:

```python
viewer, controller, dock = launch(
    project_root=project,
    anndata_path=adata,
    masks_folder=project / "masks",
    images_folders=[project / "images"],
)
```

The live object is used directly for selectors, overlays, cohort preview, and
experiment setup. It is not modified. When the experiment is created,
NapariSBT writes an atomic experiment-owned copy to
`<experiment>/inputs/anndata.h5ad`; the manifest, restart workflow, feature
workers, annotated-copy export, and continuing GUI session use that frozen
snapshot. Later edits to the original Notebook object therefore cannot silently
change an established experiment. No copy is made merely by opening the
interface.

## Setup and the frozen cohort

Setup is designed to work without launcher path arguments. Its first section
shows the current project or standalone dataset folder, registered SBT projects,
saved workspaces detected under the configured `napari_sbt.experiment_folder`,
and mouse-driven open/create actions. Workspace discovery is bounded to the
container itself and immediate child `experiment.yaml` files; it never recursively
scans the project. Each entry reports its workflow, cohort size, ROI count, last
change time, and missing-source warnings.

New workspaces require a friendly name and receive an automatically derived,
collision-checked folder. A text-and-colour readiness banner keeps **Create
workspace and start** disabled until required inputs and the integrity check are
complete. Existing workspaces reopen directly from the detected list or a folder
chooser. Workspace persistence covers scientific state and deliberately does not
restore the complete Napari window layout.

Workflow selection uses plain-language task cards rather than requiring knowledge
of internal modes. Exploration-only workspaces still create the same
experiment-backed folder structure, allowing named views and reviewed-ROI state
to persist without exposing feature building or classifier controls. The selected
workflow is stored in the manifest and can be changed later without deleting data
from hidden tabs. The combined full workspace remains behind an advanced toggle.

AnnData, masks, staining images, optional extra images, and normalization files
have **Choose…** or **Add folder…** controls and status badges which combine
colour with Ready, Check needed, Action required, or Optional text. Conventional
ROI and object-ID observations are proposed after AnnData loads; the raw fields
are hidden under advanced cell-identity settings. **Reload all selected
components** rereads known data and current-ROI state without running a full scan.

Setup provides an explicit **Check dataset integrity and build the fast image index**
action. It is the only normal UI operation that scans complete mask/image folders
and validates every eligible mask identity. The resulting
`inputs/integrity_index.json` is reused across ROI changes and later sessions;
ordinary navigation uses cached or direct ROI-specific lookups. A current
validation result is required before creating a new workspace.

For classification, cell scope is required. Choose either:

- **All cells**, which is the replacement for the whole-mask CellPose QC use
  case.
- **Selected adata.obs values**, selecting one categorical observation and one
  or more values. This is intended for subclassifying a broad Leiden or curated
  population.

The preview reports selected/total cells, represented ROIs, per-ROI counts,
missing masks, missing object IDs, and other labels present in the full masks.
It also renders a cohort-only mask for inspection. Confirming the preview
stores a frozen identity snapshot in the experiment. A later change in AnnData
membership does not silently change the experiment; create a revision.

Setup also offers a **Feature Discovery Trial**. The complete cohort remains
frozen as the scientific target, while an independently stored trial scope
limits initial feature extraction, navigation, training, and scoring to a
configurable number of representative ROIs. ROIs can be selected manually or
suggested by eligible-cell abundance. The interface continually reports trial
cells and ROIs separately from the full target cohort.

The primary classification layer contains only eligible objects and preserves
their original mask labels. The optional context layer contains the rest of
the segmentation at low opacity. Annotation clicks on that context are ignored.
Normal ROI navigation includes only ROIs with eligible cells.

Multiple image folders are merged for each ROI. Supported layouts are
`<folder>/<ROI>/<channel>.tiff`, `<folder>/<ROI>.tiff`, and flat
`<folder>/<ROI>_<channel>.tiff`. Channel filenames are matched against
`adata.var_names` and, when available, `adata.var["channel_name"]` and
`adata.var["channel_label"]`. Duplicate channels from different folders remain
available with their source folder appended; unmatched composites remain
available as additional images.

Experiments contain two to eight stable, mutually exclusive classes. Each has
a display name, colour, numeric shortcut, and `keep` or `exclude` mask
disposition. The Segmentation QC template creates `good` and `artifact`
classes. Stable class and cohort semantics are locked after confirmed labels
exist.

The Setup class table renders each colour as a readable swatch. Double-click
the swatch, or select its row and use **Pick selected colour…**, to open the
colour picker; the saved hexadecimal value remains visible on the swatch.

Setup also owns scientific-display normalization. A Nimbus normalization JSON,
or a CSV containing `Marker` and `Value` columns, can be loaded into an editable
Marker/Value table, validated, and copied in canonical JSON form to
`display/normalization.json` inside the workspace. Fixed per-channel maxima are
used where available; unmatched channels use the configurable fallback quantile
and minimum-pixel threshold. Default contrast handles apply only when a recipe
does not contain an explicit channel range. Images and Napari's slider range stay
on 0–1, while saved recipe contrast limits take precedence over Setup defaults.
A read-only technical JSON preview remains available under an advanced toggle.
Population QC uses these Setup contrast limits to initialize an unsaved
population view, while retaining per-channel manual overrides and saved RGB
recipes. Its **Use Setup contrast defaults** button provides an explicit reset.

## Population QC

The dedicated **Population QC** tab is a focused version of the general Explore
recipe workflow. Select an AnnData observation and population, then choose up to
three red, green, and blue image channels with individual 0–1 contrast ranges.
NapariSBT can suggest the three available image channels with the highest mean
matched `adata.X` expression in that population. Each population-specific RGB
view is stored in the shared Explore state, including its colours and contrast.
Population QC has one workspace-wide 0–20 pixel outline preference for all
populations (1 pixel by default; 0 fills the labels). It is not restored from or
stored separately in each population recipe.

When no saved population recipe exists, those three marker suggestions are
calculated immediately and cached for the selected population. ROI abundance
rankings are cached too. ROI switching uses vectorized object-ID overlays, retains
the bounded image/overlay cache across ROIs, updates matching Napari layers in
place, and omits classification cohort/context arrays in exploration-only modes.
Live tracking of manual layer changes defaults off for a Population QC session
and on for Data exploration; explicit recipes and viewed-ROI history remain
available in either mode.

ROIs can be recalculated by highest abundance, lowest abundance, or a reproducible
random ordering, with a configurable result count and random seed. Buttons show
the matching-cell count and change from green to grey once the ROI has been viewed
with the exact current recipe fingerprint. Legacy IMC Explorer population-setting
CSVs can be imported, and the current observation's settings can be exported. The
global outline preference is deliberately not duplicated in those per-population
CSV rows.

Named Explore recipes and review histories persist in
`explore/review_state.json`. Recipes can also be imported and exported as JSON.
If a saved recipe references a classifier, marker, observation, or image layer
that is absent from the current workflow or ROI, the entry remains stored and is
highlighted in orange; it is retried later instead of being silently pruned.

## Feature sources and calculation

Feature work has its own **🧬 Feature Building** tab; it is not required for
ordinary image exploration. The tab separates source checking, channel
selection, recipe design, and execution:

1. Add optional imported sources. Tables use
   `source_name=/path/to/features.parquet`. AnnData and CellVision sources use
   `source_name=/path/to/data.h5ad::X`, `::obs`, or an `obsm` key such as
   `::X_cellvision`.
2. Select **Validate identities, cohort coverage, and numeric features** before
   a build. Validation runs outside the Napari process and reports whether each
   source can be read, how many frozen-cohort cells match, how many are missing,
   and how many usable numeric features it supplies. Its complete
   machine-readable report is saved as `features/source_validation.json`.
3. Refresh the channel list and select channels directly from those inferred
   from `adata.var_names` and the images available for an experiment ROI.
   Clearing the selection means “use every channel discovered by the worker.”
4. Enable feature families and tick only the individual measurements required
   for this experiment. The adjacent descriptions explain the scientific
   quantity represented by every choice.

The selectable feature families are:

- **Distribution**: fast per-channel pixel summaries such as mean, median,
  quantiles, spread, range, sum, and coefficient of variation.
- **Core/border/background/contrast**: more expensive regional measurements,
  including intensity-centroid displacement and local background comparisons.
- **Channel gradients**: the selected distribution summaries calculated over
  gradient magnitude, capturing edges and within-cell texture.
- **Original-mask morphology**: channel-independent area, perimeter,
  circularity, axes, solidity, holes, bounding-box, and edge features.
- **Full-segmentation context**: neighbour counts, nearest-cell distance, and
  ROI density calculated with every segmented cell as context.
- **Cohort-relative ROI ranks**: optional z-scores and/or percentile ranks
  among eligible cells in the same ROI.

Feature sources are joined strictly by `(ROI, ObjectNumber)`, filtered to the
frozen cohort, and namespaced before modelling. Sources may be CSV/Parquet,
AnnData `X`, numeric `obs`, a selected `obsm`, CellVision embeddings, or newly
calculated IMC features.

Synthetic extraction calculates rows only for eligible cells, but it keeps the
original full segmentation in memory:

- positive offsets block at every segmented-cell boundary by default;
- **Allow positive offsets to overlap other cells** independently expands each
  eligible measurement region through neighbouring masks. In this mode a pixel
  may contribute to multiple cells, which can be useful in dense tissue;
- negative offsets erode selected objects without changing their identities;
- background rings exclude every segmented object;
- neighbourhood features describe the full tissue segmentation;
- ranks and normalizations compare eligible cells within an ROI;
- shape features always use the original segmentation.

Stored image values are used by default. A fixed Nimbus normalization mapping
may be applied for scientific features. The quantile controls used for display
never change feature values. Image/mask shape mismatches fail rather than
resizing scientific inputs.

Local builds run in a `QProcess`; the worker uses a spawn-safe process pool with
one task per ROI. Completed Parquet fragments are reusable only when the
experiment, cohort, recipe, and input fingerprints match. Cancellation
preserves valid fragments. During a local build, the panel shows the Python
process ID and whether it is alive, heartbeat age, elapsed time, active worker
count, completed/failed/pending ROI counts, the most recent ROI result, and the
final fragment/source-combination phase. A heartbeat is emitted approximately
every two seconds while ROI work is outstanding, so a long-running ROI can be
distinguished from a dead process.

## Feature-discovery trials and refinement

The dedicated **🧪 Feature Refinement** tab evaluates broad trial features after
confirmed class labels have been collected. It uses leave-one-ROI-out splits,
performs candidate screening inside each training fold, compares elastic-net
logistic regression with Random Forest, and calculates permutation importance
only on held-out ROI cells. Results report balanced accuracy, macro-F1,
missingness, correlation redundancy, and the stability of positive importance
across fold/model evaluations.

The ranked table is checkable: the automated compact recommendation can be
restored, or changed using biological knowledge. Checked features can be used
immediately by the trial classifier. Promotion creates the next experiment
revision for the complete frozen cohort, records the trial results and checked
model inputs as provenance, reduces the extraction recipe to their required
synthetic measurements and imported columns, and requires a new full feature
build before training or scoring. Three trial ROIs support preliminary
evaluation; five or more representative ROIs are preferable when feasible.

Every workflow tab includes **Help for this tab**, and every coloured workflow
box has a prominent **❓ Help** button for focused instructions. Numbered box
titles use larger, heavier type and successive boxes use different accent colours
to make long panels easier to scan. The complete tab guides, focused box pop-ups,
and [`napari_sbt` interface help](napari_sbt_help.md) documentation all read the
same packaged Markdown sources.

The **NapariSBT Readiness** tracker is a separate Napari dock, placed beneath the
built-in Layers selector on the left by default. It paints an action-start
message before a synchronous callback begins, shows elapsed time and a heartbeat,
reports live background-process names and PIDs, and changes to a clear finished
or failed state. It can be moved, floated, or hidden like any other Napari dock;
hiding it is respected by subsequent heartbeat updates. The feature-building
panel retains its more detailed per-ROI progress display.

For large datasets, configure the active experiment and run:

```bash
sbt run cellfeat
```

## Multiclass review

Labels are either `proposed` or `confirmed`. Proposed labels are visible and
audited but do not train the model or become final assignments. Training
requires at least two confirmed examples per class and warns when class/ROI
coverage is weak.

While the **Classify** tab is active, a left click anywhere in the viewer is
resolved directly against the stored full mask and frozen cohort. The
`classification_cohort` layer therefore does not need to be selected or
visible; it is hidden by default so it does not obscure staining. The selected
eligible cell is shown with a configurable outline.

The **Click action** radio buttons control what a viewer click does using the
currently selected class: **Select only**, **Set proposed on click**, **Set
confirmed on click**, or **Clear proposed on click**. A separate **Clear proposed**
button applies the same action to the selected cell. Clearing is deliberately
proposal-specific and cannot erase a confirmed label. Proposed-on-click is the
default so cells can be labelled rapidly without repeatedly returning to a button,
while confirmation remains an explicit choice. Class shortcuts continue to change
the active class. **Clear all proposals (all ROIs)…** reports the proposal count,
requires confirmation, and clears reversible proposals across the experiment while
preserving every confirmed label.

The live label tally reports proposed and confirmed cells separately for every
class. Only confirmed labels train the model. HistGradientBoosting uses 20
samples per leaf, so the tally also shows progress towards a practical target
of at least 20 confirmed cells per class. Starting HistGradientBoosting below
that target displays a warning and offers Random Forest as the early-training
alternative; the user may still choose to train anyway.

Class selectors, the live tally, and active-learning queue entries use the
class colours defined in Setup, so the same visual identity is retained across
annotation, prediction review, and display layers.

The **Model storage** row shows the exact files for the active experiment.
`models/classifier_latest.joblib` contains the fitted imputer and classifier;
`models/classifier_latest.json` contains its model ID, class order, feature set,
training-label fingerprint, package versions, and other provenance. Retraining
replaces these `classifier_latest` files rather than creating an implicit model
inside the Napari workspace.

Single-cell clicks update only the clicked object's proposed/confirmed pixels
and the live tally. They no longer rebuild every labelled object in the ROI or
run the full freshness calculation after every annotation. Label-table and
audit writes remain synchronous, and the experiment manifest is rewritten only
when the first confirmed label locks class semantics.

Queue filters are applied automatically after scores exist. Changing the ROI,
predicted class, review state, or minimum confidence immediately rebuilds the
**Ambiguous unlabelled cells** list. **Apply queue filters / refresh** remains as
an explicit refresh action. The **Queue result** line reports both the number
shown (capped at 250) and the total number of cells matching the active filters.

**Classifier display & cell-picking options** opens the compact legacy-style
display panel. It independently controls visibility and opacity for the cohort,
excluded context, confirmed, proposed, predicted, uncertainty/probability,
selected-cell, and `noncontext_mask` layers. Label layers also expose contour
width: zero displays filled cells, while positive widths show outlines. Proposed
cells default to a two-pixel contour, and these settings are retained in the ROI
reload recipe.

`noncontext_mask` is a hidden-by-default, opaque black focus layer. It inverts the
pixels available to the classifier's cohort: original eligible cell bodies,
recipe-offset intensity regions, and local-background rings when region-image
features are enabled remain visible. Positive offsets follow the configured
overlap policy and background rings exclude all segmented cells. Negative offsets
do not hide the original cell body because original-mask shape features may still
be classifier inputs. The layer is rebuilt per ROI and never modifies source masks.

HistGradientBoosting is the default model. Random Forest, XGBoost, and LightGBM
are explicit alternatives. Scores contain the predicted class, every class
probability, maximum probability, probability margin, and normalized entropy.
Cells without usable features stay visible but are marked unscorable.
The uncertainty/probability image layer masks all pixels outside scored cells,
so its colormap never colours the tissue background; genuine zero-valued cell
scores still use the colormap's lowest colour.

The uncertainty queue ranks ambiguous unlabelled cohort cells. High-confidence
cells can be bulk-added as proposals for one predicted class. Confirmed labels
always override predictions.

The Classify pane contains a three-step workflow: **Annotate**, **Train & review
predictions**, and **Finalize & export**. A display-only minimum/maximum confidence
range controls which raw argmax predictions appear in `predicted_classes`; it does
not alter scores or final identities. Queue confidence and bulk-proposal confidence
remain independent controls for their respective actions.

Final identities use three explicit rules: minimum maximum-class probability,
maximum normalized entropy, and minimum top-two probability margin. All three must
pass for a model prediction to be accepted. Confirmed labels take precedence,
proposals are ignored, and other cohort cells remain unassigned. **Create / refresh
final cell identities** writes a canonical Parquet table and decision JSON with
thresholds and counts in the experiment exports folder. Exports are blocked if a
confirmed label, score, class definition, or threshold changes afterward.

The same pane exports CSV/Parquet, writes an atomic annotated AnnData copy, or—when
an AnnData object is live in a notebook—applies the annotations to that in-memory
object without writing its source file. Thresholds and precedence are stored with
the output provenance.

## Building explicit cell lists with Labeler

The **Labeler** tab reuses Classify's direct mask-based cell picking for tasks that
do not need a trained model. Define one or more named, coloured labels, choose an
assign/select/clear click action, and click eligible cells anywhere in the viewer.
The cohort layer can remain hidden. Labeler assignments are independent from
classifier proposals and confirmations, and assigning a second label to a cell
replaces its first Labeler assignment.

The live tally reports cell count, represented eligible ROIs, and current-ROI count
for every label. Previous/Next controls use the shared ROI ordering. **Next ROI
without this label** searches forward for an eligible ROI that has not contributed
a cell to the current label, making balanced visual sampling easier without
claiming that every biological population must occur in every ROI.

The in-tab results table shows AnnData cell identity, ROI, original mask object ID,
label name, stable label ID, and assignment time. Results can be atomically exported
to CSV or explicitly applied as a categorical observation to the live AnnData;
unlabelled and out-of-cohort cells remain missing. Existing observations require an
explicit overwrite choice. Labeler data is session-local until one of these actions
is used, so it does not reintroduce general workspace persistence.

## Explore, regions, layers, and export

The **Explore** tab provides cohort-aware ROI navigation, raw and extra image
loading, RGB views, categorical/numeric AnnData overlays, and abundance-ranked
population views. Previous/Next buttons follow the current ROI ordering.
**Hide all**, **Show all**, and **Delete all** act on every Napari layer; loading
the ROI again reconstructs the cohort and classification layers after deletion.
New layers have opacity 1.0 by default, except for the deliberately dimmed
full-segmentation context layer.

Explore and Population QC also show a compact current-ROI metadata table.
NapariSBT detects `adata.obs` columns that are constant within every ROI, excluding
the ROI and object-ID columns, and formats categorical, text, Boolean, numeric,
date/time, duration, and missing scalar values. This detection is cached when the
AnnData changes; ordinary ROI navigation only retrieves the current ROI's values.

Selected two-dimensional image channels can be loaded as independent greyscale
layers or assigned, in displayed selection order, to repeating red, green,
blue, cyan, yellow, and magenta colormaps. The three-channel RGB composite
remains available. Image choices are stored by logical channel name rather
than path, so the same recipe can be replayed against another ROI and missing
channels are reported rather than substituted.

Categorical overlays and separate population layers first look for Scanpy's
`adata.uns["<observation>_colors"]` palette (with common mapping-style
alternatives accepted) and otherwise use a stable fallback palette. Specific
populations can be selected as individual contour layers. Variables in
`adata.var_names` can also be loaded from cell-level `adata.X` values and
mapped onto eligible mask objects as quantitative overlays.

Observation overlays are cohort-restricted by default. Enable **Include cells
outside the classification cohort** to map an observation over every
identity-matched object in the ROI. Population naming enables this scope when
showing a newly crafted observation, allowing the refined population to be seen
alongside the unchanged broad populations in the tissue.

With **Reload the current Explore view when ROI changes** enabled, navigation
reconstructs the same images, colormaps, observation overlay, population
layers, and marker overlays for the new ROI. A view fingerprint tracks which
ROIs have been rendered with exactly those settings: ROI selector entries are
green when viewed and amber when not yet viewed. Changing any component creates
a different review set.

The **Layers re-added when the ROI changes** list is the explicit reload
contract. It shows every replayable Explore layer, including its colormap,
visibility, opacity, applicable contour width, and image contrast limits.
Switching recipes within one ROI first compares each requested layer with its
existing reload metadata. Matching image arrays and overlays are reused in place
and receive only display settings that actually changed. Layers removed by a
different recipe can be restored from a least-recently-used cross-ROI memory cache
instead of being read or calculated again. The cache is bounded to 48 arrays or
512 MiB and is cleared when source, normalization, or live AnnData inputs change,
not on every ROI switch. Image identity includes path, size, and modification time.
Changing source data or the requested semantic overlay still triggers the necessary
load or recalculation; revisiting a cached ROI can avoid it.
The working view can be saved as any number of named recipes. Each recipe may
have one unique F1–F12 shortcut; pressing that key while the Napari viewer has
keyboard focus immediately activates and renders the recipe for the current ROI
without moving away from the currently selected workflow tab. The selector provides
the same switching operation without a shortcut.

**Save current view as new recipe** creates a preset from the complete working
view. **Update selected recipe from current view** explicitly writes later layer,
display, name, or shortcut changes back to that preset. The active-status line
marks unsaved changes as `MODIFIED`; ordinary layer edits never silently overwrite
the stored recipe. Deleting a preset requires confirmation, unbinds its F-key, and
does not delete currently displayed layers. Duplicate names and shortcuts are
rejected.

Select one or more entries and use **Delete/reset selected recipe items** to
remove Explore entries from both the recipe and the current replayed view;
selecting a managed classifier entry resets its display settings to the
defaults. **Update from current layers** rebuilds the recipe from supported
layers currently present in Napari, capturing manual colormap, visibility,
opacity, contour, and contrast-limit changes. The eligible cohort,
excluded-cell context, confirmed, proposed, predicted, and
uncertainty/probability layers are also shown explicitly. Their scientific data
continue to be regenerated from masks, labels, and scores, but their display
state is captured and replayed. Changing a tracked display property in Napari
updates the active recipe
immediately. Manual-region and unsupported derived layers remain separately
managed or are reported as ignored rather than being silently added to the
recipe.

Recipes persist the actual continuous and direct-label colormap values currently
used by their Napari layers, not merely the palette assigned when a marker or
population was first loaded. Loading a named or population recipe therefore
restores later recolouring. Saving and explicit updating take a final live-layer
snapshot as protection against colormap changes that a Napari version does not
emit as a layer event.

The first rendered contrast limits for each image layer are frozen into the
recipe, even if the contrast slider is not touched. This prevents a channel
from being automatically rescaled when the next ROI is loaded. Later manual
contrast changes replace those stored limits. Scalar disk-image channels are
normalized to 0â€“1, so recipe replay restores the saved contrast handles while
keeping Napari's slider range at 0â€“1. Do not use Napari's contrast-reset action to
inspect that scale: reset deliberately replaces the saved handles.

**Save current view for selected population** stores a population-specific
verification recipe inside the experiment. Selecting that observation and
population later automatically retrieves the recipe; the explicit load button
does the same. Named recipes, population verification recipes, their shortcut
assignments, and viewed-ROI sets are stored in `explore/review_state.json`. This
is intentional Explore-view persistence, not a general Napari workspace save/load
mechanism.

**Use this population as classification cohort** transfers the population
selector back to Setup and requires a new preview and confirmation.

The **Regions & Export** tab stores manual polygon regions and synchronizes
their contained cell identities. Classification table and AnnData outputs now
live in **Classify → Finalize & export**. Regions & Export provides:

- optional cohort-only masks preserving original IDs;
- optional cleaned masks for classes marked `exclude`.

The annotated copy includes subclass, assignment source, confidence, and
uncertainty observations; a full probability matrix with `NaN` outside the
cohort; and an optional combined broad-population/subclass column. Original
AnnData and masks are never overwritten by default.

## Naming, merging, and subclustering populations

The **Population naming** tab crafts a named `adata.obs` column from one original
categorical observation such as `leiden`. Only the new obs-column name is needed;
there is no separate human display name. Changing the saved-work dropdown loads
that draft immediately. The first draft freezes a fingerprint of the cell
identities and source labels, and all sibling drafts remain remappings of that
same source.

The base table starts as an identity mapping. Editing **Proposed name** renames
a cluster. Assigning the same final name to multiple rows is an explicit merge:
all contributors are highlighted and listed in the effective-label preview.
Contributors to one merge automatically receive one shared colour across both
base and subcluster tables. Different merge groups receive distinct colours,
preferring the first available contributor colour before using the fallback
categorical palette. The selected merge colour is displayed in the preview and
saved into the derived observation's Scanpy palette. Changing a final label's
colour propagates that choice to every matching base and subcluster row.
Preliminary mappings can be imported from CSV, including common source columns
such as the selected obs name, `source_value`, `cluster`, or `leiden`, and label
columns such as `proposed_label`, `final_population`, `label`, or `name`.

Population source labels are matched after removing leading and trailing
whitespace. The immutable source fingerprint still records the exact AnnData
values, and labels that would collide after trimming are rejected rather than
silently merged.

Subclusters use a second component table because one original population can
produce several cell groups. The table retains parent population, method, raw
component, run ID, cell count, editable final name, colour, and notes. Both the
monitored Scanpy worker and an imported NapariSBT/image-classifier assignment
table feed this same representation. Later cell assignments replace prior split
membership only for overlapping cells. **Use current classifier assignments**
also transfers confirmed/model image classifications directly from the active
experiment, with confirmed labels taking precedence and unassigned cells omitted.

Interactive Scanpy subclustering defaults to rebuilding neighbours within the
selected cells from `adata.obsm["X_biobatchnet"]`, using
`scanpy.pp.neighbors(..., n_neighbors=15)` before Leiden. The cells can therefore
form new nearest-neighbour relationships after the broad population is isolated,
while the input representation remains BioBatchNet corrected. Normalization,
scaling, PCA, BioBatchNet itself, and UMAP are not rerun. The representation and
neighbour count are selectable, and the effective count is reduced to
`n_cells - 1` for small subsets. Reusing an existing `adata.obsp` graph remains
an explicit conservative alternative. The default runs each selected broad
population separately; an opt-in mode clusters selected populations together.

The readiness indicator reports unsaved edits separately from a saved draft that
has not yet reached the rest of the app. **Save and update Explore / Population
QC** persists the edits, creates or refreshes the derived observation and Scanpy
colour array in the live AnnData, and selects the corresponding renamed
population in Population QC where possible. Its current RGB view is carried to a
renamed population when that target has no saved recipe. An obs owned by the same
draft can be revised safely; unrelated existing columns retain an advanced
overwrite guard. **Open these labels in Scanpy plotting** saves and synchronizes
pending edits when approved, opens the dedicated plotting tab, and preselects the
derived observation.

**View history…** shows a concise recent-action list. The full append-only
`provenance.jsonl` remains on disk for reproducibility without occupying a main
workflow tab. Subcluster runs retain their request, assignments, and preprocessing
safety declaration. Curated AnnData export writes a new file and refuses to
replace an existing path.

## Scanpy plotting

The **Scanpy plotting** tab replaces the small Live QC plot box previously nested
inside Population naming. It is visible in every AnnData-based workflow and works
with original, renamed, merged, subclustered, classifier-derived, or manually
applied observations. Plot generation is read-only: it does not normalize or
scale AnnData, rebuild neighbours, recalculate PCA or UMAP, rerun BioBatchNet, or
change clustering.

Common controls choose the primary label observation, all cells versus the frozen
classification cohort versus selected populations, optional ROIs, and an
expression source from `adata.X`, `adata.raw`, or `adata.layers`. Categorical
ordering and `adata.uns["<obs>_colors"]` palettes are reused throughout. A live
summary reports the selected cell and group counts before a plot can be opened.
**Clear ROI selection** removes the optional ROI filter and returns the plot scope
to every available ROI.

An additional metadata filter lists observations that are constant within each
ROI, such as patient, condition, tissue, treatment, or acquisition batch. Users can
select one or more values without changing AnnData; the active filter is included
in automatic plot titles. It is separate from composition grouping, so a plot can,
for example, show per-ROI bars after first restricting the cells to one condition.

The initial plot collection includes:

- native `scanpy.pl.embedding` views of any existing `adata.obsm` embedding,
  coloured either by populations or by one or more expression variables from the
  selected matrix source, with selectable components, display limits, point size,
  opacity, expression colour map, and configurable multi-panel columns;
- native `scanpy.pl.matrixplot`, `scanpy.pl.dotplot`, and
  `scanpy.pl.stacked_violin` expression views from a searchable marker list;
- population counts or within-sample percentages as stacked bars or heat maps,
  grouped by ROI, patient, condition, or another observation;
- old-versus-new label cross-tabulation heat maps using counts, row percentages,
  or column percentages to make merges and splits explicit.

Embedding previews use deterministic population-stratified downsampling above the
configurable point limit. Expression code slices selected markers before making a
dense working array, rather than expanding the complete AnnData matrix. Aggregated
plots continue to use all selected cells. Matrix and dot-plot colours can use
marker-wise z-scores, marker-wise 0–1 scaling, or stored means; stacked violins
show the stored matrix values directly. Common native controls include colormap,
axis swapping, and one optional side annotation: a dendrogram or population cell
totals. Totals may retain population order or sort by abundance.

Expression plots can optionally cluster the selected marker order using
``SpatialBiologyToolkit.utils.reorder_vars_by_expression``. This operates on the
temporary marker-only AnnData containing the current cell/ROI selection and chosen
expression source, rather than mutating or reordering the source AnnData. Matrix,
dot, and stacked-violin plots can also display a narrow population-colour strip
using the live ``adata.uns["<observation>_colors"]`` mapping, following the visual
convention used by ``matrixplot_with_row_colors``. The strip reserves a separate
band from the population tick labels so long names do not overlap the colours.
Users can adjust both the colour/label gap and colour-box width in points; these
sizes remain stable when the plot window is resized.

Every plot also has common presentation controls. Legends or continuous colour
scales can be hidden; categorical embedding and stacked-bar legends can be placed
in the margin, on the data, automatically, or at a chosen corner. X and Y axis
titles and X and Y ticks can each be shown or hidden independently. For Scanpy
matrix, dot, and violin plots, hiding ticks also hides the associated marker or
population tick text. Expression embedding exports include every displayed
variable alongside the plotted coordinates. Plot titles can be automatic, custom,
or hidden while retaining a meaningful popup-window name.

Native expression heat maps retain their expression colour-map, live population
colour-strip, and fresh-dendrogram controls. Composition and label-comparison heat
maps expose their own colour map and optional population-colour strip. Configurable
edge colour and weight can outline heat-map cells or composition bars. Composition
bars also expose bar width, independent padding before and after the bar sequence,
optional fixed Y-axis minimum and maximum, and ascending or descending sorting by
a selected population subgroup.

When requested, a dendrogram is always recalculated from the currently selected
cells and markers in the temporary marker-only AnnData. Correlation (Pearson,
Spearman, or Kendall), linkage (complete, average, or single), and optimal leaf
ordering are configurable. NapariSBT never reads, replaces, or writes a dendrogram
stored in the source AnnData, so changing labels, populations, ROI scope, cohort,
expression source, or markers cannot silently reuse stale hierarchy state.

Each plot opens in a modeless resizable window. The Matplotlib canvas follows the
window size. Matrix plots, dot plots, and stacked violins measure their rendered
labels and Scanpy legend axes, reserve the necessary outer margins, and repeat that
fit after the popup is resized; this avoids the clipping that ordinary Matplotlib
tight layout can produce with Scanpy's nested plotting grids. The initial popup
also retains the requested Scanpy figure width where the available screen permits.
The window includes Matplotlib's standard zoom, pan, reset, and save toolbar. The
underlying plotted points or aggregate values can be exported to CSV. The main tab
tracks open windows and can focus, close, or close all of them. Plots are snapshots;
when the live AnnData is reloaded or synchronized, open windows are visibly marked
out of date instead of being silently redrawn.

## Dataset Maintenance

The **Dataset Maintenance** workflow exposes Setup, Dataset Maintenance, and
Layers & Status; the same tab is also available in the full workspace. It is a
controlled place for deriving synchronized scientific assets rather than a
general-purpose file browser.

The readiness dashboard consumes the mask/image index already built in Setup.
Opening the tab never scans dataset folders. **Rebuild mask/image index now** is an
explicit potentially expensive scan and reports unique identities, mask coverage,
and image coverage.

The initial maintenance tools include:

- atomic saving of the exact AnnData currently in memory, including newly created
  observations, with existing-file replacement disabled by default;
- validated `var_names` renaming, synchronized exact marker metadata,
  normalization, Explore/Population QC recipes, synthetic-feature channel names,
  and copy-on-write image filename changes;
- removal of selected AnnData variables while deliberately retaining images, with
  explicit `adata.raw` handling;
- categorical, numeric-range, and missing-value cell filtering with retained-cell
  and represented-ROI previews;
- derived mask construction that either preserves retained ObjectNumbers or
  compacts them to `1…N` within each ROI, updates the live AnnData, and writes an
  identity crosswalk; and
- protected observation-column rename/removal plus repair of conventional Scanpy
  categorical colour palettes.

Image copying produces a complete derived collection: unchanged indexed channels
are copied alongside renamed channels. A requested rename is blocked if the old
logical channel cannot be found unambiguously in its filename. Mask rebuilding
reads every represented mask only after the user explicitly requests validation
or execution. Original images and masks are never modified.

Cell filters and compact ObjectNumber remapping invalidate a frozen classification
universe. NapariSBT disables the current in-memory classifier state rather than
silently adapting it; create a new experiment revision or workspace for subsequent
classification. Completed operations append compact JSON-lines audit events under
`dataset_maintenance/audit.jsonl`, while derived image and mask folders contain
their own CSV crosswalks.

The copy-on-write operations are also importable for scripted use:

```python
from SpatialBiologyToolkit.napari_sbt import (
    CellFilterRequest,
    apply_cell_filter,
    apply_var_rename,
    atomic_write_anndata,
    rebuild_masks_and_object_numbers,
    remove_anndata_vars,
)
```

Each function returns a new AnnData object or a derived output path; callers must
explicitly save or adopt the result.

## Experiment layout

```text
napari_sbt/<experiment>/
├── experiment.yaml
├── cohort/eligible_cells.parquet
├── features/
│   ├── fragments/
│   ├── feature_table.parquet
│   ├── feature_dictionary.csv
│   ├── coverage_report.csv
│   ├── source_validation.json
│   └── feature_manifest.json
├── feature_refinement/
│   ├── feature_ranking.csv
│   ├── fold_metrics.csv
│   └── summary.json
├── labels/labels.parquet
├── explore/review_state.json
├── models/
├── scores/scores.parquet
├── annotations/
├── dataset_maintenance/audit.jsonl
├── cohort_masks/
└── exports/
```

There is deliberately no viewer-workspace save/load mechanism. Scientific
state is captured in explicit experiment, label, feature, model, and export
assets instead.
