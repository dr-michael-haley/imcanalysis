# Setup

Setup defines the kind of NapariSBT session and its common dataset inputs. Only a
classification workflow asks for a frozen cohort, feature-discovery mode, and
classes; exploration and population-QC workspaces keep those controls out of the
way while retaining the same experiment-backed folders and recipes.

## Workflow selection

Choose the main task before creating or loading a workspace. **Data exploration**
shows general images, overlays, recipes, regions, and layer tools. **Population
QC** concentrates on population-specific RGB review. **Cell classification** adds
feature building, active learning, and prediction export. **Manual cell labeling**
provides simple hand-assigned identity lists, while **Population curation** adds
naming, merging, and subclustering. **Full workspace** exposes every tab.

Changing this selection hides irrelevant tabs; it does not delete their data or
saved recipes. The selection is stored in the experiment manifest, so reopening
the workspace restores the intended interface. A recipe may still contain layers
from another workflow, and those unavailable layers remain stored.

**Live recipe tracking** copies manual Napari layer visibility, opacity, colour,
contour, and contrast changes into the working recipe. It defaults on for Data
exploration and off for the lightweight Population QC workflow. Explicit recipe
controls, saved recipes, and Population QC review history continue to work when
live tracking is disabled.

## Dataset inputs

After choosing AnnData, masks, image folders, and identity observations, click
**Validate integrity and build fast asset index**. This is the explicit expensive
check: it scans the configured folders, validates eligible object IDs against all
relevant masks, reports missing coverage, and builds an ROI-to-file index. The
index is stored at `inputs/integrity_index.json` inside the workspace and reused
when its configured inputs still match.

Normal ROI navigation never performs another complete folder scan. It uses the
saved index, or fast direct lookups for conventionally named masks and nested
`images/<ROI>/` folders. Re-run validation deliberately after changing files or
folders. Creating a new workspace requires a current validation result, avoiding
an unexpected repeat of the expensive scan when **Create** is pressed.

Enter a descriptive experiment name, then identify the AnnData, masks, and image
folders. Add one image folder per line; NapariSBT searches all of them and matches
channels to the AnnData panel. Extra-image folders are for non-panel images that
should remain available in Explore. Set the ROI observation to the field containing
image/mask names and the object-ID observation to the integer mask-label field.

The experiment folder is NapariSBT's working area for the manifest, frozen cohort,
features, labels, and models. It should be a new or existing NapariSBT experiment
folder, not the source image or mask directory. Source images, masks, and AnnData
are not overwritten.

When launched from a Notebook with a live `AnnData`, selectors and previews use
that object directly and its path field is intentionally blank. Creating the
experiment writes a frozen copy under `inputs/anndata.h5ad` so the experiment
can be reopened and used by feature-building subprocesses. The GUI then uses
that snapshot too; opening the dock alone does not write the object.

## Image normalization and default display

Choose either a Nimbus normalization JSON or a CSV containing `Marker` and
`Value` columns. Both formats load the channel-to-maximum mapping into the
editable JSON pane. **Validate edited JSON** checks the values without writing
them. After a workspace exists, **Save edited copy into experiment** stores the
mapping in canonical JSON form at `display/normalization.json` and records it in
the manifest. Workspace creation also writes this experiment-owned copy,
including an empty mapping when no fixed maxima are supplied.

For channels without a fixed maximum, the fallback quantile and minimum-pixel
threshold reproduce the legacy IMC Explorer display normalization. Default lower
and upper contrast values apply to newly loaded scalar images whose recipe has no
explicit range. Recipe-specific contrast always takes precedence. Pixel values
and Napari's contrast slider range remain normalized to 0–1, even when the handles
are restored to a narrower default or recipe range.

## Cell scope

Choose **All cells** for whole-segmentation QC, or **Selected adata.obs values**
to subclassify one or more existing populations. Previewing validates identities,
mask coverage, represented ROIs, and eligible-cell counts. The eligible identity
snapshot is frozen when the experiment is created; original masks and AnnData are
never modified.

Click **Load AnnData selectors**, choose the observation and one or more values,
then click **Validate integrity and preview cohort**. Read the preview before continuing:
unexpectedly low counts, missing masks, duplicate identities, or unrepresented
ROIs usually indicate an identity-column or filename mismatch.

## Full experiment or Feature Discovery Trial

A full experiment builds and scores every eligible ROI. A Feature Discovery Trial
keeps that same full cohort as the eventual target but initially builds features
only for a configurable representative ROI subset. Select the largest eligible
ROIs automatically or choose them manually. Large ROIs are computationally
efficient, but manual selection is preferable when staining, tissue type, batch,
or disease state varies substantially.

Three ROIs are sufficient for a preliminary leave-one-ROI-out analysis; five or
more usually provide a more credible estimate of generalisation. Every target
class should be represented in at least two trial ROIs.

## Classes

Define two to eight mutually exclusive classes. Stable IDs are written to models
and exports; names and colours are presentation fields. Shortcuts select classes
in the Classify tab. A class marked `exclude` can later be removed from cleaned
mask exports. Class semantics become locked after confirmed labels exist.

Use **Add class** and **Remove selected class** to edit the table, or start from
**Segmentation QC template** for a good-versus-artifact task. Apply class edits
before creating the experiment. The Colour column is a true swatch: double-click
it, or select a row and click **Pick selected colour…**, to open the system colour
picker. The chosen hexadecimal value is shown on the swatch. Choose distinct
colours and shortcuts because both are reused throughout annotation and review.

Create the experiment only after checking the cohort preview, trial scope, and
class table. Loading an existing experiment restores these definitions.
