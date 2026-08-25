# Setup

Setup is a guided start screen for opening or creating a NapariSBT workspace,
choosing a task, connecting a dataset, and checking that it is ready. A workspace
stores the scientific manifest, cohort, recipes, labels, models, and exports; it
does not save and restore the complete Napari window layout.

## Start or resume

The first box is the normal starting point, even when NapariSBT was launched
without command-line path options. **Project or dataset** shows the current folder
and projects already present in the SBT project register. Use **Choose project
folder…** for an unregistered or standalone dataset. When an initialized SBT
project is selected, its configured AnnData, masks, image folder, normalization
file, and NapariSBT workspace location are proposed automatically.

For standalone folders, or when a configured source is missing, Setup performs a
cheap automatic lookup for conventional assets. It checks `.h5ad` files at the
project root and in conventional immediate data folders, and recognizes immediate
mask and image folders by names such as `masks`, `cell_masks`, `images`,
`processed`, or `matrix_images_alligned`. It does not inspect their contents. One
unambiguous AnnData is filled automatically; if several are found, a chooser is
shown and Setup remains incomplete if it is cancelled. Existing valid configured
or manually selected sources take precedence. Use **Automatically detect missing
inputs** to repeat this lookup after adding or moving assets.

NapariSBT looks only in the configured workspace folder (normally
`<project>/napari_sbt`) and one directory level below it for `experiment.yaml`.
This bounded lookup is deliberately cheap and is not a recursive scan of the
dataset. The workspace list shows its workflow, eligible-cell and ROI counts,
last modification time, and a green, amber, or red state. Amber workspaces can
open but have a missing configured source; red entries contain an unreadable
manifest and remain visible so the problem is not hidden.

Select a workspace and click **Open selected workspace**, or use **Browse
elsewhere…**. To create one, enter a descriptive **New workspace name**. Its folder
is generated automatically under the configured workspace location; **Change
location…** is available when required. NapariSBT refuses to overwrite an
existing workspace. The readiness banner explains why **Create workspace and
start** is disabled and enables it only after all required checks pass.

When a workspace is open, its immutable name and location are protected from
accidental edits. **Set up a new workspace** leaves the current saved files
untouched and returns to new-workspace setup using the same dataset as a starting
point. Experiment-specific state is not inherited: the cohort preview, integrity
result, current ROI, Population QC selections, and classification scope are
cleared. The new workspace starts explicitly at **All cells**, so a restricted
cohort from the previous workspace cannot silently carry into the new one.

## Workflow selection

Choose the plain-language card that describes the main task. **Explore my images
and cells** shows images, overlays, recipes, regions, and layer tools. **Check
existing cell populations** concentrates on population-specific RGB review.
**Train a cell classifier** adds feature building, active learning, and prediction
export. **Manually collect labelled cells** creates hand-assigned identity lists,
while **Rename, merge, or subcluster populations** opens population curation.
**Show every tool** is hidden behind the advanced-workflow checkbox because it
exposes the most complicated interface.

Changing this selection hides irrelevant tabs; it does not delete their data or
saved recipes. The selection is stored in the experiment manifest, so reopening
the workspace restores the intended interface. A recipe may still contain layers
from another workflow, and those unavailable layers remain stored.

**Live recipe tracking** copies manual Napari layer visibility, opacity, colour,
contour, and contrast changes into the working recipe. It defaults on for Data
exploration and off for the lightweight Population QC workflow. Explicit recipe
controls, saved recipes, and Population QC review history continue to work when
live tracking is disabled. The same session switch is also shown in Explore and
Population QC, and can be changed at any time without revising the workspace.

## Dataset inputs

The main data rows have mouse-driven file/folder choosers and accessible status
badges: green **Ready**, amber **Check needed**, red **Action required**, or grey
**Optional**. Colour is never the only signal. **Processed cell data** loads an
`.h5ad` file or uses the AnnData object supplied by a notebook. Add one or more
**Staining image folders**; additional-image folders are optional.

Channel filenames are matched to `adata.var_names` after punctuation-insensitive
normalization. Standard IMC isotope prefixes are understood, so files such as
`141Pr_Ly6G.tiff`, `Pr141_Ly6G.tiff`, and `143Nd_HLA_DR.tiff` match marker-only
variables such as `Ly6G` and `HLA-DR`. Ambiguous aliases are left as additional
images instead of being assigned to the wrong variable.

AnnData columns used to match cells to images and integer mask labels are proposed
from conventional names such as `ROI` and `ObjectNumber`. Their current meaning
is summarized in plain language. Use **Show advanced cell-identity settings** only
when the proposed columns are wrong.

After choosing AnnData, masks, and image folders, click **Check dataset integrity
and build the fast image index**. This is the explicit expensive check: it scans
the configured folders, validates eligible object IDs against all relevant masks,
reports missing coverage, and builds an ROI-to-file index. The index is stored at
`inputs/integrity_index.json` after workspace creation and reused when its
configured inputs still match.

Normal ROI navigation never performs another complete folder scan. It uses the
saved index, or fast direct lookups for conventionally named masks and nested
`images/<ROI>/` folders. Re-run validation deliberately after changing files or
folders. Creating a new workspace requires a current validation result, avoiding
an unexpected repeat of the expensive scan when **Create workspace and start** is
pressed.

Use **Reload all selected components** to reread a loaded workspace, AnnData,
normalization values, saved review state, and current ROI without running the
expensive folder scan. Use the integrity button separately after files, folders,
or identity columns change. **Automatically detect missing inputs** is also cheap:
it searches only bounded conventional locations and never substitutes for the
integrity check.

The workspace folder is NapariSBT's working area for the manifest, frozen cohort,
features, labels, and models. It is not the source image or mask directory. Source
images, masks, and AnnData are never overwritten.

When launched from a Notebook with a live `AnnData`, selectors and previews use
that object directly. Creating the workspace writes a frozen copy under
`inputs/anndata.h5ad` so the workspace can be reopened and used by feature-building
subprocesses. Opening the dock alone does not write the object.

## Image normalization and default display

Choose either a Nimbus normalization JSON or a CSV containing `Marker` and
`Value` columns. Both formats load the channel-to-maximum mapping into an editable
two-column table. Add or remove marker rows without writing JSON. **Show technical
JSON preview** provides a read-only representation for troubleshooting.
**Validate edited values** checks the table without writing it. After a workspace
exists, **Save edited copy into experiment** stores the mapping in canonical JSON
form at `display/normalization.json` and records it in the manifest. Workspace
creation also writes this experiment-owned copy, including an empty mapping when
no fixed maxima are supplied.

For channels without a fixed maximum, the fallback quantile and minimum-pixel
threshold reproduce the legacy IMC Explorer display normalization. Default lower
and upper contrast values apply to newly loaded scalar images whose recipe has no
explicit range. They also initialize Population QC RGB contrast controls for
populations without a saved recipe. Population QC values can be changed per
channel, and manual or saved overrides are not replaced by later Setup changes.
Recipe-specific contrast always takes precedence. Pixel values and Napari's
contrast slider range remain normalized to 0–1.

## Cell scope

Choose **All cells** for whole-segmentation QC, or **Selected adata.obs values**
to subclassify one or more existing populations. Previewing validates identities,
mask coverage, represented ROIs, and eligible-cell counts. The eligible identity
snapshot is frozen when the workspace is created; original masks and AnnData are
never modified.

Choose the observation and one or more values, then click **Check dataset integrity
and build the fast image index** (or **Validate integrity and preview cohort** in
the classification scope box). Read the preview before continuing: unexpectedly low counts,
missing masks, duplicate identities, or unrepresented ROIs usually indicate an
identity-column or filename mismatch.

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
before creating the workspace. The Colour column is a true swatch: double-click
it, or select a row and click **Pick selected colour…**, to open the system colour
picker. Choose distinct colours and shortcuts because both are reused throughout
annotation and review.
