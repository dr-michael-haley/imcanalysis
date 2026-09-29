# Setup

Setup is a guided start screen for opening or creating a NapariSBT workspace,
choosing a task, connecting a dataset, and checking that it is ready. A workspace
stores the scientific manifest, cohort, recipes, labels, models, and exports; it
does not save and restore the complete Napari window layout.

## Start or resume

The dock always includes **Home** and **Workspace**. Home offers **Create a
workspace**, **Open a workspace…**, recent workspaces, and **Return to current
workspace**. Visiting Home leaves the current workspace open.

Workspace setup shows one page at a time: **Dataset**, **Task**, **Check data**,
and **Save workspace**. Back and Continue stay visible beneath the scrolling
page. Supplied Python paths and live notebook AnnData prefill these pages.
Analysis tabs appear after a workspace opens.

Choose the dataset folder, or use the supplied inputs. An initialized SBT project
provides configured defaults. Standalone folders use bounded discovery: immediate
conventional folders and `.h5ad` files, with a chooser for ambiguous cell data.
The source dataset can be on a shared drive while saved work is in a separate,
writable location. Desktop welcome mode defaults saved work to
`~/NapariSBT Workspaces`; it does not inspect the application's launch folder.

Choose a task before checking data. Exploration, Population QC, manual labeling,
population curation, and dataset maintenance offer **Open quickly (check as I go)**.
Quick opening prepares and freezes cell identities without a
complete image/mask scan. Each opened mask is checked against that region's cell
IDs. A persistent status explains that the complete dataset has not been checked.
A live notebook AnnData is still saved as a workspace-owned snapshot at creation;
copying a large cell table can take time independently of asset validation.

**Check entire dataset** explicitly scans file paths, reads masks, and checks cell
IDs and image coverage. It runs in the background; cancellation takes effect
between discovery/region operations, after any current filesystem read finishes.
Missing masks, images, or eligible cell IDs produce **Needs review**, never a
successful validation status. This coverage check does not decode every image;
feature extraction checks image contents when it consumes them.

**Find files only** builds the path index without reading mask contents. This is
useful for flat or irregular layouts that direct region lookup cannot resolve.
It does not count as full validation. Nested `images/<ROI>/` folders and directly
named masks support lazy lookup without a dataset-wide index.

Classification and the combined full workspace require a successful full check
before creation. Feature workers also check their inputs independently; a build
with failed regions cannot activate a feature set for training. Training rejects
confirmed cells without feature rows. Derived mask exports validate the requested
cell identities before writing outputs.

On the final page, name the workspace and review its save location. **Saved
normalisation dictionary** lets you reuse channel intensity limits. **Other
display settings** contains automatic brightness and contrast controls. Existing
workspace files are never overwritten by creation. Once open, Workspace becomes
a summary with links to task controls, settings, checks, and the results folder.

Validation history is stored separately from the file index. A reopened workspace
reports previous checks without claiming that externally changed files were
rescanned. Changing inputs invalidates the current check. A workspace switch is
blocked while background work or unsaved maintenance/population edits remain.

## Advanced controls

The Dataset page contains **Additional image folders**, **Show advanced
cell-identity settings**, registered projects, and **Troubleshooting**. Reloading
components and detecting missing paths remain available there. Neither action
counts as full validation. The Task page retains classification cell selection,
class definitions, and expandable Feature Discovery Trial settings. **Prepare
cell selection** updates trial-region choices without scanning image assets.

## Workflow selection

Choose the plain-language task from the menu. **Explore my images
and cells** shows images, overlays, recipes, regions, and layer tools. **Check
existing cell populations** concentrates on population-specific RGB review.
**Train a cell classifier** adds feature building, active learning, and prediction
export. **Manually collect labelled cells** creates hand-assigned identity lists,
while **Rename, merge, or subcluster populations** opens population curation.
**Advanced: show every tool** exposes the complete interface.

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

For an explicit full check, use **Check entire dataset** on the Check data page. This is the explicit expensive check: it scans
the configured folders, validates eligible object IDs against all relevant masks,
reports missing coverage, and builds an ROI-to-file index. The index is stored at
`inputs/integrity_index.json` after workspace creation and reused when its
configured inputs still match.

Normal ROI navigation never performs another complete folder scan. It uses the
saved index, or fast direct lookups for conventionally named masks and nested
`images/<ROI>/` folders. Re-run validation deliberately after changing files or
folders. Creating a classification workspace requires a current successful check. Quick
exploration can create a workspace without that scan.

Use **Reload all selected components** to reread a loaded workspace, AnnData,
normalization values, saved review state, and current ROI without running the
expensive folder scan. Use Check entire dataset separately after files, folders,
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

Open **Saved normalisation dictionary** and choose a CSV or JSON file containing
your channel intensity limits. Reusing these limits makes brightness easier to
compare across regions. The table shows **Channel**, **Max. intensity**, and **Min. intensity**:
Minimum removes low background signal, and Maximum sets the bright end of the
range. Channels without saved limits are adjusted automatically. Expand **Edit
channel limits** to see the table and make changes.

Use **Check edited values** after editing the table. After a workspace
exists, **Save limits to workspace** stores a copy in canonical JSON
form at `display/normalization.json` and records it in the manifest. Workspace
creation also saves a copy when channel limits have been supplied.

For matched channels, NapariSBT uses
`clip((image - lower_threshold) / (vmax - lower_threshold), 0, 1)`. For channels
without fixed bounds, the fallback quantile and minimum-pixel threshold reproduce
the legacy IMC Explorer display normalization. Default lower and upper contrast
values apply to newly loaded scalar images whose recipe has no explicit range.
They also initialize Population QC RGB contrast controls for populations without
a saved recipe. Population QC values can be changed per channel, and manual or
saved overrides are not replaced by later Setup changes. Recipe-specific contrast
always takes precedence. Pixel values and Napari's contrast slider range remain
normalized to 0–1.

## Cell scope

Choose **All cells** for whole-segmentation QC, or **Selected adata.obs values**
to subclassify one or more existing populations. Previewing validates identities,
mask coverage, represented ROIs, and eligible-cell counts. The eligible identity
snapshot is frozen when the workspace is created; original masks and AnnData are
never modified.

Choose the observation and one or more values, then use **Prepare cell selection** to preview the cohort and **Check entire
dataset** to validate its assets. Read the preview before continuing: unexpectedly low counts,
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
