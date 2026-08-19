# Dataset Maintenance

Dataset Maintenance creates deliberate, synchronized changes to the AnnData,
image, and mask assets connected in Setup. It is intended for repairing or
deriving datasets, not for routine exploration. The live AnnData can change in
memory, but no AnnData change reaches disk until **Save current AnnData** is
pressed.

New image and mask folders are always created separately. Existing image and mask
files are never edited by these tools. AnnData replacement is available only
through its explicit replacement checkbox.

## Dataset and synchronization readiness

The dashboard reports the current AnnData dimensions, identity validity, and the
coverage of the currently cached mask and image indexes. **Refresh from current
index** is fast and never scans folders. **Rebuild mask/image index now** is the
explicit potentially expensive scan; it is not triggered merely by opening or
moving around the tab.

A green item is ready, amber needs review, and red blocks the corresponding
operation. The unsaved-change message refers to the live AnnData object only.

## Save current AnnData

This writes the exact AnnData currently in memory, including population names,
manual labels, filters, renamed variables, and other new observations. The write
uses a temporary file followed by an atomic replacement. By default an existing
destination is rejected. Enable replacement only when the exact selected file is
intended to be replaced.

**Use the saved file as the Setup AnnData path** makes subsequent reloads use the
new file and updates the active workspace source when a workspace is open.

## Rename variables and images

Enter new names only in rows that should change. **Preview rename** validates
AnnData uniqueness and matches the requested channels against the explicit image
index. The complete derived image collection contains unchanged channels as well
as renamed channels, preserving the source folder layout inside numbered
subfolders.

Filename renaming is deliberately conservative: the old logical channel must be
present in the filename. An ambiguous or unmatched filename blocks image copying
and is shown in readiness rather than being guessed. Variable renaming can also be
applied to exact matching `adata.raw` names. Current normalization keys, Explore
recipes, Population QC recipes, and synthetic-feature channel selections are
updated with the same mapping.

The recommended order is preview, copy the derived image collection, then rename
the live AnnData variables. A valid preview is retained if the in-memory rename is
applied first.

## Remove variables

Select one or more AnnData variables and remove them from the live object. AnnData
automatically subsets `X`, `layers`, `var`, `varm`, and `varp`. Images remain
untouched by design and therefore become available orphan channels. `adata.raw`
normally remains the original reference; enable the raw option when it should be
restricted to variables retained in the live AnnData.

Removing every variable is refused. Classifier scores and models are marked stale
after a variable change.

## Filter cells

Choose an observation and either keep/remove selected values, keep/remove a
numeric range, or filter missing values. Preview reports retained and removed cell
counts and represented ROIs. The exact current settings must be previewed before
the confirmation action is enabled. Applying the filter slices all
observation-aligned AnnData slots in memory.

The original mask files still contain removed cells. Use **Rebuild masks and align
ObjectNumbers** afterward when exporting a synchronized derived dataset. A cell
filter in a dedicated Dataset Maintenance workspace automatically creates a new
identity revision and snapshot. In a classification/full workspace it instead
invalidates the active frozen classifier state; create a new experiment revision
or workspace before further classification.

## Rebuild masks and ObjectNumbers

This operation explicitly reads all represented ROI masks, checks that every
AnnData `(ROI, ObjectNumber)` exists in its mask, and writes a new mask folder.
Mask-only objects outside the current AnnData become background.

- **Preserve retained ObjectNumbers** keeps the original IDs and leaves gaps.
- **Compact to 1…N within each ROI** assigns consecutive IDs and updates the live
  AnnData ObjectNumber observation.

Both modes produce `object_number_crosswalk.csv` and a maintenance manifest. The
new folder becomes the current Setup mask source. Original masks remain untouched.

## Manage observations

Observation columns can be renamed or removed in memory. ROI, ObjectNumber, the
observation defining a frozen cohort, and active Population naming source/output
columns are protected. A conventional
`adata.uns["<obs>_colors"]` palette is moved with a renamed observation and removed
with a deleted one.

**Repair categorical colours** converts the selected observation to categorical
when needed and stores one valid colour per category using the standard Scanpy
palette location. Save AnnData afterward to persist these changes.

## Audit and recovery

Completed actions append a compact JSON-lines audit under the active workspace's
`dataset_maintenance` folder, or under `napari_sbt/dataset_maintenance` when no
workspace is open. Image and mask operations also write explicit crosswalks in
their derived folders.

Before saving AnnData, reload the original Setup AnnData to discard unwanted
in-memory changes. Derived output folders can be inspected independently and do
not replace their sources automatically.
