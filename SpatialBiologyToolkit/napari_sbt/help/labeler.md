# Labeler

Labeler is a lightweight way to make explicit lists of cells without training or
scoring a classifier. It uses the active experiment's frozen cohort, ROI identity,
object identity, masks, images, and Explore recipes. Choose an **all cells**
experiment when every segmented cell should be available, or a selected cohort
when the list should be restricted to a population.

Labeler definitions and assignments are kept in memory for the current session.
They do not alter classifier proposals or confirmations. Export a CSV or explicitly
apply a categorical observation to the live AnnData object before closing the app.

## Define labels

Create one or more mutually exclusive labels. A cell can have one active Labeler
assignment; assigning another label replaces its previous assignment. Names and
colours are editable. Stable IDs are generated once and remain fixed so a cosmetic
rename does not change the meaning of existing cell records.

Double-click a colour swatch, or select a row and use **Pick selected colour**, to
choose the display colour. Click **Apply label edits** after changing names or
colours. An assigned label cannot be removed until its cells have been cleared or
reassigned.

## Label cells

Activate the Labeler tab, choose a current label and a click action, then click
cells directly in the Napari viewer. The full source mask is inspected under the
cursor, so `classification_cohort` can remain hidden and does not need to be the
selected layer. Clicks outside the frozen cohort are ignored.

**Assign selected label** is the fast annotation mode. **Select only** lets you
inspect a cell before using the button. **Clear label** removes the clicked cell's
Labeler assignment. The coloured `labeler_assignments` layer uses outlines by
default so staining remains visible, and `labeler_selected_cell_outline` marks the
current cell.

## ROI sampling guidance

The ROI selector controls the same viewer ROI used by Explore and Classify. Use
**Previous ROI** and **Next ROI** for ordinary navigation. The tally reports, for
each label, the total number of cells, the number of eligible ROIs represented,
and the number of matching cells in the current ROI.

**Next ROI without this label** searches forward, wrapping around once, for an
eligible ROI that has not contributed any cells to the current label. This is a
sampling guide, not a statistical claim that every label must occur in every ROI.
Use biological judgement where a population is genuinely absent.

## Results and export

The results table is the current in-memory cell list. It includes AnnData cell
identity when the frozen cohort contains `obs_name`, ROI, original mask object ID,
label name, stable label ID, and assignment time.

**Export CSV** atomically writes this table to the chosen destination. **Apply as
categorical obs to live AnnData** creates the named observation in the AnnData held
by this NapariSBT session: labelled cells receive their label and all other cells
remain missing. Existing observations are protected unless the explicit overwrite
box is enabled, and a confirmation dialog is always shown. This action changes only
the in-memory object; save or export AnnData separately if the change should persist.

The AnnData `uns["napari_sbt"]["labeler"]` entry records label definitions,
experiment revision, cohort fingerprint, time, and assignment count for provenance.
