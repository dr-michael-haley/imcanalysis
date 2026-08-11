# Setup

Setup defines the scientific question before any classification begins. Load the
AnnData, full segmentation masks, and one or more IMC image folders. ROI and
object-ID observations must identify every cell uniquely as `(ROI, ObjectNumber)`.

## Dataset inputs

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

## Cell scope

Choose **All cells** for whole-segmentation QC, or **Selected adata.obs values**
to subclassify one or more existing populations. Previewing validates identities,
mask coverage, represented ROIs, and eligible-cell counts. The eligible identity
snapshot is frozen when the experiment is created; original masks and AnnData are
never modified.

Click **Load AnnData selectors**, choose the observation and one or more values,
then click **Preview and validate cohort**. Read the preview before continuing:
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
