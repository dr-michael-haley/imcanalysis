# Setup

Setup defines the scientific question before any classification begins. Load the
AnnData, full segmentation masks, and one or more IMC image folders. ROI and
object-ID observations must identify every cell uniquely as `(ROI, ObjectNumber)`.

## Cell scope

Choose **All cells** for whole-segmentation QC, or **Selected adata.obs values**
to subclassify one or more existing populations. Previewing validates identities,
mask coverage, represented ROIs, and eligible-cell counts. The eligible identity
snapshot is frozen when the experiment is created; original masks and AnnData are
never modified.

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

Create the experiment only after checking the cohort preview, trial scope, and
class table. Loading an existing experiment restores these definitions.
