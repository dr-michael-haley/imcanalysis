# Layers & Status

Layers & Status contains general display utilities and a compact audit of
experiment freshness.

## Readiness and activity dock

**NapariSBT Readiness** is a separate Napari dock positioned beneath the Layers
selector on the left when the interface opens. It reports the action currently
running, elapsed time, a live heartbeat, background-process names and process
IDs, and the most recent finished or failed state. You can resize, move, float,
or hide it using Napari's normal dock controls. Heartbeat updates do not reopen
the dock after you hide it. Detailed feature-extraction progress remains in the
Feature Building tab.

## Selected-layer utilities

Select the target layer in Napari before using a utility. Recolouring and colormap
transfer modify display settings. Flipping, resizing, label expansion, and cohort
masking create derived display layers so the original scientific images and masks
remain unchanged. Check the new layer name and alignment before using it for QC.

## Experiment freshness status

The freshness report compares the loaded cohort snapshot with its fingerprint,
counts confirmed and proposed labels by class, identifies the active feature set,
checks the selected model-feature fingerprint, and verifies whether model and
scores match the current cohort, features, and confirmed labels. It also reports
reviewed ROI coverage.

`STALE` is a safety state rather than an error. Retrain after changing confirmed
labels or selected model features. Rebuild features after changing the synthetic
recipe, source tables, trial scope, or promoted experiment revision. Re-score
after retraining. Do not rely on prediction layers until all relevant freshness
indicators are current.

Click **Refresh experiment freshness status** after changing labels, features, or
models. Work from upstream to downstream: resolve cohort/feature staleness first,
then retrain the model, then rescore. Warnings identify incomplete coverage even
when the corresponding asset is technically current.
