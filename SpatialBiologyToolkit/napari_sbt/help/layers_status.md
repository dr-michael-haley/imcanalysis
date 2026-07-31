# Layers & Status

Layers & Status contains general display utilities and a compact audit of
experiment freshness. Operations such as recolouring, flipping, resizing, label
expansion, colormap transfer, and cohort masking create or modify Napari display
layers; they do not rewrite scientific source images or original masks.

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
