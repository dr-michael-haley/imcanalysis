# Feature Refinement

Feature Refinement is available for Feature Discovery Trials after broad features
have been built and cells have confirmed class labels. Its purpose is to find a
smaller, more stable model-input set before paying to process the full cohort.

## Readiness

The class table reports confirmed cells and represented ROIs. At least two
feature-bearing confirmed cells per class are required, with each class present
in at least two ROIs. Twenty to thirty cells per class is a useful initial target;
more labels and more representative ROIs improve stability.

## Evaluation

The analysis holds out an entire ROI, trains on the remaining ROIs, and repeats
this for each usable ROI. Candidate screening is fitted inside each training fold
to avoid using the held-out ROI to choose features. Elastic-net logistic
regression supplies an interpretable sparse model and Random Forest captures
nonlinear interactions.

Held-out permutation importance measures how much balanced accuracy is lost when
a feature is shuffled. Stability is the proportion of evaluated fold/model pairs
where its importance is positive. High-missingness and constant features are
screened out; highly correlated alternatives are marked redundant.

Set the maximum candidate count to control the cost within each training fold,
the requested recommendation to control the final compact set, and permutation
repeats to trade speed for stability. Missingness and correlation thresholds are
screening controls; stricter values reduce the candidate set. Defaults are a
sensible starting point unless the trial is very small or exceptionally wide.

## Analysis progress

Click **Run leave-one-ROI-out feature refinement** only after the readiness table
passes. The progress bar and log report each held-out ROI/model evaluation and the
final aggregation. Cancellation stops the monitored subprocess without changing
the feature table or current model selection. **Reload saved results** restores a
completed report after restarting the GUI.

If progress stops, inspect the most recent log entry and confirm that the worker
process is still live. Failures leave the source features and confirmed labels
untouched, so settings can be adjusted and the analysis rerun.

## Choosing and promoting features

The result table is deliberately checkable. Restore the automated recommendation,
or add/remove features using biological knowledge and extraction cost. **Use
checked features for trial classifier** changes only model inputs; it does not
delete the broad trial table.

Promotion creates the next experiment revision, retains the frozen full cohort
and trial report as provenance, and records the checked model features. The new
revision reduces the synthetic recipe to the required channels and feature
families and disables imported sources that contributed no checked features. It
clears the active feature-build identity, requiring a new full-cohort build before
training or scoring. Reported performance remains exploratory until confirmed on
independent ROIs or data not used during feature refinement.

Review importance, stability, missingness, source, and family together rather than
selecting solely by rank. **Restore recommended checks** returns to the automated
selection. **Use checked features for trial classifier** is reversible; promotion
is the explicit step that creates a full-cohort revision from the checked set.
