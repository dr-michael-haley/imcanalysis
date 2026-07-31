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
