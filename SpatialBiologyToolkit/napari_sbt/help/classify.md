# Classify

Classify is the human-in-the-loop annotation and active-learning workspace. Click
an eligible cell anywhere in the viewer; the cohort label layer may remain hidden
and does not need to be selected. Choose whether clicks only select cells, create
proposals, or immediately create confirmed labels.

Proposed labels are reversible working decisions and never become final exports
unless confirmed. Confirmed labels train the classifier and override model
predictions. The tally shows label coverage per class; representation across ROIs
is as important as raw cell count.

The classifier uses the active model-feature selection recorded by Feature
Refinement, or every eligible numeric feature when no compact selection exists.
Models are stored inside the experiment `models` folder with cohort, feature,
label, class, package-version, and feature-selection fingerprints. Any relevant
change makes the previous model stale and requires retraining.

After scoring, queue filters are applied immediately. The ambiguity queue ranks
unlabelled cells by normalized entropy. High-confidence predictions can be
proposed in bulk by class, but confirmation remains a human decision. Probability
and uncertainty layers colour only scored cell pixels; background is transparent.

Classifier display options control visibility, opacity, and label contours for
cohort, confirmed, proposed, predicted, selected-cell, and uncertainty layers.
These display settings participate in the Explore reload recipe.
