# Classify

## Cell annotation

Classify is the human-in-the-loop annotation and active-learning workspace. Click
an eligible cell anywhere in the viewer; the cohort label layer may remain hidden
and does not need to be selected. Choose whether clicks only select cells, create
proposals, or immediately create confirmed labels.

Proposed labels are reversible working decisions and never become final exports
unless confirmed. Confirmed labels train the classifier and override model
predictions. The tally shows label coverage per class; representation across ROIs
is as important as raw cell count.

Choose the current class and click action before selecting cells. **Select only**
is safest while inspecting; **Set proposed on click** is the fastest reversible
annotation mode; **Set confirmed on click** should be used only for clear examples.
**Clear proposed on click** and the selected-cell **Clear proposed** button remove
only reversible proposals; a confirmed label is left unchanged. The display-options
popup controls the annotation overlays without changing label records. Mark an ROI
reviewed only after checking it with the intended view.

Each class has a number-key hotkey, shown explicitly in the **Hotkeys** row and
beside its name in the class selector. With the Napari viewer canvas focused,
press that number to change the current class. The selected click action is left
exactly as it was, so repeated clicks can continue proposing, confirming, clearing,
or selecting cells without returning to the controls. A status message confirms
both the new class and the unchanged click action.

**Clear all proposals (all ROIs)…** removes every reversible proposal in the
experiment after showing the exact count and asking for confirmation. It is useful
when abandoning a bulk-proposal or early review pass. It never removes confirmed
labels.

## Models and active-learning queues

The classifier uses the active model-feature selection recorded by Feature
Refinement, or every eligible numeric feature when no compact selection exists.
Models are stored inside the experiment `models` folder with cohort, feature,
label, class, package-version, and feature-selection fingerprints. Any relevant
change makes the previous model stale and requires retraining.

After scoring, queue filters are applied immediately. The ambiguity queue ranks
unlabelled cells by normalized entropy. High-confidence predictions can be
proposed in bulk by class, but confirmation remains a human decision. Probability
and uncertainty layers colour only scored cell pixels; background is transparent.
The shared `uncertainty_or_probability` image layer uses additive blending whenever
it is created or refreshed, so the underlying staining remains visible.

Classifier display options control visibility, opacity, and label contours for
cohort, confirmed, proposed, predicted, selected-cell, uncertainty, and
`noncontext_mask` layers. These display settings participate in the Explore
reload recipe.

`noncontext_mask` is an optional opaque black focus layer. It leaves visible the
original eligible cell bodies, their recipe-defined intensity measurement regions,
andâ€”when region-image features are enabledâ€”their local-background rings. Everything
else is covered. Positive offsets respect the recipe's overlap policy, background
rings exclude all segmented cells, and a negative intensity offset never hides the
original cell body because original-mask shape features may still inform the model.
The layer is derived for each ROI, never changes the source mask, and is hidden by
default. Enable it in **Classifier display & cell-picking options** or in Napari's
layer list.

Train only after every class has enough confirmed, feature-bearing examples and
reasonable ROI representation. Then score the cohort. Queue filters apply when
their values change or when **Apply queue filters / refresh** is clicked; selecting
a queue entry navigates to that cell. Use the probability-class control to inspect
one class at a time, and review bulk proposals before confirmation.

## Prediction display controls

The visible confidence range controls only the `predicted_classes` layer. Use it
to inspect low-, intermediate-, or high-confidence raw argmax predictions without
changing the stored score table. **Show all predictions** resets the range to
0–1. This display range does not decide which predictions become final identities.

The probability and uncertainty layers remain separate scientific views. Queue
minimum confidence filters the active-learning list, while the bulk-proposal
minimum confidence controls only the bulk-proposal action. These controls are
deliberately independent.

## Final identities and export

After training and scoring, open **Finalize & export**. A model prediction becomes
a final identity only if it meets the minimum maximum-class probability, maximum
normalized entropy, and minimum top-two probability-margin rules. All three rules
must pass. Confirmed labels override the model, proposals are ignored, and rejected
or unscorable cells remain unassigned.

Click **Create / refresh final cell identities** before exporting. This writes a
canonical cohort-only Parquet table and a decision-provenance JSON inside the
experiment exports folder, and shows counts for confirmed, accepted-model, and
unassigned cells. Changing a rule, confirmed label, score table, class definition,
or model input makes that result stale and blocks export until it is refreshed.

## Integrate with existing labels

Subset-classification experiments normally refine one broad Leiden or named
population rather than replace every population in the dataset. Enable **Create a
full-dataset integrated population label** to make that handoff explicit. Choose
the existing observation that supplies labels outside the cohort and enter a new
output observation name. The source observation is never overwritten.

Choose how accepted classifier classes should be named:

- **Use class names** inserts the class display names from experiment setup.
- **Prefix with the source label** produces names such as `Broad myeloid → DC`.
- **Define names here** makes each integrated class name editable in the table.

Cells outside the frozen classification cohort always retain their source label.
Cohort cells without a confirmed or threshold-accepted model identity also retain
their source label. Only accepted final identities replace it. Matching names are
therefore an explicit merge. The overlap/confusion preview counts existing source
labels against the accepted integrated labels; it describes label overlap, not
classifier accuracy.

Click **Build / refresh integrated labels** after creating final identities. Any
change to the final identities, source observation, output name, naming strategy,
or class-name mapping makes the integration stale and blocks integrated export.
Disable the option if a cohort-only output is intended.

The final table may be exported as CSV or Parquet. With integration enabled, this
is the full-dataset integrated table; otherwise it is the cohort-only table. The
annotated-AnnData action writes a new atomic copy and adds the explicitly named
integrated observation only when integration is enabled and current. In notebook
sessions, **Apply to live AnnData object** performs the same update in memory after
confirmation, without writing the source file.
