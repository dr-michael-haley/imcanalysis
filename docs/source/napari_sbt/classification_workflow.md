# From example cells to final identities

Use classification when you want to teach a model to distinguish cell categories.
If you only need a hand-picked cell list, use [Labeler](labeler.md). If you only
need to rename whole existing clusters, start with [Population naming](populations.md).

## 1. Decide which cells the task covers

Create a workspace with **Train a cell classifier**. Define the cell scope in
Workspace: all cells, or selected values from an existing population column.
Define two to eight classes with distinct names and colours. Check the displayed
cell and ROI counts, then complete the required full dataset check.

## 2. Build the measurements

In [Feature Building](feature_building.md), select relevant channels and features
or connect existing measurements. Run the build and wait for saved feature
readiness to report **Ready for classification**.

Use image features when intensity distribution, morphology, or local context
matter. Imported expression or embedding features can be useful when those
measurements already exist. The feature descriptions explain what each measures.

## 3. Confirm representative examples

In [Classify](classify.md), choose a class and click behaviour, inspect staining,
and label clear examples. A proposal is a working choice; confirmation makes it
a training example. Monitor the tally and sample several ROIs for every class.

## 4. Train, score, and review

Train the model, then score the cohort. Review predictions in the tissue and use
the ambiguity queue to find uncertain cells. Correct or add confirmed examples,
retrain, and score again. The confidence display controls which predictions you
see; it does not decide what will be exported.

## 5. Create final identities

Open **Finalize & export**. Set the confidence, uncertainty, and probability-margin
rules, then **Create / refresh final cell identities**. Check accepted,
confirmed, and unassigned counts.

For a subset experiment, optionally integrate accepted identities into an
existing population column. Cells outside the cohort retain their original labels.
Review the naming table and label-overlap preview, then build the integrated labels.

## 6. Export and inspect

Export the current final table or an annotated AnnData copy. With integration
enabled, the output contains the combined whole-dataset labels; otherwise it is
the cohort result. Check the output summary and filename before using it downstream.

## When to trial features first

For an expensive feature set, create a **Feature Discovery Trial** with a
representative selection of ROIs. Build broad features, confirm examples across
ROIs, then use [Feature Refinement](feature_refinement.md). Promotion selects a
smaller feature recipe for a subsequent full-cohort build. Trial performance is
an exploratory estimate, so review predictions on independent tissue too.
