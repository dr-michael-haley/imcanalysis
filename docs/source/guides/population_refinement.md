# Exemplar-guided population refinement

`SpatialBiologyToolkit.population_refinement` splits a selected AnnData population
using explicit exemplar classes and image features. It runs without Napari or Qt.
MaxFuse supplies candidate labels and sampling priorities; manual tables and custom
source adapters use the same interfaces. Input AnnData, masks and images are read-only.

The executable example is `Tutorials/Population_refinement.ipynb`. The notebook
contains a small synthetic example that runs without external data. Replace its
setup cell with a live AnnData and the image/mask paths for an actual analysis.

## Scientific contract

Specify two to eight mutually exclusive classes for each parent population.
`ClassSpec.source_labels` maps one or more MaxFuse labels to a class. Labels outside
that mapping do not train a classifier. They remain in the target cohort and audit.
Small excluded groups are not assumed to be errors or merged into an "other" class.

Use matching scores as priorities, not calibrated probabilities. By default,
sampling is weighted by within-class, within-stratum score ranks. It accepts
negative scores and does not require a universal cutoff. `candidate_fraction`
restricts the eligible upper score pool; `minimum_score` is an optional explicit
floor. Relative ranks cannot establish that a poorly matched stratum has good
exemplars: inspect coverage, absolute scores and image evidence before fitting.

Balanced hierarchical allocation across `("Case", "ROI")` first balances cases,
then ROIs within each case. Proportional allocation, uniform or top-ranked sampling,
per-class budgets and per-stratum caps are also supported. Sampling uses no
replacement, has an explicit seed, is independent of input row order, and reports
shortfalls. Eligibility requirements are never silently relaxed to fill a quota.

## Marker rules are independent

Each `MarkerRule` refers to an explicitly named evidence column, which can contain
an AnnData measurement or an extracted image feature. Negative evidence uses an
upper bound. Threshold units belong to the supplied representation; do not mix
raw intensity, Nimbus-normalized values and display scaling.

| Role | Behaviour |
|---|---|
| `required` | Reject that class's exemplar if its interval check fails |
| `supportive` | Increase sampling priority when the check passes; observed failure is not a veto |
| `feature_only` | No agreement requirement; document a marker used for prediction |

Weights control the relative contribution of supportive checks. Missing per-cell
measurements can explicitly reject the candidate or be ignored. Missing evidence
columns are configuration errors for required/supportive checks. A feature-only
rule requires no evidence column. Rules do not automatically select classifier
columns: the final feature list is explicit and recorded separately.

For example, require a myeloid marker, give moderate support to a second marker,
and allow morphology or a state marker to inform prediction without gating seeds.
No rule requires every measured marker to agree with a transferred annotation.

`rank_reference_markers` proposes mapped image channels from pairwise RNA contrasts
between the chosen classes. Use unique reference cells and an explicit expression
layer. The supplied gene-to-channel mapping and channel availability restrict the
result. These are candidate markers, not claims of protein specificity. Custom
marker tables and manually specified rules are equally valid inputs.

## Notebook API

```python
from SpatialBiologyToolkit import population_refinement as pr

spec = pr.SplitSpec(
    population_key="leiden", populations=("mixed",),
    classes=(
        pr.ClassSpec(class_id="myeloid", name="Myeloid",
                     source_labels=("Myeloid",)),
        pr.ClassSpec(class_id="glial", name="Glial",
                     source_labels=("AC-progenitor-like",)),
    ),
)
source = pr.MaxFuseSource(label_key="atlas_label", score_key="maxfuse_score")
candidates = source.candidates(adata, spec)

# An external transfer table must be indexed by unique obs_name, not row position.
# Use TableSource(table, class_key="class_id") for manual/custom labels.
evidence = pr.expression_evidence(adata, ["IBA1", "SOX2"])
rules = (
    pr.MarkerRule(class_id="myeloid", feature="IBA1", role="required", minimum=0.2),
    pr.MarkerRule(class_id="glial", feature="SOX2", role="supportive", minimum=0.2),
)
# Bounds above illustrate syntax; establish appropriate units/bounds for your data.
selection = pr.select_exemplars(
    candidates, class_ids=spec.class_ids, evidence=evidence, rules=rules,
    sampling=pr.SamplingSpec(strata=("Case", "ROI"), per_class=500, seed=42),
)
display(selection.coverage)
print(selection.warnings)
```

For image agreement, extract a small initial image recipe, index the feature table
by `obs_name`, and supply its actual column names in marker rules. A second feature
experiment can then build a broader recipe. Freeze the target cohort per ROI in
both experiments; sampling only exemplars during extraction changes the meaning
of cohort-relative features.

```python
experiment = pr.create_feature_experiment(
    adata, spec, directory="refinement_assets/cluster_mixed",
    images=["Images"], masks="Masks",
    recipe=pr.image_recipe(["IBA1", "SOX2"],
                           normalization_dict_path="normalization_dict.csv"),
)
features = pr.build_feature_table(experiment, workers=2)
columns = pr.image_feature_columns(features)
validation = pr.evaluate_refinement(
    features, selection, spec, group_key="Case",
    feature_sets={"means": pr.image_feature_columns(features, means_only=True),
                  "image": columns},
)
result = pr.fit_refinement(features, selection, spec, feature_columns=columns)
writer = result.save("refinement_review")
validation.save(writer)
integrated = result.integrate(adata, output_key="refined_population")
```

`create_feature_experiment` saves the existing NapariSBT experiment schema, frozen
cohort and channel aliases from live AnnData. It requires no intermediate H5AD.
Reusing an unchanged directory resumes compatible feature fragments. Changed
cohorts, recipes, classes or sources require a new workspace. Extraction uses full
segmentation masks for boundaries/background, but returns only target-cell rows.

For managed extraction, set `napari_sbt.active_experiment` to this workspace and
use `sbt run cellfeat`. The existing stage supplies planning, environment capture,
reporting and local/SLURM execution. These image builds are substantial scientific
work; run them on a workstation or compute node. CPU and memory scale with the
number of concurrent ROI workers and image dimensions. No GPU or new environment
is required. The notebook API uses the existing analysis environment.

## Validation and feature selection

Hold out entire cases where possible. `evaluate_refinement` compares the supplied
feature sets and existing Random Forest/HistGradientBoosting implementations. It
reports per-class metrics, confusion tables, predictions, acceptance coverage and
the exact training groups/features for each fold. Groups lacking enough training
classes are reported as skipped; no valid fold is an error.

Optional `refine_features=True` runs existing permutation-based feature refinement
inside each outer training fold, with the specified group key. This is more
expensive and needs at least three represented groups. The final compact feature
set can be selected on development groups using
`cell_classification.feature_refinement.refine_trial_features(..., examples=...,
group_column="Case", feature_columns=...)`. Keep any independent final review
groups outside all learned marker rules, selection tuning and feature refinement.

Metrics estimate agreement with held-out exemplar labels. MaxFuse and marker gates
share evidence, so these metrics do not establish independent biological accuracy.
Compare mean-only and richer-image baselines, vary sampling seeds, inspect performance
by case, and review random held-out cells plus disagreements and rejected cells.
High-confidence training examples alone cannot validate the whole parent population.

## Assignments and provenance

Automatic examples never receive human-confirmed overrides. They pass the same
prediction acceptance criteria as other cells. Defaults check maximum probability,
entropy, probability margin, finite-feature coverage and the fraction of features
outside fitted training quantile bounds. These are configurable starting values,
not calibrated guarantees. Training-range checks do not reliably detect all unseen
cell types; restricted-class models can be confidently wrong.

Unassigned target cells keep their parent label when integrated. Cells outside the
cohort keep their source labels. `result.integrate()` returns a full-dataset table;
`apply=True` adds a new in-memory observation and refuses an existing output name.
No method overwrites the source H5AD. Persist an explicitly chosen annotated copy
through the toolkit's AnnData writer after review.

For a complete partition, use `AssignmentPolicy(assignment_mode="complete")`.
Every scorable target receives its highest-scoring class, including cells below
the acceptance thresholds. `assignment_needs_review`, `model_prediction_accepted`
and `prediction_rejection_reason` retain the original diagnostics; a forced
assignment has `assignment_source="model_forced_review"`. Missing predictions
raise an error instead of silently retaining the parent or inventing a label.
These scores are model outputs, not calibrated probabilities of biological truth.

To prioritise exemplar quality before representation, set
`SamplingSpec(quality_pool_scope="class", candidate_fraction=0.25,
strategy="top_ranked")`. This filters each class to its strongest score quartile
before allocating case/ROI quotas. Required marker rules apply first. Cases or
ROIs without eligible cells receive no exemplars; unused quotas do not relax the
criteria. Use this only when source scores are meaningfully comparable across
strata. The default `quality_pool_scope="stratum"` retains local score filtering.

`result.save()` uses `PopulationQCArtifactWriter` for candidates, exemplars,
per-rule checks, sampling coverage, scores, assignments and settings. The model
bundle records input fingerprints, feature support ranges, class definitions,
random seed and package versions. Model assets remain reusable artifacts; the
writer maintains the standard artifact manifest for human review.

The feature workspace can be opened in NapariSBT. For review of final proposals,
apply the new observation to live AnnData and open it using `launch_notebook`.
Automatic training examples are not inserted into the GUI's confirmed-label table.
The automatic model is saved separately from `classifier_latest.joblib`, preserving
the GUI's manual-training and staleness semantics.

## Extension points and compatibility

Implement `ExemplarSource.candidates(adata, spec)` to supply another label source.
`TableSource` accepts externally sampled labels already indexed by cell identity.
Alternative samplers can return `SelectionResult`; fitting and evaluation consume
the same selected-example contract. Marker rules, image recipes, class definitions,
sampling and classifier selection are separate choices.

The canonical feature kernels, catalogue, classifier and feature-refinement code
now live in `SpatialBiologyToolkit.cell_classification`. Existing
`napari_sbt.features`, `classifier`, `feature_catalog` and `feature_refinement`
imports remain compatibility paths. Workspace storage and worker orchestration
continue to use the existing headless NapariSBT services. Saved class/model formats
are preserved; no second feature engine or classifier implementation is maintained.
New fits apply class balancing once (previous Random Forest/LightGBM fits combined
balanced estimator weights with balanced sample weights). Existing saved estimators
are loaded unchanged. Infinite measurements are treated as missing values, and
training rejects examples with no finite predictor values.
