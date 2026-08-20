# Cox survival analysis

## What this stage does

The `cox` stage builds one auditable case-level table from one or more AnnData
`.obs` sources, adds encoded clinical covariates, and compares conventional Cox
PH, Ridge Cox, and CoxNet models. It supports:

- population abundances from several analyses, such as Nimbus cell labels and
  HyPERSTAC patch clusters;
- direct case identifiers or ROI-to-case mapping through clinical metadata;
- a sweep over matching population columns, such as every HyPERSTAC Leiden
  setting;
- image-only, clinical-only, and combined clinical-plus-image feature sets;
- univariate feature ranking, correlation diagnostics, regularization paths,
  repeated held-out case-level validation, fitted-risk diagnostics, and
  held-out risk-group plots.

Run it independently with:

```bash
sbt run cox
```

HyPERSTAC uses these outputs when `hyperstac-stability` is requested, but Cox is
not otherwise tied to HyPERSTAC.

## Why it is performed

Spatial analyses often produce several correlated case-level summaries. A
single unpenalized model can be unstable when the number of candidate features
is large relative to cases or events. The stage therefore reports three
complementary Cox views:

- Cox PH is the conventional hazard-ratio reference.
- Ridge Cox retains correlated features and shrinks their effects.
- CoxNet uses elastic-net regularization; the default lasso path
  (`l1_ratio=1`) is suitable for sparse feature-path interpretation.

These models quantify association and risk ranking. They do not establish
causality, and fitted-data separation is not evidence of generalization.

## Main inputs

- `cox.feature_sources`: one or more named AnnData sources.
- A categorical `population_obs`, a `population_obs_search` term, numeric
  `continuous_obs`, or a combination for every source.
- Source `case_obs`, or source `roi_obs` plus clinical ROI-to-case metadata.
- Clinical metadata from `cox.clinical_adata_path` or
  `cox.clinical_csv_path`.
- `case_col`, `duration_col`, `event_col`, optional `roi_col`, and optional
  `covariate_cols`.

If `feature_sources` is empty, the runtime can infer:

- `general.anndata_path` with `general.population_obs_primary`; and/or
- an existing `hyperstac/imc_hyperstac_representations.h5ad` with the configured
  HyPERSTAC cluster-column search.

Explicit sources are recommended for final analyses.

Example combining a cell-level and patch-level feature source:

```yaml
cox:
  clinical_adata_path: anndata.h5ad
  case_col: Case
  roi_col: ROI
  duration_col: Survival_diagnosis
  event_col: Event
  covariate_cols: [Age_at_diagnosis, Sex]
  feature_sources:
    - name: nimbus
      adata_path: anndata.h5ad
      population_obs: [spatial_cluster_Nimbus]
      case_obs: Case
      roi_obs: ROI
    - name: hyperstac
      adata_path: hyperstac/imc_hyperstac_representations.h5ad
      population_obs_search: leiden
      case_obs: null
      roi_obs: roi
```

Only one source may use `population_obs_search` in a run. This prevents an
ambiguous Cartesian product of independent clustering sweeps.

## Reusable assets produced or modified

None. Input AnnData and clinical files are read-only. Case tables, model
summaries, predictions, and plots are human-facing execution results.

## Human-facing outputs produced

The report contains:

- the combined case-level feature table and source/case coverage counts;
- univariate Cox rankings and the selected image-feature list;
- a feature-correlation heatmap;
- per-model coefficient or forest plots and regularization paths;
- fitted risk tables, distributions, risk-versus-survival plots, and
  Kaplan-Meier diagnostics;
- repeated held-out C-index tables, predictions, and held-out risk-group plots;
- a model/feature-set comparison table;
- compatibility tables used by the HyPERSTAC cross-Leiden stability report.

When a population-column search matches several columns, each gets a separate
analysis subfolder and the root contains `survival_all_cluster_summary.csv`.

## Important configuration options

- `models`: any ordered subset of `coxph`, `ridge`, and `coxnet`.
- `feature_sets`: any ordered subset of `image`, `clinical`, and
  `clinical_image`.
- `covariate_cols` is empty by default so the general stage does not assume
  disease-specific clinical fields. Add fields such as age and sex to enable
  clinical-only and combined comparisons.
- `feature_selection_top_n`, `coxph_max_features`, and
  `ridge_max_features` control dimensionality after univariate ranking.
- `ridge_alphas` defines the primary Ridge cross-validation grid.
- `coxnet_l1_ratio=1.0`, `coxnet_n_alphas=200`, and
  `coxnet_alpha_min_ratio=0.001` reproduce the sparse lasso-path analysis used
  for the multi-source comparison.
- `validation_folds=5` and `validation_repeats=10` control repeated held-out
  case validation. Reduce folds when the cohort is small, but retain at least
  two.
- `risk_group_quantiles=[0, 0.33, 0.67, 1]` creates low, middle, and high risk
  groups.
- `censored_case_ids` can derive event status when a legacy table lacks an
  event column. `assume_all_events` should be enabled only when censoring is
  genuinely absent.
- `metadata_conflict=mode` handles repeated cell/ROI metadata. Use `error` for
  strict final auditing.

## How to interpret the results

Use held-out validation first. A mean C-index above 0.5 indicates better than
random risk ranking, while fold/repeat spread shows uncertainty. Compare the
same feature set across model families and compare image-only against clinical
and combined models.

Coefficient direction is conditional on the other included features. Correlated
population fractions can share, suppress, or exchange effects. Ridge paths are
useful for direction stability; sparse CoxNet coefficients identify a compact
view but can select one member of a correlated block. Cox PH confidence
intervals remain valuable but become unreliable when too many features are
fitted for the number of events.

## Common problems and limitations

- Every source must map to the same case outcome. Disagreements stop the run.
- Inner joining sources retains only cases represented in every configured
  source; inspect the case audit before interpreting results.
- Event status and duration must be consistent within case.
- Very small cohorts or folds with too few events can make individual model
  comparisons fail. Failures are recorded; the stage fails if every requested
  model fails.
- One-hot clinical encoding drops the first categorical level, so coefficients
  are relative to that reference.
- Repeated cross-validation estimates internal performance only. Independent
  cohort validation remains necessary for a predictive biomarker claim.
- Cox execution uses the shared, repository-managed `sbt-tensorflow`
  environment because lifelines and scikit-survival are not present in the
  lightweight CLI environment. The external `sbt-hyperstac` environment
  remains available only as an explicit rollback override.
