# AI Interpretation

```{warning}
AI interpretation is a convenience relabelling step, not a validated cell-type
annotation method. Its main purpose is to put rough, readable names beside
numeric Leiden clusters while a scientist reviews the data. Do not use an
`*_AIlabel` column as ground truth, a final biological conclusion, or a basis
for clinical or quantitative claims without independent validation.
```

## What this stage does

Leiden clustering divides a cell graph into groups and gives those groups
arbitrary identifiers such as `0`, `1`, and `2`. Those identifiers say nothing
about the biological identity of the cells. This optional stage asks a large
language model to suggest a short name for each existing Leiden cluster from a
compact summary of its marker expression.

For every observation column whose name begins with `leiden_`, the current
implementation:

1. calculates a cluster-level marker summary;
2. constructs a text prompt containing the supplied tissue description, panel
   marker names, cluster sizes, selected marker statistics, and ROI composition;
3. sends that prompt to an OpenAI model;
4. requests a label, confidence value, rationale, and alternative labels for
   every cluster; and
5. maps the returned label back to all cells in that cluster in a new
   `<leiden column>_AIlabel` observation column.

The stage does **not** change the Leiden clustering or replace its original
column. Every cell assigned to one Leiden cluster receives the same suggested
name. It does not annotate cells individually, split a mixed cluster, merge
related clusters, or correct a poor clustering result.

The result is therefore best understood as an automatically generated alias:

```text
leiden_0.5 = "3"  ->  leiden_0.5_AIlabel = "Macrophage-like"
```

Even if the returned text omits words such as *likely* or *-like*, the identity
remains only a suggestion.

## Why the stage exists

Numeric cluster identifiers are inconvenient when first navigating UMAPs,
heatmaps, tissue overlays, or summary tables. A rough name can make preliminary
review faster by indicating what the cluster might represent and which clusters
deserve attention first. The saved rationale and alternatives can also expose
one possible reading of the marker pattern.

This modest role is intentional. Rigorous annotation requires knowledge that a
general language model cannot obtain from the compact summary used here:
antibody performance, staining controls, segmentation quality, marker
localisation, tissue morphology, expected populations, disease context, and
visual inspection of the actual images.

## Information calculated for each cluster

The input expression matrix is `adata.X`, with `adata.var_names` treated as the
marker names. The code converts the matrix to a dense table, calculates a
global mean and standard deviation for each marker across all cells, and then
calculates the following for every cluster.

### Number of cells

`n_cells` is the number of cells assigned to the cluster. It provides scale but
does not establish that the population is real. Very small clusters are more
sensitive to outliers, debris, segmentation errors, and unusual individual
samples.

### Mean expression

For each marker, `mean` is the arithmetic mean within the cluster. A high mean
may reflect consistently positive cells, a subset of very bright cells, or an
artefact; the summary does not distinguish these possibilities.

### Relative z-score of the cluster mean

The code calculates:

```text
(cluster marker mean - whole-dataset marker mean)
-------------------------------------------------
          whole-dataset marker standard deviation
```

Markers are ranked by this value and only the top eight are included with
detailed statistics in the prompt. This score describes how elevated a cluster
mean is relative to the complete dataset. It is not a per-cell z-score, a test
of differential expression, a p-value, or evidence that a marker defines a
cell lineage.

The ranking is dataset-relative. The same biological population may acquire a
different list of top markers when the abundance or composition of other cell
types changes.

### Fold change against the global mean

`fold_change_vs_global` divides the cluster mean by the mean across all cells.
It is descriptive only. Ratios can become large when the global mean is close
to zero, and no statistical uncertainty or minimum-expression requirement is
applied.

### Percentage positive

`pct_positive` is not based on a panel-specific positivity threshold or a
negative control. For each marker within each cluster, the code finds the 10th
percentile among values greater than zero, then reports the fraction of cluster
cells above that value. Consequently, its threshold can differ between clusters
and markers, and it should not be interpreted as the biological percentage of
marker-positive cells or compared as though every cluster used one common gate.

### ROI composition

When `adata.obs["ROI"]` exists, the prompt includes the proportion of cluster
cells contributed by up to the ten most represented ROIs. This may reveal that
a proposed cluster is dominated by one region, but the model receives no case,
condition, or experimental-design analysis. ROI composition does not determine
whether the concentration is biological or technical.

## What the language model receives

For each detected Leiden resolution, the prompt contains:

- the free-text value of `visualization.tissue`;
- the names of every marker in the panel;
- the Leiden column name;
- the number of cells in each cluster;
- the top eight markers ranked by relative z-score and their four summary
  statistics; and
- the cluster's ROI composition, when an `ROI` column is present.

It does not receive the IMC images, segmentation masks, cell morphology, spatial
coordinates, neighbourhood structure, UMAP coordinates, per-cell expression
profiles, or marker localisation. It also does not receive statistics for low
or absent markers unless a marker happens to enter the top-eight list. Missing
negative evidence is particularly important because many cell identities are
distinguished by combinations of present **and absent** lineage markers.

The prompt asks for a JSON array with `cluster`, `label`, `confidence`,
`rationale`, and `alt_labels`. The current adapter calls the model named
`gpt-4o-mini` at temperature `0.2`. The model name, temperature, prompt wording,
top-eight limit, ROI key, and selected Leiden resolutions are fixed in the
current code rather than exposed as configuration options.

Lower temperature may reduce variation in wording, but it does not make a
suggestion biologically correct, reproducible across model updates, or
statistically validated.

## Why the proposed labels are unreliable

A plausible label can still be wrong. Important causes include:

- **Limited evidence.** Only aggregate statistics for eight relatively elevated
  markers are presented in detail. A decisive lineage marker may be absent from
  the panel, fail staining, or fall outside this list.
- **Cluster impurity.** One Leiden cluster can contain several cell states or
  types. One label assigned to all its cells conceals that heterogeneity.
- **Upstream errors.** Spillover, background, denoising artefacts, failed
  segmentation, doublets, batch effects, normalisation, and the selected graph
  or Leiden resolution all affect the summary.
- **Contextual bias.** The tissue description guides the answer. An inaccurate
  description can push the model towards an expected but unsupported identity.
- **Terminology ambiguity.** The code imposes no controlled vocabulary or cell
  ontology. Closely related labels, state labels, lineage labels, and informal
  synonyms may be mixed across clusters or runs.
- **No biological consistency checks.** The response is not checked against
  marker rules, mutually exclusive identities, known controls, morphology, or
  spatial localisation.
- **No calibrated confidence.** `confidence` is text generated by the same
  model. It is not derived from replicate agreement, classification accuracy,
  posterior probability, or a validation set. A value of `0.9` must not be read
  as a 90% probability that the label is correct.
- **Response variability.** Provider-side model changes and generative sampling
  can change labels, rationales, or spelling even when the input data are
  unchanged.

The stage also performs only minimal parsing of the returned JSON. It does not
require one valid and unique response per real cluster, validate confidence
values, or verify that the proposed cluster identifiers exist. Cells whose
cluster has no usable returned label remain missing in the `*_AIlabel` column.

## Appropriate and inappropriate uses

Reasonable uses include:

- making an initial UMAP or tissue plot easier to navigate;
- choosing clusters for expert review;
- generating tentative hypotheses to check against marker plots and images; and
- providing temporary names during exploratory discussion.

Do not use the generated labels, without a separate validated annotation step,
to:

- report final cell-type frequencies or differential abundance;
- merge, remove, or biologically redefine populations automatically;
- support publication claims, clinical decisions, or diagnostic conclusions;
- create ground-truth or machine-learning training labels; or
- substitute for review by scientists familiar with the tissue, panel, and
  experiment.

## How to validate and replace the rough labels

Review each proposed population as if the AI text were an unverified hypothesis:

1. inspect expression heatmaps or matrix plots for both positive and negative
   lineage markers, not only the model's highlighted markers;
2. examine distributions at cell level to detect bimodality, outliers, and
   mixed populations hidden by means;
3. backgate cells to the source images and check morphology, localisation,
   segmentation, and staining quality;
4. examine representation across ROIs, specimens, batches, and biological
   groups so that one technical region is not mistaken for a population;
5. compare neighbouring Leiden resolutions and determine whether the identity
   is stable or whether a cluster should be split or merged; and
6. have an appropriately experienced scientist assign a separate curated label
   with an agreed vocabulary and recorded evidence.

Retain the original Leiden columns and keep curated annotations distinct from
`*_AIlabel`. This preserves the difference between a computational community,
an AI-generated suggestion, and a reviewed biological identity.

## Configuration

The active settings are in the `visualization` section:

- `enable_ai`: when `true`, permit the external model call and creation of rough
  labels. An `OPENAI_API_KEY` environment variable is required.
- `tissue`: free-text tissue context included in the prompt. Keep this accurate,
  concise, and free of patient-identifying information.
- `repeat_ai_interpretation`: when `false`, the stage skips if **any** AnnData
  observation column ends in `_AIlabel`; when `true`, it attempts to run again.

There is no configuration setting in the active path for a label vocabulary,
marker rules, confidence threshold, model, prompt, top-marker count, or selected
Leiden resolutions.

The automatic Leiden discovery expects numeric column names of the form
`leiden_<resolution>`, for example `leiden_0.5` or `leiden_1.0`. Other columns
that begin with `leiden_` but do not end in a number can cause discovery to
fail. Existing `leiden_<resolution>_AIlabel` columns also match that broad
prefix test, so forced re-interpretation of an already labelled AnnData object
may fail unless those columns are handled before the run. This is a current
implementation limitation, not a biological requirement.

## Outputs and audit trail

For each successfully processed Leiden resolution, the stage writes:

- `<leiden key>_prompt.txt`: the exact prompt sent to the model;
- `<leiden key>_raw.json`: the returned text as received, even if it was not
  valid JSON;
- `<leiden key>_interpretation.tsv`: cluster, suggested label, generated
  confidence, rationale, and alternative labels; and
- `adata.obs["<leiden key>_AIlabel"]`: the suggested label copied onto each cell
  according to its Leiden cluster.

The confidence, rationale, and alternative labels remain in the TSV and are not
copied to individual cells. The updated AnnData is saved back to the AnnData
path loaded for the stage, with the original Leiden columns retained.

Prompt and response files are valuable for audit, but they do not establish
correctness. Record the model call alongside the final manual decisions so that
future readers can distinguish generated suggestions from accepted labels.

## Privacy and operational considerations

This is an external API operation. The code sends aggregate cluster statistics,
panel marker names, the tissue description, and ROI category names in the
prompt. It does not deliberately send raw images or the full cell-by-marker
matrix, but ROI names or free text can still contain sensitive or identifying
information. Review local data-governance requirements and pseudonymise those
values before enabling the stage.

The API key is read from `OPENAI_API_KEY`; it should not be written into the YAML
configuration, notebooks, prompt files, or repository. Network, authentication,
provider, or response-format failures can leave a resolution without labels.
Because this stage is optional and its labels are provisional, such a failure
should not prevent expert annotation from proceeding from the original Leiden
results.
