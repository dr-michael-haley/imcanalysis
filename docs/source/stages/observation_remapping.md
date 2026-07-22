# Observation Remapping

## Purpose of the stage

Observation remapping converts one existing categorical cell annotation into one or more curated annotations using an explicit CSV lookup table. Common uses include:

- replacing numerical Leiden clusters with biological population names;
- merging several fine clusters into a broader lineage;
- recording both a fine cell type and a broad compartment from the same source labels;
- translating historical names into a consistent project vocabulary; and
- attaching an independently reviewed label while preserving the original clustering column.

The operation is deterministic and transparent. It does not inspect individual cells when applying labels, alter marker values, recalculate clusters, or infer biology. Every cell with the same source value receives the same mapped value. The scientific reasoning lives in the reviewed CSV, which should be retained as a reusable and auditable project asset.

This is different from subclustering. Subclustering creates new within-population partitions from marker values; observation remapping only renames, merges, or hierarchically organises categories that already exist.

## Two-stage human curation workflow

### Generate a template

With `mode: generate_blank`, the stage reads the configured `source_obs` and writes one CSV row for each non-null source category. The first column is the source key. Blank target columns are then added for the scientist to complete. If no target names are configured, the default is `<source_obs>_label`.

The generated table can include three types of helper information:

- total cell count for each source category;
- markers with the largest mean difference between that category and all remaining cells; and
- a normalised measure of how evenly the category is distributed across ROIs.

Human note columns can record uncertainty, evidence, alternative names, or follow-up checks. Helper and note columns are ignored when the remap is later applied, provided their names match the configured ignore rules.

By default, regenerating a template preserves existing edited values matched by source key. This is useful when clusters change incrementally or helper statistics are refreshed. Additional non-helper columns from the old file are also retained; review them because, unless explicitly ignored, they become target observations in apply mode.

### Apply the reviewed mapping

With `mode: apply`, the first CSV column identifies the source observation and all remaining non-ignored columns become new `adata.obs` annotations. The first-column header is used as `source_obs` when no explicit value is configured. If `source_obs` is configured, it must match the first header, preventing accidental application of a table designed for another clustering.

Each target column is applied independently. This allows a single table such as:

| leiden | population_fine | population_broad | notes |
|---|---|---|---|
| 0 | CD8 T cell | Lymphoid | CD3+ CD8+ |
| 1 | Macrophage | Myeloid | CD68+ CD163+ |

to produce both a detailed and a broad biological annotation while keeping the original `leiden` values. The `notes` column is retained in the CSV for provenance but ignored by the default name-fragment rule.

By default the stage refuses to overwrite an existing observation column. Enabling overwrite is a deliberate destructive annotation change and should be used only when the replacement table has been checked and the original annotation remains recoverable.

## Understanding the template helpers

### Cell counts

`n_cells` is the number of cells carrying each source label across the loaded AnnData. It helps identify tiny clusters that need especially cautious interpretation, but it is not a prevalence estimate across independent samples. A large count can be dominated by one large ROI or one case; a small but consistent population can be more biologically credible than a larger population confined to a single technical batch.

### Top-marker hints

For each category, the stage calculates the mean of each marker in that category minus the mean in all other cells, then lists the markers with the largest differences. This is a simple descriptive ranking. It does not perform a statistical test, calculate a P value, control for patient or ROI, or require a positive/large effect.

The ranking is sensitive to the selected matrix and its scale. It can use `adata.X`, `adata.raw.X`, or an explicit layer. Markers with larger numeric ranges can dominate if values are not comparably transformed. The “rest” group may also combine biologically unrelated cell types, making broad lineage markers rank above subtle state markers. Treat the list as a prompt for reviewing matrix plots, UMAPs, and source images—not as enough evidence to name a population.

An optional variable annotation can provide human-readable marker names in place of `var_names`. Missing or blank display names fall back to the original variable name.

### ROI-distribution evenness

The ROI helper uses normalised Shannon entropy. A value near 0 means that the population's cells are concentrated in one ROI; a value near 1 means they are distributed equally across all ROIs represented in the dataset. Even spread across only a subset of all ROIs produces an intermediate value.

This metric is a warning and context signal, not a biological quality score. It does not account for ROI area, total cells sampled per ROI, case identity, group assignment, or whether a population is expected to be localised. A tertiary lymphoid structure, tumour-restricted clone, or treatment-specific population can be genuinely concentrated. Conversely, a technical staining artefact may recur evenly across a batch. Rare populations also have unstable evenness estimates.

## Mapping keys and data types

CSV software and pandas can represent cluster keys inconsistently—for example `1`, `1.0`, and the text `"1"`. Source observations whose names contain `leiden` automatically use integer-like string normalisation, and the same behaviour can be enabled for other sources with `force_string_mapping`. Whitespace is stripped and integer-like values are converted to a common string key.

This solves common spreadsheet conversions but can also collapse labels that were intended to differ only by numeric formatting. Non-integer floating-point keys otherwise use their normal string representation, which can be fragile. Stable, explicit text labels are preferable for reusable curation tables.

Duplicate source keys in an applied CSV are rejected, even when their target values agree. One row must unambiguously describe each source category. Blank source keys are ignored.

## Missing mappings and categorical output

For a target column, a cell is considered unmapped when its non-null source key has no row or when the corresponding target cell in the CSV is blank. With `require_complete_mapping: false`, the stage warns and writes missing values for those cells. With it enabled, the stage fails before that incomplete target column is accepted.

For final biological annotations, complete mapping is usually the safer choice. Applying a freshly generated but unedited template with permissive completeness can create an entirely missing target annotation. Always inspect the report's missing counts and examples.

When `set_output_as_categorical` is enabled, target values are stored as pandas categorical observations. Category order follows the first occurrence of each non-null target value in the CSV, giving the curator control over legend and plot order. Blank strings become missing values rather than literal empty categories.

## Biological curation principles

Cluster-to-population naming should integrate several types of evidence:

- coordinated lineage and state-marker expression;
- source-image staining and cell morphology;
- segmentation quality and possible cell mixing;
- distribution across ROIs, cases, and batches;
- tissue location and expected spatial context; and
- stability across reasonable clustering or preprocessing choices.

Avoid assigning a highly specific identity when the panel supports only a broader lineage. It is often more honest to retain labels such as “T cell, unclassified” or “myeloid, uncertain” than to force every cluster into a canonical subtype. Multiple target columns make that uncertainty manageable: a confident broad annotation can coexist with a provisional fine annotation.

Merging labels changes downstream abundance and spatial results. A broad label increases cell counts and statistical power but can conceal functionally distinct subsets; a very fine label can fragment evidence and create unstable spatial associations. Choose a vocabulary suited to the measured panel, cohort size, and downstream scientific question.

## Provenance and outputs

In generation mode, the primary output is the reusable CSV and the AnnData is not modified. In apply mode, the loaded AnnData is saved back to its input path with the new observation columns. The stage report records:

- mapping and source paths;
- applied and ignored columns;
- key-normalisation behaviour;
- overwrite and completeness settings; and
- missing-mapping counts and examples.

The stage intentionally processes the external CSV even when the normal AnnData stage-run policy suggests skipping, because a curator can change the table without changing pipeline configuration. This makes it important to version or archive the table alongside the resulting AnnData.

## Limitations and common failures

- The method cannot assign different target labels to cells sharing the same source category.
- It cannot repair an incorrect or internally heterogeneous source clustering.
- Helper marker rankings are descriptive cell-level summaries, not differential-expression evidence.
- ROI evenness ignores tissue area, sampling effort, case structure, and expected localisation.
- Incomplete mappings become missing values unless strict completeness is enabled.
- Existing target annotations are protected unless overwrite is explicitly allowed.
- Relative CSV paths resolve from the process working directory, so execution from an unexpected location can select the wrong file.
- Spreadsheet programs can alter numeric-looking identifiers; stable text keys reduce this risk.

Observation remapping is most reliable when the CSV is treated as a scientific decision record: explicit, reviewable, versioned, and traceable to the plots and images used to justify each label.
