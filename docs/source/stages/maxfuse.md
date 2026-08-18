# MaxFuse cross-modality matching

## What this stage does

`maxfuse` matches one scRNA-seq reference to one target IMC AnnData using weakly
linked protein-gene features. It transfers configured reference annotations to
target cells, preserves target-unique match identities and scores, and creates
an execution report adapted from the earlier GBM MaxFuse notebooks.

Run it with:

```bash
sbt run maxfuse
```

The stage is CPU- and memory-intensive and runs through SLURM. It is not a
workflow-mode member and has no fixed upstream stage: either AnnData input may
come from an SBT stage or a separately curated analysis.

## Why it is performed

IMC measures a small protein panel while scRNA-seq measures a much larger gene
space. MaxFuse uses a set of mapped protein-gene links for initial alignment,
then refines those matches using the richer active spaces within each modality.
The resulting transfers can provide atlas evidence for interpreting IMC
populations and can link selected IMC cells to reference transcriptomes.

MaxFuse scores are matching similarities. They are not calibrated probabilities
or proof of a biological identity.

## Main inputs

- `maxfuse.reference_adata_path`: exactly one scRNA-seq reference AnnData.
- `maxfuse.target_adata_path`, falling back to `general.anndata_path`: the
  target IMC AnnData.
- `maxfuse.feature_mapping_path`: a CSV containing the configured target and
  reference feature columns, defaulting to `IMC` and `snRNAseq`.
- Reference annotation columns listed in `maxfuse.reference_transfer_obs`.
- The target population column selected by `maxfuse.target_population_obs`,
  falling back to `general.population_obs_primary`.

Observation and feature names must be unique. Mapped features must exist in
both inputs and pass the configured variability filters. At least
`min_shared_features` links must remain before matching starts.

## Reusable assets produced or modified

The input AnnData files are read-only. Reusable outputs are written beneath
`maxfuse.asset_folder` (default `maxfuse_results`):

- `maxfuse_matches.csv.gz`: target-unique reference/target indices, stable
  observation names, match source, score, legacy modality-index aliases, and
  transferred annotations;
- `maxfuse_transfer.h5ad`: annotation-only target-indexed AnnData containing
  `<reference>_maxfuse_score` and transferred label columns, with MaxFuse
  parameters and feature provenance in `.uns`;
- `feature_mapping_used.csv`: the exact weak links retained after input and
  variability checks;
- optional `maxfuse_matched_transcriptomes.h5ad`, enabled explicitly for
  confident matches intended for `MaxFuseSCRNASeq`.

The canonical match and transfer assets retain all matches returned after
MaxFuse pivot/propagation filtering. The reporting threshold does not remove
rows from those assets.

## Human-facing outputs produced

The managed execution report contains:

- an annotated target-population versus transferred-reference concordance
  heatmap and its count/fraction tables;
- an annotated mean MaxFuse score heatmap by target population;
- three-panel target and reference UMAPs showing original labels, transferred
  labels, and matching score;
- a linked-gene expression matrix and stacked-violin companion grouped by the
  transferred target population;
- Wilcoxon one-versus-rest reference RNA DEG tables and top-gene figures;
- optional supplied-gene-list annotation, legend, overlap table, and
  hypergeometric enrichment heatmap;
- score distribution and post-hoc threshold-coverage diagnostics;
- population, sample, and ROI coverage/score summaries when those observation
  columns are available;
- the full feature-mapping audit and per-phase runtime table.

UMAP rendering uses deterministic stratified downsampling when more than
`max_umap_points` matched cells are available. This affects only the figure; the
tables and reusable assets use all cells.

## Important configuration options

The defaults reproduce the successful spatial-omics GBM recipe: cyclic
batching, 1,000 reference active features, metacell size 2, 15-neighbour
graphs, strong initial/refinement smoothing (`0.3` raw-data weights), one
refinement iteration, 0.5 pivot filtering, and 0.3 propagated filtering.

```yaml
maxfuse:
  reference_name: gbm_atlas
  reference_adata_path: maxfuse/gbm_atlas.h5ad
  target_adata_path: null
  feature_mapping_path: maxfuse/feature_mapping.csv

  reference_smoothing_obs: annotation_level_3
  reference_transfer_obs:
    - annotation_level_3
    - annotation_level_4
  target_population_obs: leiden_1.0

  reference_active_features: 1000
  reference_shared_sd_min: 0.20
  target_shared_sd_min: 0.025
  min_shared_features: 20

  batching_scheme: cyclic
  max_outward_size: 2000
  matching_ratio: 4
  metacell_size: 2
  refine_iterations: 1
  pivot_filter_fraction: 0.5
  propagated_filter_fraction: 0.3

  report_score_threshold: 0.30
  gene_lists:
    - path: genelists/neftel.csv
      name: neftel
      format: wide
    - path: genelists/custom.tsv
      name: custom
      format: long
      group_column: programme
      gene_column: gene
```

Historical wide gene-list files use one group per column. Long files must
contain the configured group and gene columns. If no gene lists are supplied,
the ordinary DEG analysis still runs.

## Environment and resources

The stage uses the dedicated `sbt-maxfuse` environment with Python 3.10 and
`maxfuse==0.0.2`. MaxFuse itself supports Python 3.8 and newer, but current SBT
requires Python 3.10 or newer.

The wrapper initially requests 8 CPUs, 128 GB RAM, and 24 hours on the CPU
high-memory partition. Prior million-cell GBM runs took approximately 35–47
minutes, but memory and runtime depend on cell counts, active dimensions, and
batching. Review SLURM MaxRSS and elapsed time before reducing the allocation.

## How to interpret the results

Start with target coverage, the score distribution, and the annotated
concordance heatmap. Look for target populations with coherent transferred
labels and adequate representation across cases and ROIs. Then compare the
linked-gene matrix and reference RNA DEGs with known biology and supplied gene
programmes.

The default report threshold of 0.30 follows the historical plotting notebook.
It is a visualization/DEG inclusion threshold, not a universal definition of a
correct match. Use the threshold-sensitivity table to see how conclusions
depend on that choice.

## Common problems and limitations

- Fewer than 20 retained links usually indicates a mapping, preprocessing, or
  feature-naming problem.
- Requested SVD/CCA dimensions must not exceed the available active or linked
  feature dimensions.
- Missing `X_umap` keys skip the relevant UMAP figure without invalidating the
  matching assets.
- DEG analysis assumes the selected reference layer contains values suitable
  for Scanpy's Wilcoxon comparison.
- One reference cell may annotate several target cells. Reference-space plots
  and DEGs use the best-scoring target projection per reference cell to avoid
  duplicate transcriptomes.
- The optional matched-transcriptome asset can become very large because
  repeated reference matches are materialized under target cell identities.
- No multi-reference execution or automatic parameter scan is currently
  implemented.
