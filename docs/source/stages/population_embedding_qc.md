# Population embedding and clustering QC

## What this stage does

This stage assesses how strongly each reference cluster is supported by the existing neighbour graph, stored UMAP, optional stored PCA, and optional precomputed Leiden resolution sweep. It reports raw structural metrics, normalized QC concern scores, independent raw-threshold flags, competitors, transitions, figures, and deterministic per-cluster interpretations.

It never recalculates Leiden clustering, PCA, UMAP, or Scanpy neighbours. It does not perform marker assessment or differential expression and does not claim biological validity.

## Why it is performed

UMAP appearance alone can overstate or hide population separation. This stage brings graph separation, embedding separation, embedding reliability, and resolution stability together while keeping those evidence groups distinct. It helps prioritize populations requiring further QC.

## Main inputs

- An AnnData `.h5ad` selected by `population_embedding_qc.input_adata_path` or `general.anndata_path`.
- A required finite UMAP in `adata.obsm[population_embedding_qc.umap_key]` with at least two dimensions.
- A reference clustering in `adata.obs`, selected explicitly or automatically from `population`, `leiden`, or the median detected sweep resolution.
- Optional PCA in `adata.obsm`, optional connectivity graph in `adata.obsp`, and optional precomputed Leiden columns such as `leiden_0.2` and `leiden_0.5`.

Missing PCA or graph evidence is reported and skipped rather than reconstructed. Cells with missing reference labels are excluded and counted; small clusters remain in all summaries.

## Reusable assets produced or modified

The input AnnData is never modified in place. By default, no reusable asset is created or changed. If `write_annotated_h5ad` is enabled, a separate copy is written to `annotated_adata_path` with namespaced `embedding_qc_*` observation columns and `adata.uns["population_embedding_qc"]`. Existing files and keys are not overwritten silently.

## Human-facing outputs produced

Managed runs write through the active execution report categories:

- `tables/`: raw cluster metrics, concern scores, raw threshold flags, cluster summary, competitors, graph connectivity, UMAP density overlap, metric definitions, and sweep tables.
- `figures/`: compact and detailed concern heatmaps, cluster overview, UMAP panels, boundary plot, competitor/density/component/silhouette summaries, and sweep transition/stability figures.
- `summaries/analysis_report.md`: deterministic interpretation and limitations.
- `files/`: metric configuration and structured run summary/provenance.

The standalone CLI uses analogous `Figures/`, `Tables/`, `Report/`, and `Run/` directories below `--output-dir`.

## Important configuration options

`mode` accepts `auto`, `single`, or `sweep`. In managed runs `population_obs` falls back to `general.population_obs_primary`. In auto mode an explicit reference wins; otherwise `population`, `leiden`, or the median numerical sweep resolution is selected. `sweep_regex` must include a named `resolution` group, and `sweep_columns` can provide an explicit list.

`umap_k`, `pca_dimensions`, `min_cluster_size`, component/purity/persistence thresholds, deterministic sampling limits, and `random_seed` control calculations. Optional `sample_obs` and `roi_obs` columns add represented-sample/ROI counts to cluster summaries and heatmap row annotations. `metric_config_path` may point to YAML or JSON overrides for metric anchors, thresholds, heatmap inclusion, and weights.

Example pipeline configuration:

```yaml
population_embedding_qc:
  enabled: true
  mode: auto
  population_obs: population
  umap_key: X_umap
  pca_key: X_pca
  umap_k: 15
  min_cluster_size: 20
  silhouette_max_cells: 10000
  density_max_cells_per_cluster: 5000
  write_per_cell_metrics: false
  write_annotated_h5ad: false
  random_seed: 42
```

Run the managed stage with `sbt run popqc`. Run a generic file independently with:

```bash
python -m SpatialBiologyToolkit.population_embedding_qc \
  --input project.h5ad \
  --output-dir population_embedding_qc \
  --population-obs population \
  --mode auto
```

## How to interpret the results

The main heatmap has reference clusters as rows and structural QC metrics as columns. Colour is a normalized concern score from zero (low concern) to one (high concern). Grey cells are unavailable evidence, not zero concern. A cyan `X` is calculated independently from the raw value and marks a configured starting threshold failure.

Metric glossary:

- Graph impurity/boundary/conductance/entropy/component loss: local and cluster-wide graph mixing or fragmentation.
- Strongest competitor fraction: the most prominent external graph relationship.
- UMAP impurity, silhouette, isolation, density overlap, and components: separation visible in the stored two-dimensional embedding.
- PCA silhouette: separation in the existing higher-dimensional PCA representation.
- UMAP-graph preservation: agreement between UMAP and source-graph neighbourhoods.
- Sweep Jaccard, retention, persistence, split/merge entropy, consensus, and external co-assignment: stability across precomputed resolutions only.

Concern anchors and thresholds are configurable QC starting points. Group scores report their contributing metric count and available weight; missing evidence is never treated as reassuring zero concern.

## Common problems and limitations

- A missing or invalid UMAP stops the run; the stage cannot construct one.
- A missing graph or PCA reduces evidence and produces a warning, but UMAP and sweep analyses continue.
- Sweep mode requires at least two columns matching the configured regular expression or explicit list.
- Silhouette and density metrics use deterministic stratified samples above configured limits; every sampling decision is recorded.
- UMAP separation is not proof of biological distinctness, Leiden reflects the existing graph, small populations are less reliable, and marker plausibility and technical/sample-specific artefacts remain outside this stage.
