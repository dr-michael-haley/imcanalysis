# Cell2location: scRNA-seq reference and Visium mapping

## What this stage does

SBT provides importable analysis functions and one independently runnable stage:
`sbt run cell2location`. The `cell2location.action` setting selects `reference`,
`map`, or `full`. It is deliberately absent from the standard IMC pipeline modes
and has no upstream IMC dependencies. It can be used with a minimal SBT project
containing only configuration and externally prepared H5AD files.

The implementation targets **cell2location 0.1.5**, using its negative-binomial
reference regression and spatial mapping models. It supports joint mapping of
multiple Visium libraries, with a separate detection batch for every library.
Both a combined H5AD and multiple separate H5ADs are supported.

## Why it is performed

Reference regression estimates cell-type expression signatures from labelled
scRNA-seq counts while accounting for configured technical batches/covariates.
Spatial mapping uses those signatures to estimate cell-type abundance at Visium
spots. Reference fitting can be performed once and reused for further cohorts.
These models require RNA counts, not IMC protein intensity measurements.

## Main inputs

- A reference H5AD with raw RNA counts, a cell-type label in `obs`, and optional
  sample/batch and categorical technical covariates.
- One or more Visium H5AD files with raw RNA counts and finite two-dimensional
  `obsm['spatial']` coordinates. Optional `uns['spatial']` images/scalefactors
  are retained. Space Ranger directories are not read directly in this version;
  import them to AnnData first.
- Gene IDs in `var_names`, or explicit `*_gene_id_key` columns. Both modalities
  must use the same namespace. Duplicate IDs fail rather than receiving suffixes
  that would change their biological identity. Aggregate duplicates explicitly
  upstream if that is scientifically appropriate.
- For mapping-only runs, a signature CSV with gene IDs as its first column and
  cell types as the remaining columns. SBT reference output already has this form.
- Optional registered serial-section IMC cell counts, in `obs` or a CSV keyed
  by library and barcode. Image registration and segmentation are upstream tasks.

Each library must occur in only one input file. For combined files use
`library_key`. A single-library file may instead specify `library_id` in its
input entry. Conflicting IDs are rejected. Original barcodes default to
`obs_names`; set `barcode_key` if an earlier concatenation appended sample suffixes.
Output spot IDs are escaped `library::barcode` pairs; the original barcode and
input observation name remain in `_sbt_c2l_barcode` and `_sbt_c2l_source_obs_name`.

### Minimal full analysis

```yaml
cell2location:
  action: full
  reference_adata_path: inputs/scrna_reference.h5ad
  reference_labels_key: cell_type
  reference_batch_key: donor
  reference_counts_layer: counts
  visium_inputs:
    - path: inputs/case_a.h5ad
      library_id: case_a
    - path: inputs/case_b.h5ad
      library_id: case_b
  library_key: library_id
  visium_counts_layer: counts
  asset_folder: cell2location/cohort_v1
  n_cells_per_location: 30
  detection_alpha: 20
  accelerator: gpu
```

For one combined file replace the list with
`visium_inputs: [{path: inputs/combined_visium.h5ad}]`; its `obs['library_id']`
must identify the libraries. Counts selectors default to `X`; use a named layer
or the literal `raw` for `adata.raw`. `.raw` is not assumed to contain counts:
the selected matrix is still validated.

```bash
sbt project init /path/to/visium_project
# Edit the project's config.yaml, then:
sbt plan cell2location --project /path/to/visium_project
sbt run cell2location --project /path/to/visium_project --dry-run
sbt run cell2location --project /path/to/visium_project
```

For direct work in an allocated GPU session or on a workstation, activate the
dedicated environment and run:

```bash
python -m SpatialBiologyToolkit.scripts.cell2location_analysis --config config.yaml --action reference
python -m SpatialBiologyToolkit.scripts.cell2location_analysis --config config.yaml --action map
```

The direct override is recorded in report notes and output AnnData provenance.
Managed execution reads the action from configuration. Planning validates file
presence and destination safety; matrix validation occurs on the compute node.

### Reuse a reference

Run with `action: reference`, then set `action: map`. By default mapping reads
`<asset_folder>/reference/signatures.csv`. To reuse signatures from an earlier
analysis with a new mapping destination, set:

```yaml
cell2location:
  action: map
  signatures_path: cell2location/reference_v1/reference/signatures.csv
  asset_folder: cell2location/cohort_v2
  visium_inputs:
    - path: inputs/combined_visium.h5ad
```

Mapping-only runs do not need the original reference H5AD. Existing `reference/`
or `mapping/` result directories are never overwritten; choose a new asset folder
for refitting. A failed save retains a hidden temporary sibling for recovery.

## Preprocessing and transformations

The fitting functions copy inputs and select the requested raw-count matrix.
Negative, nonfinite and fractional expression values fail validation. Integer
validation cannot prove biological provenance: the user must select original
counts, rather than rounded normalized values.

No library-size normalization, log transformation, scaling, batch correction or
highly-variable-gene restriction is applied to model inputs. Reference filtering
matches the tutorial: retain genes expressed in more than 3% of cells, or in
more than 5 cells with mean nonzero expression greater than 1.12. These thresholds
are configurable. Mitochondrial symbols starting with `MT-` or `mt-` are excluded;
when matching by Ensembl IDs, supply `gene_symbol_key` for this filtering.
Set `mitochondrial_prefixes: []` to disable it deliberately.

If present, `in_tissue` restricts Visium input to spots labelled 1. Files are joined
on the intersection of genes, then aligned to the shared expressed signature
genes. Missing genes are not interpreted as biological zeroes. At least
`min_shared_genes` (default 100) must remain. Zero-count cells or spots after
filtering fail with a curation message. Gene and observation counts are recorded.

## Registered IMC cell-count priors

The standard model uses a global expected count. SBT optionally extends the
existing `n_s_cells_per_location` prior to use a distinct mean for each spot:

```text
mu_s = max(registered_count_s * cell_count_prior_scale, cell_count_prior_floor)
n_s ~ Gamma(shape = mu_s * r, rate = r)
r = cell_count_prior_mean_var_ratio
E[n_s] = mu_s; Var[n_s] = mu_s / r
```

This changes the prior on the latent abundance scale, not the RNA likelihood,
guide, or other model terms. It does **not** clamp the sum of posterior cell-type
abundances to the observed IMC count. Training and posterior minibatches look up
the prior by global spot index. Initialization of detection sensitivity accounts
for the mean prior density in each library.

Supply one source:

```yaml
cell2location:
  cell_count_prior_obs_key: imc_registered_cells
  cell_count_prior_scale: 1.0
  cell_count_prior_floor: 0.1
  cell_count_prior_mean_var_ratio: 1.0
```

Or set `cell_count_prior_csv: inputs/registered_spot_counts.csv` instead of the
obs key, with this exact schema:

```csv
library_id,barcode,n_cells
case_a,AAACAACGAATAGTTC-1,12
case_b,AAACAACGAATAGTTC-1,7
```

Rows are aligned using **both** library and barcode, never row order. Every
retained spot must have one finite nonnegative count; missing and duplicate keys
fail. Extra rows, such as excluded off-tissue spots, are counted in the report.
Zero counts receive the explicit positive floor required by a Gamma distribution.
Fractional expected cell counts are allowed. Raw counts, adjusted prior means,
scale, floor and prior strength are saved.

Serial sections sample different cells and may differ in tissue thickness,
registration accuracy and segmentation completeness. Start with a weak prior
and compare with the global-prior fit. Lower `r` means greater uncertainty;
larger `r` constrains the latent scale more strongly. The extension is an SBT
model variant requiring scientific validation on your paired sections; it is
not the standard global-prior model validated in the paper.

## Reusable assets produced or modified

All results are beneath the configured `asset_folder`:

| Path | Contents |
|---|---|
| `reference/model/` | Reloadable upstream RegressionModel checkpoint |
| `reference/posterior.h5ad` | Filtered count data and reference posterior summaries |
| `reference/signatures.csv` | Mean expression per gene and reference cell type |
| `mapping/model/` | Reloadable Cell2location checkpoint, including the spot-prior model class when used |
| `mapping/posterior.h5ad` | Joint counts, coordinates, library metadata and posterior abundance summaries |
| `mapping/signatures.csv` | Exact ordered signatures/genes used in mapping |

Input files are never rewritten. Results retain upstream `uns['mod']`,
`varm['means_per_cluster_mu_fg']` for the reference, and
`obsm['means_cell_abundance_w_sf']`, `stds`, `q05` and `q95` abundance summaries
for mapping. `uns['sbt_cell2location']` records settings, library/count selection,
filtering statistics, prior adjustments and package versions.

## Human-facing outputs produced

SBT's shared reporter owns the execution directory and environment provenance.
Managed runs use `outputs/<execution_id>_Cell2location_Visium/`; direct runs use
the framework's `outputs/direct/` location. Reports include training histories,
reference cell-type support, retained gene QC, signature heatmaps, per-library
coordinate maps, mean/q05 abundance tables, RNA-versus-total-cell QC, and
prior-versus-posterior diagnostics. Coordinate maps have no histology overlay.
Fitting and report failures propagate as failed executions.

## Notebook API

```python
import anndata as ad
from SpatialBiologyToolkit.config import Cell2locationConfig
from SpatialBiologyToolkit.cell2location_analysis import (
    fit_reference, prepare_visium, fit_mapping, save_result,
)

settings = Cell2locationConfig(reference_labels_key="cell_type", accelerator="gpu")
reference = fit_reference(ad.read_h5ad("reference.h5ad"), settings)
visium = prepare_visium(
    [ad.read_h5ad("case_a.h5ad"), ad.read_h5ad("case_b.h5ad")],
    settings, library_ids=["case_a", "case_b"],
)
mapping = fit_mapping(visium, reference.signatures, settings)
save_result(mapping, "cell2location/notebook_run/mapping")
# Optional per-library export:
case_a = mapping.adata[mapping.adata.obs[settings.library_key] == "case_a"].copy()
```

`fit_mapping(..., counts=count_dataframe)` accepts the same prior table schema.
Notebook functions return model objects for further upstream plots/diagnostics.
Use the script entry point when an SBT execution report is wanted automatically.
For model loading use the dedicated environment, current SBT, and upstream
`Cell2location.load(path, adata=posterior)` or `RegressionModel.load(...)`.

## Environment and resources

The dedicated `cell2location` environment key resolves to `sbt-cell2location`.
It uses Python 3.11, cell2location 0.1.5, scvi-tools 1.3.3, PyTorch 2.7.1 and a
pinned compatible JAX family. Conda intent/lock and pip extras are maintained in
`HPC_env_files/sbt-cell2location/`; SBT itself is an editable `--no-deps` overlay.
Do not install the broad SBT dependencies over this scientific stack.

```bash
sbt env validate-spec cell2location
sbt env sync cell2location
sbt env test cell2location
```

The registered checks verify imports and package versions. On a compute or
development node, run `python HPC_env_files/sbt-cell2location/smoke_test.py` for
tiny CPU reference and mapping fits, indexed-prior checks, posterior export and
model reload. It is not a test of scientific convergence.
Validate CUDA on a compute node before production fitting. The default wrapper
requests one GPU, eight CPUs, 64 GiB RAM and two days, following SBT's long-running
GPU analyses; cohort size may require resource adjustment. Training defaults
are 250 reference epochs and 30,000 mapping epochs. Full-batch mapping is the
upstream default; `mapping_batch_size` and `posterior_batch_size` can bound GPU
memory. Do not run these fits on a cluster login node.

## Important configuration options

See the generated [cell2location configuration reference](../reference/configuration/sections/cell2location.md)
for all fields. Choose `n_cells_per_location` for your tissue rather than accepting
30 without review. `detection_alpha=20` allows more within-library technical
variation than 200. A separate library should represent a technical Visium
experiment/section, even if several libraries come from one biological case.

## How to interpret the results

Posterior means estimate abundance; q05 supplies a conservative abundance summary,
not a probability that a type is present. Cell-type identifiability depends on
the reference and distinguishable expression signatures. Inspect convergence,
reference coverage, per-library maps and sensitivity to prior settings. Matching
the serial-section prior is not independent evidence of mapping accuracy.

## Common problems and limitations

- A small shared-gene set usually indicates mismatched gene IDs or already
  filtered/HVG-only inputs. No silent gene-symbol conversion is attempted.
- Reference cell types missing from the tissue or tissue types absent from the
  reference can bias decomposition; curate a representative reference.
- Missing prior values fail rather than being replaced by global averages.
- GPU availability is checked when requested; use `accelerator: cpu` for explicit
  CPU work. Very short fits are suitable only for smoke testing.
- This version supports Visium H5AD inputs and joint library fitting. It does not
  perform registration, segment IMC images, read Space Ranger directories, or
  implement NMF/tissue-zone downstream analyses.

## Sources

- [cell2location tutorial](https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html)
- [Upstream model and installation](https://github.com/BayraktarLab/cell2location)
- Kleshchevnikov et al. (2022), [Cell2location maps fine-grained cell types in spatial transcriptomics](https://doi.org/10.1038/s41587-021-01139-4).
