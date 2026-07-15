# SLURM pipeline workflow

SpatialBiologyToolkit provides SLURM wrappers for the IMC pipeline, tested on
the University of Manchester Computational Shared Facility. Run each stage
from the dataset directory containing `config.yaml`.

The [generated stage reference](stages/index.md) is the source of truth for
aliases, wrappers, environments, inputs, outputs, and config sections. It is
generated from `SLURM_scripts/pipeline.conf` and the metadata in each wrapper,
so it cannot silently omit a newly registered stage.

## Suggested run order

The normal Nimbus workflow is:

1. `config` - refresh missing/default config keys after repository updates.
2. `prep` - import IMC files and create image and metadata inputs.
3. `denoise` - denoise channel TIFFs.
4. `dnqc` - inspect denoising and panel consistency.
5. `cellpose` - preprocess DNA and generate CellPose-SAM masks.
6. `nimbus` - quantify cells and create AnnData.
7. Choose one of `bint`, `rapids`, or `bbn` for batch-aware processing.
8. Optionally run `remap` and `subcl` to curate/refine labels.
9. Optionally branch to `cchar`, `starling`, `pairsp`, or `nxsp` analyses.
10. Optionally run `aiinter` for AI-assisted cluster labels.
11. Run `vis` for the standard figures and QC outputs.
12. Optionally run `reint`, `slogs`, and `zipqc` for marker reintegration, log organisation, and packaging.

For example:

```bash
pl config prep denoise dnqc cellpose nimbus rapids remap subcl pairsp vis slogs zipqc
```

Use `rebuildmeta` only to reconstruct metadata tables from an existing AnnData
file; it is a recovery/maintenance stage rather than part of a fresh run.

## Alternative and optional stages

- `bint`, `rapids`, and `bbn` are alternative processing routes. Run more than one only when deliberately comparing methods.
- `remap`, `subcl`, `aiinter`, and `reint` are optional core stages.
- `cchar`, `starling`, `pairsp`, and `nxsp` are independent analysis branches after processing.
- `scport` is an external single-cell portrait branch after denoising and mask generation. Its environment and external converter are not installed by the standard environment setup.
- `aiinter` needs `OPENAI_API_KEY` when AI interpretation is enabled.

The [subclustering guide](subclustering.md) explains the deliberate edit-and-rerun checkpoints in `subcl`.

## Common commands

List every registered stage and its wrapper metadata:

```bash
pl --list
```

Preview a dependency chain without submitting jobs:

```bash
pl --dry-run prep denoise dnqc cellpose nimbus
```

Run a single wrapper locally for debugging:

```bash
pll prep
```

Run the environment/module diagnostics on SLURM:

```bash
pl debug
```

## Example RAPIDS configuration

```yaml
rapids:
  run_harmony: true
  batch_correction_obs: slide
  filter_obs_key: mask_area
  filter_min_value: 25
  filter_max_value: 1000
```

A parameter scan is the Cartesian product of every non-empty list:

```yaml
rapids:
  parameter_scan_dict:
    n_neighbors: [10, 20]
    n_for_pca: [20, 30]
    umap_min_dist: [0.05, 0.1]
    run_harmony: [false, true]
  parameter_scan_save_anndata: false
```

## Example STARLING configuration

```yaml
general:
  population_obs_primary: final_population
starling:
  initial_clustering_method: User
  initial_label_obs: final_population
  use_layer: null
  cell_size_col_name: mask_area
  output_prefix: starling
  store_assignment_prob_matrix: true
```

## Default paths

Unless overridden in `config.yaml`, `GeneralConfig` uses:

| Key | Default |
|---|---|
| `general.imc_files_folder` | `IMC_files` |
| `general.metadata_folder` | `metadata` |
| `general.raw_images_folder` | `tiffs` |
| `general.denoised_images_folder` | `processed` |
| `general.masks_folder` | `masks` |
| `general.celltable_folder` | `cell_tables` |
| `general.qc_folder` | `QC` |

See the generated [general config table](../reference/configuration/sections/general.md)
for the complete set of defaults.

## Adding or changing a pipeline stage

1. Add or update a `SLURM_scripts/job_*.sh` wrapper.
2. Include `#@DESC`, `#@IN`, `#@OUT`, `#@ENV`, `#@MODULE`, and `#@CONFIG` records where applicable.
3. Register its alias in `SLURM_scripts/pipeline.conf`.
4. Run `make docs-generate`, then `make docs-check`.

The generator creates a compact table and one focused page per alias. Put
cross-stage concepts and ordering advice in this workflow page; put exact
stage facts in the wrapper metadata.

## Operational notes

- Job email notifications use `IMC_EMAIL`.
- Environment names can be overridden with the `IMC_ENV_*` variables used by the wrappers.
- `rapids_singlecell`, `imc_starling`, and `scPortrait` are external/pre-existing environments rather than environments created by `make envs`.
- If executable bits are missing, run `chmod +x ~/imcanalysis/SLURM_scripts/*.sh`.
