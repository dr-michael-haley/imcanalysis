# Pipeline scripts

Stage-based entry points that can run locally or via the SLURM templates. They share `config_and_utils.py`, which loads `config.yaml` (override with `--config`) and supports inline overrides via `--override key.path=value`.

## How to run
1. Work inside a dataset folder containing `config.yaml` and the expected subfolders (e.g. `IMC_files`, `metadata`, `QC`, masks, etc.).
2. Submit on HPC with the `pl` helper (for example `pl prep denoise aiinter vis`) or call a single stage locally with `python -m SpatialBiologyToolkit.scripts.<module> --config config.yaml`.
3. Use `pll <stage>` to run a stage locally while keeping the same stage aliases defined in `SLURM_scripts/pipeline.conf`.

## Notable stages
- Setup and QC: `generate_config.py`, `check_files.py`, `check_panel_consistency.py`, `harmonize_filenames.py`, `recursive_rename.py`, `update_config.py`.
- Image prep and segmentation: `preprocess.py`, `segmentation.py`, `segmentation_nimbus.py`, `cellpose_sam.py`, `createmasks_cellpose3.py`, `preprocess_dna.py`.
- Denoising and QC: `denoising.py`, `denoising_qc.py`.
- Core analysis: `basic_process_batch_integration.py` (Harmony / BBKNN), `basic_process_rapids.py` (optional cell filtering, rapids-singlecell GPU PCA / optional Harmony / neighbors / UMAP / Leiden, QC MatrixPlots, parameter scans), the staged `cellvision_extract.py` / `cellvision_embed.py` / `cellvision_cluster.py` / `cellvision_plot.py` image-representation workflow, `basic_process_biobatchnet.py`, and `ai_interpretation.py`.
- Legacy entrypoints retained for older workflows: `basic_process.py`, `basic_process_ai.py`.
- Visualization and downstream: `basic_visualizations.py`, `cellcharter_neighborhoods.py`, `starling_analysis.py`, `pairwise_spatial.py`, `networkx_spatial.py`, `remap_obs.py`, `reintegrate_markers.py`.

The alias and module name are not always identical (`prep` invokes the
`preprocess` module, for example). Use the generated [SLURM stage
reference](stages/index.md) rather than guessing a module from an alias.

