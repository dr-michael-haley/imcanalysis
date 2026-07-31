# Pipeline scripts

Stage-based entry points that can run locally or via the SLURM templates. They share `config_and_utils.py`, which loads `config.yaml` (override with `--config`) and supports inline overrides via `--override key.path=value`.

## How to run
1. Initialize or adopt a dataset project containing `config.yaml` and the expected assets.
2. Validate and preview a stage with `sbt plan <stage>` and `sbt run <stage> --dry-run`.
3. Submit it on HPC with `sbt run <stage>`. Use direct
   `python -m SpatialBiologyToolkit.scripts.<module> --config config.yaml`
   invocation only for stage development or focused debugging.

The older `pl` and `pll` entry points remain compatibility interfaces. They do
not provide the project validation, dependency planning, run records, or
structured report control of `sbt`.

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

