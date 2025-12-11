# Pipeline scripts

Stage-based entry points that can run locally or via the SLURM templates. They share `config_and_utils.py`, which loads `config.yaml` (override with `--config`) and supports inline overrides via `--override key.path=value`.

## How to run
1. Work inside a dataset folder containing `config.yaml` and the expected subfolders (e.g. `IMC_files`, `metadata`, `QC`, masks, etc.).
2. Submit on HPC with the `pl` helper from `Bash_scripts` (for example `pl preprocess denoise aiinter vis`) or call a single stage locally with `python -m SpatialBiologyToolkit.scripts.<stage> --config config.yaml`.
3. Use `pll <stage>` to run a stage locally while keeping the same stage aliases defined in `SLURM_scripts/pipeline.conf`.

## Notable stages
- Setup and QC: `generate_config.py`, `check_files.py`, `check_panel_consistency.py`, `harmonize_filenames.py`, `recursive_rename.py`, `update_config.py`.
- Image prep and segmentation: `preprocess.py`, `segmentation.py`, `segmentation_nimbus.py`, `cellpose_sam.py`, `createmasks_cellpose3.py`, `preprocess_dna.py`.
- Denoising and QC: `denoising.py`, `denoising_qc.py`.
- Core analysis: `basic_process.py`, `basic_process_ai.py`, `basic_process_biobatchnet.py`, `ai_interpretation.py`.
- Visualization and downstream: `basic_visualizations.py`, `reintegrate_markers.py`.

