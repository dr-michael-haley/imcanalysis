#! /bin/bash --login
#SBATCH --job-name=imc_scportrait
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Generate single-cell portrait outputs via external scPortrait converter
#@IN:   general.denoised_images_folder and general.masks_folder
#@OUT:  scportrait.projects_root (default scPortrait/) project outputs
#@OUT:  outputs/<execution_id>_scPortrait_Export/ stage report under sbt
#@ENV:  sbt-scportrait
#@CONFIG: general, scportrait

source "${SBT_TOOLKIT_ROOT:-$HOME/imcanalysis}/SLURM_scripts/job_env.sh"

echo "scPortrait job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

set -euo pipefail

conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_SCPORTRAIT:-sbt-scportrait}}"

python - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

from SpatialBiologyToolkit.config import PipelineConfig, load_config

config_path = Path(os.environ.get("SBT_CONFIG", "config.yaml")).expanduser()
# Preserve standalone legacy use without a config; explicit configs must exist.
config = load_config(config_path) if config_path.exists() or "SBT_CONFIG" in os.environ else PipelineConfig()
converter = Path(os.environ.get(
    "SBT_SCPORTRAIT_CONVERTER", "~/scPortrait_to_IMC/imc_to_single_cells.py"
)).expanduser()
sys.exit(subprocess.call([
    sys.executable, str(converter),
    "--channels-dir", str(Path(config.general.denoised_images_folder).expanduser()),
    "--mask-dir", str(Path(config.general.masks_folder).expanduser()),
    "--projects-root", str(Path(config.scportrait.projects_root).expanduser()),
    "--overwrite", "--mask-expand-px", "0", "--debug",
]))
PY
