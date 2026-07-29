#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 12
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Background-correct and robustly scale ROI/channel TIFF images for HyPERSTAC
#@IN:   hyperstac.input_images_folder or general.denoised_images_folder
#@OUT:  hyperstac.asset_folder/normalised_images and managed normalization QC tables
#@ENV:  hyperstac
#@MODULE:  SpatialBiologyToolkit.scripts.hyperstac_preprocess
#@CONFIG: general, hyperstac, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_HYPERSTAC:-${IMC_ENV_HYPERSTAC:-hyperstac-imc}}"
python -m SpatialBiologyToolkit.scripts.hyperstac_preprocess
