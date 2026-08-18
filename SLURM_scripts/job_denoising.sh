#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 12

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Denoise channel TIFFs and compute denoising QC metrics
#@IN:   general.raw_images_folder (default tiffs/)
#@IN:   general.metadata_folder/panel.csv
#@OUT:  general.denoised_images_folder (default processed/)
#@OUT:  outputs/<execution_id>_Denoising/ plus reusable denoised channel images
#@ENV:  sbt-denoise
#@MODULE:  SpatialBiologyToolkit.scripts.denoising
#@CONFIG: general, denoising, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Denoising job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_DENOISE:-sbt-denoise}}"
python -m SpatialBiologyToolkit.scripts.denoising
