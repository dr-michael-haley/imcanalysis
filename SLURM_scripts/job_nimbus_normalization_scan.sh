#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH --cpus-per-task=6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Scan marker-wise Nimbus normalization values before AnnData or clustering
#@IN:   general.masks_folder, general.metadata_folder/panel.csv, general.metadata_folder/metadata.csv
#@IN:   general.denoised_images_folder (or raw fallback per nimbus settings)
#@OUT:  outputs/<execution_id>_Nimbus_Normalization_Scan/{figures,tables,summaries,files}/
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.nimbus_normalization_scan
#@CONFIG: general, segmentation, nimbus, nimbus_normalization_scan, logging


source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and ${SLURM_CPUS_PER_TASK:-1} CPU core(s)"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_SEGMENTATION:-imc_segmentation}}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.nimbus_normalization_scan
