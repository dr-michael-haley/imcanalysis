#! /bin/bash --login
#SBATCH -p himem 
#SBATCH -t 2-0
#SBATCH -n 2

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Update config.yaml by syncing all default config sections/keys
#@IN:   config.yaml (created if missing)
#@OUT:  config.yaml (updated in place)
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.update_config
#@CONFIG: all blocks (sync defaults): general, preprocess, denoising, createmasks, segmentation, nimbus, process, visualization, cellcharter, pairwise_spatial, networkx_spatial, subclustering, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Config update job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Generate or update config 
python -m SpatialBiologyToolkit.scripts.update_config
