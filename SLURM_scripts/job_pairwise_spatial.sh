#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Run pairwise spatial analyses (Squidpy interactions, distance bootstrap, and PCF) with plots/raw exports
#@IN:   pairwise_spatial.input_adata_path (fallback: process.output_adata_path, then process.input_adata_path)
#@IN:   adata.obs keys configured in pairwise_spatial (population_obs, roi_obs, X/Y coords, optional groupby_obs)
#@OUT:  general.qc_folder/pairwise_spatial.output_subdir (default QC/Pairwise_Spatial)
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.pairwise_spatial
#@CONFIG: general, process, pairwise_spatial, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Pairwise spatial job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_CELLCHARTER:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.pairwise_spatial
