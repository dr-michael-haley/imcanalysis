#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 2

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Apply a simple CSV-based remap onto adata.obs, or generate a blank remap template from an existing obs column
#@IN:   remap_obs.input_adata_path (fallback: general.anndata_path), remap_obs.remap_csv_path (default metadata/remap.csv)
#@OUT:  Updated AnnData at general.anndata_path (apply mode) and/or remap CSV at remap_obs.remap_csv_path (generate_blank mode)
#@OUT:  outputs/017_Observation_Remapping/<run_id>/ stage report
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.remap_obs
#@CONFIG: general, remap_obs, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Remap obs job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.remap_obs
