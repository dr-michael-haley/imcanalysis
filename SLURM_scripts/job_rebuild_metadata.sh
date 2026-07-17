#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 0-4
#SBATCH -n 2

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Rebuild metadata folder tables from an existing AnnData file
#@IN:   general.anndata_path (or rebuild_metadata.input_adata_path override)
#@OUT:  general.metadata_folder/{metadata.csv,dictionary.csv,panel.csv}
#@OUT:  outputs/<execution_id>_Metadata_Rebuild/ stage report
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.rebuild_metadata
#@CONFIG: general, rebuild_metadata, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Rebuild metadata job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_SEGMENTATION:-imc_segmentation}}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.rebuild_metadata
