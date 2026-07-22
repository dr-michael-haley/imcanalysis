#! /bin/bash --login
#SBATCH -p himem 
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Reintegrate markers previously removed from the processed AnnData
#@IN:   general.anndata_path + segmentation.removed_markers_anndata_path
#@OUT:  general.anndata_path (updated in place)
#@OUT:  outputs/<execution_id>_Marker_Reintegration/ stage report
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.reintegrate_markers
#@CONFIG: general, segmentation, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Reintegration job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_SEGMENTATION:-imc_segmentation}}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.reintegrate_markers
