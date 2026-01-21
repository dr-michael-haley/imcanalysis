#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Cell mask creation using CellPose-SAM
#@IN:   processed/
#@OUT:  anndata_processed.h5ad, /QC/BioBatchNet/
#@ENV:  imc_cellposesam
#@MODULE:  SpatialBiologyToolkit.scripts.cellpose_sam

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "CellPose-SAM job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.preprocess_dna

conda activate "${IMC_ENV_CELLPOSESAM:-imc_cellposesam}"
python -m SpatialBiologyToolkit.scripts.cellpose_sam