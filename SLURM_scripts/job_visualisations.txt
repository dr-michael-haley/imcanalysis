#! /bin/bash --login
#SBATCH -p himem 
#SBATCH -t 2-0
#SBATCH -n 2

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Produces visualisations for populations
#@IN:   anndata_processed.h5ad, masks, processed
#@OUT:  QC/BasicProcess_QC

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Visualisations job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.basic_visualizations