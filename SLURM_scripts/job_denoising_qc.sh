#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Produces QC plots and analyses for denoisinng
#@IN:   processed
#@OUT:  QC/denoising/
#@ENV:  imc_denoise
#@MODULE:  SpatialBiologyToolkit.scripts.denoising_qc 

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_DENOISE:-imc_denoise}"
python -m SpatialBiologyToolkit.scripts.denoising_qc

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.check_panel_consistency 