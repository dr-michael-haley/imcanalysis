#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 12

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Denoising using IMC_Denoise
#@IN:   tiffs
#@OUT:  processed, /QC/denoising/

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Denoising job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_DENOISE:-imc_denoise}"
python -m SpatialBiologyToolkit.scripts.denoising
