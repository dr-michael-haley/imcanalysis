#! /bin/bash --login
#SBATCH --job-name=imc_scportrait
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=michael.haley@manchester.ac.uk
#SBATCH --mail-type=ALL

#@DESC: scPortrait
#@IN:   denoised images, masks
#@OUT:  single cell portraits

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "scPortrait job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

set -euo pipefail

conda activate "${IMC_ENV_SCPORTRAIT:-scPortrait}"

python ~/scPortrait_to_IMC/imc_to_single_cells.py \
  --channels-dir processed \
  --mask-dir masks \
  --projects-root scPortrait \
  --overwrite \
  --mask-expand-px 0 \
  --debug