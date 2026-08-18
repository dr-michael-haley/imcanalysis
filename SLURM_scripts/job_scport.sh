#! /bin/bash --login
#SBATCH --job-name=imc_scportrait
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Generate single-cell portrait outputs via external scPortrait converter
#@IN:   processed/ and masks/ (hard-coded CLI args in this job)
#@OUT:  scPortrait/ project outputs (--projects-root scPortrait)
#@OUT:  outputs/<execution_id>_scPortrait_Export/ stage report under sbt
#@ENV:  sbt-scportrait
#@CONFIG: none (does not read config.yaml)

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "scPortrait job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

set -euo pipefail

conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_SCPORTRAIT:-sbt-scportrait}}"

python ~/scPortrait_to_IMC/imc_to_single_cells.py \
  --channels-dir processed \
  --mask-dir masks \
  --projects-root scPortrait \
  --overwrite \
  --mask-expand-px 0 \
  --debug
