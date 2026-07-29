#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 8
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Combine case-level features from one or more AnnData obs sources and compare Cox models
#@IN:   cox.feature_sources plus configured clinical AnnData or CSV metadata
#@OUT:  managed case-feature audit, Cox PH/Ridge/CoxNet comparisons, validation tables, and plots
#@ENV:  hyperstac
#@MODULE:  SpatialBiologyToolkit.scripts.cox_survival
#@CONFIG: general, cox, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_HYPERSTAC:-${IMC_ENV_HYPERSTAC:-hyperstac-imc}}"
python -m SpatialBiologyToolkit.scripts.cox_survival
