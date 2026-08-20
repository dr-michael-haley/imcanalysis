#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 8
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Compare HyPERSTAC Leiden parameter settings, with optional perturbation and Cox overlays
#@IN:   latest managed hyperstac-visualise report and reusable clustered representation; optional cox report
#@OUT:  managed parameter scorecard, agreement/support tables, figures, Markdown, and optional survival reports
#@ENV:  sbt-tensorflow
#@MODULE:  SpatialBiologyToolkit.scripts.hyperstac_stability
#@CONFIG: general, hyperstac, cox, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_TENSORFLOW:-sbt-tensorflow}}"
python -m SpatialBiologyToolkit.scripts.hyperstac_stability
