#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 8
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Cross-reference HyPERSTAC Leiden marker environments, perturbation sensitivity, and Cox effects
#@IN:   latest managed hyperstac-visualise and cox execution reports
#@OUT:  managed cross-Leiden stability tables, figures, Markdown, and per-clustering HTML reports
#@ENV:  hyperstac
#@MODULE:  SpatialBiologyToolkit.scripts.hyperstac_stability
#@CONFIG: general, hyperstac, cox, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_HYPERSTAC:-${IMC_ENV_HYPERSTAC:-hyperstac-imc}}"
python -m SpatialBiologyToolkit.scripts.hyperstac_stability
