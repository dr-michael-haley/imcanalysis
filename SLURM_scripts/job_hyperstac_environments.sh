#!/bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 4
#SBATCH --mem=16G
set -e
#@DESC: Discover physical-radius environments from saved HyPERSTAC patch embeddings
#@IN: HyPERSTAC embeddings, metrics, sensitivity and ROI-case mapping
#@OUT: Separate environment AnnData, radius graphs, GMM models, summaries and stability
#@ENV: sbt-analysis
#@MODULE: SpatialBiologyToolkit.scripts.hyperstac_environments
#@CONFIG: general, hyperstac, hyperstac_environments, logging
source "${SBT_TOOLKIT_ROOT:-$HOME/imcanalysis}/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}}"
python -m SpatialBiologyToolkit.scripts.hyperstac_environments
