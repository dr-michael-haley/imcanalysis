#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 8
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Build resumable cohort-only IMC cell features using full-segmentation context
#@IN:   napari_sbt.active_experiment manifest, frozen cohort, masks, and selected channel images
#@OUT:  napari_sbt experiment feature Parquet, dictionary, coverage, failure, and provenance assets
#@ENV:  sbt-analysis
#@MODULE:  SpatialBiologyToolkit.scripts.cell_features
#@CONFIG: general, napari_sbt, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m SpatialBiologyToolkit.scripts.cell_features
