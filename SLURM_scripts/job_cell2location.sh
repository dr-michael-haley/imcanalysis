#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 1
#SBATCH -c 12

set -euo pipefail

#@DESC: Fit reference signatures and map multiple Visium libraries with optional IMC spot-count priors
#@IN: cell2location.reference_adata_path for reference/full; signatures_path for map
#@IN: cell2location.visium_inputs and optional cell_count_prior_csv or obs column
#@OUT: cell2location.asset_folder/{reference,mapping} with models, posterior.h5ad and signatures.csv
#@OUT: Execution report with training histories, library abundance maps and cell-count prior QC
#@ENV: sbt-cell2location
#@MODULE: SpatialBiologyToolkit.scripts.cell2location_analysis
#@CONFIG: general, cell2location, logging

source "${SBT_TOOLKIT_ROOT:-$HOME/imcanalysis}/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_CELL2LOCATION:-sbt-cell2location}}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.cell2location_analysis
