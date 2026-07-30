#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -euo pipefail

#@DESC: Discover or explicitly select spatial assets and optionally build a validated SpatialData Zarr
#@IN:   spatialdata.root plus optional explicit AnnData, masks, image panels, histology, region-label, and MaxFuse paths
#@OUT:  spatialdata.output_path when spatialdata.action=build
#@OUT:  outputs/<execution_id>_SpatialData_Assembly/{tables,summaries}/
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.spatialdata_builder
#@CONFIG: general, spatialdata, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_SEGMENTATION:-imc_segmentation}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg

python -m SpatialBiologyToolkit.scripts.spatialdata_builder
