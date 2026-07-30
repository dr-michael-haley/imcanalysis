#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -euo pipefail

#@DESC: Match one scRNA-seq reference to IMC cells with MaxFuse and generate transfer/QC assets
#@IN:   maxfuse.reference_adata_path, maxfuse.target_adata_path (fallback: general.anndata_path)
#@IN:   maxfuse.feature_mapping_path with target and reference linked-feature columns
#@OUT:  maxfuse.asset_folder with match table, transfer AnnData, and retained feature mapping
#@OUT:  outputs/<execution_id>_MaxFuse_Matching/{figures,tables,summaries,files}/
#@ENV:  imc_maxfuse
#@MODULE:  SpatialBiologyToolkit.scripts.maxfuse_matching
#@CONFIG: general, maxfuse, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_MAXFUSE:-imc_maxfuse}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg

python -m SpatialBiologyToolkit.scripts.maxfuse_matching
