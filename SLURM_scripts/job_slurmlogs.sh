#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 0-4:00
#SBATCH -n 1

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Organize SLURM output files using AnnData pipeline run metadata and verify against recorded jobs
#@IN:   general.anndata_path with general.anndata_uns_log_key run_log entries containing slurm.job_id/slurm.job_name
#@IN:   Current working directory containing slurm-<job_id>.out files
#@OUT:  general.slurm_logs_folder with renamed logs and *_Unverified flags for unmatched files
#@OUT:  general.slurm_logs_folder/slurmlogs_manifest.csv
#@OUT:  outputs/<execution_id>_Legacy_SLURM_Log_Migration/ stage report under sbt
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.slurmlogs
#@CONFIG: general, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "SLURM log organizer is using $SLURM_NTASKS CPU core(s)"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_SEGMENTATION:-imc_segmentation}}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.slurmlogs
