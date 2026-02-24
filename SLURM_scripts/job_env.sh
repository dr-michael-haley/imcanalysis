# Environment used by IMC analysis jobs

module purge
export MPLBACKEND="Agg"
export QT_QPA_PLATFORM="offscreen"
unset DISPLAY

# Normalize SLURM metadata into stable env vars for pipeline logging.
export IMC_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export IMC_SLURM_JOB_NAME="${SLURM_JOB_NAME:-}"

if [[ -n "${IMC_SLURM_JOB_ID}" || -n "${IMC_SLURM_JOB_NAME}" ]]; then
    echo "SLURM context: job_id=${IMC_SLURM_JOB_ID:-NA}, job_name=${IMC_SLURM_JOB_NAME:-NA}"
fi
