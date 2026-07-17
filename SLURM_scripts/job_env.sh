# Environment used by IMC analysis jobs

set -eo pipefail

module purge
export MPLBACKEND="Agg"
export QT_QPA_PLATFORM="offscreen"
unset DISPLAY

# Normalize SLURM metadata into stable env vars for pipeline logging.
export IMC_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export IMC_SLURM_JOB_NAME="${SLURM_JOB_NAME:-}"
export SBT_SLURM_JOB_ID="${SBT_SLURM_JOB_ID:-${SLURM_JOB_ID:-}}"

if [[ -n "${IMC_SLURM_JOB_ID}" || -n "${IMC_SLURM_JOB_NAME}" ]]; then
    echo "SLURM context: job_id=${IMC_SLURM_JOB_ID:-NA}, job_name=${IMC_SLURM_JOB_NAME:-NA}"
fi

# Managed sbt runs finalize their structured stage report from the environment
# active at job exit. The original scientific exit code is always preserved.
if [[ -n "${SBT_STAGE:-}" && -n "${SBT_STAGE_OUTPUT_DIR:-}" ]]; then
    export SBT_STAGE_STARTED_AT="${SBT_STAGE_STARTED_AT:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
    _SBT_REPORTING_PYTHON="${SBT_REPORTING_PYTHON:-python}"

    if ! "$_SBT_REPORTING_PYTHON" -m SpatialBiologyToolkit.reporting.events --start; then
        echo "WARNING: SBT stage report initialization failed for ${SBT_STAGE}." >&2
    fi

    _sbt_finalize_stage_report() {
        local stage_exit_code=$?
        trap - EXIT
        if ! "$_SBT_REPORTING_PYTHON" -m SpatialBiologyToolkit.reporting.events --exit-code "$stage_exit_code"; then
            echo "WARNING: SBT stage report finalization failed for ${SBT_STAGE}." >&2
        fi
        return "$stage_exit_code"
    }

    trap _sbt_finalize_stage_report EXIT
fi
