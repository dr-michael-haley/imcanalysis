#!/usr/bin/env bash
# Copy this file and edit the commands to choose your stages and their order.
# Usage: bash run_local.sh /path/to/project
set -e

cd "${1:?Usage: bash run_local.sh /path/to/project}"
export SBT_CONFIG="${SBT_CONFIG:-$PWD/config.yaml}"
if [[ ! -f "$SBT_CONFIG" ]]; then
    echo "Config not found: $SBT_CONFIG" >&2
    exit 1
fi

log_dir="${SBT_LOG_DIR:-logs}"
mkdir -p "$log_dir"
log_dir="$(cd "$log_dir" && pwd)"
log_file="$log_dir/run-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "Saving console output to $log_file"
exec >"$log_file" 2>&1

source "${SBT_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

conda activate "${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}"
echo "Running preprocess"
python -m SpatialBiologyToolkit.scripts.preprocess

conda activate "${SBT_CONDA_ENV_TENSORFLOW:-sbt-tensorflow}"
echo "Running denoising"
python -m SpatialBiologyToolkit.scripts.denoising

conda activate "${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}"
echo "Running preprocess_dna"
python -m SpatialBiologyToolkit.scripts.preprocess_dna

conda activate "${SBT_CONDA_ENV_CELLPOSESAM:-sbt-cellpose-sam}"
echo "Running cellpose_sam"
# Preserve the CSF3 C++ runtime fix only for this process/environment.
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    python -m SpatialBiologyToolkit.scripts.cellpose_sam

conda activate "${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}"
echo "Running segmentation_nimbus"
python -m SpatialBiologyToolkit.scripts.segmentation_nimbus
