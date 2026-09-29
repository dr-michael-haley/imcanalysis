#!/usr/bin/env bash
# Nimbus -> BioBatchNet -> RAPIDS -> visualisation.
# Run from an already-active scientific Conda environment.
# Usage: bash run_local_analysis.sh /path/to/project
set -e

cd "${1:?Usage: bash run_local_analysis.sh /path/to/project}"
export SBT_CONFIG="${SBT_CONFIG:-$PWD/config.yaml}"
if [[ ! -f "$SBT_CONFIG" ]]; then
    echo "Config not found: $SBT_CONFIG" >&2
    exit 1
fi

log_dir="${SBT_LOG_DIR:-logs}"
mkdir -p "$log_dir"
log_dir="$(cd "$log_dir" && pwd)"
log_file="$log_dir/analysis-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "Saving console output to $log_file"
exec >"$log_file" 2>&1

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "Running Nimbus"
python -m SpatialBiologyToolkit.scripts.segmentation_nimbus

echo "Running BioBatchNet"
python -m SpatialBiologyToolkit.scripts.basic_process_biobatchnet

echo "Running RAPIDS"
python -m SpatialBiologyToolkit.scripts.basic_process_rapids

echo "Running visualisation"
python -m SpatialBiologyToolkit.scripts.basic_visualizations
