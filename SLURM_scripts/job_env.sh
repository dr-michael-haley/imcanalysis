# Environment used by IMC analysis jobs

# This was causing a crash
# module load gcc/13.3.0
module purge

export MPLBACKEND="Agg"
export QT_QPA_PLATFORM="offscreen"
unset DISPLAY
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
