# Environment used by IMC analysis jobs

module purge
export MPLBACKEND="Agg"
export QT_QPA_PLATFORM="offscreen"
unset DISPLAY
