# Environment used by IMC analysis jobs

# This was causing a crash
# module load gcc/13.3.0
module purge

export MPLBACKEND="Agg"
export QT_QPA_PLATFORM="offscreen"
unset DISPLAY

if [[ -f "$HOME/.imc_config" ]]; then
	source "$HOME/.imc_config"
fi