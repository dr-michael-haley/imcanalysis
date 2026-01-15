# Environment used by IMC analysis jobs

module load gcc/13.3.0

export MPLBACKEND="Agg"
export QT_QPA_PLATFORM="offscreen"
unset DISPLAY

if [[ -f "$HOME/.imc_config" ]]; then
	source "$HOME/.imc_config"
fi