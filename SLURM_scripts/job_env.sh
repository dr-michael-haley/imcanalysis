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

echo "========== DEBUG: ENVIRONMENT =========="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo

echo "Shell: $SHELL"
echo "BASH_VERSION: $BASH_VERSION"
echo "Login shell: $([[ $- == *l* ]] && echo yes || echo no)"
echo

echo "---- MODULES (after purge) ----"
if command -v module >/dev/null 2>&1; then
    module list 2>&1
else
    echo "module command not available"
fi
echo

echo "---- KEY ENV VARS ----"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo

echo "---- WHICH BINARIES ----"
command -v python || true
command -v conda || true
echo

echo "========== END DEBUG =========="
