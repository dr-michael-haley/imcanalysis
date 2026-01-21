#! /bin/bash --login
#SBATCH -p serial
#SBATCH -t 2-0
#SBATCH --job-name=imc_env_test
#SBATCH --output=imc_env_test_%j.out

#@DESC: Runs debugging on all the environments and job scripts
#@IN:   None
#@OUT:  imc_env_test_%j.out

set -euo pipefail

BASE_DIR="$HOME/imcanalysis"
JOB_DIR="$BASE_DIR/SLURM_scripts"
IMPORT_MAP="$JOB_DIR/env_imports.yaml"

echo "=============================================="
echo " IMC JOB + ENVIRONMENT DIAGNOSTIC"
echo " Host: $(hostname)"
echo " Date: $(date)"
echo "=============================================="
echo

############################################
# Load base job hygiene
############################################
module purge
unset DISPLAY
export MPLBACKEND="Agg"
export QT_QPA_PLATFORM="offscreen"

############################################
# Load conda properly (CRITICAL)
############################################
source "$HOME/miniconda3/etc/profile.d/conda.sh"

############################################
# Helper: test a single job
############################################
test_job () {
    local JOB_FILE="$1"
    local ENV_NAME="$2"
    local MODULE_NAME="$3"

    echo "----------------------------------------------"
    echo "▶ Job: $(basename "$JOB_FILE")"
    echo "  ENV: $ENV_NAME"
    echo "  MODULE: $MODULE_NAME"
    echo "----------------------------------------------"

    if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        echo "❌ Environment '$ENV_NAME' does not exist"
        echo
        return
    fi

    ############################################
    # Activate env
    ############################################
    conda activate "$ENV_NAME"

    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        echo "❌ conda activate failed"
        conda deactivate || true
        echo
        return
    fi

    ############################################
    # Critical HPC fix (ctypes / llvmlite)
    ############################################
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

    ############################################
    # Identity checks
    ############################################
    echo "CONDA_PREFIX=$CONDA_PREFIX"
    echo "Python: $(command -v python)"
    python -V

    ############################################
    # libstdc++ sanity
    ############################################
    echo
    echo "Checking libstdc++ ABI:"
    if [[ ! -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
        echo "❌ libstdc++.so.6 missing"
    else
        strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep -F "GLIBCXX_3.4.30" >/dev/null \
            && echo "✔ GLIBCXX_3.4.30 present" \
            || echo "⚠ GLIBCXX_3.4.30 missing"
    fi

    ############################################
    # Runtime linker truth
    ############################################
    echo
    echo "Runtime libstdc++ mapping:"
    python - <<'PY'
import sys
maps = open("/proc/self/maps").read().splitlines()
for l in maps:
    if "libstdc++.so.6" in l:
        print(" ", l)
PY

    ############################################
    # Python import tests (env + job specific)
    ############################################
    echo
    echo "Import tests (from env_imports.yaml):"

    python - <<PY
import yaml, importlib, sys

env = "$ENV_NAME"
module = "$MODULE_NAME"
import_map_file = "$IMPORT_MAP"

def ok(msg): print(f"✔ {msg}")
def fail(msg): print(f"❌ {msg}")

with open(import_map_file) as f:
    import_map = yaml.safe_load(f)

# Test job entry point
print("\\n--- Job entry point ---")
try:
    importlib.import_module(module)
    ok(module)
except Exception as e:
    fail(f"{module}: {e}")

# Test env-level imports
print("\\n--- Environment imports ---")
for pkg in import_map.get(env, []):
    try:
        importlib.import_module(pkg)
        ok(pkg)
    except Exception as e:
        fail(f"{pkg}: {e}")
PY

    ############################################
    # Cleanup
    ############################################
    conda deactivate
    echo
}

############################################
# Loop through job scripts
############################################
for JOB in "$JOB_DIR"/*.sh; do
    ENV=$(grep '^#@ENV:' "$JOB" | cut -d':' -f2 | xargs || true)
    MODULE=$(grep '^#@MODULE:' "$JOB" | cut -d':' -f2 | xargs || true)

    if [[ -z "$ENV" || -z "$MODULE" ]]; then
        echo "⚠ Skipping $(basename "$JOB") (missing #@ENV or #@MODULE)"
        echo
        continue
    fi

    test_job "$JOB" "$ENV" "$MODULE"
done

echo "=============================================="
echo " IMC JOB DIAGNOSTIC COMPLETE"
echo "=============================================="
