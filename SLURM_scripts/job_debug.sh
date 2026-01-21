#! /bin/bash --login
#SBATCH -p serial 
#SBATCH -t 2-0
#SBATCH --job-name=imc_env_test
#SBATCH --output=imc_env_test_%j.out

#@DESC: Runs debugging on all the environments
#@IN:   Nones
#@OUT:  imc_env_test_%j.out

set -euo pipefail

echo "=============================================="
echo " IMC ENVIRONMENT DIAGNOSTIC"
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
# Environments to test
############################################
ENVS=(
  imc_segmentation
  imc_cellposesam
  imc_biobatchnet
  imc_denoise
)

############################################
# Helper function
############################################
test_env () {
    local ENV_NAME="$1"

    echo "----------------------------------------------"
    echo "▶ Testing environment: $ENV_NAME"
    echo "----------------------------------------------"

    if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        echo "❌ Environment '$ENV_NAME' does not exist"
        echo
        return
    fi

    ############################################
    # Activate
    ############################################
    conda activate "$ENV_NAME"

    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        echo "❌ conda activate failed"
        conda deactivate || true
        echo
        return
    fi

    ############################################
    # Critical HPC fix
    ############################################
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

    ############################################
    # Basic identity checks
    ############################################
    echo "CONDA_PREFIX=$CONDA_PREFIX"
    echo "Python: $(command -v python)"
    python -V

    ############################################
    # Check libstdc++
    ############################################
    echo
    echo "Checking libstdc++ resolution:"
    if [[ ! -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
        echo "❌ libstdc++.so.6 missing from env"
    else
        strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep -F "GLIBCXX_3.4.30" >/dev/null \
            && echo "✔ GLIBCXX_3.4.30 present" \
            || echo "⚠ GLIBCXX_3.4.30 missing"
    fi

    ############################################
    # Runtime linker truth
    ############################################
    echo
    echo "Runtime libstdc++ mapping (Python process):"
    python - <<'PY'
import os
import sys
print("sys.executable:", sys.executable)
maps = open("/proc/self/maps").read().splitlines()
hits = [l for l in maps if "libstdc++.so.6" in l]
for h in hits[:10]:
    print(" ", h)
PY

    ############################################
    # Import tests (env-specific)
    ############################################
    echo
    echo "Import tests:"

    python - <<'PY'
import sys
env = sys.prefix

def ok(msg):
    print(f"✔ {msg}")

def fail(msg):
    print(f"❌ {msg}")

try:
    import numpy
    ok("numpy")
except Exception as e:
    fail(f"numpy: {e}")

try:
    import scipy
    ok("scipy")
except Exception as e:
    fail(f"scipy: {e}")

try:
    import sklearn
    ok("scikit-learn")
except Exception as e:
    fail(f"scikit-learn: {e}")

# numba / llvmlite (biggest troublemaker)
try:
    import llvmlite.binding
    ok("llvmlite")
except Exception as e:
    fail(f"llvmlite: {e}")

try:
    import numba
    ok("numba")
except Exception as e:
    fail(f"numba: {e}")

# scanpy stack
try:
    import scanpy
    ok("scanpy")
except Exception as e:
    print("(scanpy not present or failed)")

# torch stack
try:
    import torch
    print("torch:", torch.__version__)
    ok("torch")
except Exception:
    print("(torch not present)")

# tensorflow stack
try:
    import tensorflow as tf
    print("tensorflow:", tf.__version__)
    ok("tensorflow")
except Exception:
    print("(tensorflow not present)")
PY

    ############################################
    # Clean up
    ############################################
    conda deactivate
    echo
}

############################################
# Run tests
############################################
for ENV in "${ENVS[@]}"; do
    test_env "$ENV"
done

echo "=============================================="
echo " IMC ENVIRONMENT TEST COMPLETE"
echo "=============================================="