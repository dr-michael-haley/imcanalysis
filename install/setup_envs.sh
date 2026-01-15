#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/../HPC_env_files" && pwd)"
CONDA_BASE="$(conda info --base)"
ENV_ROOT="$CONDA_BASE/envs"

# List of all environments
ENVS=("imc_segmentation" "imc_denoise" "imc_cellposesam" "imc_biobatchnet")

########################################
# Ensure conda is available
########################################
if ! command -v conda >/dev/null 2>&1; then
    echo "❌ ERROR: conda not found in PATH."
    echo "Please load or install conda before running this script."
    exit 1
fi

########################################
# Ensure conda-lock is installed (in base)
########################################
if ! command -v conda-lock >/dev/null 2>&1; then
    echo "⚠️  conda-lock is not installed in your base environment."
    read -rp "Install conda-lock into base now? [y/N]: " reply
    case "$reply" in
        [Yy]*)
            echo "📦 Installing conda-lock into base..."
            conda activate base
            conda install -y -c conda-forge conda-lock
            ;;
        *)
            echo "❌ Cannot continue without conda-lock."
            exit 1
            ;;
    esac
fi

########################################
# Detect mamba for faster installs
########################################
if command -v mamba >/dev/null 2>&1; then
    INSTALL_CMD="conda-lock install --mamba --name"
    echo "🚀 Using conda-lock with mamba backend"
else
    INSTALL_CMD="conda-lock install --name"
    echo "🐍 Using conda-lock with conda backend"
fi
echo

########################################
# Create one environment
########################################
create_env() {
    local env="$1"
    local env_dir="$BASE_DIR/$env"
    local lockfile="$env_dir/conda-linux-64.lock"
    local extras="$env_dir/pip-extras.txt"

    echo "============================="
    echo "📦 Installing environment: $env"
    echo "============================="

    if [[ ! -f "$lockfile" ]]; then
        echo "❌ ERROR: Missing lockfile: $lockfile"
        exit 1
    fi

    if conda env list | awk '{print $1}' | grep -Fx "$env" >/dev/null; then
        echo "⏩ Environment '$env' already exists — skipping creation."
    else
        echo "🌱 Creating environment '$env' from lockfile..."
        $INSTALL_CMD "$env" "$lockfile"
    fi

    if [[ -f "$extras" ]]; then
        echo "📦 Installing pip extras for '$env'..."
        conda run -n "$env" pip install -r "$extras"
    fi

    echo "✔ Finished environment: $env"
    echo
}

########################################
# Install all environments
########################################
for env in "${ENVS[@]}"; do
    create_env "$env"
done

echo "🎉 All conda environments installed successfully!"
