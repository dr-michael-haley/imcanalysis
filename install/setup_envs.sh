#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/../HPC_env_files" && pwd)"
CONDA_BASE="$(conda info --base)"
ENV_ROOT="$CONDA_BASE/envs"
CONFIG_FILE="$HOME/.imc_config"

# List of all environments
ENVS=("imc_segmentation" "imc_denoise" "imc_cellposesam" "imc_biobatchnet")

########################################
# Write env names to ~/.imc_config
########################################
update_config_var() {
    local key="$1"
    local value="$2"

    touch "$CONFIG_FILE"

    if grep -q "^export ${key}=" "$CONFIG_FILE" 2>/dev/null; then
        sed -i "s|^export ${key}=.*|export ${key}=\"${value}\"|" "$CONFIG_FILE"
    else
        echo "export ${key}=\"${value}\"" >> "$CONFIG_FILE"
    fi
}

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

    local created=false
    if conda env list | awk '{print $1}' | grep -Fx "$env" >/dev/null; then
        echo "⏩ Environment '$env' already exists — skipping creation."
    else
        echo "🌱 Creating environment '$env' from lockfile..."
        $INSTALL_CMD "$env" "$lockfile"
        created=true
    fi

    if [[ -f "$extras" && "$created" == "true" ]]; then
        echo "📦 Installing pip extras for '$env'..."
        conda run -n "$env" pip install -r "$extras"
    elif [[ -f "$extras" ]]; then
        echo "⏭️  Skipping pip extras for existing environment '$env'."
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

########################################
# Persist environment names to config
########################################
update_config_var "IMC_ENV_SEGMENTATION" "imc_segmentation"
update_config_var "IMC_ENV_DENOISE" "imc_denoise"
update_config_var "IMC_ENV_CELLPOSESAM" "imc_cellposesam"
update_config_var "IMC_ENV_BIOBATCHNET" "imc_biobatchnet"
update_config_var "IMC_ENV_SCPORTRAIT" "scPortrait"

echo "🎉 All conda environments installed successfully!"
