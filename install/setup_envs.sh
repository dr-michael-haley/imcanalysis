#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(dirname "$0")/../HPC_env_files"
ENV_ROOT="$HOME/miniconda3/envs"

# List of all environments
ENVS=("imc_segmentation" "imc_denoise" "imc_cellposesam" "imc_biobatchnet")

# Detect mamba or conda
if command -v mamba >/dev/null 2>&1; then
    CREATE="mamba create -y"
else
    CREATE="conda create -y"
fi

echo "🔧 Using create command: $CREATE"
echo

create_env() {
    local env="$1"
    local env_dir="$BASE_DIR/$env"
    local lockfile="$env_dir/conda-linux-64.lock"
    local extras="$env_dir/pip-extras.txt"

    echo "============================="
    echo "📦 Installing environment: $env"
    echo "============================="

    if [ ! -f "$lockfile" ]; then
        echo "❌ ERROR: Missing lockfile: $lockfile"
        exit 1
    fi

    if conda env list | awk '{print $1}' | grep -Fx "$env" >/dev/null; then
        echo "⏩ Environment '$env' already exists — skipping create."
    else
        echo "🌱 Creating conda environment '$env'..."
        $CREATE -n "$env" --file "$lockfile"
    fi

    if [ -f "$extras" ]; then
        echo "📦 Installing pip extras for '$env'..."
        source "$ENV_ROOT/$env/bin/activate"
        pip install -r "$extras"
        deactivate || true
    fi

    echo "✔ Finished environment: $env"
    echo
}

for env in "${ENVS[@]}"; do
    create_env "$env"
done

echo "🎉 All conda environments installed!"
