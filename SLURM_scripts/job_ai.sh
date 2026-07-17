#! /bin/bash --login
#SBATCH -p himem 
#SBATCH -t 2-0
#SBATCH -n 2

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: AI labeling of Leiden populations and writing *_AIlabel columns
#@IN:   process.output_adata_path (default anndata_processed.h5ad) with Leiden columns
#@IN:   OPENAI_API_KEY environment variable (if visualization.enable_ai=true)
#@OUT:  process.output_adata_path (updated in place)
#@OUT:  outputs/<execution_id>_AI_Interpretation/ (legacy direct fallback: general.qc_folder)
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.ai_interpretation
#@CONFIG: general, visualization, process, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "AI interpretation job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# AI interpretation (optional, adds *_AIlabel columns)

python -m SpatialBiologyToolkit.scripts.ai_interpretation
