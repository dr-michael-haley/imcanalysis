#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Run STARLING segmentation-aware probabilistic phenotyping from IMC AnnData marker expression
#@IN:   starling.input_adata_path (fallback: general.anndata_path)
#@IN:   starling.initial_label_obs or general.population_obs_primary when starling.initial_clustering_method=User
#@IN:   adata.X by default, or starling.use_layer if set; optional starling.marker_include/marker_exclude
#@OUT:  starling.output_adata_path (default general.anndata_path)
#@OUT:  outputs/<execution_id>_STARLING_Phenotyping/
#@ENV:  imc_starling
#@MODULE:  SpatialBiologyToolkit.scripts.starling_analysis
#@CONFIG: general, starling, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "STARLING job is using ${SLURM_GPUS:-0} GPU(s) with ID(s) ${CUDA_VISIBLE_DEVICES:-none} and ${SLURM_NTASKS:-${SLURM_NTASKS_PER_NODE:-1}} CPU core(s)"

conda activate "${IMC_ENV_STARLING:-imc_starling}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.starling_analysis
