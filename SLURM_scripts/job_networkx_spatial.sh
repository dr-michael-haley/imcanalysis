#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 2-0
#SBATCH -n 16

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Run per-ROI Squidpy/NetworkX spatial graph metrics (assortativity, per-population clustering, bootstrapped nulls, and case aggregation)
#@IN:   networkx_spatial.input_adata_path (fallback: general.anndata_path)
#@IN:   adata.obs keys configured in networkx_spatial (population_obs, roi_obs, case_obs optional, X/Y coords)
#@OUT:  outputs/<execution_id>_NetworkX_Spatial_Analysis/
#@ENV:  imc_cellcharter
#@MODULE:  SpatialBiologyToolkit.scripts.networkx_spatial
#@CONFIG: general, process, networkx_spatial, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "NetworkX spatial job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_CELLCHARTER:-imc_cellcharter}}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.networkx_spatial
