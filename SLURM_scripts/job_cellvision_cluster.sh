#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 12
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Fuse CellVision morphology and BioBatchNet intensity graphs, then run RAPIDS UMAP and Leiden
#@IN:   cellvision.asset_folder/cellvision_embeddings.h5ad
#@IN:   cellvision.fusion_intensity_adata_path or CellVision source AnnData with the configured BioBatchNet obsm representation
#@OUT:  cellvision.asset_folder/cellvision_clustered.h5ad
#@ENV:  rapids_singlecell
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_cluster
#@CONFIG: general, cellvision, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_RAPIDS:-${IMC_ENV_RAPIDS_SINGLECELL:-rapids_singlecell}}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.cellvision_cluster
