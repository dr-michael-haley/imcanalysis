#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 12
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Run CellVision extraction, VICReg embedding, RAPIDS clustering, and plotting in one GPU job
#@IN:   general.anndata_path, cellvision.input_adata_path override, and configured cellvision population selection
#@IN:   configured CellVision source/fusion AnnData containing cellvision.fusion_intensity_representation when fusion is enabled
#@IN:   general.denoised_images_folder/general.masks_folder or cellvision image/mask overrides
#@OUT:  cellvision.asset_folder with H5SC, identity table, VICReg checkpoint, embeddings, and clustered AnnData
#@OUT:  outputs/<execution_id>_CellVision_Full/ training diagnostics, cluster-explanation QC, comparisons, projections, and galleries
#@ENV:  sbt-scportrait
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_extract
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_embed
#@ENV:  sbt-analysis
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_cluster
#@ENV:  sbt-scportrait
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_plot
#@CONFIG: general, cellvision, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "CellVision full job is using ${SLURM_GPUS:-0} GPU(s) with ID(s) ${CUDA_VISIBLE_DEVICES:-none} and ${SLURM_NTASKS:-1} CPU core(s)"

SCPORTRAIT_ENV="${SBT_CONDA_ENV_SCPORTRAIT:-sbt-scportrait}"
ANALYSIS_ENV="${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}"

conda activate "$SCPORTRAIT_ENV"
python -m SpatialBiologyToolkit.scripts.cellvision_extract
python -m SpatialBiologyToolkit.scripts.cellvision_embed

_CELLVISION_ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
conda activate "$ANALYSIS_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.cellvision_cluster

conda activate "$SCPORTRAIT_ENV"
if [[ -n "$_CELLVISION_ORIGINAL_LD_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH="$_CELLVISION_ORIGINAL_LD_LIBRARY_PATH"
else
    unset LD_LIBRARY_PATH
fi
python -m SpatialBiologyToolkit.scripts.cellvision_plot
