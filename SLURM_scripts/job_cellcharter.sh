#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Identify spatial neighborhoods with CellCharter (TRVAE reduction by default) and save QC summaries
#@IN:   cellcharter.input_adata_path (fallback: process.output_adata_path, then process.input_adata_path)
#@IN:   adata.obs sample key (default cellcharter.sample_key=ROI) and spatial coords (obsm['spatial'] or X_loc/Y_loc)
#@OUT:  cellcharter.output_adata_path (default anndata_cellcharter.h5ad)
#@OUT:  outputs/<execution_id>_CellCharter_Neighbourhoods/ (legacy direct fallback: general.qc_folder)
#@ENV:  imc_cellcharter
#@MODULE:  SpatialBiologyToolkit.scripts.cellcharter_neighborhoods
#@CONFIG: general, process, cellcharter, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "CellCharter job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_CELLCHARTER:-imc_cellcharter}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.cellcharter_neighborhoods
