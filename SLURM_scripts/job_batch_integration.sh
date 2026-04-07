#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Batch integration with Harmony and/or BBKNN, followed by UMAP/Leiden post-processing
#@IN:   batch_integration.input_adata_path (fallback: general.anndata_path)
#@IN:   batch_integration.batch_correction_obs must exist in AnnData.obs for Harmony/BBKNN modes
#@OUT:  batch_integration.output_adata_path (default general.anndata_path)
#@OUT:  general.qc_folder/batch_integration.qc_output_subdir (default QC/BatchIntegration)
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.basic_process_batch_integration
#@CONFIG: general, batch_integration, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Batch integration job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_CELLCHARTER:-imc_cellcharter}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.basic_process_batch_integration
