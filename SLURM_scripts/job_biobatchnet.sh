#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Batch correction with BioBatchNet + UMAP/Leiden post-processing
#@IN:   process.input_adata_path (default anndata.h5ad)
#@IN:   process.batch_correction_obs must exist in AnnData.obs
#@OUT:  process.output_adata_path (default anndata_processed.h5ad)
#@OUT:  general.qc_folder/BioBatchNet/
#@ENV:  imc_biobatchnet
#@MODULE:  SpatialBiologyToolkit.scripts.basic_process_biobatchnet
#@CONFIG: general, process, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "BioBatchNet job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_BIOBATCHNET:-imc_biobatchnet}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.basic_process_biobatchnet
