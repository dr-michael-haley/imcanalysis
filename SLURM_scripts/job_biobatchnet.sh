#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 4-0
#SBATCH -n 12

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Batch correction with BioBatchNet + UMAP/Leiden post-processing
#@IN:   biobatchnet.input_adata_path (fallback: general.anndata_path)
#@IN:   biobatchnet.batch_correction_obs must exist in AnnData.obs
#@OUT:  biobatchnet.output_adata_path (default general.anndata_path)
#@OUT:  outputs/008_BioBatchNet_Integration/<run_id>/ (legacy direct fallback: general.qc_folder)
#@ENV:  imc_biobatchnet
#@MODULE:  SpatialBiologyToolkit.scripts.basic_process_biobatchnet
#@CONFIG: general, biobatchnet, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "BioBatchNet job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_BIOBATCHNET:-imc_biobatchnet}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.basic_process_biobatchnet
