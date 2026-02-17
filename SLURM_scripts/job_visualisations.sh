#! /bin/bash --login
#SBATCH -p himem 
#SBATCH -t 2-0
#SBATCH -n 2

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Generate UMAP/matrix/overlay/population visualization outputs
#@IN:   visualization.input_adata_path or process.output_adata_path
#@IN:   general.masks_folder, general.denoised_images_folder, general.metadata_folder
#@OUT:  general.qc_folder/BasicProcess_QC/
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.basic_visualizations
#@CONFIG: general, visualization, process, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Visualisations job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.basic_visualizations
