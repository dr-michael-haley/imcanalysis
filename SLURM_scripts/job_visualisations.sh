#! /bin/bash --login
#SBATCH -p multicore 
#SBATCH -t 1-0
#SBATCH -n 8

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Generate UMAP/matrix/overlay/population visualization outputs
#@IN:   visualization.input_adata_path or process.output_adata_path
#@IN:   general.masks_folder, general.denoised_images_folder, general.metadata_folder
#@OUT:  outputs/<execution_id>_Visualisation/
#@ENV:  sbt-analysis
#@MODULE:  SpatialBiologyToolkit.scripts.basic_visualizations
#@CONFIG: general, visualization, process, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Visualisations job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.basic_visualizations
