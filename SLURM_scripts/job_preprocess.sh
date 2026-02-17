#! /bin/bash --login
#SBATCH -p himem 
#SBATCH -t 2-0
#SBATCH -n 4

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Import IMC files, export TIFF stacks, unstack channels, and build metadata/panel tables
#@IN:   general.imc_files_folder (default IMC_files/, fallback general.mcd_files_folder)
#@OUT:  general.tiff_stacks_folder (default tiff_stacks/)
#@OUT:  general.raw_images_folder (default tiffs/)
#@OUT:  general.metadata_folder/{metadata.csv,dictionary.csv,panel.csv[,panel_mapping.csv]}
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.preprocess
#@CONFIG: general, preprocess, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Preprocessing job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.preprocess
