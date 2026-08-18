#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: DNA preprocessing + CellPose-SAM mask generation
#@IN:   general.denoised_images_folder ROI folders (default processed/)
#@IN:   createmasks.dna_image_name channel (default DNA1)
#@OUT:  createmasks.dna_preprocessing_output_folder_name/ (default preprocessed_dna/)
#@OUT:  general.masks_folder/ (default masks/)
#@OUT:  outputs/<execution_id>_Segmentation/ plus reusable preprocessed DNA and masks
#@ENV:  sbt-analysis
#@MODULE:  SpatialBiologyToolkit.scripts.preprocess_dna
#@ENV:  sbt-cellpose-sam
#@MODULE:  SpatialBiologyToolkit.scripts.cellpose_sam
#@CONFIG: general, createmasks, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "CellPose-SAM job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.preprocess_dna

conda activate "${SBT_CONDA_ENV_CELLPOSESAM:-sbt-cellpose-sam}"
python -m SpatialBiologyToolkit.scripts.cellpose_sam
