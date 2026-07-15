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
#@OUT:  general.qc_folder/DNA_preprocessing_QC/ and general.qc_folder/CellposeSAM_QC/
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.preprocess_dna
#@ENV:  imc_cellposesam
#@MODULE:  SpatialBiologyToolkit.scripts.cellpose_sam
#@CONFIG: general, createmasks, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "CellPose-SAM job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.preprocess_dna

conda activate "${IMC_ENV_CELLPOSESAM:-imc_cellposesam}"
python -m SpatialBiologyToolkit.scripts.cellpose_sam
