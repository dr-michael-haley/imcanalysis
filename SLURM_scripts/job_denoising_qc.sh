#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Generate denoising side-by-side QC and panel consistency checks
#@IN:   general.raw_images_folder + general.denoised_images_folder
#@IN:   general.metadata_folder/panel.csv
#@OUT:  general.qc_folder/denoising/
#@OUT:  panel_consistency_report_*.csv (+ *_pixel_qc.csv)
#@ENV:  imc_denoise
#@MODULE:  SpatialBiologyToolkit.scripts.denoising_qc
#@CONFIG: general, denoising, logging (plus check_panel_consistency defaults)

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_DENOISE:-imc_denoise}"
python -m SpatialBiologyToolkit.scripts.denoising_qc

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.check_panel_consistency 
