#! /bin/bash --login
#SBATCH -p gpuA 
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Segment/quantify cells with Nimbus and build AnnData outputs
#@IN:   general.masks_folder, general.metadata_folder/panel.csv, general.metadata_folder/metadata.csv
#@IN:   general.denoised_images_folder (or raw fallback per nimbus settings)
#@OUT:  nimbus.output_dir/ (default nimbus_output/) and general.celltable_folder/nimbus_cell_tables/
#@OUT:  segmentation.anndata_save_path or nimbus.anndata_output (default anndata.h5ad)
#@OUT:  optional segmentation.removed_markers_anndata_path + general.qc_folder/nimbus_normalization_qc/
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.segmentation_nimbus
#@CONFIG: general, segmentation, nimbus, logging


source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.segmentation_nimbus
