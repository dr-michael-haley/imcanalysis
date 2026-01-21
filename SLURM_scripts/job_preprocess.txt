#! /bin/bash --login
#SBATCH -p himem 
#SBATCH -t 2-0
#SBATCH -n 4

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Extracts TIFFs from MCD files and sets up metadata folders, including panel file
#@IN:   /MCD_files/
#@OUT:  /tiffs, /tiff_stacks, /metadata, /metadata/dictionary.csv, /metadata/panel.csv

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Preprocessing job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_SEGMENTATION:-imc_segmentation}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m SpatialBiologyToolkit.scripts.preprocess