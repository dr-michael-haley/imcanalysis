#! /bin/bash --login
#SBATCH -p serial 
#SBATCH -t 2-0

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Zip selected QC output directories for download
#@IN:   QC paths defined in Bash_scripts/zipqc (default set)
#@OUT:  <dataset_dir>_<set>_<YYYY-MM-DD>.zip
#@CONFIG: none

echo "ZIP folder job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

zipqc
