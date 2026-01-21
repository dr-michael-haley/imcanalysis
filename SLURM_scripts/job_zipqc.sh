#! /bin/bash --login
#SBATCH -p serial 
#SBATCH -t 2-0

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: ZIPs up directories ready for download
#@IN:   Several QC directories
#@OUT:  zip file

echo "ZIP folder job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

zipqc