#! /bin/bash --login
#SBATCH -p serial 
#SBATCH -t 2-0

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Zip selected sequential execution output directories for download
#@IN:   outputs/ stage paths defined in Bash_scripts/zipqc (legacy QC fallback retained)
#@OUT:  outputs/<execution_id>_Output_Archive/files/<dataset>_<set>_<date>.zip under sbt
#@CONFIG: none

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "ZIP folder job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

zipqc
