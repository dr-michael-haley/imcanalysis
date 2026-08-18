#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 2-0
#SBATCH -n 6

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Run checkpointed population subclustering (template generation, subclustering QC, optional remap integration)
#@IN:   subclustering.input_adata_path (fallback: process.output_adata_path, then process.input_adata_path)
#@IN:   subclustering/sublustering_settings.csv and subclustering/marker_list.csv (created automatically on first run)
#@OUT:  subclustering.output_adata_path (default anndata_subclustered.h5ad)
#@OUT:  subclustering.output_subdir (default subclustering/) with settings, marker list, figures, remap CSVs
#@OUT:  outputs/<execution_id>_Subclustering/figures/ for human-facing plots under sbt
#@ENV:  sbt-analysis
#@MODULE:  SpatialBiologyToolkit.scripts.subclustering
#@CONFIG: general, process, subclustering, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Subclustering job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.subclustering
