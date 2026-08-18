#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=256G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -euo pipefail

#@DESC: Learn empirical marker halos, calculate neighbour-attributable fractions, and retain spatial source-cell provenance
#@IN:   general.anndata_path with marker-aligned X, optional manual exemplar_obs, and ROI/object-label mapping
#@IN:   general.raw_images_folder ROI/channel TIFFs and general.masks_folder label masks
#@OUT:  neighbour_signal.output_adata_path with halo scores, raw-intensity layers, and marker_halo provenance
#@OUT:  neighbour_signal.source_target_table_path with sparse source-target provenance for max aggregation
#@OUT:  outputs/<execution_id>_Neighbour_Attributable_Signal/{figures,tables,summaries}/
#@ENV:  sbt-analysis
#@MODULE:  SpatialBiologyToolkit.scripts.neighbour_signal
#@CONFIG: general, neighbour_signal, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_ANALYSIS:-sbt-analysis}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python -m SpatialBiologyToolkit.scripts.neighbour_signal
