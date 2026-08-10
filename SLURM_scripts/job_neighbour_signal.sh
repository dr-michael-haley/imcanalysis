#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -euo pipefail

#@DESC: Learn empirical marker halos and calculate cell-by-marker neighbour-attributable fractions
#@IN:   general.anndata_path with neighbour_signal.exemplar_obs and ROI/object-label mapping
#@IN:   general.raw_images_folder ROI/channel TIFFs and general.masks_folder label masks
#@OUT:  neighbour_signal.output_adata_path with halo scores, raw-intensity layers, and marker_halo provenance
#@OUT:  outputs/<execution_id>_Neighbour_Attributable_Signal/{figures,tables,summaries}/
#@ENV:  imc_segmentation
#@MODULE:  SpatialBiologyToolkit.scripts.neighbour_signal
#@CONFIG: general, neighbour_signal, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_SEGMENTATION:-imc_segmentation}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python -m SpatialBiologyToolkit.scripts.neighbour_signal
