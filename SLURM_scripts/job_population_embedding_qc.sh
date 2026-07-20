#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: Assess population support from existing graph, UMAP, PCA, and precomputed clustering-sweep state
#@IN:   population_embedding_qc.input_adata_path (fallback: general.anndata_path)
#@IN:   Existing UMAP, optional PCA/connectivities, and population or precomputed Leiden obs columns
#@OUT:  outputs/<execution_id>_Population_Embedding_QC/{figures,tables,summaries,files}/
#@OUT:  population_embedding_qc.annotated_adata_path only when write_annotated_h5ad is enabled
#@ENV:  imc_cellcharter
#@MODULE:  SpatialBiologyToolkit.scripts.population_embedding_qc
#@CONFIG: general, population_embedding_qc, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "Population embedding QC is using ${SLURM_CPUS_PER_TASK:-1} CPU core(s); no GPU is requested"

conda activate "${SBT_CONDA_ENV:-${IMC_ENV_CELLCHARTER:-imc_cellcharter}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m SpatialBiologyToolkit.scripts.population_embedding_qc
