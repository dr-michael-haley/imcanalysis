#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 12

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

#@DESC: GPU processing with rapids-singlecell: optional cell filtering, PCA, optional Harmony, neighbors, UMAP, Leiden, optional parameter scan
#@IN:   rapids.input_adata_path (fallback: general.anndata_path)
#@IN:   rapids.batch_correction_obs must exist in AnnData.obs when rapids.run_harmony=true
#@IN:   rapids.filter_obs_key plus optional rapids.filter_min_value/filter_max_value filters cells after load
#@OUT:  rapids.output_adata_path (default general.anndata_path)
#@OUT:  general.qc_folder/rapids.qc_output_subdir (default QC/RapidsProcess)
#@OUT:  general.qc_folder/rapids.qc_output_subdir/Matrixplots/ Leiden MatrixPlots
#@OUT:  general.qc_folder/rapids.qc_output_subdir/ParameterScan/ when rapids.parameter_scan_dict is set
#@ENV:  rapids_singlecell
#@MODULE:  SpatialBiologyToolkit.scripts.basic_process_rapids
#@CONFIG: general, rapids, visualization, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"

echo "RAPIDS single-cell job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

conda activate "${IMC_ENV_RAPIDS_SINGLECELL:-rapids_singlecell}"
# Fix ctypes error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m SpatialBiologyToolkit.scripts.basic_process_rapids
