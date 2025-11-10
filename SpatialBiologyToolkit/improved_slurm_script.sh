#!/bin/bash --login
#SBATCH -p himem 
#SBATCH -t 2-0
#SBATCH -o slurm_output_%j.out         # Capture stdout to file with job ID
#SBATCH -e slurm_error_%j.err          # Capture stderr to file with job ID  
#SBATCH --open-mode=append             # Append to output files if they exist
#SBATCH -J backgating_analysis         # Job name for easier identification

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "========================"

# Activate conda environment
echo "Activating conda environment..."
conda activate segmentation
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Set environment variables
export OPENAI_API_KEY="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"
export MPLBACKEND="Agg"
export QT_QPA_PLATFORM="offscreen"
export PYTHONUNBUFFERED=1              # Force unbuffered Python output
export PYTHONIOENCODING=utf-8          # Ensure proper encoding
unset DISPLAY

echo "Environment variables set."

# Run the analysis with proper output handling
echo "Starting backgating analysis..."
echo "Output will be captured in slurm_output_${SLURM_JOB_ID}.out"
echo "Errors will be captured in slurm_error_${SLURM_JOB_ID}.err"

# Use -u flag for unbuffered output and explicit flushing
python -u -m SpatialBiologyToolkit.scripts.basic_visualizations 2>&1 | tee -a "analysis_log_${SLURM_JOB_ID}.txt"

# Check exit status
EXIT_CODE=$?
echo "=== Job Completion ==="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully!"
else
    echo "Job failed with exit code $EXIT_CODE"
fi

echo "Check these files for output:"
echo "  - slurm_output_${SLURM_JOB_ID}.out (stdout)"
echo "  - slurm_error_${SLURM_JOB_ID}.err (stderr)"  
echo "  - analysis_log_${SLURM_JOB_ID}.txt (combined output)"
echo "  - pipeline.log (application log)"