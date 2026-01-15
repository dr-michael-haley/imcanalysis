# SLURM job templates

Batch scripts for running pipeline stages on HPC (tested on CSF3). Each job calls a module from `SpatialBiologyToolkit/scripts`.

- `pipeline.conf` maps short stage names (for example `preprocess`, `denoise`, `aiinter`, `vis`) to job files. The `pl` helper reads this file.
- `job_*.txt` files are sbatch templates; metadata lines starting `#@DESC`, `#@IN`, `#@OUT`, `#@CONFIG` are shown by `pl --list`.
- `job_env.sh` sets module loads and headless plotting variables used by each job.
- The job scripts must be executable. `make install` or `make envs` will set permissions automatically. If you see `Permission denied`, run:
	- `chmod +x ~/imcanalysis/SLURM_scripts/*.txt`

## Typical use
1. Install the helper scripts (`make install`) so `pl`, `pll`, and `pls` are on PATH.
2. From a dataset folder with `config.yaml`, submit a chain of stages, e.g. `pl preprocess denoise aiinter vis`. Use `--dry-run` to review without submitting.
3. Inspect or edit individual job files to change resources, partitions, or the conda environment they activate.
4. Check recent submissions with `pls` (reads `pipeline_log.log` in the working directory).

