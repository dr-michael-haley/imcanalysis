# Conda environments

Environment specs used for the toolkit.

- `conda_environment.yml`: current lightweight environment (Python 3.12) recommended for most users before running `pip install -e .`.
- `conda_environment_specific.yml`: fully pinned, heavier environment kept for reproducibility; use if you need exact versions or hit compatibility issues.
- `athena_environment.yml`: older Python 3.8 environment kept for Athena/legacy systems.

These files are not auto-synced with the latest code; treat the pinned ones as reference when debugging dependency problems.

