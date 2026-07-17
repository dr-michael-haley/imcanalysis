# Environment Diagnostics

## What this stage does

Checks registered job modules and selected imports inside their configured Conda environments.

## Why it is performed

It identifies environment, ABI, import, and wrapper drift before expensive pipeline execution.

## Main inputs

SLURM wrapper metadata and the environment import map.

## Reusable assets produced

No scientific asset.

## Human-facing outputs produced

The technical diagnostic log and a concise stage report.

## Important configuration options

Environment names and import lists maintained by the installation.

## How to interpret the results

Resolve failed entry-point or dependency imports before running the affected stage.

## Common problems and limitations

The diagnostic confirms imports, not full GPU, data, or scientific correctness.
