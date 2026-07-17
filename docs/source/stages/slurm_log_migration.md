# Legacy SLURM Log Migration

## What this stage does

Organises legacy SLURM logs using transitional AnnData pipeline metadata.

## Why it is performed

It helps reconcile projects created before project-scoped `.sbt/runs/` records existed.

## Main inputs

Legacy SLURM output files and AnnData stage-log entries.

## Reusable assets produced

An organised legacy log folder and verification manifest.

## Human-facing outputs produced

Verification tables and links to migrated technical logs.

## Important configuration options

Legacy log folder and AnnData provenance key.

## How to interpret the results

Unverified files require manual matching; new managed runs already record exact log paths.

## Common problems and limitations

Incomplete historical job IDs can prevent confident matching.
