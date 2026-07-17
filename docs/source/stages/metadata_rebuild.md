# Metadata Rebuild

## What this stage does

Reconstructs metadata, dictionary, and panel tables from an existing AnnData object.

## Why it is performed

It repairs or recreates project metadata when the AnnData is the most complete surviving source.

## Main inputs

AnnData observations, variables, and existing metadata where preservation is enabled.

## Reusable assets produced

Rebuilt project metadata, dictionary, and panel CSV files.

## Human-facing outputs produced

The run report records selected columns, row counts, and warnings.

## Important configuration options

Input path, output folder, include/exclude patterns, panel mappings, and preservation settings.

## How to interpret the results

Inspect rebuilt tables before using them to drive image or quantification stages.

## Common problems and limitations

Only information represented consistently in AnnData can be reconstructed reliably.
