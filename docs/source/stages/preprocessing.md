# Preprocessing

## What this stage does

Imports raw IMC acquisitions, exports multiplex TIFF stacks and per-channel images, and builds the project metadata and panel tables.

## Why it is performed

It converts instrument-oriented input into stable, readable assets used by all later image and cell-analysis stages.

## Main inputs

Raw MCD or TXT acquisitions in the configured IMC input folder.

## Reusable assets produced

TIFF stacks, unstacked channel images, and metadata, dictionary, and panel tables.

## Human-facing outputs produced

The run report summarises imported files, generated assets, warnings, and relevant import settings.

## Important configuration options

Input/output folders and preprocessing ROI-size checks.

## How to interpret the results

Confirm that expected acquisitions and channels were imported and that the panel table describes the intended markers.

## Common problems and limitations

Malformed acquisitions, inconsistent channel names, and very small ROIs can prevent complete import.
