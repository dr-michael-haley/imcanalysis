# scPortrait Export

## What this stage does

Invokes the external scPortrait converter to generate single-cell portrait assets.

## Why it is performed

It supports image-based inspection and downstream workflows centred on individual segmented cells.

## Main inputs

Processed channel images and segmentation masks.

## Reusable assets produced

The external scPortrait project output.

## Human-facing outputs produced

The report links to the generated project and technical job logs.

## Important configuration options

The wrapper reads `config.yaml` (or `SBT_CONFIG`) using the typed configuration:

- `general.denoised_images_folder`: channel images, default `processed`.
- `general.masks_folder`: segmentation masks, default `masks`.
- `scportrait.projects_root`: reusable output, default `scPortrait`.

Set `SBT_SCPORTRAIT_CONVERTER` to change the external converter location; its
default remains `~/scPortrait_to_IMC/imc_to_single_cells.py`. Paths containing
spaces are passed as single arguments. Relative dataset paths use the project
working directory. Without a config, standalone legacy calls retain all defaults;
an explicitly selected `SBT_CONFIG` must exist. Existing `--overwrite`,
`--mask-expand-px 0`, and `--debug` behaviour is preserved.

## How to interpret the results

Validate cell-image crops against masks and source images before downstream use.

## Common problems and limitations

This stage depends on an external repository and the `sbt-scportrait` environment.
The [Linux/EC2 guide](../getting_started/ec2.md) describes machine path overrides.
