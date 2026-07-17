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

The current wrapper uses external command-line defaults and environment configuration.

## How to interpret the results

Validate cell-image crops against masks and source images before downstream use.

## Common problems and limitations

This stage depends on an external repository and currently has limited project-config integration.
