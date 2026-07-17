# Output Archive

## What this stage does

Creates a portable ZIP archive from selected human-facing stage outputs.

## Why it is performed

It simplifies transfer, review, and handover of report material without duplicating large reusable assets.

## Main inputs

Numbered stage output folders, with deprecated QC paths supported as a fallback.

## Reusable assets produced

No scientific computational asset.

## Human-facing outputs produced

A dated ZIP archive recorded in this stage's run folder.

## Important configuration options

Archive set selection in the `zipqc` helper.

## How to interpret the results

Treat the archive as a convenience copy; canonical reports remain in the project outputs folder.

## Common problems and limitations

Missing optional folders are skipped, and very large output sets may be slow to compress.
