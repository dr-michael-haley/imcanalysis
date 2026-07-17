# Observation Remapping

## What this stage does

Generates an editable remap template or applies CSV mappings to AnnData observation columns.

## Why it is performed

It supports transparent curation and consolidation of cluster or population labels.

## Main inputs

AnnData and the configured remap CSV.

## Reusable assets produced

The remap table and updated AnnData annotations.

## Human-facing outputs produced

Template summaries, counts, marker hints, and the execution report.

## Important configuration options

Mode, source column, generated columns, completeness, overwrite policy, and marker summaries.

## How to interpret the results

Review mappings for missing or duplicated labels before applying them.

## Common problems and limitations

Partial mappings and unexpected type conversion can leave values unmapped.
