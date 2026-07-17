# Marker Reintegration

## What this stage does

Rejoins markers previously stored outside the processed AnnData.

## Why it is performed

Some markers may be excluded from clustering but needed later for interpretation or visualisation.

## Main inputs

The processed AnnData and the configured AnnData containing removed markers.

## Reusable assets produced

An updated canonical AnnData with aligned reintegrated markers.

## Human-facing outputs produced

The report records paths, settings, and completion status; no major figures are expected.

## Important configuration options

Processed and removed-marker AnnData paths and marker alignment settings.

## How to interpret the results

Confirm that cells and marker names align and that the expected variables are present after reintegration.

## Common problems and limitations

Changed cell indexes or incompatible AnnData objects prevent safe reintegration.
