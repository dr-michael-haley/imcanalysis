# External and old code

This directory contains unmaintained, experimental, and third-party-derived
material. It is excluded from the supported pipeline and Python API. Review
the source and upstream licence before reusing it.

## IMC_Denoise

The bundled notebook implementation of
[IMC_Denoise](https://github.com/PENGLU-WashU/IMC_Denoise/) targeted older
Bodenmiller-pipeline outputs. The supported route is now the `denoise` SLURM
stage and its QC companion `dnqc`.

The retained upstream notes are in
[`External_and_old_code/IMC_Denoise`](https://github.com/dr-michael-haley/imcanalysis/tree/main/External_and_old_code/IMC_Denoise).

## REDSEA

The bundled [REDSEA](https://github.com/labsyspharm/redseapy) adaptation was
an experiment in segmentation-overlap correction. Internal testing found that
it could remove too much positive signal, so it is not recommended for current
production analyses.

## Mike_old_code and Mike_scripts

These directories are historical holding areas. Much of the reusable logic has
since moved into `SpatialBiologyToolkit`; code remaining here has no current
compatibility guarantee.
