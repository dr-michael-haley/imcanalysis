# HyPERSTAC spatial environments

## What this stage does

`sbt run hyperstac-environments` discovers spatial environments from existing
HyPERSTAC patch representations. It uses CellCharter one-hop aggregation on
physical-radius graphs, followed by seeded CPU scikit-learn Gaussian mixtures.
It does not train a new image encoder or recalculate patch perturbations.

## Why it is performed

Compare focal-patch clustering with tissue context at explicitly stated physical
scales. Original Leiden labels are interpretation references, not input features.

## Main inputs

Under `hyperstac.asset_folder`: representation, patch-metric and permutation
AnnData. Unique patch identifiers must match exactly. Coordinates are micrometres
on a regular square grid of non-overlapping tiles. Supply a ROI/case CSV and an
existing `hyperstac_environments.reference_cluster`. Missing case mappings remain
in discovery and use ROI groups in resampling; they are excluded from case tables.

## Reusable assets produced or modified

A separate `hyperstac_environments.output_folder` contains spatial_environments.h5ad,
PCA, graphs, mixture models, input hashes and per-setting checkpoints. Original
AnnData is read-only. Resume requires exactly matching source hashes/config;
change the output folder for a different experiment. Models are trusted local
joblib artifacts and must not be loaded from untrusted sources.

## Human-facing outputs

The managed SBT report contains the scorecard, patient-resampling results and
stability figure. Detailed reusable tables include central and surrounding metric
profiles (mean, median, quartiles, finite counts, equal-ROI and equal-case means),
profiles stratified by original Leiden label, reference-label compositions,
assignment probabilities, ROI maps and case fractions under both patch and
equal-ROI weighting. Each centre contributes once to abundance; overlapping
neighbourhood members are not counted again as tissue area.

## Important configuration options

Use radii 0,100,150,200,300 micrometres as an example for a 100-micrometre grid;
zero uses only focal features. Nonzero radii concatenate focal PCs and neighbour
mean PCs. No per-sample scaling, whitening or batch integration is performed.
Component counts, fit repeats, PCA dimensions, seed, CPU threads, covariance
regularization and patient subsampling are explicit typed settings. The default
diagonal covariance is a bounded baseline; full covariance can be compared later.
Run `sbt plan hyperstac-environments --backend local` then a managed dry run before
launching. Existing `analysis` environment supplies CellCharter; no new environment
or image-model dependency is required. The wrapper requests four CPUs and 16 GiB;
local numerical thread pools are limited independently by `cpu_threads`.

## How to interpret the results

Radius is centre-to-centre Euclidean distance, not a graph-hop count. Links stay
within each ROI and within connected components of the edge-adjacent retained
grid. This prevents links between disconnected islands but can cross an internal
hole within a connected component. Counts and ROI-edge truncation are reported;
tissue-mask coverage is not inferred. Isolates are unassigned at nonzero radii.

Mixture probabilities are model membership, not calibrated biological confidence.
Repeated-fit ARI measures initialization sensitivity. Patient subsampling refits
PCA and GMM, predicts omitted patients, and compares to the full-data reference.
This is sensitivity to cohort composition, not independent generalization:
the reference partition and pretrained encoder have seen the full cohort.
Whole-patient grouping keeps all available ROIs together. Unmapped ROIs are
identified separately. Smooth maps are an expected consequence of aggregation,
not evidence of biological truth. BIC is only comparable within a radius/feature
representation, not across different-dimensional radius-zero and contextual fits.

Patch sensitivity summaries describe the original encoder; neighbourhood averages
are not a causal perturbation experiment on environment assignments. Stratified
profiles expose mixtures of patch identities hidden by overall averages.

## Common problems and limitations

The stage fails on identity mismatch, duplicate coordinates, an invalid grid,
insufficient supported centres, or no converged mixture initialization. Individual
nonconverged fits are counted and omitted from stability averages. Numerical errors
propagate to SBT. Source normalization and clipping remain unchanged. Select
environments using stability, case support and matched image review before using
outcome associations. Existing Cox machinery can consume the output AnnData's
`env_` columns in a separate configured stage.
