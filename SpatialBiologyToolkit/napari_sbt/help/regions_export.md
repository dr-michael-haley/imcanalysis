# Regions & Export

Regions & Export connects manual spatial annotation with cell classifications and
produces interoperable outputs.

## Manual tissue regions

Enter a region name and click **Create/select regions layer**. Draw or edit shapes
using Napari's Shapes controls, then click **Synchronize regions to cell table**.
Membership is assigned from cell centroids, so inspect boundary cells when regions
touch or overlap. Synchronize again after editing shapes.

## Cohort results and exports

Final classification decisions, CSV/Parquet output, and AnnData output now live
in **Classify → Finalize & export**, immediately after model review. This keeps
threshold selection, final identity creation, and export in one guided sequence.
This Regions & Export box is reserved for derived mask outputs.

Cohort-mask export preserves original object IDs while setting other pixels to
zero. Cleaned-mask export is available only for classes marked `exclude`.
Prediction-based exclusion additionally requires the current final-identity
confidence, entropy, and probability-margin rules. Original masks, images, and
source AnnData are never overwritten by default.

Cohort masks are safe subset views. Create or refresh final identities in Classify,
then review class dispositions and the reported decision thresholds carefully
before requesting cleaned masks.
