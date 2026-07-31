# Regions & Export

Regions & Export connects manual spatial annotation with cell classifications and
produces interoperable outputs. Create or select a tissue-regions layer, draw the
regions in Napari, then synchronize membership back to the cell table using cell
centroids.

Assignment exports are cohort-only and contain identity, source population,
confirmed or predicted class, assignment source, confidence, uncertainty, and
per-class probabilities. Confirmed labels override model predictions; proposals
are not final assignments.

Annotated AnnData export writes a new copy by default. Eligible cells receive
subclass, source, confidence, and uncertainty observations; cells outside the
cohort remain missing for subclass fields. A combined observation retains the
original broad population outside the cohort and substitutes subclasses inside
it. Probability matrices contain `NaN` outside the cohort, and provenance is
stored under `uns["napari_sbt"]`.

Cohort-mask export preserves original object IDs while setting other pixels to
zero. Cleaned-mask export is available only for classes marked `exclude`.
Prediction-based exclusion additionally requires the selected confidence
threshold. Original masks, images, and source AnnData are never overwritten by
default.
