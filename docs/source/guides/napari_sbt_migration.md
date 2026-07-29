# Migrating the legacy Napari workflows to `napari_sbt`

The legacy module paths remain available while experiments move to
`napari_sbt`. Existing QC folders and explorer annotations are not rewritten.
Create a new experiment beside them and validate its identity joins first.

## From CellPose good/artifact QC

1. Select **All cells** in Setup.
2. Use the **Segmentation QC template**.
3. Add the existing CellPose metrics table as an imported table feature source.
4. Keep CellPose probability, flow, flow-error, and segmentation-configuration
   columns as imported features. They are not recalculated from ordinary IMC
   channels.
5. Confirm the cohort and train the multiclass service with the two template
   classes.

Old binary label CSVs are not silently imported because they may use historical
identity fallbacks. Convert them explicitly to records with `ROI`,
`ObjectNumber`, `class_id`, `state`, `source`, `user`, and `timestamp`, then
validate them against the frozen cohort.

## From `napari_imc_explorer`

Use the same AnnData, mask, image, extra-image, ROI, and object-ID sources.
Categorical/numeric overlays, population RGB inspection, abundance-ranked ROI
navigation, manual regions, and display-layer utilities are represented in the
unified tabs.

New experiments do not allow AnnData row-order fallback. Every eligible cell
must have a unique `(ROI, ObjectNumber)` identity. This prevents a filtered or
reordered AnnData object from being mapped to the wrong mask label.

Population QC now connects directly to Setup. Choose a population and use
**Use this population as classification cohort**; inspect the cohort-only mask
and confirm it before calculating features or labelling cells.

## Deliberate changes

- Viewer workspace persistence is removed.
- Classes are stable records rather than binary constants.
- Exclusion during cleaned-mask export is class-specific.
- Model-based exclusion requires the predicted exclude class and the configured
  confidence threshold.
- Display quantile normalization is never reused for scientific features.
- The original full segmentation remains authoritative for boundaries,
  backgrounds, expansion, and neighbourhood context even when only a subset is
  visible.
- SpatialData-native experiments remain deferred; use identity-aligned AnnData
  during this migration.

Keep the two legacy workflows for an existing analysis until a `napari_sbt`
experiment reproduces its selected identities and imported feature coverage.
The new assignment and annotated-AnnData exports can then become the auditable
handoff to downstream analysis.
