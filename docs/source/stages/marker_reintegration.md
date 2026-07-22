# Marker Reintegration

## Why markers are removed and later restored

An IMC panel often contains channels that remain biologically informative but are unsuitable for defining cell-to-cell similarity. Examples include a marker with high background, staining available for only part of a cohort, a technically variable antibody, a channel dominated by segmentation artefact, or a functional marker whose strong treatment response would overwhelm the lineage structure needed for population annotation.

During segmentation, markers listed in `segmentation.remove_and_store_markers` are copied into a separate AnnData object and removed from the main AnnData feature matrix. The main object then proceeds through integration, dimensionality reduction, clustering, and annotation without those variables. Marker reintegration adds the stored variables back after those representation-defining steps are complete.

This separation supports an important analytical distinction:

- **features used to construct populations** determine the neighbour graph, embedding, or clustering; and
- **features used to describe populations** can be overlaid afterwards as independent biological context.

For example, a highly treatment-responsive phosphoprotein may be inappropriate for defining cell identity but useful for comparing signalling within an already established population. Restoring that marker lets it appear in UMAP colouring, matrix plots, image validation, and targeted downstream summaries without retroactively changing the clusters.

Exclusion should not be used to hide inconvenient biology. The reason for removing each marker should be recorded, and its quality should still be inspected in the source images. A biologically central lineage marker with poor technical performance may make the corresponding cell type intrinsically uncertain, even if removing it produces a visually cleaner embedding.

## What the stage does

The stage loads two objects:

1. the current canonical AnnData at `general.anndata_path`, containing the processed cells and retained markers; and
2. the stored AnnData at `segmentation.removed_markers_anndata_path`, containing the markers separated during segmentation.

It verifies that both objects have exactly the same observation names in exactly the same order. It then concatenates them along the variable axis, so the removed markers become additional columns in `.X`. AnnData layers are concatenated at the same time. The combined object replaces the canonical AnnData in place, while the normal execution report records the removed-marker input path.

The stage does not recluster cells, recompute neighbours, change the UMAP, rerun batch integration, renormalise intensities, or create population labels. Existing observations, embeddings, graphs, and unstructured metadata are taken from the processed main object. Consequently, existing clusters and embeddings remain representations of the marker set that was present *before* reintegration.

## Cell identity and ordering are strict

Safe reintegration requires a one-to-one correspondence between the rows of the two AnnData objects. The implementation deliberately requires `obs_names` to be identical and ordered identically; it does not attempt to sort, join, or guess correspondence from another identifier.

This protects against a serious error in which a removed marker value from one segmented cell is attached to a different cell. It also means that filtering cells, changing observation names, or reordering the main AnnData after the split prevents reintegration unless the same operation was applied identically to the stored object.

The removed-marker object should therefore be treated as a row-aligned companion to the segmentation output. Do not independently subset or reorder it. A shared number of cells is not enough: the observation names and their sequence must match.

## Marker names and AnnData layers

If a marker name is already present in the processed object, the copy from the removed-marker object is skipped to avoid duplicate variables. Reintegration fails if every stored marker overlaps, because there is then nothing unique to add. Review unexpected overlaps: they can indicate that reintegration was already performed, that marker naming changed, or that the wrong files were paired.

Common layers are joined marker by marker. If a layer exists in only one object, AnnData's outer concatenation retains the layer and fills the unavailable side with missing values. This is scientifically preferable to silently inventing measurements, but it means a downstream function may encounter `NaN` values for some marker–layer combinations. Check the final layer structure and use only a layer that contains meaningful values for the markers under study.

The main object's `.obs`, `.uns`, `.obsm`, and `.obsp` information is preserved. Thus sample metadata, population labels, UMAP coordinates, and neighbour graphs continue to describe the processed dataset. The stored object's embeddings or graphs are not used.

## Processing history and measurement scale

The removed markers are frozen when the segmentation stage creates their companion AnnData. They do not participate in later operations applied only to the main feature matrix. Reintegration simply appends the stored arrays; it does not transform them to match changes that may have happened to retained markers.

This can produce a combined AnnData in which different variables have different analytical histories. Whether this matters depends on the upstream pipeline: an integrated representation stored in `obsm` may leave `.X` unchanged, whereas a procedure that modifies `.X` or a layer can make retained and restored columns non-comparable on that matrix. Always establish what `.X` and each layer represent before comparing marker magnitudes.

Reintegrated values are most defensible for:

- colouring an existing embedding to ask where high- and low-valued cells occur;
- describing or comparing values within pre-existing biological populations;
- selecting informative source-image channels for backgating;
- generating marker summaries with an explicitly appropriate layer; and
- retaining measurements for specialist analyses that account for their technical limitations.

They should not automatically be included in a new unsupervised neighbour graph, clustering, or batch-integration fit. Once restored, they are ordinary AnnData variables and the code does not add a dedicated “excluded from clustering” flag. Any downstream method that consumes all variables may use them unless features are explicitly selected. Keep the configured removal list as provenance and make feature selection deliberate.

## Biological interpretation

Because the clusters were formed without the restored marker, association between that marker and a cluster can provide useful descriptive evidence rather than circularly restating a clustering feature. For example, elevated activation-marker values in one T-cell cluster may support interpreting it as an activated state.

However, the association is not automatically independent in a causal or statistical sense. The restored marker may correlate with retained markers, batch, treatment, ROI, tissue compartment, or staining quality. Cell-level tests can also produce extremely small P values by treating thousands of cells from the same patient as independent. Biological comparisons should retain case and ROI structure and inspect whether the effect is consistent across biological replicates.

A marker excluded because of poor signal remains poor after reintegration. Restoring it preserves access; it does not repair staining, normalise batches, or validate specificity. Apparent cluster differences should be checked against raw or denoised images, background patterns, missingness, and sample-level distributions.

## Outputs and checks

The principal output is the canonical AnnData with additional variables in `.X` and compatible layers. No scientific figure is generated by this stage; figures are created by later visualisation or targeted analysis stages.

After reintegration, verify:

- the expected number and names of restored markers;
- exact preservation of the cell count, observation order, annotations, embeddings, and graphs;
- the presence and missing-value pattern of each relevant layer;
- the numeric scale and distribution of restored versus retained markers;
- that clustering and integration outputs were not unintentionally recomputed; and
- that downstream feature-selection code still excludes technically unsuitable channels where required.

## Limitations and failure modes

- Cell filtering, renaming, or reordering after the original split breaks the required row alignment.
- Pairing objects from different runs can be catastrophic and is rejected only when their observation indexes differ; meaningful stable identifiers remain essential.
- Relative removed-marker paths must resolve to the file produced during segmentation. Moving the canonical AnnData or running stages from inconsistent working directories can make the companion file difficult to locate.
- Layers present on only one side contain missing values for the other marker set after the outer join.
- The stage does not harmonise processing history or measurement scale.
- Restored markers can accidentally enter later all-feature analyses unless feature selection is explicit.

Marker reintegration is therefore best understood as a controlled restoration of descriptive measurements, not as a reversal of the scientific decision to exclude those measurements from population construction.
