# Visualisation

## Purpose of the stage

The visualisation stage converts a processed single-cell IMC dataset into views that can be inspected by a biologist. It does not perform one single scientific test. Instead, it brings together several complementary questions:

- Do the proposed populations have coherent marker phenotypes?
- Do cells with the same label occupy plausible regions of the existing cell-state embedding?
- Where do those cells occur in the original tissue?
- Do the segmented objects and source-channel staining support their assigned labels?
- How common are the populations in each ROI, case, and biological group?

These views are designed for quality control, biological interpretation, and communication. Agreement between several views is more convincing than any one plot. A visually separated UMAP group with an implausible image phenotype should not be accepted simply because it is separated; similarly, overlapping UMAP positions do not disprove a spatially or phenotypically meaningful population.

The stage mainly reads existing analyses. It does not compute a new UMAP, create new population labels, or correct technical effects. It can add or refresh category colour information in AnnData when plotting and saves the AnnData through the normal pipeline checkpoint mechanism.

## Inputs and automatic annotation selection

The central input is an AnnData object containing cells in rows, measured markers in columns, and relevant annotations in `adata.obs`. Different visualisation modules also use:

- `adata.obsm['X_umap']` for all UMAP plots;
- labelled segmentation masks for tissue reconstruction and density estimation;
- denoised channel images for backgating;
- ROI, case, comparison-group, and optional compartment observations; and
- optional ROI areas in `adata.uns['sample']['mm2']`.

Population columns should normally be set explicitly through `visualization.population_columns` or the shared population settings. If they are not supplied, the code searches observation names for terms such as `population`, `cluster`, `leiden`, `louvain`, `phenotype`, `cell_type`, or `annotation`, retaining columns with between two and `max_categories` unique values.

Metadata columns likewise use explicit settings first. Automatic detection favours columns recorded in `metadata/dictionary.csv` and common names describing ROI, sample, patient, batch, replicate, condition, treatment, or group. Other low-cardinality non-numeric columns can also be included. Automatic selection is convenient but imperfect: identifiers or technical annotations can be mistaken for biology, so the selected columns should be checked in the execution log.

## UMAP views

### Population and metadata UMAPs

For each selected population annotation, the stage colours the existing UMAP by population and can save an individual highlighted view for every category. It can also colour the same coordinates by categorical metadata such as treatment, patient, batch, or ROI.

UMAP is a two-dimensional representation of a higher-dimensional cell-similarity structure. Nearby cells generally had similar values in the representation used when the UMAP was constructed, but distances and empty spaces are not quantitatively preserved. The shape, orientation, islands, and apparent size of gaps can change with UMAP parameters and random initialisation.

These plots are useful for asking whether:

- a population is compact or spread across a phenotypic continuum;
- two labels occupy almost identical regions and may be over-separated;
- one label contains visibly distinct regions that merit closer review;
- batches, patients, or ROIs segregate in the embedding; or
- rare populations are represented by enough cells to inspect.

The stage reuses `X_umap`; it does not verify which markers, transformation, batch correction, or neighbour graph created it. A metadata gradient may indicate genuine biology, technical confounding, or both. Conversely, good batch mixing is not proof that biological structure was preserved.

### Marker UMAPs

The stage colours the UMAP by every marker in `adata.X`, creates a combined marker gallery, and creates additional galleries for every AnnData layer. This helps relate population boundaries to measured protein programmes and compare different stored measurement representations—for example, processed scores in `adata.X` and reintegrated intensities in a layer.

The numeric meaning of colour is entirely determined by the selected matrix or layer. The default gallery colour-bar label is “Nimbus-Inference Score”, but that label must be changed if `adata.X` contains another scale. A shared `vmax` improves visual comparability but can saturate highly expressed markers or make low-range markers appear uninformative. Marker UMAPs should therefore be read together with matrix plots and source images rather than as quantitative expression tests.

## Marker matrix plots

Matrix plots show the mean value of every marker for cells in each population or metadata category. A dendrogram orders groups by the similarity of their mean profiles, and markers are ordered using the toolkit's expression-ordering helper. When segmentation-only or removed markers are configured, an additional filtered version excludes them.

Two versions answer different questions:

- The **scaled matrix plot** rescales each marker across groups (`standard_scale='var'`). It emphasises which groups are relatively high or low for that marker and is useful for identifying phenotypic patterns. It does not preserve absolute differences between markers; a weak but variable marker can look as visually strong as an abundant marker.
- The **unscaled, capped matrix plot** retains the values in `adata.X` but limits the colour range at `matrixplot_vmax`. It is closer to the stored scale, although values above the cap are indistinguishable.

Use coordinated marker programmes when assigning biological meaning. A proposed cell identity supported by several compatible membrane, lineage, and functional proteins is stronger than a label based on one marker. Unexpected combinations can reflect real transitional biology, but also antibody background, spillover, segmentation mixing, doublets, or incorrect masks. Means can conceal heterogeneous or bimodal groups, so inspect individual marker UMAPs and backgated cells when a matrix row looks surprising.

## Returning labels to tissue

### Population mask overlays

The tissue-overlay module maps each cell's categorical population label back onto its labelled segmentation object for every ROI. The resulting image preserves the physical arrangement of cells while using population colours rather than channel intensity.

These overlays are appropriate for recognising structures such as epithelial nests, immune aggregates, stromal bands, invasive margins, or dispersed infiltrates. They can reveal obvious annotation or registration problems—for example, a lymphocyte label consistently occupying large epithelial-shaped masks.

An overlay is still a categorical rendering of the segmentation and annotation. It does not show marker evidence, cell uncertainty, tissue morphology, or unsegmented space. Dense visual contact also does not establish a statistically enriched interaction: cell size, tissue architecture, and population abundance all influence apparent proximity. Use the dedicated spatial stages for quantitative interaction analysis.

### Backgating to source images

Backgating is the strongest visual link between a population label and the original measurement. For each population, it locates annotated cells in the denoised channel images using ROI and coordinate observations, optionally uses the segmentation masks, and creates cell thumbnails, channel composites, and population overlays.

By default, candidate display markers are selected using a one-population-versus-rest Scanpy differential-expression ranking. The configured method is usually Wilcoxon. Markers with adequate positive log fold change are preferred, but if too few pass, the code falls back to the highest-ranked markers. The adjusted-P-value threshold is reported for context and is not a strict selection filter. DNA channels can be excluded from ranking while one DNA marker remains fixed as the blue channel to provide nuclear and morphological context.

This differential-expression calculation is a display heuristic, not a valid biological differential-expression study. It treats individual cells as observations, does not model patients or ROIs, and searches for markers that visually distinguish an existing label. Its purpose is to choose informative RGB channels for image review.

Backgating supports a human checkpoint:

1. `save_markers` calculates mean-expression tables and editable RGB settings without generating images.
2. A scientist can replace automatically selected channels with markers that better test the proposed identity.
3. `load_markers` generates images from those reviewed settings. `full` performs selection and imaging in one run.

When reviewing backgated cells, ask whether the source staining is correctly localised, whether the mask follows a credible cell, whether neighbouring cells have contaminated the measurement, and whether the examples are consistent across ROIs. The thumbnail gallery is sampled and should not be mistaken for the complete population. Limiting saved ROIs reduces output volume, but intensity normalisation still uses the full eligible ROI set for consistency.

## Population abundance analysis

When `groupby_obs` is configured, the stage runs a structured abundance analysis. Otherwise it falls back to descriptive count, proportion, and stacked plots across automatically selected metadata columns.

### Unit of measurement

The structured analysis first counts cells for each population within each ROI. It then calculates:

- **ROI-level proportion:** population cell count divided by the total analysed cell count in that ROI;
- **cells per mm²:** population cell count divided by the ROI area; and
- **case-average values:** the arithmetic mean of ROI-level proportions or densities for each case.

Proportion and density describe different biology. Proportions express tissue composition conditional on all detected cells in the analysis scope; an increase can arise because the target expanded or because another population contracted. Cells per mm² expresses absolute detected density within the imaged area, but depends on tissue coverage, segmentation sensitivity, and correct area calibration.

ROI area is taken from `adata.uns['sample']['mm2']` when available. Otherwise the code uses the mask image dimensions and assumes one image pixel represents one micrometre, calculating height × width / 1,000,000. If that pixel-size assumption is incorrect, the resulting cells/mm² values are incorrectly scaled. Store validated physical areas in AnnData for quantitative density work.

When a case identifier is configured, case-average plots reduce the influence of cases contributing many ROIs. ROI-level plots expose within-case variability but do not make ROIs biologically independent. Both levels are saved because they answer related but different questions.

Optional compartment settings repeat the proportion analysis within selected compartments as well as across all cells. A compartment-specific proportion is conditional on the cells retained in that compartment. Density is omitted for compartment scopes because the implementation does not calculate the physical area of each compartment; dividing by whole-ROI area would answer a different and potentially misleading question.

### Statistical tables

Formal statistics are attempted only when a case column is present and exactly two comparison groups remain.

At ROI level, each population is analysed using a mixed linear model with comparison group as a fixed effect and case as the grouping/random-effect structure. ROI is included as an additional variance component where the data contain repeated rows within a case–ROI combination. The output also reports ordinary independent-sample t-tests and Mann–Whitney U tests. P values are corrected across populations using Benjamini–Hochberg false-discovery-rate correction.

For the case-average analysis, ROI values are averaged within each case before the t-test and Mann–Whitney U test; the mixed-model result is removed from that output. These tests can disagree because they make different distributional and independence assumptions.

Statistical output must be interpreted at the biological-replicate level. A small P value does not measure effect size, biological importance, annotation reliability, or generalisability. Mixed models can also be unstable with few cases, strongly unbalanced sampling, zero-inflated rare populations, or boundary fits; model warnings are retained in the full tables and should be reviewed. The numerous figures are exploratory and should not be used to select a favourable comparison without appropriate validation.

### Plot styles and axis scaling

Bar plots summarise the distribution; strip or swarm plots show individual ROI or case values with mean and standard-error overlays. Showing individual points is usually more informative when sample size permits. Linear axes preserve absolute differences, whereas logarithmic axes can reveal rare populations spanning a wide range but cannot display zero or negative values. The `intelligent` option switches only for positive distributions with sufficient dynamic range and improved skew; the selected axis changes presentation, not the underlying test.

## Output organisation

For an `sbt run`, outputs are written beneath the active execution's `figures`
directory (for example, `outputs/007_Visualisation/figures/`). Direct reported
runs use their corresponding `outputs/direct/.../figures/` directory. Calls made
without any reporting context retain the legacy `QC/BasicProcess_QC` fallback.
The organised output tree includes:

- `UMAPs` for categorical, highlighted, marker, and layer-specific views;
- `Matrixplots` for scaled, unscaled, and marker-filtered summaries;
- `Population_images` for population-labelled segmentation masks;
- `Backgating` for marker settings, composites, thumbnails, and source-image overlays;
- `Population_Analysis_Figures` for plots, raw tables, and statistical results; and
- `Color_legends` for reusable category-to-colour keys.

Many submodules catch errors independently so that, for example, missing images do not prevent matrix plots from being created. A completed stage can therefore contain only a subset of requested outputs. The execution log and report are part of the result and should be checked for skipped modules and row-level failures.

## Practical interpretation checklist

Before using a population label or abundance result downstream, verify that:

- its marker matrix profile is biologically coherent;
- individual marker distributions support, rather than merely average into, that profile;
- backgated source images show credible staining and segmentation;
- its tissue distribution is plausible and not driven by damaged areas or ROI edges;
- it is represented across appropriate biological replicates;
- apparent group differences are not explained by batch, ROI selection, tissue area, or segmentation yield; and
- the AnnData matrix or layer used by each plot has the intended biological scale.

Visualisation is therefore not the decorative final step of the pipeline. It is a structured audit of the chain from protein measurement and cell segmentation to annotation, spatial context, and cohort-level inference.
