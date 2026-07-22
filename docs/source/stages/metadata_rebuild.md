# Metadata Rebuild

## Purpose of the stage

Metadata rebuild reconstructs three project-control tables—`metadata.csv`, `dictionary.csv`, and `panel.csv`—from an existing AnnData object. It is a recovery and synchronisation tool for situations in which the AnnData has become the most complete surviving record of ROI annotations and measured channels, or when these CSV files need to be regenerated after a dataset has been transferred or reorganised.

This direction of travel matters. In the usual early pipeline, project metadata helps create the cell-level AnnData. During rebuild, the code works in reverse and infers ROI-level facts from observations repeated across cells. It can only recover information that is present, correctly named, and internally consistent in AnnData. It cannot reconstruct lost acquisition metadata, antibody details, physical calibration, or curation decisions that were never stored there.

The three outputs serve different purposes:

- `metadata.csv` is the operational ROI import table used by image-oriented pipeline stages;
- `dictionary.csv` is an ROI-indexed table of descriptive and experimental metadata; and
- `panel.csv` describes the current AnnData variables and controls which image channels are denoised or used downstream.

Because these tables can drive earlier processing steps, rebuilt files should be reviewed before they replace a curated metadata folder or are used to rerun image analysis.

## Reconstructing ROI-level metadata from cells

The stage requires the observation named by `general.roi_obs`. It groups cells by that value and searches for other observation columns that are invariant within every ROI: after missing values are ignored, each ROI must have no more than one distinct value.

This is appropriate for information that should be shared by all cells in an image, such as patient, treatment, tissue region, staining batch, acquisition run, or clinical group. For each selected column, the rebuilt ROI row receives the first non-missing value observed among that ROI's cells.

The invariant rule is a consistency check, not proof that a field is genuinely ROI-level. A cell-derived measurement can happen to be constant, especially in a small or filtered ROI. Conversely, a valid sample annotation containing inconsistent spelling or trailing whitespace can be rejected because it has more than one recorded value. Warnings and the resulting column list should therefore be inspected.

### Inclusion and exclusion rules

Cell identifiers, spatial coordinates, and population labels should not become ROI metadata. The stage excludes:

- configured ROI, coordinate, and master-index observations;
- shared primary and alternative population observations;
- exact names such as `ObjectNumber`, `CellID`, and `Master_Index`; and
- names containing fragments such as `population`, `leiden`, `cluster`, `nhood`, or `neighborhood`.

`include_obs_patterns` can provide a regular-expression allowlist. When set, a column must match at least one pattern as well as pass invariance and exclusion checks. This is the safest way to restrict recovery to known sample-level fields, such as `Case`, `Treatment`, or `Batch`.

If `general.case_obs` is missing, the stage warns. If it exists but varies within an ROI, it is not written. A case identifier that changes within one image usually signals corrupted metadata or an incorrect ROI key and should be resolved rather than forced into the output.

## Building `metadata.csv`

The rebuilt operational table contains one sorted row per ROI and the following core columns:

- `unstacked_data_folder`: the ROI name taken from `general.roi_obs`;
- `description`: a configured invariant observation, an invariant column literally named `description`, or the ROI name as fallback;
- `width_um` and `height_um`: dimensions read from the matching segmentation-mask array;
- `import_data`: set to `true`; and
- optionally, all other selected invariant observations.

Mask files are matched to ROI names by filename stem and can use `.tif` or `.tiff`. Missing or unreadable masks leave dimensions blank.

Despite the `_um` column names, the implementation writes the mask width and height in **pixels** without applying a pixel-size conversion. These values are physically correct in micrometres only when the image sampling is exactly 1 µm per pixel. If the acquisition or resampling scale differs, correct the table using validated calibration before using dimensions for area or density calculations.

The current implementation always sets `import_data` to `true` for every reconstructed ROI. The `preserve_existing_import_data` setting is retained in configuration for compatibility but is not read by the active code. Any intentionally excluded ROI will therefore be re-enabled unless the rebuilt file is edited afterwards.

Existing `metadata.csv` descriptions, non-AnnData columns, and manual edits are not merged. The file is rewritten from reconstructed content.

## Building `dictionary.csv`

`dictionary.csv` uses the ROI as its index. It includes the same resolved description and, when enabled, the selected invariant observations. It does not include the operational import and dimension columns added specifically to `metadata.csv`.

The table provides a convenient source for visualisation metadata discovery and downstream grouping. Its values should still be checked against the study design. Repetition across cells can amplify an original metadata error without making it more reliable.

Existing `dictionary.csv` content is overwritten. Definitions, units, explanatory fields, or variables not represented in AnnData cannot be recovered automatically and should be restored from a version-controlled or archived source.

## Reconstructing `panel.csv`

The rebuilt panel has one row for each variable currently present in AnnData. Channel names and labels are chosen in this order:

1. the explicitly configured `adata.var` column;
2. existing `channel_name` or `channel_label` variable annotations; or
3. `adata.var_names`.

Channel labels have non-word characters removed because they are used in filenames and pipeline matching. Empty labels fall back to the original text, and duplicate cleaned labels receive suffixes such as `_2` or `_3`. Review these transformations carefully: two biologically different channel labels can become identical after punctuation is removed, and suffixing preserves uniqueness but does not establish the intended identity.

Four Boolean control columns are created:

- `use_denoised`: prefer the denoised image for quantification;
- `to_denoise`: include the channel in denoising;
- `use_raw`: permit or request raw-image use according to the consuming stage; and
- `remove_outliers`: enable the configured pre-denoising outlier rule.

Defaults come from the rebuild configuration. When `preserve_existing_panel_flags` is enabled, parseable values from an existing `panel.csv` are carried forward for rows whose *cleaned, uniqueness-adjusted channel labels* match. Accepted Boolean forms include true/false, 1/0, and yes/no variants.

Only those four flags are preserved. Other manually curated panel fields—such as antibody descriptions, metal isotopes, outlier thresholds, channel-specific parameters, or notes—are not reconstructed by this implementation and are lost when the file is overwritten. Preserve the original panel separately and merge scientifically important fields deliberately.

The panel reflects only variables present in the source AnnData. If marker reintegration has not yet restored separately stored channels, those markers will not appear. Conversely, an AnnData variable that is a derived feature rather than an image channel may be incorrectly emitted as a panel row. Confirm that the source object's variables correspond to the image channels expected by downstream stages.

## Biological and analytical interpretation

Metadata are not merely administrative. They define biological replication, experimental comparisons, compartments, batch structure, and the denominators used by abundance and spatial analyses. A wrong case or treatment value propagated to every cell in an ROI can create a confident but false cohort-level result.

After rebuilding, verify at minimum:

- one and only one row exists for each intended ROI;
- every ROI maps to the correct case, condition, batch, and tissue context;
- case and group assignments agree with the experimental design;
- intentionally excluded images have the correct `import_data` status;
- mask dimensions have been converted to physical units when necessary;
- channel names match actual TIFF filenames and biological targets;
- raw/denoised and denoising flags are appropriate for each marker; and
- all necessary markers, including separately stored markers where relevant, are represented.

Metadata reconstructed from the final cell table also inherit upstream selection. An ROI that produced no retained cells cannot appear in `adata.obs` and therefore cannot be recovered from AnnData alone. Similarly, filtering all cells with missing or failed segmentation can make a source image disappear from the rebuilt tables.

## Outputs and overwrite behaviour

The destination is `rebuild_metadata.output_metadata_folder` or, when omitted, `general.metadata_folder`. The stage writes the three fixed filenames directly:

- `metadata.csv`;
- `dictionary.csv`; and
- `panel.csv`.

These writes replace existing files. Panel flag preservation reads the old panel before replacement, but there is no equivalent general merge for metadata or dictionary content. Use version control, a backup, or a separate output folder when auditing the result for the first time.

The run log reports the source AnnData, output paths, ROI and channel counts, number of invariant observations, missing masks, and inconsistent case metadata. The stage does not modify the AnnData.

## Limitations and common failure modes

- Only data present in the source AnnData can be reconstructed.
- ROI-invariant detection can accept accidentally constant cell-level fields or reject inconsistently encoded sample fields.
- Missing ROI identifiers can form invalid groups and should be cleaned before rebuilding.
- ROIs with no retained cells are absent from the reconstructed tables.
- Mask dimensions are pixel counts labelled as micrometres and assume 1 µm/pixel unless corrected externally.
- All ROIs are marked for import regardless of `preserve_existing_import_data`.
- Metadata and dictionary edits not represented in AnnData are overwritten.
- Only four existing panel Boolean flags are preserved; other panel columns are lost.
- Cleaned channel-label matching can fail or become ambiguous after naming changes.

Metadata rebuild is therefore a controlled recovery mechanism, not a substitute for the original study manifest and acquisition panel. The reconstructed tables should be reconciled with those primary records before they are allowed to govern a new pipeline run.
