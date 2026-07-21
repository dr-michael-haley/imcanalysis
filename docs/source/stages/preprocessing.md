# Preprocessing

## What this stage does

The preprocessing stage converts raw imaging mass cytometry (IMC) acquisition
files into ordinary TIFF images and small CSV tables that the rest of the
pipeline can read.

An IMC acquisition is naturally stored as a stack: every channel records the
signal from a different metal-labelled antibody across the same two-dimensional
region of interest (ROI). The instrument's MCD container also stores acquisition
and channel metadata. This is efficient for acquisition software, but most image
analysis programs work more easily with one TIFF per marker and explicit tables
describing the samples and panel.

The stage therefore performs four organizational tasks:

1. read each MCD or TXT acquisition;
2. save one channel-by-height-by-width TIFF stack per ROI;
3. split each stack into one two-dimensional TIFF per channel; and
4. build metadata, panel, and ROI-dictionary tables.

It does **not** change the biological signal. In particular, this stage does not
normalize intensities, remove hot pixels, denoise images, correct isotope
spillover, subtract background, segment cells, or quantify marker expression.
Those operations either occur in later pipeline stages or require a separate,
explicit analysis decision.

## IMC files and the Bodenmiller tools

The [Bodenmiller Group `imctools` project](https://github.com/BodenmillerGroup/imctools)
established a widely used open workflow for converting proprietary MCD and TXT
files into accessible image formats and metadata. Its own repository now marks
`imctools` as unmaintained and directs users to the maintained
[`readimc` parser](https://github.com/BodenmillerGroup/readimc).

The current SpatialBiologyToolkit implementation imports `MCDFile` and `TXTFile`
from `readimc`; it does not import the older `imctools` Python package directly.
The design and file-conversion role are closely aligned with the `imctools`
workflow supplied as the reference, while the live code uses its maintained
Bodenmiller Group parser.

There is also a format distinction worth making. `imctools` can create
metadata-rich OME-TIFF intermediates. This pipeline stage instead writes a plain
multipage TIFF for each ROI, with `CYX` axes recorded by `tifffile`, plus separate
CSV metadata and panel files. Downstream users should therefore keep the image
files and their accompanying tables together.

## Why this conversion is needed

### Accessible image data

Later stages operate on standard image arrays rather than communicating with the
instrument container. The multipage stack preserves all channels for an ROI,
while the unstacked files allow denoising, segmentation, visualization, and
quantification code to locate an individual marker without repeatedly decoding
the MCD.

### Explicit sample identity

An MCD can contain many acquisitions, each representing a different ROI. The
pipeline needs stable folder names and a table linking those folders back to the
original acquisition description and source file. Without this mapping, results
from different tissue cores, sections, or patients can easily be confused.

### An explicit antibody panel

The acquisition records both a channel name, normally the metal or isotope, and a
channel label, normally the antibody target. Downstream analysis needs both. The
metal identifies the detector channel; the target supplies the biological name
used to find and interpret an image.

The generated panel also contains workflow flags that tell later stages whether a
channel should be denoised and whether raw or denoised images should be used.
These flags are starting assumptions, not a substitute for checking the antibody
panel.

## Main inputs

The preferred input location is `general.imc_files_folder`, which defaults to
`IMC_files/`. If that configured path is absent, the script falls back to the
legacy `general.mcd_files_folder`, normally `MCD_files/`.

The stage searches the top level of that directory for files ending in lowercase
`.mcd` or `.txt`. It does not recursively search nested input folders. On the
Linux HPC system, differently capitalized extensions such as `.MCD` may not be
discovered, so standard lowercase extensions are recommended.

### MCD input

An MCD is the instrument-oriented container and can contain several acquisitions.
For each acquisition on the first slide, `readimc` supplies:

- a channel-by-height-by-width intensity array;
- metal or detector channel names;
- antibody target labels;
- the acquisition identifier and description; and
- its recorded width and height.

Acquisition entries containing zero channels are treated as empty stage movements
and ignored. The implementation assumes such empty acquisitions occur at the end
of the acquisition list. Each nonempty acquisition is handled independently, so
one malformed ROI can be recorded as an import error while other ROIs from the
same MCD continue.

### TXT input

A TXT export is treated as one acquisition. Its cleaned filename becomes the ROI
description. Because TXT input does not provide the same acquisition dimensions,
the script fills the compatibility columns `width_um` and `height_um` from the
array width and height. These values are therefore pixel dimensions despite the
historical column names.

The MCD minimum-size filter is not applied to TXT files. An error while processing
a TXT file is logged and raised rather than added to the per-acquisition MCD error
table.

## How the stage processes the data

### 1. Read and screen acquisitions

For MCD input, an ROI is exported only when both its `readimc` `width_um` and
`height_um` values are strictly greater than
`preprocess.minimum_roi_dimensions`. The default is 200. Thus a dimension exactly
equal to 200 does not pass the test.

This filter is intended to exclude accidental, test, or extremely small
acquisitions that are unlikely to support reliable segmentation and spatial
analysis. It is not a tissue-quality measurement. A large empty or damaged ROI
passes, whereas a small but biologically valuable ROI can be excluded. The list
of rejected MCD acquisitions is written to the import-error table for review.

In many IMC datasets one pixel corresponds approximately to one micrometre, so
physical and pixel dimensions have similar numerical values. The code nevertheless
uses the dimensions reported by `readimc`; the threshold should not be interpreted
as a universal pixel rule for data acquired or exported at another resolution.

### 2. Write one TIFF stack and panel per ROI

Every accepted MCD acquisition is saved as:

```text
tiff_stacks/<MCD-name>/<MCD-name>_acq_<acquisition-index>.tiff
```

Its neighbouring CSV contains the ordered `channel_name` and `channel_label` rows
needed to interpret the planes. MCD acquisition metadata are collected in a CSV
for the source file. Failures or size-filtered acquisitions are written to a
source-specific errors CSV.

A TXT file produces a similar directory, but its filename is cleaned by replacing
runs of punctuation or whitespace with underscores:

```text
tiff_stacks/<clean-TXT-name>/<clean-TXT-name>.tiff
```

The TIFF values are the array returned by `readimc`. The code does not rescale,
clip, transform, or convert them to display intensities. Very dark-looking raw
TIFFs in a standard image viewer can still contain valid low-count IMC data; an
appropriate display contrast is required for visual inspection.

### 3. Build the cohort metadata table

Metadata from successfully exported ROIs are merged into
`general.metadata_folder/metadata.csv`. Important columns include:

| Column | Meaning |
|---|---|
| `id` | Acquisition index within the source file; TXT files use 0 |
| `description` | Instrument acquisition description or cleaned TXT filename |
| `width_um`, `height_um` | MCD dimensions reported by `readimc`; TXT array dimensions for compatibility |
| `source_file` | Stem identifying the original MCD or cleaned TXT source |
| `file_type` | `mcd`, `txt`, or `unknown` |
| `import_data` | Initial inclusion flag, set to `True` for every successfully exported ROI |
| `unstacked_data_folder` | ROI folder name used under the raw image directory |
| `tiff_stacks` | Relative path to the ROI's multipage TIFF stack |

`import_data` is deliberately editable. It gives the analyst a place to exclude
test scans, damaged tissue, failed staining, or other unsuitable ROIs after
review. Preprocessing itself sets it to `True`; it does not make those biological
or quality-control judgments.

If `dictionary.csv` does not exist, the stage creates it with one row per ROI,
indexed by `ROI`. It initially contains the acquisition description and placeholder
`Example_*` columns. These example columns are prompts, not experimental
metadata. They should be replaced with meaningful case, tissue, treatment, batch,
or outcome columns before analyses that depend on sample grouping.

If a dictionary already exists, preprocessing leaves it unchanged. This protects
manual annotations, but also means newly imported or renamed ROIs are not
automatically reconciled with an older dictionary.

### 4. Reconcile panels across files

The stage compares all ROI panel CSVs exactly and groups identical tables using a
hash. If every ROI has the same panel, the merged result is
`general.metadata_folder/panel.csv`. If different panel layouts or labels are
detected, it writes `panel_1.csv`, `panel_2.csv`, and so on, plus
`panel_mapping.csv` linking source files to their panel file.

Multiple panels are not automatically harmonized. A difference may be intentional,
such as a revised antibody panel, or may indicate inconsistent naming, missing
channels, or acquisition setup errors. The mapping should be inspected before
downstream analysis assumes a common feature set.

The merged panel contains these workflow flags:

- `use_denoised` is initially `True` for channels that appear to have a real
  target label;
- `to_denoise` initially copies `use_denoised`;
- `use_raw` is initially `False`; and
- `remove_outliers` is initially `False`.

A channel is treated as blank when its target label is missing or when the target
and metal names contain exactly the same characters, such as alternative orderings
of an isotope name. This is a pragmatic naming rule, not an examination of image
signal. Blank or incorrectly labelled channels can therefore be classified
wrongly and must be reviewed manually.

The stage removes spaces and other non-word characters from `channel_label`.
This makes marker filenames easier for later scripts to match, but can change
familiar antibody names and can make distinct labels converge on the same cleaned
name. The final `channel_name`/`channel_label` pairs should be checked for
uniqueness.

### 5. Split stacks into individual channel images

The second half of the stage recursively finds every `.tiff` stack in
`general.tiff_stacks_folder`. For each ROI it creates a directory under
`general.raw_images_folder`, normally `tiffs/`, and writes every plane as a
two-dimensional TIFF.

A typical filename is:

```text
00_00_191Ir_DNA1.tiff
```

The components are the zero-padded channel position, a zero-padded ROI enumeration
used during this run, the channel name, and the cleaned biological label. The ROI
directory—not the second number alone—is the stable sample context.

All planes are written, including channels classified as blank. Blank-channel
flags affect later selection but do not discard the source image. Before writing,
the script verifies that the panel has exactly as many rows as the TIFF stack has
planes. A mismatch stops unstacking because labels could otherwise be assigned to
the wrong images.

The raw image directory also receives `channels_list.csv`, which lists the unique
channel name/label combinations and a `contains_data` flag. A compatibility copy
of the source-level metadata is written there as `metadata.csv`; the cohort table
under `general.metadata_folder` remains the table designed for pipeline metadata
editing.

## Reusable assets produced

The main reusable outputs are:

```text
tiff_stacks/
  <source>/
    <ROI>.tiff
    <ROI>.csv
    <source>_meta.csv
    <source>_errors.csv          # when an MCD acquisition failed or was too small

tiffs/
  <ROI>/
    <channel-index>_<ROI-index>_<metal>_<marker>.tiff
  channels_list.csv
  metadata.csv                  # compatibility copy

metadata/
  metadata.csv
  dictionary.csv
  panel.csv                     # when all panels match
  panel_1.csv, panel_2.csv, ... # when panels differ
  panel_mapping.csv             # when panel information is available
  errors.csv                    # merged MCD acquisition errors, when present
```

Folder names are configurable through the `general` section. The stage's managed
execution report records its lifecycle and generated-file inventory, while these
TIFFs and CSVs remain reusable project assets for later stages.

## What to check before continuing

Preprocessing is successful only when the conversion is biologically correct, not
merely when the job exits without an error. Review at least the following:

1. **ROI count:** compare `metadata.csv` with the acquisition plan. Investigate
   missing ROIs and every row in `errors.csv`.
2. **ROI identity:** confirm that acquisition descriptions and dictionary rows map
   to the correct case, tissue, core, and condition.
3. **Dimensions:** look for unexpectedly small, truncated, or differently sized
   acquisitions.
4. **Panel order:** confirm that every stack plane matches its metal and antibody
   label. A channel-order error contaminates all later results.
5. **Cleaned labels:** check that punctuation removal has not created ambiguous or
   duplicate marker names.
6. **Panel differences:** if multiple panel files were generated, determine
   whether the differences are expected and how they will be harmonized.
7. **Image appearance:** inspect representative nuclear, membrane, lineage, and
   low-abundance markers from every acquisition batch using suitable contrast.
8. **Inclusion flags:** set `import_data: false` for ROIs that should not enter the
   biological analysis.
9. **Workflow flags:** decide which markers genuinely require denoising, raw-image
   use, or outlier handling instead of accepting the automatic naming heuristic
   uncritically.

These checks are especially important because a metal/target naming error can
look plausible after segmentation and may not become obvious until biological
interpretation.

## The preprocessing configuration

`preprocess.minimum_roi_dimensions` is the only stage-specific setting. It must
be positive and defaults to 200. It applies only to MCD acquisitions, and both
recorded dimensions must be strictly greater than the threshold. Input and output
folders are configured in `general`.

See the [preprocess configuration reference](../reference/configuration/sections/preprocess.md)
for the canonical default and field description.

## Common problems and limitations

- **No raw files are found:** confirm the input folder, top-level file placement,
  and lowercase `.mcd` or `.txt` extension. An existing but empty
  `general.imc_files_folder` prevents fallback to the legacy folder.
- **An ROI is missing:** inspect both the size threshold and the source-specific or
  merged errors table. A rejected small MCD acquisition is recorded as an error.
- **Zero-channel acquisitions occur in the middle of an MCD:** the current counting
  logic assumes empty acquisitions occur at the end and can mis-handle unusual
  acquisition sequences.
- **An MCD contains several slides:** the current reader path processes the first
  slide's acquisition list.
- **Old outputs are present:** the stage creates and overwrites expected files but
  does not clear the stack or raw-image directories first. Files from removed or
  renamed acquisitions can remain and be unstacked on a later run. Use a clean,
  deliberately managed output location when the source set changes.
- **Panels differ:** the stage records the difference but does not decide how
  markers should be matched, excluded, or treated as missing across panels.
- **A marker appears blank:** preprocessing does not measure staining quality.
  Review the image, acquisition settings, antibody performance, and metal/target
  annotation separately.
- **Raw intensities look dark or have a long-tailed range:** no display scaling or
  normalization is applied. This is expected for count-like IMC channels and is
  not by itself evidence of an import failure.
- **TXT dimensions appear in micrometre-named columns:** they are inferred from
  array shape for compatibility and should not be treated as calibrated physical
  measurements without independent resolution metadata.
- **Successful conversion does not establish scientific quality:** staining,
  spillover, ablation quality, tissue integrity, and acquisition artifacts remain
  biological and technical QC responsibilities.

## Further reading

- [Bodenmiller Group `imctools`](https://github.com/BodenmillerGroup/imctools),
  the original conversion toolkit and workflow reference.
- [Bodenmiller Group `readimc`](https://github.com/BodenmillerGroup/readimc), the
  maintained parser used directly by this stage.
- Windhager *et al.* (2023), [An end-to-end workflow for multiplexed image
  processing and analysis](https://doi.org/10.1038/s41596-023-00881-0),
  *Nature Protocols*.
