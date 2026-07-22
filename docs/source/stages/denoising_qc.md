# Denoising QC

## Purpose of the stage

Denoising QC is the checkpoint between image restoration and analyses that treat the restored images as measurements. Denoising is intended to suppress isolated hot pixels and shot noise while retaining biologically meaningful staining. If it is too aggressive, weakly positive cells, membrane boundaries, punctate signals, and real cell-to-cell heterogeneity can be removed. If it is insufficient, technical noise can inflate cell intensities, disrupt segmentation, and create false marker-positive populations.

The stage therefore combines two different forms of quality control:

1. **side-by-side image review**, comparing raw and denoised images for selected channels and ROIs; and
2. **panel consistency checking**, verifying that the denoised ROI folders contain the channel files expected from `metadata/panel.csv` and summarising their pixel distributions.

These checks are complementary. The images address whether restoration appears biologically plausible. The panel audit addresses whether the dataset is structurally complete and consistently named. Neither check, by itself, proves that denoising preserved quantitative marker measurements.

For the scientific basis and implementation of DIMR and DeepSNiF, see the [Denoising guide](denoising.md).

## Why denoising must be reviewed biologically

IMC measurements contain several kinds of image variation that should not all be treated as noise. An isolated, extremely bright pixel is unlikely to represent an entire protein-positive cell and is a plausible technical outlier. In contrast, a small punctum, a thin membrane, or a weak cluster of pixels aligned with a cell can be genuine biology. The distinction depends on marker localisation, tissue context, and spatial coherence rather than intensity alone.

Denoising can affect every later stage:

- DNA-channel restoration influences the images available for nuclear segmentation.
- Membrane and lineage-marker changes alter per-cell quantification and population annotation.
- Loss of weak signal can move cells below an informal positivity threshold.
- Residual bright outliers can dominate mean intensities or colour scales.
- Spatially blurred staining can transfer signal between neighbouring cells and change apparent interactions.

Quality review should therefore include markers with different expected patterns: nuclear, membrane, cytoplasmic, diffuse, punctate, abundant, and weakly expressed. A method that performs well for a bright structural marker may be inappropriate for a sparse signalling marker.

## Side-by-side image comparisons

For each configured channel, the stage loads matching TIFFs from `general.raw_images_folder` and `general.denoised_images_folder`. When no explicit channel list is supplied, channels are constructed from `panel.csv` rows marked `to_denoise`, falling back to `use_denoised` for older panel files.

One figure is saved per channel. Each row contains a raw image and its denoised counterpart. Up to `denoising.qc_num_rois` ROIs are selected randomly for that channel; setting it to null includes all available ROIs. Sampling has no fixed random seed, so repeated runs can show different regions, and different channels are not guaranteed to display the same ROI subset.

The “raw” image is the file currently present in the pipeline raw-image folder. If the optional panel-driven outlier-removal step previously modified that file in place, it is not necessarily the pristine acquisition image. The comparison then shows the state after that preprocessing step versus the final DIMR or DeepSNiF output.

### How the images are scaled

Every raw and denoised panel is displayed with its own colour range from zero to half of that image's maximum value. A separate colour bar is drawn for each panel. This makes spatial structure visible when the two images have different numeric ranges, but it prevents direct visual comparison of brightness.

Two images can therefore look equally bright even when denoising substantially changed their absolute intensities. Conversely, one remaining hot pixel can increase an image maximum and compress the appearance of all other staining. Use the side-by-side figures to assess morphology and localisation, not quantitative intensity preservation.

The configured colour map and DPI affect figure rendering only. They do not change the stored TIFF values or denoising model.

### Features that should usually be preserved

Review whether the denoised image retains:

- the same positive cells and tissue compartments as the raw image;
- expected nuclear, membrane, cytoplasmic, or extracellular localisation;
- thin boundaries and small structures that are credible for that marker;
- meaningful differences between strongly, weakly, and negatively stained cells;
- spatial gradients and heterogeneous regions present across multiple neighbouring pixels; and
- alignment with tissue morphology and other biologically related channels.

Successful denoising commonly removes isolated extreme pixels and reduces fine-grained background variation while leaving coherent cellular structures recognisable.

### Warning signs of over-denoising

Potential over-denoising includes:

- disappearance of weak but spatially coherent positive cells;
- loss or thinning of biologically expected membranes;
- erasure of punctate staining that aligns with cells or subcellular structures;
- excessive uniformity within or between cells;
- blurred boundaries or signal spreading into neighbouring objects;
- loss of rare tissue-specific structures; and
- different biological regions becoming artificially similar.

### Warning signs of insufficient denoising

Potential under-denoising includes persistent isolated hot pixels, salt-and-pepper texture in otherwise negative regions, extreme colour-scale compression, or high-frequency variation that does not follow tissue or cellular structure. Some true markers are naturally punctate, so isolated-looking signal should be checked across related channels and source morphology before being classified as noise.

## Image pairing limitations

The visualisation code loads raw and denoised collections separately and pairs them by their list position. It does not explicitly join the two collections by ROI name before plotting. If a channel is missing from one folder, or filesystem ordering differs, a raw ROI can potentially be paired with the wrong denoised ROI; with unequal collection lengths, `zip` also stops at the shorter list. The subsequent panel-consistency audit detects many missing-file cases but does not retroactively validate each plotted pair.

For that reason, confirm that row labels and visible tissue morphology correspond, particularly when the panel audit reports missing files. A visually implausible pair may be a file-organisation problem rather than a denoising failure.

The standalone Denoising QC stage compares the normal denoised-image directory. Parameter-scan-specific comparison folders are generated during the denoising stage itself; this standalone invocation does not iterate over configured scan values.

## Panel consistency audit

After the image figures, the wrapper switches to the segmentation environment and runs a structural audit of the denoised-image folder. For every ROI directory, it compares the TIFF filenames with the non-deleted rows in `metadata/panel.csv`.

The expected filename structure is:

```text
<channel-index>_<roi-index>_<channel-name>_<channel-label>.tiff
```

Channel labels may contain additional underscores, but the parser treats only the third underscore-separated component as `channel_name`. Filenames with fewer than four components are reported as unparseable.

For each ROI, the audit records:

- actual and expected file counts;
- missing panel channel–label combinations;
- additional channel–label combinations not represented in the panel;
- missing or additional channel names;
- TIFF filenames that cannot be parsed; and
- files whose byte size differs substantially from other channel files in the same ROI.

Any missing, extra, unparseable, or size-flagged file marks that ROI as having an issue. The checker returns a non-zero exit status when at least one ROI has an issue, so the stage can be reported as failed even though its QC tables were successfully created. This is an intentional prompt for review rather than evidence that every flag represents corrupted biology.

### Biological importance of a consistent panel

A channel missing from only some ROIs creates sample-dependent measurement availability. Downstream code may fail, discard those ROIs, insert missing values, or compare cells measured with different feature sets. If missingness follows batch, treatment, or patient group, it can become indistinguishable from a biological difference.

Extra files can also be dangerous if their names collide with expected channel substrings or if an old processing output remains in an ROI folder. A naming mismatch can cause the wrong TIFF to be selected for quantification even when the intended data are present.

The checker expects every non-deleted panel row in every denoised ROI. It does not use `to_denoise`, `use_denoised`, or `use_raw` to restrict that expectation. An intentionally raw-only channel absent from the denoised folder can therefore be reported as missing. Such a flag should be reconciled with the intended folder contract rather than automatically “fixed” by duplicating or fabricating data.

### File-size flags are heuristic

Within an ROI, the checker compares TIFF file sizes in bytes and flags files more than 10% larger than the smallest channel file. Byte size is not the same as image dimensions or scientific validity. TIFF compression, metadata, data type, and image complexity can all change file size. A flagged channel may be valid, while identically sized files can still contain incorrect data.

Treat file-size results as prompts to inspect image shape, metadata, and content. They are not proof of truncation, corruption, or abnormal staining.

## Pixel-distribution summary

The panel checker also reads denoised images and calculates simple per-channel statistics. For each ROI image, it removes 20% from every border and analyses the remaining central 60% of the image width and height. It calculates that central region's mean, standard deviation, minimum, and maximum, then reports the unweighted mean of each statistic across the available ROI images.

The central crop reduces the influence of some image-edge artefacts, but it can also exclude biologically relevant peripheral tissue or contain little tissue in an off-centre acquisition. These summaries do not use tissue masks and therefore include background pixels.

A channel receives `low_std` when its mean within-image standard deviation is below 1. This is a fixed threshold in the stored TIFF units. It is not normalised to the channel's dynamic range and is not compared with the raw image. Consequently:

- a genuinely absent or uniformly low marker can be flagged;
- an over-smoothed image can be flagged;
- a channel stored on a 0–1 scale will almost inevitably be flagged;
- a noisy channel with standard deviation above 1 can pass; and
- variation driven by background or a few extreme pixels can prevent a flag.

The mean, minimum, maximum, and standard deviation are descriptive screening values, not acceptance criteria. They should be interpreted per marker and measurement scale, alongside the side-by-side images and biological expectations.

Channel images are located by case-insensitive substring matching, and only the first matching TIFF in each ROI is used. Ambiguous or overlapping channel names can therefore select an unintended file. Exact, unique channel-name–label combinations reduce this risk.

## Outputs

The stage produces human-facing QC artifacts rather than a new reusable scientific dataset:

- one raw-versus-denoised PNG per selected channel;
- a timestamped panel-consistency CSV with one row per ROI;
- a matching pixel-QC CSV with one row per successfully read panel channel; and
- execution logs summarising structural issues and low-standard-deviation channels.

The denoising stage itself also writes `denoised_pixel_qc.csv`. That related table is calculated from the channels processed during denoising; the Denoising QC panel checker independently recalculates pixel summaries for the panel channels it can locate.

In a managed `sbt` run, these outputs are routed into the Denoising QC execution report. Direct compatibility runs use the configured legacy QC location and current working directory conventions.

## Recommended review workflow

1. **Resolve structural errors first.** Check missing, extra, and unparseable files before trusting visual pairs or downstream quantification.
2. **Review every marker class.** Include bright and weak markers and different expected subcellular localisations.
3. **Review across the cohort.** Sample multiple tissues, staining batches, acquisition runs, and biological groups; the default random subset may not cover all of them.
4. **Compare morphology, not apparent brightness.** The side-by-side panels use independent colour scales.
5. **Investigate low-variance flags in context.** Inspect the stored scale, tissue content, raw image, and expected biology.
6. **Inspect downstream cell measurements.** After segmentation, compare raw and denoised per-cell distributions for important markers when quantitative preservation matters.
7. **Record acceptance decisions.** Note channels, ROIs, and known limitations so later population and spatial analyses can be interpreted appropriately.

## Important limitations

- Visual ROIs are randomly sampled without a fixed seed and may miss rare failure modes.
- Raw and denoised images are independently colour-scaled and are not quantitatively comparable in the figure.
- Raw and denoised collections are paired positionally rather than explicitly by ROI identifier.
- No difference image, correlation, structural-similarity measure, or signal-retention statistic is calculated.
- The panel audit relies on a strict filename convention and substring matching.
- The audit expects all non-deleted panel channels, including channels not intended for denoising.
- File-size differences are only an indirect heuristic.
- Pixel statistics include background, average across ROIs, and use a scale-dependent low-variance threshold.
- The checks do not establish antibody specificity, compensate spillover, or validate biological identity.

Denoising QC should therefore be treated as a structured scientific review rather than an automated pass/fail test. Its purpose is to expose restoration failures and dataset inconsistencies before those problems become embedded in segmentation, phenotyping, and spatial conclusions.
