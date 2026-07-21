# Segmentation

## What this stage does

This stage identifies individual nuclear objects in each imaging mass cytometry
(IMC) region of interest (ROI). It deliberately combines two generations of
[Cellpose](https://github.com/mouseland/cellpose):

1. **Cellpose3 image restoration** deblurs and upsamples the DNA channel to make
   small or poorly sampled nuclei easier for a segmentation model to recognize.
2. **Cellpose-SAM instance segmentation** assigns foreground pixels to separate
   nuclear objects on the restored image.

The labels are then resized back to the original IMC dimensions, cleaned, and
saved as one integer mask image per ROI. A pixel value of zero is background;
each positive integer identifies a different object.

These are **nuclear-centred masks**, not measurements of complete cell
boundaries. The default one-pixel expansion adds only a narrow region around
each nucleus. It does not recover the cytoplasm or plasma membrane, and it should
not be described as whole-cell segmentation without an independent validation
showing that the chosen expansion represents the cells in that tissue.

## Why nuclear segmentation is performed

Most downstream analysis needs a reproducible definition of an individual
biological object. The mask provides that definition: it determines which image
pixels belong to each object, where its centroid lies, its size and shape, and
which other objects are nearby. Marker intensities and Nimbus phenotype scores
can then be summarized object by object rather than across a mixed tissue image.

DNA is a useful anchor because most nucleated cells contain a compact,
high-contrast nucleus and because nuclear staining does not depend on the cell's
phenotype. It is not, however, a perfect one-to-one label for cells. Closely
packed nuclei can merge; lobulated nuclei can split; dividing cells may contain
condensed or paired DNA structures; multinucleated cells contain several nuclei;
and apoptotic debris may be sufficiently DNA-bright to form an object. Conversely,
weakly stained, sectioned, or out-of-focus nuclei can be missed. The mask should
therefore be understood as an algorithmic census of nuclear profiles in a
two-dimensional section, not automatically as a ground-truth cell census.

Segmentation errors propagate. A merged pair combines marker measurements and
removes a cell from density and neighbourhood estimates. A fragmented nucleus
creates extra objects, each with incomplete marker signal. A shifted boundary
changes both object morphology and the pixels assigned during quantification.
For this reason, segmentation QC is a biological validation step rather than a
purely technical check.

## Why the DNA image is restored first

IMC commonly samples tissue at approximately one micrometre per pixel. A small
lymphocyte nucleus may consequently span only a modest number of pixels, while
ablation, sectioning, ion-counting noise, prior denoising, and optical or
instrumental blur can weaken its boundary. Direct segmentation of such images
can be unstable even when a human can infer the likely nuclear profile.

The [Cellpose3 restoration paper](https://www.nature.com/articles/s41592-025-02595-5)
addresses noisy, blurred, and undersampled microscopy images. Its restoration
models were not trained only to reproduce target pixel values. They combine
objectives that preserve perceptual similarity with an objective that rewards
images that a fixed Cellpose model can segment. This is important: the restored
image is optimized as an aid to object detection and boundary formation.

The pipeline applies the `deblur_nuclei` model first when `run_deblur` is true.
It then applies `upsample_nuclei` by default when `run_upscale` is true. The
nuclei restoration model was trained around a mean nuclear diameter of 17
pixels. With the pipeline's default estimated input diameter of 10 pixels, the
expected enlargement is therefore 17 / 10, or 1.7-fold. The script records the
actual change in array dimensions rather than assuming that the requested ratio
was achieved exactly.

Cellpose3 trained its upsampling networks on images degraded by decimation,
blurring, interpolation, and noise. It recovered features that improved
segmentation in the paper's test data, but a learned upsampler cannot recreate
an observation that the IMC instrument never made. It uses patterns learned from
training images to infer a segmentation-friendly image. Plausible-looking fine
structure may therefore reflect the model's prior rather than measured DNA.

The restored TIFFs are intermediate segmentation assets. They must not be used
as if they contained newly measured DNA intensity, and they should not replace
the original or denoised marker images for quantitative biology. The conservative
use of restoration is to derive masks from the restored DNA image and measure
biology from the aligned source channels. That is how this pipeline is designed.

The restoration paper reported improved segmentation across several degraded
microscopy benchmarks, but it also showed that restoration is not uniformly
beneficial for every image. Performance on a new IMC tissue, staining protocol,
or acquisition batch must be established by comparing boundaries with the
source DNA image.

## How Cellpose-SAM creates nuclear instances

[Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1)
combines a customized vision-transformer encoder derived from the Segment
Anything Model (SAM) with Cellpose's representation of objects. In this context,
“SAM” does **not** mean that the pipeline asks a user for points or boxes and
then uses SAM's prompt-based mask decoder. The model is run automatically on
each single-channel DNA image.

For every image location, the network predicts two biologically useful kinds of
information:

- a **cell-probability value**, indicating whether the pixel is likely to be
  inside an object; and
- a two-dimensional **flow field**, pointing pixels towards the centre of their
  predicted object.

Pixels above the foreground threshold follow the predicted flow. Pixels that
converge on the same centre are assembled into one candidate mask. This
centre-seeking representation helps separate touching objects because adjacent
nuclei can have flows directed towards different centres even when their
intensities meet at the boundary.

The `cellprob_threshold` controls which pixels can enter the foreground. Lower
values are more permissive and commonly produce more or larger masks; higher
values are more conservative and can erode weak objects. The `flow_threshold`
is a consistency filter on completed masks. Candidate objects whose geometry is
poorly explained by the predicted flows have a high flow error. Lowering this
threshold rejects more such objects, while raising it retains more candidates,
including potentially irregular failures. The two thresholds act on different
parts of mask formation and should be assessed together.

The supplied Cellpose-SAM manuscript is a version 1 bioRxiv preprint. Its authors
report training across more than three million manually drawn regions from
diverse datasets and robustness to object size, channel order, noise,
downsampling, and several kinds of blur. They also report performance exceeding
the agreement between individual annotators on their benchmark. These results
support the use of a generalist starting model; they do not establish
“superhuman” accuracy for IMC nuclear profiles, which differ from many of the
training images and still require domain-specific QC.

## The implemented workflow

The active `cellpose` pipeline stage is a two-program SLURM job. Separate
environments are necessary because the restoration API belongs to Cellpose3,
whereas `cpsam` belongs to Cellpose4:

1. **Locate DNA.** In each denoised ROI folder, the stage requires exactly one
   filename containing the case-sensitive `dna_image_name` substring.
2. **Restore DNA with Cellpose 3.1.0.** Optional `deblur_nuclei` restoration is
   followed by optional learned upsampling. One floating-point TIFF is written
   to `dna_preprocessing_output_folder_name` for each successful ROI.
3. **Normalize for inference.** Cellpose-SAM receives a grayscale image. By
   default, each restored ROI is normalized between its 0th and 99.9th
   percentiles. This changes model input scaling, not the saved source channels.
4. **Segment with Cellpose 4.0.7.** The wrapper loads `cpsam` (or a compatible
   custom model), passes the configured diameter and thresholds, and predicts
   flows and instance masks. Although the preprint describes scale-robust native
   inference, the pinned 4.0.7 API used here still accepts a diameter and the
   pipeline supplies one; the implementation therefore matters more than the
   latest repository behaviour when reproducing a run.
5. **Return to the IMC grid.** If the DNA image was upsampled, categorical masks
   are resized to the original DNA dimensions with nearest-neighbour
   interpolation. This preserves integer identities and aligns the labels with
   the images used for downstream measurement.
6. **Clean each label.** Enclosed holes are optionally filled, border-touching
   objects can be removed, and labels can be expanded without allowing
   neighbouring labels to overlap. Objects are then filtered by area and the
   retained labels are numbered consecutively.
7. **Write masks and evidence.** The stage saves masks, ROI summaries, per-object
   diagnostic measurements, and optional overlays on both restored and source
   DNA images. A failure in one ROI is logged and the remaining ROIs continue.

The diameter supplied to Cellpose-SAM is `cellpose_cell_diameter` when learned
upsampling is disabled. With supported upsampling models, it is the restoration
target diameter: 17 pixels for `upsample_nuclei` or 30 for `upsample_cyto3`.
Masks always return to the original image dimensions before final area filtering
and saving.

The stage reads the `createmasks` configuration section. The separately named
`segmentation` section configures the later construction of cell tables and
AnnData objects; it does not control Cellpose3 restoration or Cellpose-SAM mask
generation.

## Choosing biologically appropriate settings

### DNA channel and starting diameter

Choose a DNA channel with strong nuclear contrast, low non-nuclear staining,
and comparable acquisition across ROIs. `dna_image_name` is a substring match,
so it must identify exactly one TIFF in every processed ROI. If two filenames
contain the value, the stage stops that ROI rather than choosing silently.

`cellpose_cell_diameter` should approximate the median nuclear diameter on the
**original** image grid. Measure several representative nuclei in distinct
tissue compartments rather than estimating from the largest epithelial cells.
An incorrect diameter changes the restoration scale and, when upsampling is
off, the scale at which Cellpose-SAM evaluates the image. Very heterogeneous
nuclear sizes may not be summarized well by a single number, so QC should cover
small lymphocytes as well as larger stromal or malignant nuclei.

For IMC DNA, `upscale_nuclei` is the biologically coherent default. The
`upscale_cyto3` model was trained around larger cellular objects and targets 30
pixels; it should not be selected merely to obtain a larger-looking image.

### Normalization and thresholds

Percentile normalization is performed separately for each ROI. This makes the
model less sensitive to absolute signal range, but it can also make a weak or
background-dominated ROI appear deceptively high contrast. Inspect both the
source and normalized/restored views. If the upper percentile is lowered, more
bright pixels are clipped; if the lower percentile is raised, dim background
and weak nuclear signal are compressed.

Start threshold assessment with the defaults (`cellprob_threshold: 0.0` and
`flow_threshold: 0.4`) and change them only in response to a reproducible error:

- widespread missing weak nuclei suggests testing a lower cell-probability
  threshold;
- foreground spreading into background suggests a higher cell-probability
  threshold;
- many implausible or flow-inconsistent shapes suggests a lower flow threshold;
  and
- loss of valid but unusual nuclear shapes may justify a higher flow threshold.

These are tendencies, not universal rules. Changing a threshold can fix one
compartment while damaging another, which is why representative multi-ROI QC is
more informative than optimizing the total object count.

### Area filtering, edges, holes, and expansion

`min_cell_area` removes very small objects and debris. It is passed to Cellpose
during inference and is applied again after masks have returned to the original
grid. Set it with respect to the smallest real nuclear profiles expected after
sectioning, not just the median nucleus.

`max_size_fraction` is an upper limit expressed as a fraction of the entire ROI
area, not an absolute nuclear area. The default 0.4 principally guards against a
catastrophic foreground mask covering much of an image; it is not a sensitive
filter for moderately merged nuclei. The legacy `max_cell_area` value is not
used by the active stage.

Enabling `remove_edge_masks` excludes every object touching the image border.
This avoids partial-object measurements but systematically removes valid cells
at ROI boundaries. Use the same policy across samples whose densities or spatial
distributions will be compared.

`fill_holes` fills enclosed background inside each nuclear label. This is often
reasonable for DNA profiles, but it can erase genuine unstained regions within
large or atypical nuclei. `expand_masks` then grows each label by a specified
number of original-resolution pixels until it meets another label. Expansion
does not use membrane or cytoplasmic staining and therefore remains a geometric
approximation. Increasing it can capture more perinuclear marker signal, but it
also assigns extracellular pixels and divides the space between crowded nuclei
according to proximity rather than true cell boundaries.

## Parameter scanning

`run_parameter_scan` evaluates a Cartesian grid of two configuration fields. By
default it crosses seven cell-probability thresholds with six flow thresholds,
giving 42 combinations per selected ROI. If `specific_rois` is null, the active
implementation randomly samples `num_rois_to_scan` available restored ROIs
without a fixed seed. Use `specific_rois` when an auditable, reproducible set of
tissues is required.

Each combination receives its own masks, overlays, result table, and summary
plots. These masks are written below parameter-specific directories; a scan does
not create the canonical root-level masks expected by downstream stages. Select
parameters by examining boundary accuracy across representative tissue types,
then disable the scan and rerun the normal stage with the chosen values.

Several older configuration names remain loadable but are not read by the active
workflow: `cell_pose_model`, `max_cell_area`, `resample`, `tile_overlap`,
`window_size`, and `scan_rois`. Their presence in an older YAML file does not
mean that they affected a Cellpose-SAM result.

## Outputs and how to interpret them

The main reusable outputs are:

- `preprocessed_dna/<ROI>.tiff`: segmentation-oriented, restored DNA images;
- `masks/<ROI>.tiff`: original-resolution unsigned-integer instance masks;
- `DNA_preprocessing_results.csv`: restoration dimensions and scale information;
- `CellposeSAM_segmentation_results.csv`: per-ROI counts, filtering, parameters,
  and output paths; and
- `CellposeSAM_cell_metrics.csv` plus its feature dictionary: per-object shape,
  intensity, flow, probability, neighbourhood, and diagnostic features used to
  audit mask quality.

The configured folder names may change the first two paths. QC overlays use
green boundaries for retained objects and red boundaries for excluded objects.
`qc_boundary_dilation` only makes those outlines easier to see; it does not
alter a mask.

The per-object metrics include DNA measurements on both restored and source
images. Differences between them help diagnose what restoration changed, but
the restored intensity is not a biological abundance measurement. Cellpose
probability, flow, local-background, and morphology measurements are similarly
diagnostic features rather than cell phenotypes.

The reported object density assumes one image pixel represents one micrometre,
so `Objects_per_mm2` is meaningful only when that calibration is true. The mask
TIFF itself does not contain physical pixel-size metadata. Verify acquisition
resolution before interpreting or comparing the density column.

## Segmentation QC checklist

Review overlays on the **original denoised DNA**, not only the sharper restored
image. At minimum, include ROIs from every tissue type, staining batch,
acquisition session, and visibly different signal-quality group. Check:

- **Detection:** are weak, small, bright, mitotic, and oddly shaped nuclei all
  represented, and is DNA-positive debris excluded?
- **Separation:** are touching nuclei split at plausible boundaries, without
  dividing lobulated or textured single nuclei into fragments?
- **Merging:** do unusually large masks contain several visible DNA centres?
- **Alignment:** after downscaling, do mask edges remain aligned with source DNA
  rather than appearing shifted or blocky?
- **Compartment balance:** do thresholds work for lymphocyte-rich, stromal,
  epithelial, necrotic, and tumour regions rather than only the dominant tissue?
- **Borders and empty regions:** are border objects handled consistently and are
  tissue-free regions free of spurious objects?
- **Counts and morphology:** are discontinuities between otherwise comparable
  ROIs explained by biology, or by changes in DNA intensity, restoration, or
  filtering?

An overlay can look satisfactory while still producing biased marker summaries.
Inspect several individual masks together with the channels that will be
quantified, especially when nuclear-centred masks are used for membrane or
cytoplasmic markers. Record the selected settings and the representative ROIs
used for acceptance so that later analyses can distinguish biological changes
from a changed object definition.

## Common limitations

- Cellpose-SAM is a generalist microscopy model, not an IMC-specific nuclear
  ground truth. Performance can vary by tissue and acquisition quality.
- Learned deblurring and upsampling can introduce plausible features. Their
  outputs support segmentation but must not be interpreted as new measurements.
- Two-dimensional nuclear profiles do not map perfectly to biological cells,
  particularly for multinucleated, dividing, apoptotic, or sectioned cells.
- A nuclear-only mask incompletely samples membrane and cytoplasmic proteins.
  Geometric expansion reduces but does not solve this problem.
- Parameter scans compare internal outputs without a manual ground truth. More
  masks, smoother masks, or a favourable mean metric do not prove greater
  biological accuracy.
- The two pinned Cellpose versions are part of run provenance. Results may change
  if either restoration or segmentation is upgraded, even when YAML settings are
  unchanged.

## References

- Stringer C, Pachitariu M. [Cellpose3: one-click image restoration for improved
  cellular segmentation](https://doi.org/10.1038/s41592-025-02595-5). *Nature
  Methods* 22, 592–599 (2025).
- Pachitariu M, Rariden M, Stringer C. [Cellpose-SAM: superhuman generalization
  for cellular segmentation](https://doi.org/10.1101/2025.04.28.651001).
  bioRxiv version 1 preprint (2025).
- [Cellpose source repository](https://github.com/mouseland/cellpose).
