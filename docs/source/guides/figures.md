# Publication figures

`SpatialBiologyToolkit.figures` builds the same spatially aligned panel layout
for one or every ROI. A recipe describes **what to draw**; a `Dataset` supplies
AnnData, images, segmentation masks and spatial calibration. Rendering does not
save intermediate composites or alter AnnData/backgating settings.

The runnable notebook `Tutorials/Publication_figures.ipynb` creates a small
synthetic dataset and demonstrates all five layer types without project data.

## Short notebook workflow

Use ordinary in-memory AnnData, or `Dataset.from_h5ad` to load it. The latter
retains all expression layers, original observation IDs and embeddings; it is
not a partial loader. It supports newer nullable-string/null encodings in older
SBT environments without changing the source H5AD.

```python
from SpatialBiologyToolkit import figures as F

data = F.Dataset.from_h5ad(
    "cells.h5ad", imc_folder="Images", mask_folder="Masks",
    image_folders={"he": "Images_HE"},
).map_obs("population_names.csv", source="leiden_1.5",
          columns=["Broad", "Specific"])
# Already loaded? Use F.Dataset(adata, ...) with the same folder arguments.

imc = F.IMC.rgb("CD3", "CD68", "DNA1")
cells = F.Populations(obs="Specific", mode="outline")
figure = F.Figure.grid([
    [F.Panel(title="IMC", layers=[imc]),
     F.Panel(title="Cell identities", layers=[imc, cells]),
     F.Panel(title="Histology", layers=[F.Image(source="he")])],
], intensities=F.Intensities.from_csv("normalization_dict.csv"),
   crop=F.Crop(mode="center", size=(300, 300)))

report = figure.preflight(data)
display(report)
roi = report.select(method="first")[0]
with figure.preview(data, roi) as preview:
    display(preview)                         # in memory, 120 dpi
    preview.save_bundle("Figures/example")  # PNG + layered SVG, 300 dpi
    reviewed = preview.freeze()             # resolved settings and this ROI's crop
```

`map_obs` defaults to CSV key `source_population`. It validates unique keys,
complete mappings and nonmissing labels. Mapped categorical annotations live
in `data.obs`; the supplied `adata.obs` and source file remain unchanged. An
existing output column requires `overwrite=True`. CSV paths and fingerprints
are recorded in exports.

`Figure.grid` assigns positions and letters and adds one automatic scale bar
unless a panel already specifies one. Disable these with `letters=False` or
`scale_bar=False`. `ScaleBar.auto()` chooses a 1/2/5-series length near one fifth
of the crop width, using micrometres when calibrated and pixels otherwise.
Layer objects can be reused in multiple panels; each placement receives its
own SVG identity. Explicit duplicate IDs in saved recipes are still errors.

`Intensities.from_csv` reads named `marker`, `lower_threshold` and `vmax`
columns (custom column names are supported). Channels inherit these bounds;
explicit channel `limits=` or `scale=` takes precedence. Missing entries in a
supplied table raise before calibration. Without a shared table, omitted
channel scales retain the existing cohort-quantile behaviour. Pixel bounds
are never applied to `Values` layers.

`Intensities.scale_max(factor)` multiplies every maximum by the supplied factor
while keeping minima unchanged. It returns a validated new object; neither the
original object nor the source CSV is modified. Explicit per-channel scales
still take precedence over shared intensities. The factor must be finite and
positive, and each adjusted maximum must remain greater than its minimum.

`Scale(mode="pooled_quantile", quantiles=(0.01, 0.99))` supports exact pooled
cell quantiles for observation or expression values, including named layers.
It weights cells equally, whereas `cohort_quantile` reduces per-ROI bounds.
Pooled quantiles are unavailable for image cohorts because they would require
retaining the cohort's pixel values. Calibration uses all dataset ROIs unless
`Scale.rois` explicitly selects a reference cohort, even for one preview.
Supplied fixed bounds require no calibration image reads.

Preflight reads image headers and checks required markers, references, integer
mask types, IMC dimensions, population labels and basic crop feasibility.
Missing files are reported as exclusions; ambiguous matches and invalid inputs
raise. `report.eligible_rois` is an explicit complete-modality cohort.
`report.save(path)` writes the full availability table. Spatial hotspot
feasibility and image contents are validated during rendering.

`report.select(n, method="most_cells")` ranks qualifying whole-ROI cell counts;
it is separate from the within-ROI crop objective. `method="random", seed=42`
is reproducible. Add `balance_by="Case"` to select round-robin across cases;
the balancing column must identify exactly one group per ROI. These selectors
do not assert statistical representativeness.

`figure.with_style(title_fontsize=14)` and `figure.with_crop(new_crop)` return
validated copies. Preview writes nothing by itself. Bundle exports include
images, source binding, metadata, a reusable recipe and a per-ROI frozen recipe.
`figure.export(data, folder, roi=...)` offers single-ROI export; `export_rois`
adds a batch manifest and HTML review gallery, plus per-ROI frozen recipes.
Neither performs image registration.

`Style.letter_fontsize` and `Style.scale_bar_fontsize` control panel letters and
scale-bar text independently, in points. Both default to `None`, inheriting
`title_fontsize` and `legend_fontsize`, respectively, to preserve existing
recipes. Changing either override does not change titles or legend text.

## A reusable recipe

```python
from SpatialBiologyToolkit import figures as F

data = F.Dataset(
    adata=adata,
    imc_folder="Images_panel1_cells",  # one subfolder per ROI, one TIFF per channel
    mask_folder="Masks",             # one 2D cell-label TIFF per ROI
    image_folders={"he": "Images_HE"},
    label_folders={"regions": "Tissue_annotations"},
    roi_obs="ROI", label_obs="ObjectNumber",
    x_obs="X_loc", y_obs="Y_loc",
    pixel_size_um=1.0,                # replace with the actual reference calibration
)

figure = F.Figure(
    layout=(2, 2),                   # rows, columns; zero-based panel positions
    crop=F.Crop(mode="center", size=(300, 300)),
    style=F.Style(
        font_family="Arial", title_fontsize=14, legend_fontsize=9,
        panel_width_mm=60, panel_height_mm=60,
        row_spacing_mm=4, column_spacing_mm=4,
    ),
    panels=[
        F.Panel(row=0, col=0, title="IMC", letter="A", layers=[
            F.IMC(channels=[
                F.Channel("CD3", color="red", limits=(0, 5)),
                F.Channel("CD68", color="green", limits=(0, 10)),
                F.Channel("DNA1", color="blue", limits=(0, 20)),
            ]),
        ], scale_bar=F.ScaleBar(length=50, unit="um")),
        F.Panel(row=0, col=1, title="Populations", letter="B", layers=[
            F.Populations(obs="Specific", groups=["TAM-Mac-2", "MES-Hypoxic"],
                          mode="both", edgecolor="white", opacity=0.8),
        ]),
        F.Panel(row=1, col=0, title="Tissue annotations", letter="C", layers=[
            F.Image(source="he"),
            F.LabelMask(source="regions", labels={1: "Tumour", 2: "Stroma"},
                        colors={1: "#e45756", 2: "#00b478"}, opacity=0.35),
        ]),
        F.Panel(row=1, col=1, title="Hypoxia", letter="D", layers=[
            F.Values(value=F.obs("hypoxia_score"), cmap="magma",
                     scale=F.Scale(mode="fixed", limits=(-2, 2))),
        ]),
    ],
)

# Reuse calibration for previews and repeated ROI renders.
prepared = figure.prepare(data)
with prepared.render("ROI_1", dpi=120) as result:
    display(result)
    result.save("Figures/ROI_1.svg")

manifest = figure.export_rois(data, "Figures/All_ROIs", formats=("png", "svg"))
figure.save("figure_recipe.yaml")
restored = F.Figure.load("figure_recipe.yaml")
```

The numeric channel limits above are examples, not recommended limits for a
particular dataset. Use your established normalisation values.

## Layers and spatial coordinates

Panels contain ordered layers; later layers draw on top. `rowspan` and `colspan`
allow panels to occupy multiple slots, and unused slots remain blank. Overlap
and out-of-grid placements are validation errors. All panels use the same view
and preserve image aspect ratio; image content may occupy less than its slot.

| Layer | Content |
| --- | --- |
| `IMC(channels=[...])` | Additive composite of arbitrary marker/colour pairs. Each channel has a `Scale` and optional positive `gamma` (display exponent is `1/gamma`). |
| `Image(source="he")` | Grayscale or RGB/RGBA raster. Optional `colors={"Legend label": "#hex"}` supplies a legend. |
| `Populations(obs="Specific", ...)` | AnnData categories mapped through explicit per-ROI object IDs. `groups=None` shows all; `groups=[]` shows none. |
| `Values(value=F.obs("score"), ...)` | Floating-point cell measurements with a continuous colour bar. Use `F.var("CD3", layer="scaled")` for expression in X/a named layer. |
| `LabelMask(source="regions", labels={1:"Tumour"}, ...)` | Independent TIFF annotation labels; no AnnData object is required for each tissue region. |

Cell/label layers accept `mode="outline"`, `"fill"`, or `"both"`, `opacity`,
`linewidth` in points for vector paths, `edgecolor`, and `rendering="vector"`
(default) or `"raster"`. Raster mode is intended for dense previews; its boundary
is a source-pixel edge. Vector mode is recommended for editable publication
outlines. Label-mask background defaults to zero and stays transparent;
unlisted label IDs also remain transparent. Disconnected regions and holes are
retained. Use Matplotlib colour names or hex strings in recipes.

`Dataset` determines the reference grid from `reference_folder` if supplied,
otherwise the cell mask, IMC channels, then external image/label sources.
Coordinates use reference pixels with x increasing right and y increasing down.
IMC channels and cell masks must match this grid. External images and annotation
masks can have different pixel resolutions but **must cover the same tissue
extent and orientation**. They are resampled directly into the requested crop;
label IDs always use nearest-neighbour resampling. This is not registration.

External images support PNG/JPEG/BMP and single-plane TIFFs. Multi-page or
multichannel scientific images must first be exported as a display plane.
Whole-slide pyramid readers, affine registration and a graphical editor are not
part of this first release. Physical scale bars currently use one scalar
`pixel_size_um` per Dataset; use separate datasets for different calibrations.

Filenames are indexed once. Matching prefers exact ROI stems, then case/separator/
leading-zero equivalents, then names with prefixes or suffixes. Numeric tokens
prevent `ROI_1` matching `ROI_10`; ambiguous best matches raise an error. IMC
marker matching uses the same safeguards. `data.describe()` exposes available
ROIs, image/label sources, channel filename stems, obs, var and expression layers.

## Selecting a view

All sizes/bounds are reference pixels, independent of export DPI. Bounds use
`(x, y, width, height)`; sizes use `(width, height)`.

```python
F.Crop()  # complete ROI
F.Crop(mode="bounds", bounds=(100, 200, 300, 300))
F.Crop(mode="coordinate", size=(300, 300), center=(450, 600))
F.Crop(mode="cell", size=(300, 300), cell_id=123)
F.Crop(mode="upper_left", size=(300, 300))
# Also: center, upper_right, lower_left, lower_right.

# Most selected cells:
F.Crop.hotspot(size=(300, 300), reducer="count", where=[
    F.Condition(value=F.obs("Specific"), values=["TAM-Mac-2"]),
])

# Highest mean score among finite cells; avoid isolated extreme cells:
F.Crop.hotspot(size=(300, 300), score=F.obs("hypoxia_score"),
               reducer="mean", min_cells=30)

# Greatest proportion of high-score cells, within a population denominator:
F.Crop.hotspot(size=(300, 300), reducer="fraction", min_cells=30,
    where=[F.Condition(value=F.obs("hypoxia_score"), op="gt", values=[1.5])],
    denominator=[F.Condition(value=F.obs("Specific"), values=["MES-Hypoxic"])])

# Greatest annotated tissue-area fraction, independent of AnnData:
F.Crop.hotspot(size=(300, 300), reducer="fraction",
               mask_source="regions", mask_labels=[1])

# Cell-based selection constrained by at least 50% tumour area:
F.Crop.hotspot(size=(300, 300), score=F.obs("hypoxia_score"), reducer="sum",
               mask_source="regions", mask_labels=[1], min_coverage=0.5)
```

Conditions in a list are combined with AND. `op="in"` (default) supports several
values; `eq`, `gt`, `ge`, `lt`, `le` take one value. Precomputed per-cell scores
can be stored in obs and referenced with `F.obs`; recipes never evaluate code.

`count` and `sum` favour concentrated numbers/total signal; `mean` favours average
signal and `fraction` uses an explicit cell denominator. Means/sums exclude
non-finite scores and apply `min_cells` to contributing cells; fractions apply
it to denominator cells. Mask-only selection counts pixels or area fraction,
so `min_cells` does not apply. Out-of-image/non-finite cell coordinates are
excluded. Cell centers are rounded to reference pixel centers.

The default `stride=1` searches every integer window origin with bounded working
arrays. Larger strides are an explicitly approximate speed/precision tradeoff;
edge positions are also tested. There is no hidden random sampling. Ties prefer
proximity to the selected-cell centroid, then topmost/leftmost coordinates.
When no eligible view exists, the default is an error; `fallback="center"` is
explicit and recorded. Oversized requested sizes clamp to the ROI; explicit
out-of-bounds coordinates raise an error.

The result metadata and batch manifest record selected bounds, objective, score,
stride, denominator/contributor count and any mask coverage. Reuse them with:

```python
views = {row["roi"]: row["view"]["bounds"]
         for row in manifest["results"] if row["status"] == "ok"}
figure.crop = F.Crop(mode="bounds", roi_bounds=views)
```

Hotspots illustrate maximal activity, not necessarily representative tissue.
Review all ROIs in the generated `index.html` before selecting publication panels.

## Scaling, palettes and memory

`Scale(mode="fixed", limits=(low, high))` performs no cohort image reads.
`Scale(mode="roi_quantile", quantiles=(0, .99))` computes each ROI's bounds.
The default `cohort_quantile` averages per-ROI quantile bounds across **all Dataset
ROIs**, even when a subset is rendered. `reduction="min"` or `"max"` changes that
reduction; `rois=[...]` explicitly defines a different calibration cohort.
These are reductions of per-ROI quantiles, not a pooled-pixel quantile.

Fixed display bounds map directly to fixed intensities across images. This is
intentional for the new engine; legacy backgating retains its existing display
behaviour. A constant intensity range maps to zero in an IMC composite. All-
non-finite calibration data raises an error; explicit limits allow missing cell
values to be shown with `Values(missing_color=...)`. True zero is a valid value,
separate from mask background. Unmatched cell-mask objects remain transparent.

Population palettes are determined once from explicit colours, AnnData's stored
palette, or a deterministic fallback over the full category set. Categories
and limits therefore do not change when a different ROI is rendered. Metadata
records resolved palettes and normalisation bounds/reference ROIs.

Calibration streams one image/value vector at a time and caches only scalar
bounds. Rendering uses a bounded, per-ROI image cache (128 MiB by default; configure
`Dataset(cache_bytes=...)`). Large uncompressed TIFFs can be memory-mapped;
compressed formats may require a full decode. Destination resampling, composites
and cell geometry are limited to the visible crop (plus a boundary halo).
The largest decoded source, current figure and cache still contribute to peak
memory. Sparse expression is sliced to the requested ROI/marker before densifying.

`prepared = figure.prepare(data)` freezes the recipe and resolved cohort bounds;
reuse it for lower-DPI previews and final exports. A changed source/calibration
requires preparing again. Use `with prepared.render(...) as result` or call
`result.close()`; this releases the figure and renderer buffers. No pyplot figure
is registered or queued for notebook display by batch rendering.

## Export and reproducibility

PNG, SVG, PDF and TIFF are supported. SVG keeps named panel, image, object fill,
object outline, legend, title, letter, colour-bar and scale-bar groups. Text remains
text in the chosen font (Arial by default); fonts are referenced, not embedded.
Illustrator may place editable SVG groups under a single native layer. Dense
raster layers stay raster. PDF export is supported but SVG is the editable-group
contract.

Physical panel dimensions and spacing are preserved instead of tight-cropping
the canvas. Title and colour-bar annotation space expands when required by the
font sizes. Legends sit inside the top-right corner of their panel; disable them
per layer (`legend=False`) or panel. Scale bars are opt-in per panel. A scale bar
larger than the crop raises an error.

Batch export writes one final figure per ROI/format, `recipe.json`, per-ROI JSON
metadata, an incrementally updated `manifest.json`, and a local `index.html`.
Filenames include a safe ROI slug and short digest; the manifest preserves exact
ROI identifiers. There are no intermediate composites or modified source files.
Source metadata includes matched paths, dimensions, sizes and modification times.

The default `on_error="raise"` records the failure and stops. `on_error="skip"`
records ordinary per-ROI failures and continues; memory exhaustion and cancellation
always stop. `progress(event)` and `cancelled()` callbacks support a future GUI.
Cancellation is checked between calibration reads, crop-search rows and panels.

Recipes use schema version 1, strict validation and persistent panel/layer IDs.
`F.Figure.model_json_schema()` describes the available fields and discriminated
layer types for future editors. Dataset discovery, recipe editing, preparation,
preview and export are separate operations; no GUI framework is required.

## Running through the visualisation stage

The same engine is available through `visualization.figure_jobs` in an ordinary
`sbt` visualisation run. Existing stage/SLURM routing is unchanged. Paths are
resolved from the project working directory. Figure output uses the stage's
reporting location under population images / `PublicationFigures`.

```yaml
visualization:
  figure_jobs:
    - name: tissue_comparison
      recipe: figure_recipe.yaml
      imc_folder: Images_panel1_cells
      mask_folder: Masks
      image_folders: {he: Images_HE}
      label_folders: {regions: Tissue_annotations}
      pixel_size_um: 1.0
      formats: [png, svg]
      on_error: raise
```

This option runs independently of the existing backgating enable flag. Existing
backgating functions retain their interfaces and output paths. Shared vector
geometry and SVG export now live in `figures`, with their old plotting imports
kept available for compatibility.
