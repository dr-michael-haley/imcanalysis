---
napari_help_topic: explore
---

# Export tissue images for figures

Build a view in Explore and save a named recipe before opening **Publication
export…**. A practical sequence is:

1. Choose the recipe and whether to export the whole ROI or the current field.
2. Check pixel calibration before enabling a physical scale bar.
3. Choose a resolution, scale-bar appearance, and optional labels.
4. Render a preview and inspect it at the intended display size.
5. Save one image, or select a list of ROIs for bulk export.

An exported image records the view; its JSON sidecar records how it was made.
Keep the sidecar with the image for reproducibility.

```{include} ../../../SpatialBiologyToolkit/napari_sbt/help/explore.md
:start-after: "## Publication image export"
:end-before: "**Reapply the current Explore recipe"
```
