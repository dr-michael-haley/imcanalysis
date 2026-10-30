# MCD Explorer

[Open MCD Explorer](https://mcd.spatialbiology.app/)

Use MCD Explorer to inspect raw imaging mass cytometry acquisitions inside an
`.mcd` file. You can find regions on a slide overview, combine marker channels,
inspect pixel counts and export composed images. No TIFF conversion, cell
segmentation or AnnData file is needed.

The file is read on your computer. Display adjustments and exports do not
change the original MCD.

## Your first acquisition

1. Open the app in desktop Chrome or Edge and choose **Open MCD**, or drag one
   `.mcd` file onto the application. Use the labelled synthetic demo if you want
   to try the controls without your own data.
2. Select an acquisition from **Acquisitions**, or click its outline on the
   slide overview. An acquisition is one imaged region of interest (ROI).
3. Choose a marker in a channel slot. Begin with one channel, then add others
   to make a composite.
4. Use **Auto** to initialise contrast, then adjust **Min/Max** while watching
   the tissue image. The number fields allow exact values.
5. Drag to pan and scroll to zoom. **Fit view**, **1:1**, and double-click help
   you return to a useful view.
6. Click an image pixel to inspect the original values for the selected
   channels. These are raw measurements, even when a display transform is used.

## Find a region on the slide

In **Acquisitions → Backdrop**, choose a panorama, an embedded whole-slide
photograph or coordinates only. Available imported photographs appear by name;
the separate slide overview remains another option. Before/after-ablation
images are available when they are embedded in the MCD.

Scroll over the overview to zoom, drag to pan, and click an ROI outline to open
that acquisition. The **+**, **−** and **Fit** controls offer the same navigation
without a mouse wheel. With the overview focused, use **+ / −**, the arrow keys,
and **F** or **Home**. Switching acquisitions on the same backdrop retains the
overview zoom; changing backdrop resets it.

ROI outlines use the MCD's recorded positioning. An externally cropped or
rotated photograph may not align with them. A photo without positioning metadata
can be shown without outlines; coordinates-only navigation remains available
if an optical image cannot be decoded.

## Mix channels and adjust contrast

You can display up to **six channels**. Select a marker and a colour for each
slot, and use its visibility control to hide it without losing its settings.
The arrows beside the marker selector cycle through markers while retaining
the slot's colour and visibility. Each newly selected marker gets automatic
contrast.

- **Min/Max** set the raw-count display range. Values below or above it clip to
  the range endpoints; the measurements themselves are unchanged.
- **Auto** uses the sampled minimum/zero and 99.5th percentile. It is a starting
  display choice, not a scientific normalisation method.
- Pure colours include red, green, blue, cyan, yellow, magenta and white.
  Viridis, Plasma, Magma and Inferno are also available. Channels blend
  additively; show a single channel when you need to read its colour scale
  without contributions from other markers.
- **Advanced → Display transform** offers linear, arcsinh, log1p and square
  root. Arcsinh defaults to cofactor 5. The transform affects appearance, not
  the raw values reported by pixel inspection or histograms.
- **Advanced → Channel order** switches between alphabetical names and the
  original MCD order without changing the selected channels.

Keep contrast and transform settings consistent when comparing acquisitions.
Independently auto-scaled images can look similarly bright despite different
signal levels.

## Examine intensity distributions

1. Open **Histogram** after displaying the channels of interest.
2. Select a subset of those channels and choose **Generate histogram**.
3. Compare their pixel-count distributions. Enable log frequency if low-count
   bins are difficult to see.

The histogram uses every finite measured pixel in the acquisition, including
measured zeros, in 128 shared raw-count bins. Missing pixels are excluded.
Changing acquisition, visible channels or the histogram subset clears the
result. Contrast and colour changes do not require regeneration because they
do not change the raw measurements.

Use **Metadata** to inspect the acquisition's recorded attributes.

## Export a figure

1. Open **Image Export** and switch from **Viewer** to **Figure**.
2. Choose an aspect ratio, or lock the frame to a physical width and height.
   Pan to position the tissue within the frame.
3. Set the title, channel legend, colour scheme and text sizes. Choose an
   automatic or manual scale bar.
4. Choose the output width and export PNG, JPEG or SVG.

The IMC view uses **1 image pixel = 1 µm** for its scale bar. Optical photographs
are uncalibrated and do not have a scale bar. Check that this calibration is
appropriate before using an IMC export as a quantitative figure.

PNG and JPEG preserve the composed appearance. SVG embeds the composite image
as a raster and keeps its labels, legend and scale bar as editable vectors;
it does not turn the measured pixels into vector shapes.

Filename options can include a prefix, acquisition name, title, timestamp and
selected acquisition metadata. Optional export bookmarks save the acquisition,
channels, display settings and framing for returning to that figure **within
the current file and tab**. Returning to Viewer restores the previous framing.

**Export bookmarks and display settings are not retained after closing the
file or reloading the tab.** Download the figures you need before ending a
session.

## Limits and troubleshooting

| What you see | What to check |
| --- | --- |
| Checker pattern within an acquisition | Those pixels are missing measurements; this is distinct from measured zero signal. Check the acquisition's notices. |
| No optical image | The MCD may not contain a supported image. Choose coordinates only and navigate through the acquisition list. |
| Photo and ROI outlines do not agree | Check whether the image was cropped/rotated and whether the MCD contains positioning metadata. |
| A marker appears dark | Show it alone, choose Auto, and inspect raw pixel values before interpreting it as absent. |
| Unsupported acquisition or memory notice | The current reader supports Float32 acquisitions. One selection is limited to 16,777,216 image pixels, 16,384 pixels per side and a 256 MiB decoded-data budget. Try fewer channels if the decoded-data limit is reached. |
| Figure export is too large | Reduce output dimensions. Exports are limited to 8,192 pixels in width, 16,384 per side and 32 million pixels overall; SVG has a 128 MiB size limit. |

The viewer holds one acquisition at a time and reads selected channels as
needed. It does not segment cells, assign populations or retain a session
workspace. For analysed cells and masks, use [Tissue Explorer](tissue_explorer.md).
For browser or network problems, see [Troubleshooting](troubleshooting.md).
