# Operating the viewer

NapariSBT is the workflow panel inside Napari. Napari supplies the image canvas,
layer list, and display controls; NapariSBT connects them to your cell data and
saved work.

## Know where to look

| Area | What it does |
| --- | --- |
| Image canvas | Shows the tissue, marker images, cell outlines, and other overlays |
| Layer list, usually left | Selects an image or overlay; the eye icon controls visibility |
| Layer controls | Changes the selected layer's contrast, colour map, opacity, blending, or contour |
| NapariSBT panel, usually right | Chooses the task, ROI, recipe, populations, and analysis actions |
| Readiness dock, below the layers | Reports Ready, Working, Finished, or Failed, with time and detail |
| Cell properties dock | Shows selected cell observations when click tracking is enabled |

Drag a dock's title bar to move it or float it. Drag its edge to resize it.
If a tab's controls extend beyond the visible panel, scroll within that panel.
Reopening a workspace restores saved scientific state, not every window position.

## Move around the image

Use the mouse wheel or trackpad to zoom. In Napari's pan/zoom mode, drag the
canvas to move the view. Use the canvas reset-view button to fit the loaded
layers. When drawing shapes, select the relevant Shapes tool; return to pan/zoom
when finished.

In Classify and Labeler, clicks can assign labels. Choose **Select only** when
you want to inspect without assigning. These workflows do not require the cohort
layer to be visible or selected.

## Understand the layers

An **image layer** shows measured staining. A **labels layer** shows segmented
objects or assigned populations. A **Shapes layer** stores drawn regions.
Changing visibility does not remove the data; deleting a displayed layer removes
it from this viewer, not from the source file.

For several coloured marker channels, **additive** blending combines their
light. Opacity controls how strongly a layer contributes. A labels contour of
**1 px** is a useful thin outline; **0** fills the cells. Cells can share a colour
while retaining their individual boundaries.

Contrast changes how values are displayed. NapariSBT scalar images use a 0–1
display range. Saved normalisation values set the image's intensity scaling;
contrast handles choose the visible range within that scale. Recipe-specific
contrast settings take precedence over defaults in Workspace.

## Read colours with their labels

The guide uses the same blue, violet, teal, amber, rose, and cyan accents as
the interface boxes. These help you locate controls; they are not readiness
ratings. Status messages include words as well as colour.

ROI review colours depend on the tool:

| Control | Colour | Meaning |
| --- | --- | --- |
| Explore ROI dropdown | Green | Viewed with the current recipe |
| Explore ROI dropdown | Amber | Not yet viewed with the current recipe |
| Population QC ROI button | Green | Not yet viewed with this RGB recipe |
| Population QC ROI button | Grey | Viewed with this RGB recipe |

Changing the recipe changes what counts as reviewed. A review mark records a
viewing action, not a scientific validation of the population.

## Get focused help

Use the blue **Help** button on a box for just that section, or **Help for this
tab** for the complete workflow. Workspace's **Help** follows the current setup
step. In a help popup, use **Larger text** or **Smaller text** to suit your screen.
The popup headings use the same colours as the relevant interface boxes.

```{figure} ../_static/napari_sbt/help.png
:alt: Focused RGB recipe help with a teal heading, dark text on a light background, and text size controls.
:width: 760px

The real focused Help popup: matching section colour, readable text, and text-size controls.
```

## Keyboard shortcuts

Click the image canvas before using shortcuts. Saved recipes can use **F1–F12**.
In Classify, the displayed number keys change the current class while preserving
the chosen click action. Typing in a text field should remain ordinary text entry.

For multi-selection lists, use Ctrl-click on Windows/Linux or Command-click on
macOS to select several entries; Shift-click selects a range.
