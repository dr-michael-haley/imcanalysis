# Your first session

The aim of this first session is simple: open a dataset, see a marker image with
cell outlines, and save a view you can use again.

## Before you begin

You need NapariSBT installed and three components from a processed dataset:

| Component | What it is | What to select |
| --- | --- | --- |
| Cell data | Measurements and labels for each cell | The intended `.h5ad` file |
| Cell outlines | Segmentation masks linking pixels to cells | The masks folder |
| Staining images | Marker images for each region | One or more image folders |

Your analyst or imaging colleague can identify these files. The cell data and
masks must describe the same cells. If there are several analysis versions, ask
which one to use before combining them.

## 1. Open NapariSBT

On Windows, open the **NapariSBT** shortcut if it was installed. On another
workstation, use the launcher provided by your administrator. If you have no
launcher, see [Installation and launching](installation.md).

```{figure} ../_static/napari_sbt/home.png
:alt: NapariSBT Home with Create a workspace, Open a workspace, and recent workspaces.
:width: 640px

Home is the starting point for both new and saved work.
```

Click **Create a workspace**. A workspace is the folder that keeps your review
settings and analysis work. Give separate review tasks separate workspaces.

## 2. Choose the dataset

On **Workspace → Dataset**, click **Choose dataset…**. Expand **Review dataset
files** and confirm the detected cell-data, mask, and image paths. Use the chooser
buttons to correct missing paths. If several AnnData files are found, select the
one you intend to review. Click **Continue**.

## 3. Choose exploration

On **Task**, choose **Explore my images and cells**. This shows the tools needed
for image review. Click **Continue**.

On **Check data**, you may leave **Open quickly (check as I go)** enabled for a
first exploration. NapariSBT checks each region when opened. For a complete asset
check, use **Check entire dataset** and review its result. **Find files only**
locates paths but does not check mask contents.

On **Save workspace**, enter a name such as `Initial tissue review`, check the
save location, and click **Create workspace**. Wait for the workflow tabs to appear.

## 4. Display one marker and cell outlines

Open **Explore**, select an ROI, and click **Load ROI**. Scroll to the amber
image-channel box, select one marker, and click **Load selected greyscale**.

Select that image layer in Napari's left-hand list. Adjust the contrast handles
until you can see the staining. Add the **all-cells mask** using the button at
the top of Explore; choose an outline contour in Napari's labels controls if
filled labels obscure the image.

For named population colours, use the teal **AnnData overlays** box: choose the
population observation and click **Render observation overlay**.

## 5. Save your view and inspect another ROI

In Explore's violet recipe box, enter a descriptive name and click **Save current
view as new recipe**. Enable **Reapply the current Explore recipe after changing
ROI**, then use **Next ROI**. The same markers, colours, and contrast settings
should appear wherever those channels are available.

You have now created a reusable image-review workspace. Continue with
[Viewer basics](viewer_basics.md), or use [Population QC](population_qc.md) to
review one cell population at a time.

## Finish the session

Saved named recipes return when you reopen this workspace from Home.
If you also created labels, check [Saving and resuming work](saving.md): some
tools update only the live cell table until you explicitly save or export.
