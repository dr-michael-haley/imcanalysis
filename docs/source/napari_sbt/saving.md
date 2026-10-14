# Saving and resuming work

A workspace, the cell-data file, and an exported result serve different purposes.
Before closing NapariSBT, check which one your current tool saves.

| Your work | How to keep it | What returns later |
| --- | --- | --- |
| Named image recipe | **Save current view as new recipe**, or **Update selected recipe from current view** | Saved channels, colours, visibility, opacity, contours, contrast, and shortcut |
| Population QC RGB view | **Save RGB recipe for population**, or load its population view | That population's channel combination and contrast ranges |
| Population names and merges | **Save and Update** | Saved naming draft; its derived labels are synchronized into the app |
| Classifier annotations | Propose/confirm actions write workspace annotation records | Proposals and confirmed training labels |
| Labeler cell list | **Export CSV** before closing | A portable list of explicitly labelled cells |
| New column in live AnnData | Save an AnnData copy using the relevant export or Dataset Maintenance | Labels available in the saved cell-data file for other tools |
| Scanpy figure | Use the plot toolbar's save action; optionally export plotted data | An image and/or CSV; open popup windows are session snapshots |
| Tissue image for a figure | **Explore → Publication export…** | Image plus a settings sidecar; bulk export also records a result manifest |

## Reopen the workspace

On Home, choose a recent workspace or **Open a workspace…**. Select the saved
workspace folder, which contains `experiment.yaml`. Select the workflow from
**Workspace → Change task** if you need a different set of tabs.

The workspace still refers to its dataset. If files moved, update the paths in
**Dataset settings / troubleshooting** and check them again. Reopening a
workspace does not prove that externally modified source files still match.

## Save a cell-data file

Population naming and other tools can add columns to the AnnData held in memory.
To share those labels with another application, save a new `.h5ad` file. In
Dataset Maintenance, choose **Save current AnnData** and a clear output filename.
Read the saved-path message, then keep that file with the corresponding image
and mask assets.

An export to CSV is a cell list or assignment table; it is not a complete AnnData
file. A saved recipe stores display choices; it is not a copy of the images.

## What “up to date” means

Changing labels can make a model stale. Retraining makes earlier scores stale.
Changing final-decision thresholds requires refreshing final identities before
export. Follow readiness messages in that order.

**Finished** in the activity dock describes the most recent action. The dedicated
feature, naming, or classifier readiness status tells you whether its saved result
is suitable for the next action.
