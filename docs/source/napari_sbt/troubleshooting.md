# When something looks wrong

Start with the status message in NapariSBT Readiness and the relevant tab's
readiness summary. If an action failed, keep its full message and, when offered,
the traceback for whoever supports your installation.

## I cannot see the tab I need

Workspace shows only tools relevant to the chosen task. Use **Workspace → Change
task**, or choose **Advanced: show every tool** when combining workflows.
Scroll the tab bar horizontally if some tabs are outside the dock.

## No ROI, marker image, or cell outline appears

Check the paths in **Workspace → Dataset settings / troubleshooting**.
Confirm that the correct cell-data file is selected and that its ROI and
ObjectNumber columns match the masks. **Check dataset** reports coverage issues.
For irregular image layouts, **Find files only** builds a reusable file index.

In Explore, load an ROI and explicitly load a channel. Check layer visibility,
opacity, and contrast before assuming the file failed to load. An unmatched image
can still be listed as an additional channel.

## A population seems to have disappeared

Read the Population QC scope banner. A workspace restricted to one population
does not show the entire dataset. Open an all-cells workspace to review everything.
After renaming, click **Save and Update**, then select the new label column.

## My recipe is missing a layer

An orange recipe entry means its source is unavailable in this ROI or workflow.
The entry stays saved and can be retried elsewhere. Do not delete it unless it
should no longer belong to the recipe.

If a named recipe still shows an old contrast or colour, use **Update from current
layers**, then **Update selected recipe from current view**. Live tracking updates
the working view; it does not automatically overwrite every named recipe.

## The viewer says Working for a long time

Look for changing elapsed time, completed-ROI counts, and process details.
Full dataset checks read many files. Feature builds have a separate progress log
and worker heartbeat. A live process alone does not prove useful progress; a
stale heartbeat or repeated failure message needs investigation.

Use the operation's cancel control when appropriate, and wait for its current
filesystem operation to finish. If the interface remains unresponsive, record
the current action and the latest visible status for your support contact.

## All predictions look alike

Check confirmed-label counts for every class, not just proposals. Confirm that
Feature Building reports ready, examples have usable features, and the model
and scores match the current labels. Add representative examples across ROIs,
then retrain and score. Threshold changes cannot repair an uninformative model.

## Plots have long names or no apparent filter

Enlarge the plot window. Select fewer markers, shorten display names, or adjust
the colour-strip gap and width. In Scanpy plotting, an empty ROI selection means
all ROIs. Selecting populations filters cells only with the corresponding
selected-population scope. Regenerate snapshots after updating labels.

## Labels did not survive closing the application

Review [Saving and resuming work](saving.md). In particular, Labeler assignments
are session-only, and applying a column to live AnnData does not save a file.
Use CSV export or save an AnnData copy before closing.

## Image brightness differs between ROIs

Use saved channel intensity limits in **Workspace → Display settings** for
consistent normalisation. Per-ROI automatic scaling can make different raw
intensities look similarly bright. Recipe contrast overrides workspace defaults.
Neither view alone establishes a quantitative biological difference.
