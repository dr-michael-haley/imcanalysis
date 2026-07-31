# Explore

Explore is the image and population-QC workspace. ROI navigation normally shows
only ROIs in the current experiment scope. A Feature Discovery Trial restricts
navigation to its representative ROIs so it is always obvious which cells are
being labelled and evaluated.

Load channels as greyscale, RGB, or the automatic red/green/blue/cyan/yellow/
magenta sequence. All image layers start at opacity 1.0. AnnData categorical or
numeric observations, individual populations, and `adata.X` marker values can be
rendered as cell overlays. Existing categorical colours are recovered from
`adata.uns` whenever possible.

The reload recipe records which images and overlays should reappear after moving
to another ROI, together with colours, visibility, opacity, contours, and contrast
limits. **Update from current layers** captures manual display changes. Delete
selected recipe entries to stop recreating them. Reviewed-ROI colouring applies
to the current recipe fingerprint, so changing the view creates a distinct review
context.

Population QC can rank ROIs by abundance and transfer a selected population to
Setup as a proposed cohort. Saved population verification views remember the
channels and overlays used to inspect that population.
