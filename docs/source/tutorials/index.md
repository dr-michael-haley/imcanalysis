# Tutorials

The current starting point for interactive population analysis is
[`NEW - Population identification.ipynb`](https://github.com/dr-michael-haley/imcanalysis/blob/main/Tutorials/NEW%20-%20Population%20identification.ipynb).
Copy notebooks into an analysis directory outside the repository before
editing them, so a later `git pull` cannot conflict with your work.

The notebooks under [`Tutorials/Old_Notebooks`](https://github.com/dr-michael-haley/imcanalysis/tree/main/Tutorials/Old_Notebooks)
pre-date the current config-driven pipeline. They remain useful as examples,
but may require changes for current package APIs and data layouts.

| Status | Notebook | Notes |
|---|---|---|
| Retired | 1A. ImcSegmentationPipeline Data Import | Uses the older Bodenmiller import pipeline. |
| Legacy | 1B. Steinpose and Steinbock Data Import | Example for older Steinpose/steinbock outputs. |
| Legacy | 2. Population identification | Superseded by the current population-identification notebook. |
| Retired | 3. Neighborhoods | Superseded neighbourhood workflow. |
| Experimental | 3b. QUICHE neighbourhoods | Research example; dependencies and API may change. |
| Legacy | 4. Plotting | Older plotting examples. |
| Legacy | 5. Spatial analyses | Older SpOOx-based spatial workflow. |
| Example | 6. Napari explorer | Use with the current [Napari explorer guide](../guides/napari_imc_explorer.md). |
| Example | 7. Subregion analysis | Sample subregion workflow. |
| Legacy | 8. Backgating and plotting populations | Most functionality moved into the current population-identification workflow. |

For routine processing, prefer the [SLURM pipeline](../pipeline/index.md); use
notebooks for interactive QC, exploration, and analyses that are not yet
pipeline stages.
