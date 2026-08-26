# Population QC

Population QC is a compact population-by-population image-review workflow. It
uses the same image discovery, AnnData overlays, saved recipes, display
normalization, and viewed-ROI history as Explore, while exposing the controls most
useful for rapidly checking whether a population label is biologically plausible.

## Population selection

Choose the AnnData observation containing the populations to review, then choose
one population. Categorical order is preserved when the observation is
categorical. Changing the selection restores any RGB recipe previously saved for
that exact observation/value pair. Without a saved recipe, NapariSBT immediately
calculates and caches the top three safely matched markers for that population.
The abundance ranking is also cached and is recalculated only when the population
or ranking controls change, or when **Recalculate ROI list** is pressed.

Population QC reads cells from the workflow's frozen identity scope. A prominent
banner at the top of the tab says either **WHOLE DATASET** or **LIMITED CELL
SCOPE**, with selected/total cell and ROI counts. For a limited scope it also
shows the frozen observation/value selector. Population choices, marker
suggestions, ROI abundance rankings, and overlays include only represented cells;
categories which have no cells in the active scope are not offered in the
population selector. This prevents an out-of-scope population from looking like a
broken zero-cell result.

**Variable list order** controls the RGB channel menus and is synchronized with
the same control in every other NapariSBT tab. AnnData order is the default;
alphabetical and cached `adata.X` expression-similarity ordering are also
available. Changing the order never changes a saved RGB recipe or reloads the
current ROI.

Population QC does not rename, merge, confirm, or classify cells. Use Population
naming for population names and merges, or transfer a population through Explore
when it should become a new classification cohort. To review the complete dataset
after opening a limited classification experiment, open or create an all-cells
workspace; frozen scope is not silently widened.

## ROI sample metadata

The current ROI panel shows `adata.obs` fields automatically identified as
sample-level metadata because they are constant within every ROI. The ROI and
object-ID columns are omitted from the table. Common categorical, text, Boolean,
numeric, date/time, duration, and missing scalar values are supported. Detection
is cached when the AnnData changes, so clicking the ranked ROI buttons only swaps
the small displayed value table.

## RGB verification recipe

Choose up to three distinct image channels for red, green, and blue additive
layers. Each channel has its own normalized 0–1 contrast range; these limits and
the explicit colours are stored in the shared Explore recipe state. A population
without a saved recipe starts with **Default contrast lower** and **Default
contrast upper** from Setup. Edit any red, green, or blue values here to override
those defaults for the current population. Later Setup edits update only controls
which still contain the previous default; they do not replace manual overrides or
saved recipes. **Use Setup contrast defaults** explicitly resets all three working
ranges when required.

**Outline width for all populations** controls the contour of the
selected-population label layer from 0 to 20 pixels. The default is 1 pixel; 0
displays filled labels. Original mask IDs are retained in this display layer, so
a non-zero outline also separates touching cells assigned to the same population
instead of presenting them as one merged object. This is one workspace-wide
Population QC preference, so changing populations does not replace it. It is not
stored in each population's
RGB recipe or CSV row. The current global width still contributes to the effective
view fingerprint, so changing it creates an appropriate new viewed-ROI context.

**Suggest top three markers** ranks image channels that can be matched safely to
`adata.var` by their mean `adata.X` expression in the selected population.
Suggestions are a starting point and should be reviewed biologically. If the
selected scope contains no matching cells, or image names cannot be matched to
variables, the optional suggestion action reports a warning in the tab instead of
interrupting workspace use with an error dialog; RGB channels can still be chosen
manually.

**Save RGB recipe for population** records the controls without changing the
viewer. **Load population view** saves and applies the recipe to the current ROI.
Missing channels remain in the saved recipe and are visibly marked rather than
being discarded. Legacy IMC Explorer RGB-setting CSV files can be imported, and
the saved settings for the selected observation can be exported in a compatible
one-row-per-population layout.

## ROI sampling

Choose **Top abundance**, **Bottom abundance**, or **Random**, set the number of
ROIs, and recalculate the list. Top and bottom order cells by the number of
selected-population cells in each eligible ROI, including zero-count ROIs at the
bottom or top as appropriate. Random order uses the displayed seed, so samples are
reproducible; change the seed for a different sample.

Each ROI button shows the matching-cell count. Green means that ROI has not been
viewed with the exact current channels, colours, contrasts, population overlay,
and layer settings; grey means it has. Clicking a button loads that ROI with the
current Population QC recipe and records it as viewed. Changing a channel or
contrast creates a different recipe fingerprint and therefore a separate review
history.

Population QC uses a lightweight ROI-switching path. Population masks are mapped
to pixels in one vectorized pass, matching layer objects are updated in place,
images and overlays can be restored from the bounded cross-ROI cache, and the
classification-cohort/context layers are not constructed. The review recipe and
visited ROI are persisted once after a successful switch.

Live tracking of manual Napari layer changes defaults off for a Population QC
session. Enable it in Setup when manual changes should continually update the
working Explore recipe; explicit RGB controls and saved views work either way.
