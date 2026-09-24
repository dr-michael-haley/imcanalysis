# Neighbour-Attributable Fraction (NAF): calculation and interpretation

## What does NAF measure?

The **Neighbour-Attributable Fraction (NAF)** estimates how much of a cell's measured marker signal could plausibly be explained by signal extending from nearby strongly positive cells.

It is calculated separately for **every cell and every marker**.

This addresses spatial overlap between neighbouring cells, not isotopic spillover between imaging channels.

## How we calculate it

### 1. Select positive exemplar cells

For each marker, we identify suitable positive cells using the existing expression measurement, usually the Nimbus score.

By default, exemplars must have no other cell positive for **the same marker** within 10 pixels. Cells negative for that marker can still be nearby.

We sample across image regions and positive-score ranges, rather than selecting only the brightest cells.

### 2. Learn that marker's typical "halo"

Using the original marker image, we measure signal in successive one-pixel bands **outside each exemplar's segmentation boundary**, up to 8 pixels by default.

Pixels assigned to other cells are excluded. Unassigned pixels are retained because they may contain signal from membrane or cytoplasm outside a conservative segmentation mask.

We subtract background and account for differences in exemplar brightness. Brightness is measured using a robust high-intensity statistic within the mask and a small surrounding expansion, rather than simply the mean inside the mask.

We combine the exemplar profiles using their median, retaining their variability for QC. At least five usable exemplars are required by default.

This produces a marker-specific profile describing **how much signal is typically present at each distance from a positive cell**. It need not decline continuously: membrane-associated staining may peak outside the segmentation boundary.

### 3. Identify potential neighbouring sources

We measure raw-image brightness for all retained cells.

Cells sufficiently bright relative to the exemplars become potential sources. By default, the threshold is the 10th percentile of exemplar brightness.

Nimbus helps select exemplars, but **raw image measurements determine source brightness and the final NAF**.

### 4. Project a halo around each source

We apply the learned profile around each source's actual segmentation shape, scaled to its brightness.

Two safeguards are important:

- A source's halo is never projected inside its own mask, so a cell cannot explain its own signal.
- Where halos overlap, we use the **strongest prediction at each pixel**, rather than adding predictions together.

We also record which source supplies that prediction.

### 5. Compare the predicted halo with the target's observed signal

We subtract the image-region background from the target's raw signal and set negative values to zero.

At each pixel inside the target mask, the attributable amount is the **smaller of**:

- the observed signal above background; and
- the predicted neighbouring-halo signal.

Therefore, a prediction counts only where it coincides with observed signal, and cannot explain more signal than is actually present.

We then calculate:

**NAF = attributable signal summed across the target mask ÷ total observed signal above background in that mask**

If there is no observed signal above background, NAF is set to zero.

### 6. Identify the sources and their populations

We retain the contributing neighbours and divide total NAF into:

- **Homotypic NAF:** explained by sources from the same population as the target.
- **Heterotypic NAF:** explained by sources from a different population.
- **Unknown-population NAF:** source or target population labels are missing.

These components add up to total NAF. Same-population sources are not discounted.

## How to interpret the results

**NAF describes spatial explainability, not proven contamination.**

For example, a macrophage with CD31 NAF of **0.70** has 70% of its observed, background-subtracted CD31 signal spatially explainable by the neighbouring-halo model.

If this comprises:

- homotypic NAF = 0.10;
- heterotypic NAF = 0.60;

then 10% of its observed signal is explained by same-population neighbours and 60% by different-population neighbours. Source provenance identifies the specific cells involved.

This does **not** mean there is a 70% probability that the macrophage's CD31 measurement is artefactual.

In practice:

- **High NAF:** interpret that marker measurement cautiously and inspect the image, halo profile and predicted sources. Genuine expression and neighbouring signal may coexist.
- **Low NAF:** this model finds little spatial explanation from neighbouring sources. It does not prove genuine expression or exclude other artefacts.
- **High heterotypic NAF:** particularly useful for investigating unexpected marker measurements across population boundaries.
- **High homotypic NAF:** indicates spatial ambiguity within a population, even if it does not change the cell-type interpretation.

Always consider absolute intensity alongside NAF: a high fraction of a very weak signal can represent little attributable intensity.

Population heatmaps show the **mean cell-level fraction**, including zero-scoring cells, not the fraction of pooled population intensity.

Finally, markers without enough valid exemplars are **unavailable**, not confidently unaffected, even though their stored scores use zero placeholders. There is no validated universal NAF cutoff; values such as 0.5 are descriptive QC thresholds.
