# Words used in the interface

| Word | Meaning in NapariSBT |
| --- | --- |
| ROI | A region of interest: usually one tissue image or acquisition |
| AnnData / H5AD | The cell-data object / file containing measurements, cell information, and stored analysis results |
| Observation / obs | Information attached to each cell, such as population, patient, or ROI |
| Variable / var | A measured channel or marker, such as CD3 |
| X | The stored cell-by-marker measurement table |
| Mask / segmentation | An image in which every cell has an integer object ID |
| ObjectNumber | The mask ID used to link a segmented cell with its data row |
| Population | A group of cells sharing a categorical label |
| Leiden cluster | A computational group, often initially identified by a number |
| Cohort / cell scope | The subset of cells included in the current workflow |
| Workspace | A folder holding NapariSBT's saved workflow state and results |
| Layer | One image, label overlay, or shapes collection displayed in Napari |
| Recipe | Saved display choices: channels, colours, contrast, and overlays |
| Normalisation | Rescaling intensity for display or analysis; image display settings do not automatically change stored cell measurements |
| Feature | A numerical measurement used as a classifier input |
| Proposed label | A reversible classifier annotation awaiting confirmation |
| Confirmed label | A reviewed example used for classifier training and final decisions |
| Prediction | A model's suggested class, with associated probabilities |
| Entropy / uncertainty | How spread out class probabilities are; higher values indicate less certainty |
| Stale / out of date | A saved result no longer matches its inputs and needs refreshing |
| Live AnnData | The cell data currently in memory; edits may not yet be saved to a file |
| Provenance | A record of the inputs, settings, and actions that produced a result |
