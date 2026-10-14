# NapariSBT: the no-code guide

NapariSBT lets you inspect spatial-biology images, review cell populations, label
cells, and prepare results using buttons, menus, and the image viewer. This guide
is for scientists working with an already processed dataset. You do not need to
write Python or use the command line for the workflows on these pages.

Start with **Your first session**. Once you can open tissue images, follow the
page for your biological question. Classification and feature refinement are
later topics; they are optional.

| What would you like to do? | Start here |
| --- | --- |
| See staining and cell outlines | [Explore](explore.md) |
| Check whether an existing population label fits the staining | [Population QC](population_qc.md) |
| Turn numbered clusters into names, merge or split populations | [Population naming](populations.md) |
| Collect a list of particular cells by hand | [Labeler](labeler.md) |
| Compare marker expression, samples, or existing embeddings | [Scanpy plotting](scanpy_plotting.md) |
| Teach a classifier from reviewed examples | [Classification walkthrough](classification_workflow.md) |
| Save a tissue image for a figure | [Publication images](publication_export.md) |
| Save or deliberately change the underlying dataset | [Dataset Maintenance](dataset_maintenance.md) |

## Start here

```{toctree}
:maxdepth: 1

first_session
viewer_basics
workspace
saving
```

## Everyday image and population review

```{toctree}
:maxdepth: 1

explore
population_qc
populations
labeler
scanpy_plotting
publication_export
regions_export
```

## Advanced workflows

```{toctree}
:maxdepth: 1

classification_workflow
feature_building
classify
feature_refinement
dataset_maintenance
layers_status
```

## Help and technical reference

```{toctree}
:maxdepth: 1

troubleshooting
glossary
installation
Technical launch and storage reference <../guides/napari_sbt>
CSF3 interactive sessions <../guides/napari_sbt_csf3>
Moving from the older viewers <../guides/napari_sbt_migration>
```

The coloured headings on the workflow pages match the corresponding boxes in the
interface. Colours identify sections; the words **Ready**, **Needs review**, and
**Failed** describe status. The same detailed instructions appear when you press
**Help** inside NapariSBT. Screenshots use small synthetic demonstration data;
the example cells and labels are not biological results.
