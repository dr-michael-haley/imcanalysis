# Population naming

Use this tab to turn an original categorical observation such as `leiden` into
one or more named, reviewable observations. The original observation is never
edited. The main loop is: edit names, **Save and update Explore / Population
QC**, review them in tissue, then return here to refine them.

## 1. Source workspace and drafts

Choose the original observation, enter the single **New label column
(adata.obs)** name, and create the first draft. That is the only new name needed:
the technical draft display-name field has been removed. NapariSBT still records
the original cell/label fingerprint internally so it cannot silently remap a
different dataset. Before the first draft, changing the source observation updates
the automatic suggestion to `<selected source>_named`; an explicitly typed name is
preserved while the same source remains selected.

Several saved label columns can derive from the same original observation.
Changing **Saved naming work** loads it immediately; there is no separate Load
button. The readiness indicator distinguishes unsaved edits, saved work which has
not yet been synchronized, a conflicting obs name, and a saved revision already
available throughout the app.

## Base naming and explicit merges

Every original source value has one row and initially maps to itself. Edit
**Proposed name** to assign a biological name. Select several rows and use
**Give selected rows one name / merge** to make the proposed merge explicit.
The background of each Proposed name cell uses its assigned colour, with readable
light or dark text, so names and colours can be checked together. Merges are also
listed in the Preview tab with all contributing source clusters. NapariSBT
automatically gives every contributor to one merge the same colour. Separate
merge groups are kept visually distinct; the first available contributor colour
is retained where possible and propagated across base and subcluster rows.
Editing the colour for one final label likewise updates every row carrying that
proposed label.

**Automatically colour…** opens the shared Colour Helper. Choose a Scanpy or
Matplotlib categorical palette, untick colours that should not be used, and
assign the remaining colours by abundance (largest or smallest first) or name
(A–Z or Z–A). The preview shows the exact category-to-colour assignment before it
is applied. All rows carrying the same proposed name receive one colour.

Sharing a colour is valid only when rows also share the same proposed name and
therefore form an explicit merge. Reusing one colour for different final names is
shown as a collision in the Preview and blocks saving until corrected. This
prevents distinct tissue overlays from becoming visually indistinguishable.

Colours are stored with the draft and written to Scanpy's
`adata.uns["<obs>_colors"]` convention when the draft is applied. The effective
merge colour is shown beside its name in the Preview tab and is saved with the
mapping.

**Import preliminary names from CSV** accepts common layouts. It looks for the
selected source-observation name (or `source_value`, `cluster`, or `leiden`) and
a label column such as the derived observation name, `proposed_label`,
`final_population`, `label`, `name`, or `population`. Repeated source rows are
rejected; repeating a proposed name is a valid explicit merge.

Leading and trailing whitespace in source population values is ignored throughout
draft editing, Scanpy subclustering, and label synthesis. If two distinct source
values would become identical after trimming (for example `A` and `A `), NapariSBT
stops with an explicit message instead of silently merging them.

## Subclusters

Subclusters are represented separately because one source population can now
produce several components. Each component retains its parent source value,
method, run ID, raw component label, membership count, proposed name, colour,
and notes. Component membership overrides the base mapping only for its member
cells.

The monitored Scanpy action defaults to `adata.obsm["X_biobatchnet"]`. After
isolating the selected cells, it rebuilds their neighbour graph with
`scanpy.pp.neighbors(use_rep="X_biobatchnet", n_neighbors=15)` and then runs
Leiden. This allows cells to form new nearest-neighbour relationships within the
broad population while continuing to use the batch-corrected representation.
Normalization, scaling, PCA, BioBatchNet itself, and UMAP are **not** rerun. For
populations containing 15 or fewer cells, the worker reduces the effective
neighbour count to `n_cells - 1` and records it in the assignment table.
Subsets with fewer than three cells are rejected as too small for meaningful
neighbour rebuilding.

The corrected representation and neighbour count are selectable. Reusing an
existing `adata.obsp` connectivity graph remains an explicit alternative when
the conservative induced-subgraph behaviour is desired. The default clusters
each selected source population separately; together mode can find communities
spanning the selected source populations.

For a live notebook AnnData with no file source, starting the worker first writes
one fingerprinted, workspace-owned H5AD snapshot so a separate Python process
can read it. This snapshot is not written back to the user's original object.

**Import image/other cell-level assignments** is the bridge from NapariSBT's
image classifier or another pipeline. The table must contain `obs_name` (or a
supported cell-identity alias) and a class/label column. New assignments replace
older split membership only for overlapping cells; unrelated components remain.
Rename the imported components before saving the draft.

**Use current classifier assignments** performs the same bridge without a file:
confirmed labels override model predictions, unassigned cohort cells are ignored,
and class IDs are converted to their configured class names. The imported groups
remain proposals in this population draft until you review, rename, and save it.

## Preview and QC

### Applying a draft

The preview always shows effective cell counts, split coverage, and each explicit
merge. **Save and update Explore / Population QC** now performs both actions: it
saves the draft, adds or refreshes its derived observation in the live working
AnnData, refreshes the app's observation selectors, and selects the corresponding
new population name in Population QC where possible. The current Population QC
RGB view is carried across a rename when there is no existing recipe for the new
name.

Revising a label column previously created by the same draft is safe and does not
need the overwrite checkbox. The advanced overwrite checkbox is only for
deliberately replacing an unrelated existing observation. Export creates a new
AnnData file and refuses to replace an existing file.

### Scanpy plotting handoff

**Open current labels in Explore** renders the same derived observation over the
current tissue ROI and enables the full-dataset overlay scope, so broad populations
outside a classification cohort remain visible alongside split populations.

**Open these labels in Scanpy plotting** moves expression, embedding, composition,
and old-versus-new label QC into the dedicated Scanpy plotting workspace. If the
current mapping is unsaved or has not reached the live AnnData object, NapariSBT
offers to save and synchronize it first. The plotting tab then opens with this
derived observation already selected. Existing plot windows remain as snapshots
and are marked out of date when labels change, making before-and-after comparison
possible without silently redrawing results.

## History and provenance

**View history…** opens a short human-readable list of recent actions. The full
append-only JSONL audit remains available on disk for reproducibility, but it is
no longer a permanent tab in the main naming interface. Worker run folders still
retain their exact request, assignment table, and preprocessing safety declaration.
