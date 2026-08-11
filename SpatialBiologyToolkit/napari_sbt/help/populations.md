# Population Curation

Use this tab to turn an original categorical observation such as `leiden` into
one or more named, reviewable observations. The original observation is never
edited. Each new observation is a sibling **draft** tied to the same frozen
source labels and cell identities.

## 1. Source workspace and drafts

Choose the original observation before creating the first draft. NapariSBT
records a fingerprint of every cell identity and source value. If either later
changes, the existing workspace will refuse to apply silently; start an explicit
new workspace or dataset revision instead.

The draft display name is for people. **New adata.obs name** is the exact column
that will be created in the live working AnnData. Several sibling drafts can be
kept at once, but all drafts inside one workspace derive from the same original
observation.

## Base naming and explicit merges

Every original source value has one row and initially maps to itself. Edit
**Proposed name** to assign a biological name. Select several rows and use
**Give selected rows one name / merge** to make the proposed merge explicit.
Rows participating in a shared final name are highlighted orange and listed in
the Preview tab with all contributing source clusters.

Colours are stored with the draft and written to Scanpy's
`adata.uns["<obs>_colors"]` convention when the draft is applied. Select all
contributors to a merge before setting their common colour if you want to avoid
a colour-conflict warning.

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
Rename the imported components before applying the draft.

**Use current classifier assignments** performs the same bridge without a file:
confirmed labels override model predictions, unassigned cohort cells are ignored,
and class IDs are converted to their configured class names. The imported groups
remain proposals in this population draft until you review, rename, and apply it.

## Preview, apply, and QC

### Applying a draft

The preview always shows effective cell counts, split coverage, and each explicit
merge. **Apply draft** adds or refreshes only the derived observation in the live
working AnnData and refreshes Explore and Classify selectors. Existing columns
are protected unless the overwrite checkbox is deliberately enabled.

Apply only after reviewing the base mapping, split components, and explicit merge
list. Use the overwrite checkbox only when intentionally revising the same derived
observation. Export creates a new AnnData file and refuses to replace an existing
file.

### Live QC plots

**Show in Explore** renders the same derived observation over the current tissue
ROI and enables the full-dataset overlay scope, so broad populations outside a
classification cohort remain visible alongside the split population. The
embedding pop-up uses an existing two-dimensional `obsm`; the heat-map
pop-up shows population means for selected `adata.X` markers and z-scores each
marker across populations. These plots are quick QC views, not publication
outputs, and they never recompute the embedding or clustering.

Choose an existing two-dimensional embedding for the scatter plot. Select a small,
biologically informative marker set for the heat map; regenerate these quick views
after naming, merging, or subclustering changes to check that the interpretation
still makes sense in expression and tissue space.

## Provenance

The append-only JSONL log records workspace and draft creation, every saved
naming/colour/note change, current explicit merge groups, CSV imports, subcluster-run
requests and results, cancellations/failures, component replacement, application
to the live AnnData, mapping exports, and curated AnnData exports. Worker run
folders also retain their exact request, assignment table, and preprocessing
safety declaration.
