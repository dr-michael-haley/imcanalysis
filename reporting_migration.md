You are working on my public `imcanalysis` repository. You already have access to a project skill that explains the repository’s purpose, current layout, Pydantic configuration system, `sbt` CLI/control layer, stage registry, SLURM launch model, Conda environments, scientific scripts, ReadTheDocs setup, project-folder conventions, and existing provenance mechanisms.

Your task is to perform a repository-wide standardisation of pipeline outputs, stage reporting, and provenance.

This is intentionally a substantial refactor. Do not limit the implementation to one exemplar stage. Migrate all currently supported pipeline stages in the same development effort.

The scientific behaviour of the pipeline should remain unchanged wherever possible. The primary goals are to make project folders navigable, distinguish reusable computational assets from human-facing analysis outputs, and generate detailed stage-level reports that are useful to experts, non-experts, future agents, and open-science workflows.

# Core objectives

Implement a consistent system in which:

1. Reusable pipeline assets remain in clearly named folders in the project root.
2. Human-facing figures, plots, summaries and result tables are written beneath a standard `outputs/` directory.
3. Each pipeline stage has a fixed, numbered, human-readable output folder.
4. Each execution of a stage creates a clearly linked human-facing report.
5. Technical run records remain under `.sbt/runs/`.
6. Human-facing output reports link back to their corresponding technical run records and SLURM jobs.
7. Each stage has one reusable Markdown explainer that can be used by:

   * ReadTheDocs
   * `sbt stages explain`
   * generated stage output reports
8. Scientific scripts emit structured stage-level provenance through a common reporting API.
9. The overloaded legacy `QC/` folder is retired as a general dumping ground.
10. Existing projects and legacy outputs are handled conservatively and documented clearly.

# Intended project structure

Reusable computational assets should remain in the project root for now.

Do not introduce an `assets/` directory.

A target project may look like:

```text
project_root/
  config.yaml

  IMC_files/
  metadata/

  tiff_stacks/
  tiffs/
  processed/
  masks/
  cell_tables/
  anndata.h5ad

  outputs/
    001_Preprocessing/
    002_Denoising/
    003_Segmentation/
    004_Quantification/
    005_Batch_Integration/
    006_Clustering/
    007_Spatial_Neighbourhoods/
    ...

  .sbt/
    project.yaml
    runs/
      <run_id>/
        run_manifest.yaml
        run_plan.yaml
        config.user.yaml
        config.resolved.yaml
        submitted_jobs.yaml
        status.yaml
        logs/
        stage_events/
```

The exact root-level asset names should continue to come from the existing Pydantic config wherever applicable.

Examples include:

```text
imc_files_folder
metadata_folder
tiff_stacks_folder
raw_images_folder
denoised_images_folder
masks_folder
celltable_folder
anndata_path
```

Do not create a duplicate path-definition system.

# Fundamental distinction

Apply this rule consistently:

## Reusable project assets

Files primarily consumed by later computational stages.

Examples:

* raw or extracted image stacks
* denoised images
* segmentation masks
* cell tables
* AnnData files
* cached graph or model data
* intermediate datasets required by later stages

These remain in their configured project-root locations.

## Human-facing outputs

Files primarily inspected or interpreted by a person.

Examples:

* QC plots
* diagnostic montages
* UMAPs
* summary figures
* result plots
* statistical tables
* overview CSV files
* interpretation summaries
* stage README reports

These should move beneath:

```text
outputs/<numbered_stage_folder>/
```

Do not continue using `QC/` as a catch-all destination.

Some outputs may genuinely be QC. They should still live inside the relevant numbered stage output folder rather than a universal `QC/` dumping ground.

For example:

```text
outputs/003_Segmentation/<run_id>/figures/segmentation_qc.png
```

rather than:

```text
QC/segmentation_qc.png
```

# Fixed numbered stage folders

Extend the central stage registry so each stage has stable display and reporting metadata.

For example:

```python
StageSpec(
    name="cellpose",
    display_name="Segmentation",
    display_order=3,
    output_folder="003_Segmentation",
    documentation_path="docs/stages/segmentation.md",
)
```

The numbering should describe logical pipeline order, not submission order.

Rerunning a stage should not create a new numbered stage.

Preserve the stage aliases already used by the pipeline.

Determine the complete stage numbering and display names from the current pipeline rather than inventing an arbitrary order.

It is acceptable to leave numerical gaps if that makes future insertion easier, but use one consistent convention across the repository.

# Human-facing stage output structure

Each numbered stage folder should act as a stable human-facing location.

Use a structure such as:

```text
outputs/
  003_Segmentation/
    README.md
    <run_id>/
      README.md
      stage_manifest.yaml
      figures/
      tables/
      summaries/
      files/
```

The top-level stage `README.md` should:

* explain the stage
* identify the latest recorded run
* provide a run-history index
* link to run-specific reports
* link to reusable assets produced by the stage
* link to the technical records under `.sbt/runs/`

Each run-specific folder should contain the human-facing outputs generated during that execution.

Do not duplicate large reusable assets such as masks, image stacks or AnnData objects into every stage-run folder.

Instead, record and link to their canonical project-root locations.

For example:

```yaml
produced_assets:
  - role: masks
    path: ../../../masks
  - role: anndata
    path: ../../../anndata.h5ad
```

Use project-relative links in Markdown where reliable.

Use absolute resolved paths in machine-readable manifests where appropriate.

# Relationship to technical run records

Keep the distinction clear:

```text
.sbt/runs/<run_id>/
```

is the technical operational record, containing:

* submitted commands
* resolved config
* run plan
* job IDs
* SLURM logs
* status
* backend details
* technical stage events

```text
outputs/<stage_folder>/<run_id>/
```

is the human-facing scientific record, containing:

* reusable explanation of the stage
* run summary
* relevant config
* generated figures and tables
* metrics
* warnings
* interpretation notes
* links to assets
* link to the technical run record

Both locations should share the same stable `run_id`.

Each stage report must identify:

* run ID
* project ID
* stage name
* SLURM job ID where applicable
* technical run-record path
* pipeline version
* Git commit where available

# Shared stage documentation

Create one Markdown explainer per logical pipeline stage.

Use a structure such as:

```text
docs/
  stages/
    preprocessing.md
    denoising.md
    segmentation.md
    quantification.md
    batch_integration.md
    clustering.md
    spatial_neighbourhoods.md
    ...
```

Create explainers for all currently supported user-facing pipeline stages.

Each document should contain stable, reusable scientific context:

```markdown
# Stage title

## What this stage does

## Why it is performed

## Main inputs

## Reusable assets produced

## Human-facing outputs produced

## Important configuration options

## How to interpret the results

## Common problems and limitations
```

Reuse these documents in:

1. ReadTheDocs
2. `sbt stages explain <stage>`
3. generated stage-level output reports

Avoid maintaining independent copies of the same explanatory text.

The stage registry should identify the explainer document for each stage.

When a stage report is generated, include a snapshot or rendered inclusion of the current explainer so that historical reports remain understandable even if the central documentation later changes.

# Stage reporting system

Create a central reporting package rather than allowing each script to write ad hoc Markdown.

A possible structure is:

```text
SpatialBiologyToolkit/
  reporting/
    __init__.py
    models.py
    reporter.py
    render.py
    paths.py
    events.py
    inventory.py
```

Use typed Pydantic models for structured reporting records.

Create a common `StageReporter` or equivalent API.

The intended usage should be simple enough to add to every scientific stage:

```python
from SpatialBiologyToolkit.reporting import StageReporter

with StageReporter.from_environment(stage="cellpose") as report:
    report.add_input(
        role="denoised_images",
        path=denoised_images_path,
        description="Denoised IMC image stacks used for segmentation.",
    )

    result = run_segmentation(...)

    report.add_asset(
        role="masks",
        path=masks_path,
        description="Reusable per-cell segmentation masks.",
    )

    report.add_file(
        category="figure",
        path=overview_figure,
        description="Overview of segmentation results across ROIs.",
    )

    report.add_metric("rois_processed", number_of_rois)
    report.add_metric("total_cells", total_cells)
```

The exact API can differ, but keep it:

* lightweight
* consistent
* typed
* easy for all scripts to use
* safe during exceptions
* independent of heavy analysis dependencies

# Context-manager behaviour

Prefer a context-manager or similarly robust lifecycle.

It should:

1. Record stage start.
2. Capture relevant SBT environment variables.
3. Create output/report directories.
4. Record inputs.
5. Allow outputs, assets, files, metrics and warnings to be added.
6. Record successful completion.
7. Record failure details on exception.
8. Re-raise exceptions so SLURM still reports failure correctly.
9. Write machine-readable records even if Markdown rendering fails.
10. Update the relevant stage index where safe.

Do not swallow scientific exceptions.

# Structured stage manifest

Each stage execution should write a structured file such as:

```text
outputs/<stage_folder>/<run_id>/stage_manifest.yaml
```

It should contain a versioned schema:

```yaml
schema_version: 1
project_id: ...
run_id: ...
stage: cellpose
display_name: Segmentation
status: completed

started_at: ...
completed_at: ...
duration_seconds: ...

pipeline_version: ...
git_commit: ...
slurm_job_id: ...

technical_run_record: ...

inputs: []
produced_assets: []
generated_files: []
parameters: {}
metrics: {}
warnings: []
errors: []
```

Use typed models and central serialization.

Do not scatter arbitrary dictionaries throughout scripts.

# Stage README generation

Generate:

```text
outputs/<stage_folder>/<run_id>/README.md
```

from:

1. the shared stage explainer
2. the structured stage manifest
3. relevant Pydantic config metadata
4. metrics and files reported by the scientific script

A generated README should include:

```markdown
# Segmentation Report

## Stage overview

[shared stage explanation]

## This execution

- Project
- Run ID
- Date
- Status
- SLURM job ID
- Pipeline version
- Git commit

## Why this stage was run

Use an explicit run purpose/rationale if one is available.
Otherwise provide a neutral automatic description.

## Inputs

## Reusable assets produced

## Figures and tables

## Important configuration

## Metrics and findings

## Warnings and limitations

## Technical record

## How to interpret these outputs
```

Do not include every single config option by default.

Use the existing Pydantic field metadata to choose relevant stage-specific settings.

Where possible, include:

* explicitly overridden values
* commonly tuned/basic parameters
* parameters associated with the current stage
* values materially affecting the output

# Stage-level parameter reporting

Use the Pydantic config metadata already present in the repository, including:

```text
description
level
stage
ui_group
advice
```

Build reusable helpers that can extract the settings relevant to a stage.

The stage report should provide a concise parameter summary and, where useful, short descriptions.

The full resolved config remains available in the technical run directory and should be linked rather than repeated in full.

# Dynamic scientific summaries

For this refactor, add useful script-specific reporting wherever data is already available without introducing expensive duplicate computation.

Examples:

* number of ROIs processed
* number of files generated
* number of cells identified
* numbers of masks
* cluster counts
* failed or skipped ROIs
* missing markers
* group counts
* warnings already detected by the script
* important result table paths

Do not invent scientific interpretations.

Record objective metrics and summaries that the script can reliably determine.

Create extension points so later agentic or user-authored interpretation can be appended.

# Run rationale and notes

Provide a mechanism for `sbt run` to accept an optional human-readable purpose, for example:

```bash
sbt run segmentation --reason "Repeat segmentation with a larger Cellpose diameter after fragmentation was observed."
```

Store this in:

* the technical run manifest
* stage reports
* generated README files

Also support a project/run notes mechanism that can later be used by a human or agent.

Do not require a reason for every run.

# Migrate all scientific scripts

Inventory all pipeline scripts currently exposed as stages or invoked by the main pipeline.

For every stage:

1. Identify reusable root-level assets it reads.
2. Identify reusable root-level assets it produces.
3. Identify human-facing figures and tables it produces.
4. Redirect human-facing files into the correct numbered output folder.
5. Stop writing general outputs into `QC/`.
6. Integrate the common stage reporter.
7. Register inputs, assets, output files, metrics and warnings.
8. Generate a stage manifest.
9. Generate a run-specific README.
10. Update the stage-level README/index.
11. Preserve scientific computation and existing defaults.
12. Add or update tests.

This task should cover all currently supported pipeline stages, not just one.

# Legacy `QC/` migration

Inventory every current write to the configured `QC/` folder.

Classify each item as:

```text
reusable asset
human-facing figure
human-facing table
technical log
temporary/debug file
```

Then migrate it appropriately.

Examples:

* reusable data required downstream → configured project-root asset path
* plot or diagnostic image → relevant `outputs/<stage>/<run_id>/figures/`
* summary CSV → relevant `outputs/<stage>/<run_id>/tables/`
* SLURM logs → `.sbt/runs/<run_id>/logs/`
* temporary debug files → a documented temporary path or remove if unnecessary

Do not simply rename `QC/` to `outputs/`.

Make a deliberate stage-aware mapping.

After migration:

* new runs should not use `QC/` as a generic dumping ground
* retain a compatibility/deprecation path for existing projects if needed
* document the old layout and new layout
* do not delete existing user data

Consider retaining the Pydantic `qc_folder` field temporarily for backward compatibility while clearly deprecating it.

Do not silently reinterpret it in a way that breaks existing configs.

# SLURM output and error logs

Standardise SLURM logs so they are linked to technical run records.

Prefer:

```text
.sbt/runs/<run_id>/logs/<stage>_<job_id>.out
.sbt/runs/<run_id>/logs/<stage>_<job_id>.err
```

Update SLURM submission or scripts so logs use the current run directory when `SBT_RUN_DIR` is available.

Maintain a sensible fallback for direct legacy invocation outside `sbt`.

Ensure `sbt logs` continues to find logs through recorded paths rather than expensive directory searches.

Human-facing stage reports should link to the technical logs but should not copy large logs into `outputs/`.

# Root-level reusable assets

Keep reusable assets in the project root.

Do not create:

```text
assets/
```

Continue using readable canonical folders such as:

```text
tiff_stacks/
tiffs/
processed/
masks/
cell_tables/
anndata.h5ad
```

Where current names are unclear, standardise cautiously through the Pydantic config and migration support.

Do not move raw MCD files or existing project data destructively.

For existing projects, prefer:

* adoption
* validation
* warnings
* optional migration commands
* compatibility aliases

over automatic moves.

# Project validation updates

Extend `sbt project validate` and `sbt project assets` for the new layout.

They should understand:

* root-level reusable asset roles
* numbered output stage folders
* technical `.sbt` run records
* deprecated legacy folders
* stage output/report completeness

Useful validation checks include:

* output folder names match the stage registry
* output-stage run folders contain a stage manifest
* README files exist where expected
* recorded asset paths resolve
* technical run links resolve
* stage output folders do not masquerade as reusable inputs
* no active script still targets the generic `QC/` dumping ground

Provide warnings rather than destructive automatic correction.

# Stage output indexes

For each numbered stage folder, maintain:

```text
outputs/<stage_folder>/README.md
```

This should include:

* shared stage explanation
* current/latest run
* list of recorded executions
* status of each execution
* links to run-specific reports
* links to canonical reusable assets
* link to ReadTheDocs
* link to the stage registry name/CLI command

Generate this from structured manifests rather than appending brittle free text.

Ensure updates are atomic where possible.

# Project-level output index

Generate:

```text
outputs/README.md
```

This should provide a non-expert overview of the complete project analysis.

Include:

* project title/ID
* brief description
* numbered pipeline stages
* which stages have been run
* latest run/status for each stage
* links to stage folders
* links to important reusable project assets
* explanation of the distinction between project assets, outputs and technical records

This should be usable as the starting point when handing over or publishing a project folder.

# ReadTheDocs integration

Update ReadTheDocs so the stage explainer files are included in the documentation navigation.

Document:

* standard project structure
* root-level reusable assets
* numbered human-facing outputs
* technical `.sbt` run records
* generated stage reports
* stage manifests
* how reruns are represented
* how to interpret links between outputs and run records
* legacy `QC/` migration
* how scientific scripts should use `StageReporter`

Avoid duplicating stage explanations manually across docs.

# Configuration changes

Update the Pydantic config conservatively.

Potential changes may include:

* adding `outputs_folder`, defaulting to `"outputs"`
* deprecating `qc_folder` as a general-purpose destination
* adding any minimal reporting/output settings genuinely required
* preserving legacy config compatibility
* documenting migration behaviour

Do not move folder-order or display metadata into user config if it belongs in the stage registry.

The numbered output folder associated with a stage should normally be defined centrally in code, not repeatedly configured per project.

# Backward compatibility

This is a major refactor, but existing projects must not be destroyed.

Implement appropriate compatibility measures:

* existing configs should still load
* old asset paths should be recognised
* old `QC/` outputs should be reported as legacy
* direct script execution should still work where practical
* reports should degrade gracefully when no SBT run environment is present
* scientific scripts should use sensible fallback output paths if invoked outside `sbt`
* provide clear warnings about legacy behaviour
* do not delete or automatically move existing files
* add an explicit migration command only if it can be made safe and non-destructive

If adding a migration command, prefer dry-run by default.

# Direct script execution

The pipeline scripts may sometimes be invoked directly rather than through `sbt`.

The reporting system should therefore support two contexts:

## Managed SBT run

Environment variables such as:

```text
SBT_PROJECT_ROOT
SBT_PROJECT_ID
SBT_RUN_ID
SBT_RUN_DIR
SBT_STAGE
SBT_CONFIG
SBT_SLURM_JOB_ID
```

are available.

Use the full run-aware reporting system.

## Legacy/direct execution

No SBT run context is available.

The script should:

* still run scientifically
* create a minimal reporting context where practical
* use a clear fallback output path
* emit a warning that full run provenance is unavailable
* not crash purely because reporting context is absent

# Reporting failures

Reporting should not cause successful scientific computation to be lost unnecessarily.

However, reporting errors must not be silently ignored.

Use a clear policy:

* machine-readable stage manifest writing is high priority
* Markdown rendering failures should be recorded and reported
* failure to update a stage index should not corrupt a completed stage manifest
* scientific exceptions must still fail the job
* reporting exceptions should be handled centrally and surfaced clearly
* use atomic writes for manifests and README files where practical

# Testing

Add comprehensive tests across the full refactor.

At minimum test:

## Stage registry and paths

* every registered user-facing stage has:

  * display name
  * display order
  * numbered output folder
  * documentation source
* stage output folder names are unique
* stage order is deterministic
* documentation paths exist

## Reporting models

* stage manifests validate
* schema versions are present
* manifests serialize to YAML/JSON
* success and failure states work
* unknown optional metadata does not break historical records where compatibility is intended

## Reporter lifecycle

* start and completion events are recorded
* exceptions create failed manifests and are re-raised
* inputs, assets, files, metrics and warnings are captured
* direct-execution fallback works
* managed SBT environment works

## Markdown generation

* run README is generated
* static explainer is included
* dynamic fields are rendered
* stage index is generated
* project output index is generated
* links to technical records are correct

## Output routing

* reusable assets remain in configured root-level paths
* figures/tables go to stage-specific run output folders
* new code does not use `QC/` as a generic destination
* SLURM logs resolve to the technical run directory
* reruns create separate run-specific human-facing records

## Script integrations

For every migrated stage, add at least a focused test or mocked integration test showing:

* output paths are resolved correctly
* StageReporter is invoked
* generated files are registered
* reusable assets are registered
* manifest/report generation completes

Do not require full real-world datasets for every test.

Use representative fixtures and mocks.

## Compatibility

* old configs load
* direct script invocation remains possible
* missing SBT environment does not crash scripts
* legacy folders are identified
* no tests submit real SLURM jobs

# Verification and migration audit

Before changing code, perform an audit of:

* all writes to `QC/`
* all hard-coded output paths
* all root-level generated folders
* all SLURM output/error directives
* all scripts exposed through the stage registry
* all current AnnData provenance/logging calls
* all existing figure/table export utilities
* all existing Markdown or report generation

Create a migration map before making broad edits.

For example:

```text
Current path or behaviour
  → classification
  → target path
  → responsible stage
  → compatibility action
```

Use this map to ensure no major output path is missed.

# Existing AnnData logging

The repository currently records some stage/task metadata directly in AnnData.

Do not simply delete this without understanding its purpose.

During this refactor:

1. Inventory the existing AnnData logging.
2. Preserve it where needed for compatibility.
3. Avoid adding further duplicated provenance logic.
4. Route new canonical reporting through the central reporting package.
5. Clearly document which AnnData provenance remains.
6. Identify a future migration path for consolidating or replacing it.

It is acceptable to mark existing AnnData logging as legacy or transitional.

Do not perform an unsafe wholesale removal unless the same information is demonstrably captured elsewhere and tests verify compatibility.

# Non-goals

Do not:

* implement the private LLM agent
* add agent-specific prompts or credentials
* migrate execution to Nextflow
* add a Nextflow backend
* create an `assets/` folder
* migrate AnnData to SpatialData
* change scientific algorithms
* change biological defaults without clear necessity
* merge Conda environments
* containerise all environments
* delete existing user outputs
* automatically move large legacy project files
* calculate expensive checksums on login nodes by default
* duplicate large assets into stage-run output folders
* invent scientific conclusions in reports

# Development strategy

Although this is one large repository-wide refactor, keep the implementation internally structured.

Proceed in this order:

1. Audit existing paths, outputs, QC writes, stage scripts and logging.
2. Produce a migration map.
3. Finalise the numbered stage-folder scheme.
4. Extend the stage registry.
5. Create shared stage explainer documents.
6. Implement typed reporting models.
7. Implement `StageReporter`.
8. Implement Markdown rendering and index generation.
9. Update SBT project/run models.
10. Update SLURM log routing.
11. Migrate every scientific pipeline stage.
12. Update Pydantic config compatibility.
13. Update project validation.
14. Update ReadTheDocs.
15. Add and update tests.
16. Run the repository test suite, linting and type checks.
17. Search the repository again for:

    * writes to `QC/`
    * hard-coded output paths
    * scripts missing reporting integration
    * outdated documentation
18. Produce a final migration report.

# Required final report

At completion, report clearly:

* the final project folder convention
* the complete numbered stage-folder mapping
* which root-level folders are reusable assets
* which legacy output locations remain supported
* how each stage now reports
* where SLURM logs live
* how stage reports link to technical runs
* which existing AnnData logging remains
* any scripts that could not be fully migrated
* any compatibility assumptions
* any manual migration recommended for existing projects
* test, lint and type-check results
* recommended future work for:

  * richer scientific summaries
  * user/agent interpretation notes
  * output versioning
  * SpatialData
  * Nextflow

The outcome should be a coherent, navigable and documented project structure in which a non-expert can begin at:

```text
outputs/README.md
```

and understand:

* what stages were run
* what each stage does
* when it was run
* what settings mattered
* what files were generated
* where the reusable data assets are
* where the technical run record and SLURM logs are
* how the results should be interpreted
