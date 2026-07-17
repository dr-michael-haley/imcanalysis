You are working on my public `imcanalysis` repository. You have access to the repository skill and should already understand the current `sbt` CLI, project model, stage registry, SLURM launcher, run records, reporting framework, numbered stage folders, Pydantic configuration, scientific scripts, and ReadTheDocs documentation.

I want you to perform a clean redesign of how pipeline executions are numbered, stored, displayed and removed.

This should be treated as a coherent architectural refactor rather than a small patch to the existing fixed-number stage registry.

# Current behaviour

At present, numbered output folders are permanently associated with pipeline stages through the central stage registry.

For example:

```text
prep       → 001_Preprocessing
starling   → 011_STARLING_Phenotyping
```

The number represents a fixed catalogue position rather than the order in which stages were executed in an individual project.

Human-facing outputs are currently organised approximately as:

```text
outputs/
  001_Preprocessing/
    <long_run_id>/
      README.md
      stage_manifest.yaml
      figures/
      tables/
```

The technical run ID is also exposed prominently in the human-facing output structure.

This has several disadvantages:

* the output numbering does not show the actual analysis history of a project
* a stage may be numbered `011` even when it is the first analysis performed
* the user cannot see the executed workflow order at a glance
* the additional run-ID subfolder makes paths unnecessarily deep
* the existing run IDs are long and difficult for people to read or refer to
* stable technical provenance and human-facing navigation are being conflated

# Desired behaviour

The number at the beginning of each output folder should represent the order in which that stage execution was added to the project workflow.

For example:

```text
outputs/
  001_Preprocessing/
  002_Denoising/
  003_Segmentation/
  004_Quantification/
  005_Cell_Clustering/
  006_STARLING_Phenotyping/
```

If STARLING is the first stage executed in a project, it should be:

```text
outputs/
  001_STARLING_Phenotyping/
```

If segmentation is executed again later, it should receive a new execution number:

```text
outputs/
  001_Preprocessing/
  002_Segmentation/
  003_Quantification/
  004_Segmentation/
```

The number therefore identifies a particular stage execution, not a permanent stage type.

The human-facing output folder should itself be the execution folder. Remove the additional human-facing run-ID subdirectory.

The target structure is:

```text
outputs/
  README.md

  001_Preprocessing/
    README.md
    stage_manifest.yaml
    figures/
    tables/
    summaries/
    files/

  002_Segmentation/
    README.md
    stage_manifest.yaml
    figures/
    tables/
    summaries/
    files/
```

There should not be:

```text
outputs/
  001_Preprocessing/
    <long_run_id>/
```

# Important identity model

Do not solve this by discarding stable technical identities.

Introduce a clear distinction between:

## Human-facing execution ID

A short sequential integer within the project:

```text
001
002
003
```

This is the ID users normally see and use.

It determines the output folder prefix.

It may be referred to through the CLI as the public `run_id` or execution ID, for example:

```bash
sbt status 003
sbt logs 003
sbt remove 003
```

## Internal technical run ID

A permanent, globally unique or sufficiently unique identifier used internally for provenance.

This may be a UUID, timestamp-based ID or the existing technical run identifier.

It must remain available inside:

```text
.sbt/
```

and in machine-readable manifests.

It should not be used as the primary human-facing folder name.

## SLURM job ID

The SLURM job ID is a separate execution-backend identifier.

Do not use the SLURM job ID as either the human-facing execution ID or the permanent project run identity.

The data model must keep these concepts separate:

```yaml
execution_id: 3
execution_label: "003"
technical_run_id: "..."
slurm_job_id: "..."
stage: "cellpose"
```

Use clear terminology throughout the implementation.

Avoid continuing to use the ambiguous name `run_id` internally for several different concepts.

A reasonable internal model might use:

```text
execution_id
technical_run_id
workflow_run_id
slurm_job_id
```

The CLI may continue to present the short execution ID as the normal user-facing run reference.

# Consequences to evaluate before implementation

Before changing code, audit the current implementation and document the consequences of moving from fixed stage catalogue numbers to project-specific execution numbers.

Evaluate at least:

* stage registry changes
* output path generation
* stage report generation
* project output index generation
* technical run manifests
* SLURM job submission
* multi-stage workflow submission
* status and log commands
* `latest` resolution
* project validation
* ReadTheDocs links
* existing project migration
* existing fixed-number folders
* reruns of the same stage
* failed or dependency-blocked jobs
* concurrent submissions
* stage removal and renumbering
* links between output reports and `.sbt` records

Write a concise design note or migration plan before making broad edits.

Then proceed with the implementation without waiting for further confirmation unless an unavoidable ambiguity cannot be resolved from the repository.

# Stage registry redesign

Remove fixed human-facing output numbers from the stage registry.

The stage registry should continue to define stable stage metadata such as:

```python
StageSpec(
    name="cellpose",
    display_name="Segmentation",
    output_slug="Segmentation",
    documentation_path="docs/stages/segmentation.md",
    requires_assets=[...],
    produces_assets=[...],
)
```

It should not permanently define:

```python
output_folder="003_Segmentation"
display_order=3
```

unless a separate order is genuinely needed only for documentation navigation.

If ReadTheDocs needs a logical catalogue order, represent that separately from project execution numbering.

Do not allow documentation order to control output folder IDs.

Create one central helper for generating a human-facing execution folder name:

```text
{execution_id:03d}_{stage_output_slug}
```

Examples:

```text
001_Preprocessing
002_Cellpose_Segmentation
003_STARLING_Phenotyping
```

Use filesystem-safe, stable stage slugs.

# Project execution index

Do not determine the next execution number solely by scanning folder names.

Create a typed, project-level execution index under `.sbt`, for example:

```text
.sbt/executions.yaml
```

or an equivalently appropriate lightweight format.

It should contain the active human-facing workflow sequence and map execution IDs to technical records.

For example:

```yaml
schema_version: 1

executions:
  - execution_id: 1
    technical_run_id: "..."
    stage: prep
    output_folder: outputs/001_Preprocessing
    status: completed
    created_at: "..."

  - execution_id: 2
    technical_run_id: "..."
    stage: cellpose
    output_folder: outputs/002_Segmentation
    status: running
    created_at: "..."
```

Use typed models, schema versions and atomic writes.

The output folders and execution index must agree.

Project validation should detect inconsistencies.

# Safe execution-number allocation

Execution numbers must be allocated safely.

Consider:

* two `sbt run` commands being submitted close together
* a multi-stage workflow allocating several stage executions
* interrupted submission
* partial SLURM submission failure
* existing malformed folders
* removal occurring while another process is submitting

Implement a lightweight project lock around operations that allocate, remove or renumber execution IDs.

A lock location such as:

```text
.sbt/locks/executions.lock
```

would be appropriate.

Do not rely on “scan folders, take maximum plus one” without locking and structured state.

# Multi-stage workflows

When a command submits multiple stages:

```bash
sbt run prep denoise cellpose
```

allocate one sequential human-facing execution ID to each stage in the resolved execution order.

For example:

```text
001_Preprocessing
002_Denoising
003_Segmentation
```

These stage executions may share an internal workflow-level technical run ID, but each stage must have its own human-facing execution ID and stage manifest.

Record the relationships explicitly.

Jobs that are submitted but later fail, are cancelled or remain blocked by a failed dependency should retain their execution folders and report their actual status.

Do not falsely present an attempted stage as successfully completed.

# Output structure

Each stage execution should write directly into:

```text
outputs/<execution_id>_<stage_slug>/
```

For example:

```text
outputs/003_Segmentation/
```

That folder should contain:

```text
README.md
stage_manifest.yaml
figures/
tables/
summaries/
files/
```

Do not create another run-ID directory underneath it.

Reusable project assets should continue to live in their configured project-root locations, such as:

```text
masks/
processed/
cell_tables/
anndata.h5ad
```

The stage manifest and README should link to those assets rather than copying them.

Technical records should remain under `.sbt`.

# Manifest updates

Update stage and run manifest schemas so they record:

```yaml
schema_version: 2

execution_id: 3
execution_label: "003"
stage: cellpose
stage_display_name: Segmentation
output_folder: outputs/003_Segmentation

technical_run_id: "..."
workflow_run_id: "..."
slurm_job_id: "..."

status: completed
started_at: "..."
completed_at: "..."
```

Use schema migration or backwards-compatible loading where appropriate.

Historical technical IDs must remain immutable.

The human-facing execution ID may be changed during an explicit removal-and-renumbering operation, so it must not be the sole permanent identity used for audit records.

# CLI redesign

Update all relevant CLI commands to use the short project execution ID as the normal user-facing reference.

Examples:

```bash
sbt status 003
sbt logs 003
sbt report 003
sbt remove 003
```

Also support:

```bash
sbt status latest
sbt logs latest
```

Where useful for technical troubleshooting, allow an internal technical ID to be supplied through an explicit option, but do not make it the normal interface.

Commands should print both IDs only in verbose or technical output.

Default user-facing output should emphasise:

```text
Execution 003 — Segmentation
```

rather than a long internal identifier.

# Add `sbt summary`

Implement:

```bash
sbt summary
```

This should print a concise human-readable table showing the pipeline stages executed or attempted in the current project.

Suggested default columns:

```text
ID
Stage
Status
Started
Completed or Duration
Assets
Output
```

Example:

```text
ID    Stage                    Status      Started              Assets
001   Preprocessing            completed   2026-07-10 09:12     created
002   Denoising                completed   2026-07-10 12:43     modified
003   Segmentation             failed      2026-07-11 08:20     none
004   Segmentation             completed   2026-07-11 14:05     created
```

The exact presentation may use Rich tables.

Add useful filtering and output options.

At minimum consider:

```bash
sbt summary --stage segmentation
sbt summary --status failed
sbt summary --latest
sbt summary --details
sbt summary --assets
sbt summary --include-removed
sbt summary --format table
sbt summary --format yaml
sbt summary --format json
```

Default `sbt summary` should remain concise and useful to a novice.

Machine-readable formats should be stable and useful to a future agent.

The summary must be generated from structured execution records, not inferred from README text.

# Add `sbt remove`

Implement a safe command for removing a stage execution from the visible project workflow:

```bash
sbt remove 003
```

This operation needs careful provenance handling.

It must distinguish between:

1. executions that created only human-facing outputs
2. executions that created new reusable assets
3. executions that modified existing reusable assets
4. executions for which asset effects are unknown

Use the structured stage manifest and asset records to make this classification.

## First confirmation

Every removal should require confirmation unless a deliberate non-interactive option is supplied.

The first prompt should clearly identify:

* execution ID
* stage
* status
* output folder
* technical run ID
* whether reusable assets were created or modified

Example:

```text
Remove execution 003 — Segmentation?

This will remove its human-facing output folder and remove it from the active
project workflow summary.

Continue? [y/N]
```

## Second confirmation for asset-producing stages

When the stage created or modified reusable assets, or when the effect is unknown, show a second stronger warning.

For example:

```text
This execution created or modified reusable project assets.

Removing the execution record does not necessarily restore those assets to
their earlier state. Downstream analyses may depend on them, and the project
may become internally inconsistent.

Remove this execution from the visible workflow anyway? [y/N]
```

The command should still allow the removal after explicit confirmation.

For non-interactive or agentic use, require explicit flags that make the risk clear, for example:

```bash
sbt remove 003 --yes
sbt remove 003 --yes --accept-asset-risk
```

Do not allow a simple generic `--force` to bypass all asset warnings without making the risk explicit.

# Asset deletion and rollback semantics

Do not pretend that removing an execution automatically reverses scientific changes.

There is an important distinction between:

* removing the execution from the visible workflow
* deleting human-facing outputs
* deleting newly created reusable assets
* restoring reusable assets that were modified in place

The base `sbt remove` operation should:

1. remove the human-facing output folder
2. remove the execution from the active project execution index
3. preserve an internal audit record
4. not claim to restore modified assets

If an asset was modified in place and no snapshot exists, automatic rollback is impossible.

State this clearly.

Do not delete shared or modified project assets automatically.

An optional future or carefully implemented flag may delete assets that are demonstrably:

* exclusively created by this execution
* unmodified since creation
* not referenced by later executions

However, do not implement unsafe asset deletion merely to make removal appear complete.

The immediate requirement is safe logical removal with honest warnings.

# Hidden removal audit

Removed executions should disappear from the normal human-facing workflow.

After removal:

```bash
sbt summary
```

should not show the execution.

The stage output folder should be removed.

The project-level `outputs/README.md` should no longer list it.

However, removal itself must be preserved in a separate, non-user-facing audit record.

Use a location such as:

```text
.sbt/audit/removals/
```

or an append-only technical audit log.

Record at least:

```yaml
schema_version: 1
removed_at: "..."
removed_by: "..."
previous_execution_id: 3
technical_run_id: "..."
stage: cellpose
previous_output_folder: outputs/003_Segmentation
asset_effect: modified
confirmation_mode: interactive
reason: null
```

Allow an optional reason:

```bash
sbt remove 003 --reason "Failed test segmentation using incorrect diameter."
```

This removal audit must not appear in the normal project output report or normal `sbt summary`.

It may appear only through an explicit technical option such as:

```bash
sbt summary --include-removed
```

or a dedicated audit command.

Do not destroy the immutable technical evidence that the execution occurred.

# Renumbering after removal

From the ordinary user’s perspective, a removed execution should appear as though it was never part of the active workflow.

Therefore, after removing an execution, compact and renumber the remaining active execution folders by default.

For example, before removal:

```text
001_Preprocessing
002_Denoising
003_Failed_Segmentation
004_Segmentation
005_Quantification
```

After removing execution `003`:

```text
001_Preprocessing
002_Denoising
003_Segmentation
004_Quantification
```

The stable technical IDs ensure that provenance is not lost when human-facing sequence numbers change.

Implement renumbering carefully:

* acquire the project execution lock
* validate the current index and filesystem
* use temporary names to avoid rename collisions
* update the execution index
* update stage manifests
* update generated README files
* update project and stage links
* update references from active technical records
* preserve the old-to-new mapping in the removal audit
* use atomic writes where practical
* recover safely from interruption

Do not use execution folder names as permanent technical references.

Any technical reference that must survive renumbering should use the immutable technical run or stage-execution ID.

If external paths may be broken by renumbering, document this clearly.

# Project output index

Update:

```text
outputs/README.md
```

so it presents the active analysis as an ordered project workflow.

For example:

```markdown
# Project Analysis Summary

| ID | Stage | Status | Date | Report |
|---:|---|---|---|---|
| 001 | Preprocessing | Completed | 10 July 2026 | [Open](001_Preprocessing/) |
| 002 | Denoising | Completed | 10 July 2026 | [Open](002_Denoising/) |
| 003 | Segmentation | Completed | 11 July 2026 | [Open](003_Segmentation/) |
```

This should be generated from the active execution index.

Do not use the fixed stage catalogue order.

# Shared stage documentation

Keep the existing Markdown explainer for each stage.

For example:

```text
docs/stages/segmentation.md
```

These documents continue to explain the stage type and should still support:

* ReadTheDocs
* `sbt stages explain`
* generated stage reports

However, remove fixed output-folder numbers from stage documentation unless they are explicitly described as examples.

Documentation should explain that output numbers are assigned per project according to execution order.

Update the reporting documentation currently describing fixed numbered stage folders.

# Existing project migration

Existing projects may use the old layout:

```text
outputs/
  001_Preprocessing/
    <technical_run_id>/
  011_STARLING_Phenotyping/
    <technical_run_id>/
```

Do not silently migrate existing projects on ordinary CLI startup.

Add an explicit migration or project-upgrade command, for example:

```bash
sbt project migrate-execution-layout --dry-run
sbt project migrate-execution-layout
```

The dry run should show:

* old folders
* detected stage executions
* inferred chronological order
* new execution IDs
* new target paths
* manifest changes
* ambiguous or incomplete records
* anything that cannot be migrated safely

Determine chronology from structured manifest timestamps where possible.

Do not rely primarily on filesystem modification time or lexicographic folder order.

If multiple old run-ID subfolders exist beneath the same fixed stage folder, each should become its own sequential execution folder.

Example:

```text
outputs/
  003_Segmentation/
    run_A/
    run_B/
```

might become:

```text
outputs/
  002_Segmentation/
  005_Segmentation/
```

depending on their actual execution dates relative to other stages.

Migration should preserve technical IDs and create an audit mapping.

Do not delete legacy data until the new structure has been validated.

Prefer staged temporary paths and atomic rename operations.

Abort with a clear report when chronology or identity cannot be established reliably.

# Failed and incomplete executions

Decide and document how output folders are handled for:

* submission failure before any SLURM job is accepted
* submitted jobs that fail
* jobs cancelled by the user
* jobs blocked because an upstream dependency failed
* interrupted report generation
* partially submitted multi-stage workflows

The preferred behaviour is:

* if no job was ever accepted, retain technical submission-failure evidence under `.sbt` but avoid creating a normal completed-looking human-facing stage folder
* if a stage was accepted or attempted, retain its sequential execution folder with an accurate failed, cancelled or blocked status
* make failures visible in `sbt summary`
* do not imply that a folder means successful completion

Use the audit record for events excluded from the normal visible workflow.

# Status and logs

Update:

```bash
sbt status
sbt logs
```

so they can resolve a short execution ID through the execution index.

For example:

```bash
sbt status 003
sbt logs 003
```

The commands should locate the corresponding technical run record and SLURM job without scanning large directory trees.

Verbose modes may show:

* technical run ID
* workflow run ID
* SLURM job ID
* technical run-record path

Default output should remain human-readable.

# Reporting API integration

Update the reporting framework so that `StageReporter` receives or resolves:

```text
execution_id
technical_run_id
stage
output_folder
technical_run_record
slurm_job_id
```

The reporter should write directly into the sequential execution folder.

Do not let individual scientific scripts independently calculate execution numbers or output paths.

Allocation belongs to the `sbt` project/run control layer.

Scripts should receive the resolved path through the managed SBT execution environment.

For direct legacy script execution outside `sbt`, use a clearly documented fallback that does not corrupt the managed execution index.

# Environment variables

Review and update managed execution variables as needed.

For example:

```text
SBT_PROJECT_ROOT
SBT_PROJECT_ID
SBT_EXECUTION_ID
SBT_EXECUTION_LABEL
SBT_OUTPUT_DIR
SBT_TECHNICAL_RUN_ID
SBT_WORKFLOW_RUN_ID
SBT_RUN_DIR
SBT_STAGE
SBT_CONFIG
SBT_SLURM_JOB_ID
```

Do not retain misleading variable names when they refer to a different identity type.

Maintain transitional compatibility where necessary, but clearly deprecate ambiguous names.

# Project validation

Extend:

```bash
sbt project validate
```

to verify:

* active execution IDs are sequential and unique
* output folder prefixes agree with the execution index
* stage slugs agree with the stage registry
* manifests agree with their folder IDs
* technical IDs resolve
* technical links remain valid
* no active execution is missing its output folder unexpectedly
* no duplicate execution number exists
* removed executions are absent from the active index
* audit records do not contaminate the human-facing summary

Provide a safe repair or reindex command only when the operation is deterministic.

Do not silently renumber during ordinary validation.

# Tests

Add comprehensive tests for the redesign.

At minimum test:

## Identity and numbering

* fixed stage numbers are removed from output path generation
* the first project execution receives `001`
* subsequent executions increment correctly
* executing the same stage twice creates two sequential folders
* a stage run first receives `001` regardless of its stage type
* technical IDs remain stable when human IDs change
* SLURM IDs remain distinct from both identity systems

## Concurrency and locking

* concurrent allocation cannot produce duplicate IDs
* project locks are released after errors
* interrupted allocation does not corrupt the execution index

## Multi-stage submission

* each resolved stage receives a sequential execution ID
* dependencies remain correct
* workflow and stage execution identities are recorded correctly
* partial submission failure is represented accurately

## Summary

* default table is concise and ordered
* filtering by stage and status works
* JSON and YAML output validate
* `latest` resolves correctly
* removed executions are hidden by default
* `--include-removed` works

## Removal

* output-only execution removal requires confirmation
* asset-producing execution shows a second warning
* unknown asset effect is treated as risky
* non-interactive removal requires explicit flags
* active index entry is removed
* output folder is removed
* technical evidence is preserved
* removal audit is written
* normal summary hides the removed execution
* remaining executions are renumbered
* manifests and README links are updated
* old-to-new mappings are recorded
* modified assets are not falsely claimed to be restored

## Migration

* old fixed-number stage folders are detected
* old technical run-ID subfolders are flattened
* chronology uses structured timestamps
* repeated stage executions migrate separately
* dry run is non-destructive
* ambiguous migration aborts safely
* technical IDs are preserved

## Reporting

* reports are generated directly in the execution folder
* no extra long-ID subfolder is created
* project output index uses execution order
* ReadTheDocs no longer claims numbers are fixed stage catalogue positions

## Validation

* duplicate execution IDs are detected
* folder/index mismatches are detected
* missing technical links are reported
* removed executions do not appear active

Mock SLURM calls.

Tests must not submit real jobs or modify real user projects.

# Documentation

Update all relevant documentation.

Explain clearly:

* output numbers represent project execution order
* numbers are assigned automatically
* stage types no longer have permanent folder numbers
* rerunning a stage creates another numbered execution folder
* the human-facing execution ID is short
* permanent technical IDs remain under `.sbt`
* SLURM job IDs are separate
* stage output folders no longer contain long run-ID subfolders
* `sbt summary` presents the active workflow
* `sbt remove` logically removes an execution but does not automatically restore modified assets
* removal is preserved in a hidden technical audit
* remaining stages are renumbered after removal
* existing projects require an explicit migration command

Update examples throughout ReadTheDocs.

In particular, update:

```text
pipeline/reporting.html
```

and its source Markdown or reStructuredText.

Search for all documentation and code that assumes:

```text
prep is always 001
STARLING is always 011
```

and update it.

# Backwards compatibility

This redesign should be clean, but existing projects and technical records must remain recoverable.

Provide explicit migration rather than accumulating permanent compatibility hacks throughout the new implementation.

Preserve:

* existing technical run IDs
* existing SLURM job IDs
* historical timestamps
* stage names
* configs
* reusable assets
* technical provenance

Do not preserve the old fixed-number layout as the primary new data model.

After migration, new projects should use only the new execution-order model.

# Non-goals

Do not:

* change scientific algorithms
* change biological defaults
* migrate to Nextflow
* implement the private agent
* create an `assets/` root folder
* duplicate reusable assets into output folders
* use SLURM job IDs as human execution IDs
* discard technical provenance
* automatically roll back modified scientific assets
* silently migrate existing projects
* silently delete old outputs
* infer provenance from README prose
* rely exclusively on folder names as the execution database

# Development process

Proceed in this order:

1. Audit current stage numbering, run IDs, output paths, manifests and CLI references.
2. Write a concise internal design and migration note.
3. Define the new identity terminology and typed models.
4. Remove fixed output numbers from the stage registry.
5. Implement the project execution index and lock.
6. Implement sequential allocation.
7. Refactor output path generation.
8. Update reporting manifests and README generation.
9. Update multi-stage SLURM submission.
10. Implement `sbt summary`.
11. Implement safe `sbt remove`.
12. Implement renumbering and hidden removal audit.
13. Update status, logs, reports and project validation.
14. Implement explicit existing-project migration.
15. Update tests.
16. Update ReadTheDocs.
17. Search the repository for stale fixed-number and ambiguous run-ID assumptions.
18. Run tests, linting and type checks.
19. Provide a final implementation report.

# Required final implementation report

At completion, report:

* the final identity terminology
* where human execution IDs are stored
* where permanent technical IDs are stored
* how sequential IDs are allocated safely
* how multi-stage workflows are numbered
* the final output-folder structure
* how reruns of a stage appear
* how `sbt summary` works
* how `sbt remove` handles outputs and assets
* how deletion auditing works
* how renumbering is performed safely
* how old projects are migrated
* which backward-compatibility mechanisms remain
* all tests, lint and type-check results
* any unresolved edge cases

The final user experience should make a project’s analysis history understandable at a glance:

```text
outputs/
  001_Preprocessing/
  002_Denoising/
  003_Segmentation/
  004_Quantification/
  005_Cell_Clustering/
```

while the permanent, detailed and immutable provenance remains available under:

```text
.sbt/
```
