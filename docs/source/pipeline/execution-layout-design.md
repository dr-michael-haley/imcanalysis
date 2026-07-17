# Execution identity and layout design

This note records the identity model used by the project execution layout. It
is also the migration contract for the earlier fixed stage-number layout.

## Identities

- **Execution ID** is the short, project-local sequence number shown as `001`,
  `002`, and so on. It may change when an active workflow is compacted after a
  removal.
- **Technical execution ID** permanently identifies one attempted stage
  execution. It never changes when display numbers are compacted.
- **Workflow run ID** permanently identifies one `sbt run` request and its
  technical directory under `.sbt/runs/`. A multi-stage request has one
  workflow run ID and several technical execution IDs.
- **SLURM job ID** identifies the scheduler job only. It is never used as a
  project execution identity.

The typed `.sbt/executions.yaml` index is authoritative for the active,
human-facing sequence. Output folder names are derived from the index; they are
not the execution database.

## Allocation and output paths

Allocation is serialized with `.sbt/locks/executions.lock`. A multi-stage run
reserves one consecutive execution ID per resolved stage while holding that
lock. Managed scripts receive the resolved execution identity and output path
through environment variables and never calculate sequence numbers.

Each accepted stage writes directly to:

```text
outputs/<execution_id>_<stage_slug>/
```

Direct `python -m` execution does not modify the managed index and writes below
`outputs/direct/` instead.

## Removal and renumbering

Removal deletes only the human-facing execution folder and active index entry.
It does not restore or delete reusable scientific assets. Risky or unknown
asset effects require a second explicit confirmation. The immutable technical
record remains under `.sbt`, and a removal audit is written under
`.sbt/audit/removals/`.

Remaining active executions are compacted with collision-safe temporary names
while holding the execution lock. Manifests, active technical references, and
generated indexes are then rewritten. External links to mutable numbered paths
may therefore break; permanent integrations should use technical IDs.

## Existing projects

Ordinary CLI startup never rewrites the former layout. Projects containing
fixed stage folders with technical-run subdirectories must use:

```bash
sbt project migrate-execution-layout --dry-run
sbt project migrate-execution-layout
```

Migration orders executions by structured manifest timestamps, preserves
workflow and scheduler identities, flattens each old run subdirectory into one
sequential execution folder, and records an audit mapping. Ambiguous records
cause migration to stop without modifying the project.
