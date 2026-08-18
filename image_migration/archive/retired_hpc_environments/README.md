# Retired HPC Conda environment specifications

This directory is a migration backup, not an active SBT environment catalogue.
`sbt env` reads only `HPC_env_files/environments.yaml` and will not install or
select specifications stored here.

The archived environment families were replaced by the consolidated
`analysis` / `sbt-analysis` runtime:

| Archived logical key | Archived Conda name | Replacement |
|---|---|---|
| `segmentation` | `imc_segmentation` | `analysis` / `sbt-analysis` |
| `biobatchnet` | `imc_biobatchnet` | `analysis` / `sbt-analysis` |
| `cellcharter` | `imc_cellcharter` | `analysis` / `sbt-analysis` |
| `rapids` | `rapids_singlecell` | `analysis` / `sbt-analysis` |

The first three directories contain their final repository specifications and
locks. The former RAPIDS runtime was externally managed and had no repository
specification directory; its registry entry is preserved in
`environments.pre-sbt-standardization.yaml`.

The registry snapshot also records the pre-standardization physical names of
the retained standalone environments. Do not edit archived locks to make them
look current. If an old runtime must be restored, review its dependencies,
copy it back into the active catalogue under a deliberate key and name, then
regenerate and validate its lock on the target HPC.
