# Archived managed RAPIDS candidate

This directory preserves the abandoned repository-managed `sbt-rapids`
candidate from commit `4b9855a`.

The candidate attempted to express the official RAPIDS-singlecell 26.08 CUDA
13 runtime through SBT's Conda-lock-then-pip installation model. It was retired
before production activation because Michael chose to keep the official
`rapids_singlecell` environment externally managed. The active upstream recipe
snapshot is retained under `image_migration/reference_specs/`.

These files are investigation and rollback evidence only. They are not an
active environment specification and must not be passed to `sbt env sync`.
