# Configuration Maintenance

## What this stage does

Synchronises missing default configuration fields in a run-local or legacy configuration.

## Why it is performed

It keeps older configs compatible with the current schema.

## Main inputs

The selected YAML configuration.

## Reusable assets produced

An updated configuration copy; managed SBT runs preserve the user's source config.

## Human-facing outputs produced

The report records completion and links to the technical resolved config.

## Important configuration options

No scientific settings are introduced by this maintenance stage.

## How to interpret the results

Review newly added defaults before relying on them in later runs.

## Common problems and limitations

Unknown legacy keys are retained or warned about according to the compatibility loader.
