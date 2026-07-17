# AI Interpretation

## What this stage does

Optionally proposes text labels for Leiden clusters and writes the selected labels into AnnData.

## Why it is performed

It can accelerate preliminary annotation while retaining an auditable record for expert review.

## Main inputs

Cluster marker summaries, tissue context, panel markers, and an explicitly configured API credential.

## Reusable assets produced

Updated AnnData label columns.

## Human-facing outputs produced

Per-resolution interpretation tables and the prompts/results retained by the existing stage.

## Important configuration options

Enable/repeat switches, tissue context, cluster resolutions, and model-related settings.

## How to interpret the results

Treat proposed labels as hypotheses requiring domain-expert validation.

## Common problems and limitations

Requires network/API access and must not be treated as an autonomous biological conclusion.
