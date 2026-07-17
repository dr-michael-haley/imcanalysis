#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER_ENV="${SBT_LAUNCHER_ENV:-sbt-cli}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found on PATH. Install or load Conda first." >&2
    exit 2
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$LAUNCHER_ENV"; then
    conda env create --file "$REPOSITORY_ROOT/Local_envs/sbt_cli_env.yml" --name "$LAUNCHER_ENV"
fi

conda run -n "$LAUNCHER_ENV" python -m pip install \
    -e "$REPOSITORY_ROOT" --no-deps

echo "Installed the sbt launcher in fixed environment '$LAUNCHER_ENV'."
echo "Next: conda run -n '$LAUNCHER_ENV' sbt env doctor"
echo "Then: conda run -n '$LAUNCHER_ENV' sbt env sync --all --dry-run"
