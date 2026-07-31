#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GUI_ENV="${SBT_GUI_ENV:-sbt-gui}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found on PATH. Install or load Conda first." >&2
    exit 2
fi

if conda env list | awk '{print $1}' | grep -Fxq "$GUI_ENV"; then
    conda env update --file "$REPOSITORY_ROOT/Local_envs/sbt_gui_env.yml" \
        --name "$GUI_ENV" --prune
else
    conda env create --file "$REPOSITORY_ROOT/Local_envs/sbt_gui_env.yml" \
        --name "$GUI_ENV"
fi

conda run -n "$GUI_ENV" python -m pip install \
    -e "$REPOSITORY_ROOT" --no-deps --no-build-isolation

echo "Installed the SBT Project Console in fixed environment '$GUI_ENV'."
echo "Next: activate an X11 interactive session, then run 'sbt gui project'."
