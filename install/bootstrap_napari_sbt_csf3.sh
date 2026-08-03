#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NAPARI_ENV="${SBT_NAPARI_ENV:-sbt-napari}"
ENVIRONMENT_FILE="$REPOSITORY_ROOT/HPC_env_files/sbt-napari/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found on PATH. Install or load Conda first." >&2
    exit 2
fi

if conda env list | awk '{print $1}' | grep -Fxq "$NAPARI_ENV"; then
    conda env update --file "$ENVIRONMENT_FILE" --name "$NAPARI_ENV" --prune
else
    conda env create --file "$ENVIRONMENT_FILE" --name "$NAPARI_ENV"
fi

conda run -n "$NAPARI_ENV" python -m pip install \
    -e "$REPOSITORY_ROOT" --no-deps --no-build-isolation

conda run -n "$NAPARI_ENV" python -c \
    "import SpatialBiologyToolkit, anndata, napari, pyarrow, qtpy, skimage, sklearn"

echo "Installed NapariSBT in fixed environment '$NAPARI_ENV'."
echo "Next: request an X11 interactive node, then run 'sbt gui napari --check'."
