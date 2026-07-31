#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
WARNING: install/setup_envs.sh is a compatibility wrapper.
Canonical environment management now lives in `sbt env`.
This command synchronizes every repository-managed environment.
New users should let `sbt run` offer only the environments their workflow needs.
EOF

if ! command -v sbt >/dev/null 2>&1; then
    echo "sbt is not available. Run install/bootstrap_sbt.sh first." >&2
    exit 2
fi

exec sbt env sync --all "$@"
