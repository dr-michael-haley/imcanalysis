#!/usr/bin/env python3

import ast
import re
import sys
import yaml
import importlib.util
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------
# Paths
# -------------------------------------------------
SLURM_DIR = Path(__file__).resolve().parent
OUTPUT_YAML = SLURM_DIR / "env_imports.yaml"

# -------------------------------------------------
# Regex
# -------------------------------------------------
ENV_RE = re.compile(r"^#@ENV:\s*(\S+)")
MODULE_RE = re.compile(r"^#@MODULE:\s*([A-Za-z0-9_\.]+)")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def resolve_module_to_file(module_name: str) -> Path | None:
    """Resolve python -m module.path to a .py file"""
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        return None

    if spec is None or spec.origin is None:
        return None

    origin = Path(spec.origin)
    if origin.name == "__init__.py":
        return origin
    if origin.suffix == ".py":
        return origin
    return None


def extract_top_level_imports(py_file: Path) -> set[str]:
    """Extract top-level imports using AST"""
    imports = set()

    try:
        source = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return imports

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    return imports


# -------------------------------------------------
# Main logic
# -------------------------------------------------
env_imports = defaultdict(set)

for script in SLURM_DIR.glob("*.sh"):
    env = None

    with script.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            env_match = ENV_RE.match(line)
            if env_match:
                env = env_match.group(1)
                continue

            module_match = MODULE_RE.match(line)
            if not module_match or not env:
                continue
            mod = module_match.group(1)
            py_file = resolve_module_to_file(mod)
            if not py_file:
                print(f"WARNING: Could not resolve module {mod}")
                continue

            imports = extract_top_level_imports(py_file)
            env_imports[env].update(imports)

# -------------------------------------------------
# Write YAML (stable output)
# -------------------------------------------------
output = {
    env: sorted(imps)
    for env, imps in sorted(env_imports.items())
}

with OUTPUT_YAML.open("w", encoding="utf-8") as f:
    yaml.safe_dump(output, f, sort_keys=False)

print(f"Generated {OUTPUT_YAML}")
