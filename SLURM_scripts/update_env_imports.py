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
PYMOD_RE = re.compile(r"^\s*python\s+-m\s+([A-Za-z0-9_\.]+)")

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
    modules = []

    with script.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if env is None:
                m = ENV_RE.match(line)
                if m:
                    env = m.group(1)

            m = PYMOD_RE.search(line)
            if m:
                modules.append(m.group(1))

    if not env:
        continue

    for mod in modules:
        py_file = resolve_module_to_file(mod)
        if not py_file:
            print(f"⚠ Could not resolve module {mod}")
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

print(f"✔ Generated {OUTPUT_YAML}")
