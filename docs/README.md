# SpatialBiologyToolkit documentation

The canonical Sphinx sources live under `docs/source/` and are published at
<https://imcanalysis.readthedocs.io/en/latest/>. Do not edit the legacy
generated HTML under `Documentation/`.

## Generate repository-derived pages

From the repository root:

```bash
make docs-generate
make docs-check
```

`docs/tools/generate_docs.py` creates:

- the SLURM stage table and per-stage pages from `pipeline.conf` and wrapper `#@` metadata;
- compact config tables and JSON Schema from the Pydantic models;
- one Sphinx autodoc page per package module.

Generated pages are committed so they are readable on GitHub and available to
automation. Edit their source metadata/code, not the generated files.

## Build locally

Install the small documentation environment and build:

```bash
python -m pip install -r docs/requirements.txt
make docs-html
```

On Windows without GNU Make:

```powershell
python docs/tools/generate_docs.py
python -m sphinx -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser. Read the Docs runs the generator
as a pre-build job and then uses `docs/source/conf.py`.
