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

## Maintain the NapariSBT no-code guide

The main contents entry is `docs/source/napari_sbt/index.md`. Introductory pages
assume a scientist using buttons and menus; keep commands and implementation
details in the linked technical reference. Old Technical Guides URLs remain valid.

Workflow pages include the canonical Markdown in
`SpatialBiologyToolkit/napari_sbt/help/`. Edit those files to update both published
instructions and in-app help. Keep section headings used by focused Help buttons
stable, or update their callers at the same time.

`help/presentation.py` defines the shared interface/help colour palette and section
mapping. Sphinx's local `napari_sbt_help` extension reads it without importing Qt.
Workflow pages declare `napari_help_topic` in their front matter; the extension
adds matching accessible heading colours. Do not hand-copy palette values into
individual pages. The browser CSS and popup text styles use explicit contrast and
comfortable reading sizes; help popups also provide text zoom controls.

Regenerate screenshots from the real widgets with the GUI environment's Python:

```bash
python docs/tools/capture_napari_sbt.py
```

The script uses a disposable synthetic dataset and isolated application settings,
and captures the actual Qt dock and Help dialog, not mockups. It does not capture
the OpenGL tissue canvas. Images go to `docs/source/_static/napari_sbt/`; inspect
them after UI changes. Captions must identify demonstration data and avoid
presenting these cells as biological evidence. Rebuild Sphinx after regenerating.
