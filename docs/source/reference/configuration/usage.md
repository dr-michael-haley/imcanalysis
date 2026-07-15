# Using SpatialBiologyToolkit configuration

The complete pipeline configuration is represented by
`SpatialBiologyToolkit.config.PipelineConfig`. A user `config.yaml` may contain
only values that differ from the defaults; the typed loader validates those
values and fills omitted sections and fields without modifying the source YAML.

## Load a user config

```python
from SpatialBiologyToolkit.config import load_config

config = load_config("config.yaml")
print(config.general.anndata_path)
print(config.createmasks.cellpose_cell_diameter)
```

Existing pipeline entry points retain their dictionary-loading compatibility
layer. New Python code should prefer the typed API.

## Export a resolved config

```python
from SpatialBiologyToolkit.config.export import write_resolved_config

write_resolved_config(config, "run_config.resolved.yaml")
```

The resolved YAML includes every default after user overrides are applied, so
it is suitable for run provenance.

## JSON Schema and Markdown

```python
from SpatialBiologyToolkit.config import GeneralConfig, write_config_docs
from SpatialBiologyToolkit.config.schema import write_json_schema

write_json_schema("config.schema.json")
write_config_docs("generated_config_docs", layout="table")
```

The [generated configuration reference](index.md) uses the same table layout.
It includes each field's type, default, description, user level, and advice.
Importing the package never writes documentation files; generation is always
explicit.

## Keeping the published reference current

After changing a Pydantic config model or its field metadata, run:

```bash
make docs-generate
make docs-check
```

Read the Docs runs the same generator before its Sphinx build, while the
committed Markdown remains useful to readers and coding agents on GitHub.
