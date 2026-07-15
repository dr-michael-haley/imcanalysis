Pydantic configuration
======================

The complete pipeline configuration is represented by
``SpatialBiologyToolkit.config.PipelineConfig``. Each existing YAML section and
field retains its current name and default. A user YAML may contain only the
values that differ from those defaults.

Load a user config
------------------

.. code-block:: python

   from SpatialBiologyToolkit.config import load_config

   config = load_config("config.yaml")
   print(config.general.anndata_path)
   print(config.createmasks.cellpose_cell_diameter)

The typed loader validates values and fills omitted sections and fields. It does
not modify the input YAML. Existing pipeline scripts continue to use the legacy
dictionary loader, which still creates or updates ``config.yaml`` where that was
the previous behavior.

Export a resolved config
------------------------

.. code-block:: python

   from SpatialBiologyToolkit.config.export import write_resolved_config

   write_resolved_config(config, "run_config.resolved.yaml")

The exported YAML contains every default after user overrides are applied and is
suitable for provenance records.

Export JSON Schema
------------------

.. code-block:: python

   from SpatialBiologyToolkit.config.schema import write_json_schema

   write_json_schema("config.schema.json")

The schema includes field descriptions and baseline ``level``, ``stage``,
``ui_group``, and ``advice`` metadata for future documentation and UI tooling.
