"""Experimental, removable AnnData/TIFF to Odon adapter.

Importing this package does not load scientific or GUI dependencies.
"""


def export_dataset(*args, **kwargs):
    """Export a dataset; see :func:`.exporter.export_dataset` for options."""
    from .exporter import export_dataset as export

    return export(*args, **kwargs)


__all__ = ["export_dataset"]
