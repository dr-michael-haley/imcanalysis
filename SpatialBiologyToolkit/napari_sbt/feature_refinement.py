"""Compatibility imports; implementation lives in cell_classification.feature_refinement."""
from SpatialBiologyToolkit.cell_classification import feature_refinement as _implementation

__all__ = getattr(_implementation, "__all__", [
    name for name in vars(_implementation) if not name.startswith("_")
])


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
