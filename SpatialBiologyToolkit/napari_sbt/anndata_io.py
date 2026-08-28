"""Compatibility helpers for writing live AnnData objects from NapariSBT."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def _nullable_string_write_scope() -> Iterator[None]:
    """Temporarily opt into AnnData's nullable-string on-disk encoding.

    AnnData 0.11+ deliberately requires an explicit opt-in before writing
    ``pandas.StringDtype`` arrays. Notebook users commonly hold these arrays in
    ``obs`` after modern pandas operations. The override is scoped to one
    synchronous write and AnnData restores its previous setting in ``finally``.
    Older AnnData releases do not expose the setting and retain their existing
    writer behaviour.
    """

    import anndata as ad

    settings = getattr(ad, "settings", None)
    if settings is None or not hasattr(settings, "allow_write_nullable_strings"):
        yield
        return
    override = getattr(settings, "override", None)
    if callable(override):
        with override(allow_write_nullable_strings=True):
            yield
        return

    previous = settings.allow_write_nullable_strings
    try:
        settings.allow_write_nullable_strings = True
        yield
    finally:
        settings.allow_write_nullable_strings = previous


def write_h5ad_compat(
    adata: Any,
    destination: str | Path,
    **kwargs: Any,
) -> None:
    """Write AnnData while preserving nullable strings and caller state."""

    # AnnData otherwise calls ``strings_to_categoricals`` on the live object,
    # changing notebook-owned obs/var dtypes merely because it was saved.
    kwargs.setdefault("convert_strings_to_categoricals", False)
    with _nullable_string_write_scope():
        adata.write_h5ad(destination, **kwargs)


__all__ = ["write_h5ad_compat"]
