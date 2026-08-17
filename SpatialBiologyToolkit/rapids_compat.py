"""Small runtime-compatibility helpers for RAPIDS-singlecell releases."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def resolve_harmony_runtime_kwargs(
    harmony_integrate: Callable[..., Any],
    flavor: str,
) -> dict[str, str]:
    """Translate the configured Harmony flavor to the installed API.

    RAPIDS-singlecell 0.12 exposes ``correction_method`` and implements the
    original Harmony algorithm. Newer releases expose ``flavor`` and add
    Harmony2. Harmony2 must never be silently replaced with a different
    algorithm when the older runtime is selected.
    """

    normalized = str(flavor).strip().lower()
    if normalized not in {"harmony1", "harmony2"}:
        raise ValueError("Harmony flavor must be one of: 'harmony1', 'harmony2'.")

    parameters = inspect.signature(harmony_integrate).parameters
    if "flavor" in parameters:
        return {"flavor": normalized}

    if "correction_method" in parameters:
        if normalized == "harmony1":
            return {"correction_method": "original"}
        raise RuntimeError(
            "The installed RAPIDS-singlecell runtime implements the legacy Harmony "
            "API and cannot run harmony2. Set rapids.harmony_flavor to 'harmony1' "
            "or run this stage in the newer legacy RAPIDS environment."
        )

    raise RuntimeError(
        "The installed RAPIDS-singlecell Harmony API is not recognized: expected "
        "a 'flavor' or 'correction_method' parameter."
    )
