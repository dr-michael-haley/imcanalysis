"""Shared ordering for AnnData variable selectors across NapariSBT."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

VariableOrderMode = Literal["anndata", "alphabetical", "similarity"]

VARIABLE_ORDER_OPTIONS: tuple[tuple[str, VariableOrderMode], ...] = (
    ("AnnData order", "anndata"),
    ("Alphabetical", "alphabetical"),
    ("Expression similarity", "similarity"),
)
VARIABLE_ORDER_MODES = {mode for _label, mode in VARIABLE_ORDER_OPTIONS}


def _default_similarity_orderer(adata, variables: list[str]) -> list[str]:
    """Use the same clustering helper as the matrix-plot ordering option."""

    from SpatialBiologyToolkit.utils import reorder_vars_by_expression

    return list(reorder_vars_by_expression(adata, variables))


class VariableOrderRegistry:
    """Own one session-wide variable order and cache its similarity ranking.

    The registry deliberately computes similarity from the live ``adata.X`` once,
    then reuses that ranking in every selector. Lists containing image-only or
    raw-only variables append those unmatched entries in their incoming order.
    """

    def __init__(
        self,
        *,
        mode: VariableOrderMode = "anndata",
        similarity_orderer: Callable[[object, list[str]], list[str]] | None = None,
    ) -> None:
        self._adata = None
        self._mode: VariableOrderMode = "anndata"
        self._similarity_orderer = similarity_orderer or _default_similarity_orderer
        self._similarity_order: list[str] | None = None
        self.last_warning: str | None = None
        self.set_mode(mode)

    @property
    def mode(self) -> VariableOrderMode:
        return self._mode

    def set_mode(self, mode: str) -> None:
        value = str(mode).strip().casefold()
        if value not in VARIABLE_ORDER_MODES:
            raise ValueError(f"Unknown variable-order mode: {mode!r}.")
        self._mode = cast(VariableOrderMode, value)

    def set_adata(self, adata) -> None:
        """Bind the live AnnData and invalidate expression-derived state."""

        self._adata = adata
        self.invalidate()

    def invalidate(self) -> None:
        self._similarity_order = None
        self.last_warning = None

    def _anndata_variables(self) -> list[str]:
        if self._adata is None:
            return []
        return [str(value) for value in self._adata.var_names]

    def _expression_similarity_order(self) -> list[str]:
        variables = self._anndata_variables()
        if self._similarity_order is not None:
            return list(self._similarity_order)
        if len(variables) < 2:
            self._similarity_order = variables
            return list(variables)
        try:
            ordered = [
                str(value) for value in self._similarity_orderer(self._adata, variables)
            ]
            if len(ordered) != len(variables) or set(ordered) != set(variables):
                raise ValueError("the similarity order did not return every variable")
        except Exception as exc:  # noqa: BLE001 - safe UI fallback is intentional
            self.last_warning = (
                "Expression-similarity ordering could not be calculated; AnnData "
                f"order is being used instead ({type(exc).__name__}: {exc})."
            )
            ordered = variables
        else:
            self.last_warning = None
        self._similarity_order = ordered
        return list(ordered)

    def ordered(
        self,
        values: Sequence[object],
        *,
        canonical_names: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Order unique display values using the session-wide registry mode."""

        unique = list(dict.fromkeys(str(value) for value in values))
        if self._mode == "alphabetical":
            return sorted(unique, key=lambda value: (value.casefold(), value))

        reference = (
            self._expression_similarity_order()
            if self._mode == "similarity"
            else self._anndata_variables()
        )
        rank = {value: index for index, value in enumerate(reference)}
        aliases = canonical_names or {}
        known: list[tuple[int, int, str]] = []
        unmatched: list[tuple[int, str]] = []
        for input_index, value in enumerate(unique):
            canonical = str(aliases.get(value, value))
            if canonical in rank:
                known.append((rank[canonical], input_index, value))
            else:
                unmatched.append((input_index, value))
        known.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in known] + [item[1] for item in unmatched]


__all__ = [
    "VARIABLE_ORDER_MODES",
    "VARIABLE_ORDER_OPTIONS",
    "VariableOrderMode",
    "VariableOrderRegistry",
]
