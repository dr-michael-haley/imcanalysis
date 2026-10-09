"""Draw regions on an embedding and apply categorical labels to live AnnData.

The popup uses Matplotlib and Qt, without importing or launching Napari. Drawing
and assigning labels only edit a draft; ``Apply to AnnData`` commits the column.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CohortFilter:
    """An obs value selection or inclusive numeric range in obs or X.

    Multiple filters are combined with AND; values within one filter use OR.
    ``None`` in ``values`` matches missing observations. Blank range bounds are
    unbounded. X always means the stored ``adata.X``, without transformation.
    """

    source: Literal["obs", "X"]
    key: str
    values: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None

    def describe(self) -> str:
        if self.values is not None:
            labels = ["<missing>" if pd.isna(v) else str(v) for v in self.values]
            return f"{self.source}[{self.key}] in {', '.join(labels)}"
        lower = "−∞" if self.minimum is None else f"{self.minimum:g}"
        upper = "+∞" if self.maximum is None else f"{self.maximum:g}"
        return f"{lower} ≤ {self.source}[{self.key}] ≤ {upper}"

    def to_recipe(self) -> dict[str, Any]:
        record = asdict(self)
        if self.values is not None:
            record["values"] = [None if pd.isna(v) else v for v in self.values]
        return record


class AnnotationSession:
    """A small, undoable label draft indexed by cell identity."""

    def __init__(
        self,
        adata: Any,
        *,
        basis: str = "X_umap",
        components: tuple[int, int] = (0, 1),
        obs_names: Sequence[str] | None = None,
        source_obs: str | None = None,
        filters: Sequence[CohortFilter] = (),
    ) -> None:
        if not adata.obs_names.is_unique:
            raise ValueError("Annotation requires unique obs_names.")
        if getattr(adata, "is_view", False) or getattr(adata, "isbacked", False):
            raise ValueError(
                "Use an in-memory AnnData object, not a view or backed file."
            )
        if basis not in adata.obsm:
            raise ValueError(f"AnnData has no embedding {basis!r}.")
        coordinates = np.asarray(adata.obsm[basis])
        if (
            coordinates.ndim != 2
            or len(components) != 2
            or any(not isinstance(c, (int, np.integer)) for c in components)
            or min(components) < 0
            or max(components) >= coordinates.shape[1]
            or components[0] == components[1]
        ):
            raise ValueError(
                "Choose two different, valid zero-based embedding components."
            )
        self.adata = adata
        self.obs_names = adata.obs_names.copy()
        self.basis = basis
        self.components = components
        self.coordinates = np.array(
            coordinates[:, list(components)], dtype=float, copy=True
        )
        self.scope: np.ndarray = np.ones(len(self.obs_names), dtype=bool)
        self.scope_names = None if obs_names is None else list(obs_names)
        if obs_names is not None:
            requested = pd.Index(obs_names)
            if not requested.isin(self.obs_names).all():
                raise ValueError("Some requested cells are missing from AnnData.")
            self.scope = self.obs_names.isin(requested)
        self.finite = np.isfinite(self.coordinates).all(axis=1)
        self.eligible = self.scope & self.finite
        if not self.eligible.any():
            raise ValueError(
                "No cells with finite coordinates remain in the selected scope."
            )
        self.selected: np.ndarray = np.zeros(len(self.obs_names), dtype=bool)
        self.selection_vertices: list[list[float]] | None = None
        self.regions: list[dict[str, Any]] = []
        self.history: list[tuple[np.ndarray, np.ndarray]] = []
        self._palette: dict[str, str] = {"Unassigned": "#bdbdbd"}
        self.dirty = False
        self.reset_labels(source_obs)
        self.set_filters(filters)

    def set_filters(self, filters: Sequence[CohortFilter]) -> None:
        """Change the selectable cohort without changing labels or undo history."""
        current = self.adata.obs_names
        if not current.is_unique or len(current) != len(self.obs_names):
            raise ValueError(
                "The AnnData cell set changed; reopen the annotation window."
            )
        positions = current.get_indexer(self.obs_names)
        if (positions < 0).any():
            raise ValueError(
                "The AnnData cell set changed; reopen the annotation window."
            )
        mask = self.scope & self.finite
        for rule in filters:
            if rule.source == "obs":
                if rule.key not in self.adata.obs:
                    raise ValueError(f"AnnData has no observation {rule.key!r}.")
                values = self.adata.obs[rule.key].iloc[positions]
            elif rule.source == "X":
                if rule.key not in self.adata.var_names:
                    raise ValueError(f"AnnData X has no marker {rule.key!r}.")
                column = self.adata.var_names.get_loc(rule.key)
                if not isinstance(column, (int, np.integer)):
                    raise ValueError("Marker names must be unique to filter X.")
                if self.adata.X is None:
                    raise ValueError("AnnData has no X matrix.")
                # Read only one column, including for sparse expression matrices.
                expression = self.adata.X[:, column : column + 1][positions]
                values = pd.Series(
                    np.asarray(
                        expression.toarray()
                        if hasattr(expression, "toarray")
                        else expression
                    ).ravel()
                )
            else:
                raise ValueError("Filter source must be obs or X.")
            if rule.values is not None:
                if rule.source != "obs" or not rule.values:
                    raise ValueError("Choose at least one obs value.")
                if rule.minimum is not None or rule.maximum is not None:
                    raise ValueError(
                        "Choose values or a numeric range for each filter."
                    )
                present = [v for v in rule.values if pd.notna(v)]
                matches = values.isin(present)
                if any(pd.isna(v) for v in rule.values):
                    matches |= values.isna()
                mask &= matches.to_numpy(dtype=bool)
            else:
                if rule.minimum is None and rule.maximum is None:
                    raise ValueError("Enter at least one range bound.")
                bounds = [v for v in (rule.minimum, rule.maximum) if v is not None]
                if not np.isfinite(bounds).all():
                    raise ValueError("Range bounds must be finite numbers.")
                if (
                    rule.minimum is not None
                    and rule.maximum is not None
                    and rule.minimum > rule.maximum
                ):
                    raise ValueError("The minimum must not exceed the maximum.")
                if not pd.api.types.is_numeric_dtype(values):
                    raise ValueError(
                        f"{rule.key!r} is not numeric; use obs values instead."
                    )
                numeric = values.to_numpy(dtype=float, na_value=np.nan)
                matches = np.isfinite(numeric)
                if rule.minimum is not None:
                    matches &= numeric >= rule.minimum
                if rule.maximum is not None:
                    matches &= numeric <= rule.maximum
                mask &= matches
        self.filters = tuple(filters)
        self.eligible = mask
        self.selected[:] = False
        self.selection_vertices = None

    def reset_labels(self, source_obs: str | None = None) -> None:
        """Start a fresh draft, optionally preserving existing labels elsewhere."""
        if source_obs is None:
            self.labels = pd.Series("Unassigned", index=self.obs_names, dtype=object)
        else:
            if source_obs not in self.adata.obs:
                raise ValueError(f"AnnData has no observation {source_obs!r}.")
            values = self.adata.obs[source_obs].reindex(self.obs_names)
            self.labels = values.astype(object).map(
                lambda value: str(value) if pd.notna(value) else "Unassigned"
            )
        self.source_obs = source_obs
        self.parent_labels = self.labels.copy()
        self._palette = {"Unassigned": "#bdbdbd"}
        if source_obs is not None:
            self._palette.update(self.observation_palette(source_obs))
        self.history.clear()
        self.regions.clear()
        self.selected[:] = False
        self.selection_vertices = None
        self.dirty = False

    def select_polygon(self, vertices: Sequence[Sequence[float]]) -> int:
        """Select all eligible cells inside a region, including undisplayed cells."""
        from matplotlib.path import Path

        polygon = np.asarray(vertices, dtype=float)
        self.selected[:] = False
        self.selection_vertices = None
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or len(np.unique(polygon, axis=0)) < 3
            or not np.isfinite(polygon).all()
        ):
            raise ValueError("Draw a region with at least three distinct points.")
        self.selected[self.eligible] = Path(polygon).contains_points(
            self.coordinates[self.eligible]
        )
        self.selection_vertices = polygon.tolist()
        return int(self.selected.sum())

    def assign(self, label: str, *, inherit_parent: bool = False) -> int:
        """Assign the selection; the latest assignment wins where regions overlap."""
        label = label.strip()
        if not label:
            raise ValueError("Enter a category label.")
        if not self.selected.any():
            raise ValueError("Draw around some cells first.")
        if inherit_parent and self.source_obs is None:
            raise ValueError(
                "Choose a parent label column before adding a parent prefix."
            )
        positions = np.flatnonzero(self.selected & self.eligible)
        proposed: np.ndarray = np.full(len(positions), label, dtype=object)
        if inherit_parent:
            proposed = np.asarray(
                [
                    f"{parent} / {label}"
                    for parent in self.parent_labels.iloc[positions]
                ],
                dtype=object,
            )
        changed = self.labels.iloc[positions].to_numpy() != proposed
        positions, proposed = positions[changed], proposed[changed]
        self.history.append(
            (positions, self.labels.iloc[positions].to_numpy(copy=True))
        )
        self.labels.iloc[positions] = proposed
        self.regions.append({
            "vertices": self.selection_vertices,
            "label": label,
            "inherit_parent": inherit_parent,
            "filters": [rule.to_recipe() for rule in self.filters],
        })
        self.dirty = True
        return len(positions)

    def undo(self) -> bool:
        if not self.history:
            return False
        positions, previous = self.history.pop()
        self.regions.pop()
        self.labels.iloc[positions] = previous
        self.dirty = True
        return True

    def recipe(self, key_added: str = "manual_population", **display: Any) -> dict:
        """Export assigned regions in order, with the cohort used for each one.

        A pending selection is saved for reference but is not assigned on replay.
        Display settings are descriptive and do not affect headless selection.
        """
        return deepcopy({
            "version": 1,
            "basis": self.basis,
            "components": list(self.components),
            "key_added": key_added,
            "source_obs": self.source_obs,
            "obs_names": self.scope_names,
            "regions": self.regions,
            "filters": [rule.to_recipe() for rule in self.filters],
            "selection_vertices": self.selection_vertices,
            "palette": self.palette(),
            "display": display,
        })

    def save_recipe(
        self, path: str | Path, key_added: str = "manual_population", **display: Any
    ) -> Path:
        """Save a JSON recipe without modifying AnnData."""
        destination = Path(path)
        payload = json.dumps(
            self.recipe(key_added, **display),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_recipe_json_value,
        )
        destination.write_text(payload + "\n", encoding="utf-8")
        return destination

    def palette(self) -> dict[str, str]:
        """Keep draft colours stable as labels are added or undone."""
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        for label in self.labels.unique():
            if label not in self._palette:
                self._palette[label] = to_hex(
                    colormaps["tab20"]((len(self._palette) - 1) % 20)
                )
        return self._palette

    def observation_palette(self, key: str) -> dict[str, str]:
        """Use the same parent colours before and after changing cohorts or labels."""
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        source = self.adata.obs[key]
        categories = sorted(
            source.astype(object).fillna("Unassigned").astype(str).unique()
        )
        palette = {
            name: to_hex(colormaps["tab20"](i % 20))
            for i, name in enumerate(categories)
        }
        palette["Unassigned"] = "#bdbdbd"
        saved = self.adata.uns.get(f"{key}_colors", [])
        if isinstance(source.dtype, pd.CategoricalDtype) and len(saved) == len(
            source.cat.categories
        ):
            palette.update(
                {str(k): str(v) for k, v in zip(source.cat.categories, saved)}
            )
        return palette

    def apply(self, key_added: str, *, overwrite: bool = False) -> str:
        """Write one categorical column, aligning labels by identity, not row order."""
        key_added = key_added.strip()
        if not key_added:
            raise ValueError("Enter an output observation column name.")
        current = self.adata.obs_names
        if (
            not current.is_unique
            or len(current) != len(self.obs_names)
            or not current.isin(self.obs_names).all()
        ):
            raise ValueError(
                "The AnnData cell set changed; reopen the annotation window."
            )
        if key_added in self.adata.obs and not overwrite:
            raise ValueError(
                f"obs[{key_added!r}] already exists. Choose a new name or enable overwrite."
            )
        self.adata.obs[key_added] = pd.Categorical(self.labels.reindex(current))
        palette = self.palette()
        self.adata.uns[f"{key_added}_colors"] = np.asarray(
            [palette[label] for label in self.adata.obs[key_added].cat.categories],
            dtype=str,
        )
        self.dirty = False
        return key_added


def _recipe_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(
        f"Cannot save recipe value of type {type(value).__name__}; "
        "use text, numeric, boolean or missing obs filter values."
    )


def apply_annotation_recipe(
    adata: Any,
    recipe: str | Path | Mapping[str, Any],
    *,
    key_added: str | None = None,
    overwrite: bool = False,
) -> str:
    """Replay saved regions into categorical ``adata.obs`` without importing Qt.

    Uses the supplied embedding coordinates and obs/X values. Regions run in
    assignment order; the last matching region wins. Regions matching no cells
    are skipped. Unassigned selections and display settings have no effect.
    An explicit obs-name scope is preserved; all scoped names must be present.
    Returns the output column name. Save the modified AnnData separately.
    """
    if isinstance(recipe, (str, Path)):
        recipe = json.loads(Path(recipe).read_text(encoding="utf-8"))
    if not isinstance(recipe, Mapping) or recipe.get("version") != 1:
        raise ValueError("Expected an annotation recipe with version 1.")
    try:
        session = AnnotationSession(
            adata,
            basis=recipe["basis"],
            components=tuple(recipe["components"]),
            source_obs=recipe["source_obs"],
            obs_names=recipe["obs_names"],
        )
        for region in recipe["regions"]:
            session.set_filters([CohortFilter(**rule) for rule in region["filters"]])
            if session.select_polygon(region["vertices"]):
                session.assign(
                    region["label"], inherit_parent=region["inherit_parent"]
                )
        session._palette.update(recipe.get("palette", {}))
        output_key = recipe["key_added"] if key_added is None else key_added
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid annotation recipe: {exc}") from exc
    return session.apply(output_key, overwrite=overwrite)


def annotate_embedding(
    adata: Any,
    *,
    basis: str = "X_umap",
    components: tuple[int, int] = (0, 1),
    key_added: str = "manual_population",
    color: str | None = None,
    source_obs: str | None = None,
    filters: Sequence[CohortFilter] = (),
    obs_names: Sequence[str] | None = None,
    point_limit: int = 50_000,
    layer: str | None = None,
    use_raw: bool = False,
    block: bool | None = None,
    parent: Any = None,
    on_apply: Callable[[str], None] | None = None,
    before_apply: Callable[[], None] | None = None,
) -> Any:
    """Open a standalone Qt popup for lasso/polygon annotation.

    ``components`` are zero-based. ``obs_names`` restricts selectable cells;
    ``point_limit`` only limits drawing, never selection. Colours can come from
    an observation or marker. ``source_obs`` supplies initial labels, otherwise
    every cell starts as ``Unassigned``. Later assignments replace earlier ones.
    ``filters`` restricts the annotation cohort using obs values or obs/X ranges;
    other cells remain visible in grey. Cohort changes preserve draft labels.

    In a notebook the Qt event loop is enabled automatically (a popup, not an
    inline plot). Scripts block until the window closes by default. Pass
    ``block=False`` only when a Qt event loop is already running. The returned
    dialog exposes its draft as ``dialog.session``. Applying modifies live
    ``adata.obs``; saving a new H5AD copy is a separate button.
    Use ``dialog.save_recipe("regions.json")`` or the Save recipe button to
    save settings and drawn regions. ``apply_annotation_recipe`` replays the
    assignments without a window, using the same embedding coordinate system.
    """
    if not isinstance(point_limit, int) or point_limit < 1:
        raise ValueError("point_limit must be a positive integer.")
    session = AnnotationSession(
        adata,
        basis=basis,
        components=components,
        obs_names=obs_names,
        source_obs=source_obs,
        filters=filters,
    )
    # Import the optional GUI only when a popup is requested.
    from ._annotation_popup import open_annotation_popup

    return open_annotation_popup(
        session,
        key_added=key_added,
        color=color,
        point_limit=point_limit,
        layer=layer,
        use_raw=use_raw,
        block=block,
        parent=parent,
        on_apply=on_apply,
        before_apply=before_apply,
    )


def main() -> None:
    """Standalone module entry point used by ``sbt gui annotate``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anndata", required=True)
    parser.add_argument("--basis", default="X_umap")
    parser.add_argument("--key-added", default="manual_population")
    parser.add_argument("--color")
    parser.add_argument("--source-obs")
    parser.add_argument("--recipe", help="Replay this JSON recipe without a window.")
    parser.add_argument("--output", help="New H5AD file for recipe replay.")
    parser.add_argument("--overwrite-obs", action="store_true")
    args = parser.parse_args()
    if bool(args.recipe) != bool(args.output):
        parser.error("--recipe and --output must be supplied together.")
    if args.overwrite_obs and not args.recipe:
        parser.error("--overwrite-obs requires --recipe.")
    import anndata as ad

    if args.recipe:
        from .napari_sbt.population_curation import atomic_write_curated_anndata

        if Path(args.output).exists():
            parser.error("Refusing to overwrite the output H5AD file.")
        data = ad.read_h5ad(args.anndata)
        key = apply_annotation_recipe(data, args.recipe, overwrite=args.overwrite_obs)
        atomic_write_curated_anndata(data, Path(args.output))
        print(f"Applied annotation recipe to obs[{key!r}]; saved {args.output}")
        return
    annotate_embedding(
        ad.read_h5ad(args.anndata),
        basis=args.basis,
        key_added=args.key_added,
        color=args.color,
        source_obs=args.source_obs,
        block=True,
    )


if __name__ == "__main__":
    main()
