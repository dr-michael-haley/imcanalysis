"""Reusable categorical-colour selection and validation helpers.

The pure functions in this module deliberately do not import Qt.  They support
both Population naming and Dataset Maintenance, while the shared popup is built
by :mod:`SpatialBiologyToolkit.napari_sbt.app` only after Napari has started.
"""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence


_HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")


def normalise_hex_colour(value: object) -> str:
    """Return a canonical six-digit hexadecimal colour or an empty string."""

    text = str(value).strip()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", text):
        text = "#" + "".join(character * 2 for character in text[1:])
    if re.fullmatch(r"#[0-9a-fA-F]{8}", text):
        text = text[:7]
    if _HEX_COLOUR.fullmatch(text):
        return text.lower()
    try:
        from matplotlib.colors import to_hex

        converted = to_hex(value, keep_alpha=False)
    except (ImportError, TypeError, ValueError):
        return ""
    return converted.lower() if _HEX_COLOUR.fullmatch(converted) else ""


def contrasting_text_colour(value: object) -> str:
    """Choose readable black or white text for a hexadecimal background."""

    colour = normalise_hex_colour(value)
    if not colour:
        return "#111827"
    red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
    # WCAG-style relative brightness is sufficient for compact table swatches.
    brightness = (299 * red + 587 * green + 114 * blue) / 1000
    return "#111827" if brightness >= 150 else "#ffffff"


def categorical_colour_collisions(
    labels: Sequence[object], colours: Sequence[object]
) -> dict[str, list[str]]:
    """Return colours used by more than one *different* final label.

    Repeated rows with the same final label are intentional merges and therefore
    do not count as collisions.
    """

    owners: dict[str, set[str]] = defaultdict(set)
    for label, colour in zip(labels, colours, strict=False):
        name = str(label).strip()
        canonical = normalise_hex_colour(colour)
        if name and canonical:
            owners[canonical].add(name)
    return {
        colour: sorted(names, key=str.casefold)
        for colour, names in owners.items()
        if len(names) > 1
    }


def ordered_category_names(
    labels: Iterable[object],
    counts: Mapping[str, int] | None,
    mode: str,
) -> list[str]:
    """Return unique category names in one user-facing assignment order."""

    names = list(
        dict.fromkeys(
            str(label).strip() for label in labels if str(label).strip()
        )
    )
    abundance = {str(key): int(value) for key, value in (counts or {}).items()}
    if mode == "abundance_desc":
        return sorted(
            names,
            key=lambda name: (-abundance.get(name, 0), name.casefold()),
        )
    if mode == "abundance_asc":
        return sorted(names, key=lambda name: (abundance.get(name, 0), name.casefold()))
    if mode == "alphabetical_desc":
        return sorted(names, key=str.casefold, reverse=True)
    if mode == "alphabetical_asc":
        return sorted(names, key=str.casefold)
    raise ValueError(f"Unknown colour-assignment order: {mode!r}.")


def assign_categorical_colours(
    labels: Iterable[object],
    counts: Mapping[str, int] | None,
    colours: Iterable[object],
    *,
    order: str = "abundance_desc",
) -> dict[str, str]:
    """Assign distinct selected colours to distinct final labels."""

    names = ordered_category_names(labels, counts, order)
    selected = list(
        dict.fromkeys(
            canonical
            for value in colours
            if (canonical := normalise_hex_colour(value))
        )
    )
    if not names:
        raise ValueError("There are no named populations to colour.")
    if len(selected) < len(names):
        raise ValueError(
            f"Select at least {len(names)} distinct colours for {len(names)} "
            f"populations; only {len(selected)} are currently enabled."
        )
    # Palettes commonly contain many more colours than the current categories.
    # Only the first required enabled colours are assigned; the remaining palette
    # entries stay available if more categories are added later.
    return dict(zip(names, selected[: len(names)], strict=True))


def categorical_palette_catalog() -> OrderedDict[str, tuple[str, ...]]:
    """Return common Scanpy and Matplotlib categorical palettes.

    Scanpy palettes are used when Scanpy is installed.  The Matplotlib palettes
    are loaded lazily so importing NapariSBT's non-GUI services stays lightweight.
    """

    palettes: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    try:
        from scanpy.plotting import palettes as scanpy_palettes

        for label, attribute in (
            ("Scanpy default 20", "default_20"),
            ("Scanpy default 28", "default_28"),
            ("Scanpy default 102", "default_102"),
        ):
            values = getattr(scanpy_palettes, attribute, None)
            if values:
                palettes[label] = tuple(
                    colour
                    for value in values
                    if (colour := normalise_hex_colour(value))
                )
    except ImportError:
        pass

    try:
        from matplotlib import colormaps

        for name in (
            "tab10",
            "tab20",
            "Set1",
            "Set2",
            "Set3",
            "Paired",
            "Dark2",
            "Accent",
            "Pastel1",
            "Pastel2",
        ):
            cmap = colormaps[name]
            values = getattr(cmap, "colors", None)
            if values is None:
                continue
            from matplotlib.colors import to_hex

            palettes[f"Matplotlib {name}"] = tuple(
                normalise_hex_colour(to_hex(value)) for value in values
            )
    except ImportError:
        pass

    # This fallback also makes the helper useful in lightweight installations.
    if not palettes:
        palettes["Built-in categorical 20"] = (
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
            "#c49c94",
            "#f7b6d2",
            "#c7c7c7",
            "#dbdb8d",
            "#9edae5",
        )

    combined = tuple(
        dict.fromkeys(
            colour for values in palettes.values() for colour in values
        )
    )
    palettes["All available categorical colours"] = combined
    palettes.move_to_end("All available categorical colours", last=False)
    return palettes
