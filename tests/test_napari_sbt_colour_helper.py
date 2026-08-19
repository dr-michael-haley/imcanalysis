from __future__ import annotations

import pytest

from SpatialBiologyToolkit.napari_sbt.colour_helper import (
    assign_categorical_colours,
    categorical_colour_collisions,
    contrasting_text_colour,
    normalise_hex_colour,
)


def test_colour_collisions_ignore_rows_belonging_to_one_explicit_merge():
    collisions = categorical_colour_collisions(
        ["T cell", "T cell", "B cell", "Myeloid"],
        ["#ff0000", "#ff0000", "#00ff00", "#00ff00"],
    )

    assert collisions == {"#00ff00": ["B cell", "Myeloid"]}


def test_colour_assignment_supports_abundance_and_reverse_alphabetical_order():
    counts = {"B": 20, "A": 5, "C": 10}

    abundant = assign_categorical_colours(
        ["A", "B", "C"],
        counts,
        ["#111111", "#222222", "#333333"],
        order="abundance_desc",
    )
    reverse = assign_categorical_colours(
        ["A", "B", "C"],
        counts,
        ["#111111", "#222222", "#333333"],
        order="alphabetical_desc",
    )

    assert list(abundant) == ["B", "C", "A"]
    assert list(reverse) == ["C", "B", "A"]


def test_colour_assignment_refuses_palette_reuse():
    with pytest.raises(ValueError, match="Select at least 3 distinct colours"):
        assign_categorical_colours(
            ["A", "B", "C"],
            {},
            ["#111111", "#111111", "#222222"],
        )


def test_colour_assignment_accepts_a_palette_longer_than_the_categories():
    assignment = assign_categorical_colours(
        ["A", "B"],
        {"A": 20, "B": 10},
        ["#111111", "#222222", "#333333", "#444444"],
    )

    assert assignment == {"A": "#111111", "B": "#222222"}


def test_colour_normalisation_and_contrasting_text_are_deterministic():
    assert normalise_hex_colour("#AbC") == "#aabbcc"
    assert contrasting_text_colour("#ffffff") == "#111827"
    assert contrasting_text_colour("#000000") == "#ffffff"
