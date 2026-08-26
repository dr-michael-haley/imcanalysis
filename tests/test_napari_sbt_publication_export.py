from __future__ import annotations

import numpy as np
import pytest

from SpatialBiologyToolkit.napari_sbt.explore import ExploreViewRecipe
from SpatialBiologyToolkit.napari_sbt.publication_export import (
    PixelCalibration,
    PublicationAnnotations,
    PublicationExportPreset,
    PublicationExportState,
    PublicationFrame,
    PublicationOutput,
    PublicationScaleBar,
    ResolvedPublicationFrame,
    build_publication_filename,
    camera_frame_from_canvas,
    compose_publication_image,
    detect_tiff_pixel_calibration,
    fit_frame_to_aspect,
    publication_render_geometry,
    resolve_publication_frame,
    resolve_publication_output_size,
    resolve_scale_bar_length,
    save_publication_image,
)


def _preset(**updates) -> PublicationExportPreset:
    values = {
        "preset_id": "preset-1",
        "name": "T cell panel",
        "source_recipe_name": "T cell verification",
        "recipe": ExploreViewRecipe(
            image_mode="six_colour",
            image_channels=["CD3", "CD8"],
        ),
        "output": PublicationOutput(width=800, height=600),
    }
    values.update(updates)
    return PublicationExportPreset(**values)


def test_camera_frame_is_canvas_independent() -> None:
    frame = camera_frame_from_canvas(
        center=(0.0, 40.0, 60.0),
        zoom=2.0,
        canvas_width=800,
        canvas_height=600,
    )
    assert frame == ResolvedPublicationFrame(
        center_y=40.0,
        center_x=60.0,
        field_height=300.0,
        field_width=400.0,
    )


def test_frame_crop_and_pad_never_stretch() -> None:
    source = ResolvedPublicationFrame(50, 100, 100, 200)
    cropped = fit_frame_to_aspect(
        source, output_width=100, output_height=100, mode="crop"
    )
    padded = fit_frame_to_aspect(
        source, output_width=100, output_height=100, mode="pad"
    )
    assert (cropped.field_height, cropped.field_width) == (100, 100)
    assert (padded.field_height, padded.field_width) == (200, 200)


def test_full_roi_frame_uses_target_aspect() -> None:
    output = PublicationOutput(width=800, height=600)
    frame = resolve_publication_frame(
        PublicationFrame(mode="full_roi", aspect_mode="crop"),
        output=output,
        current_frame=None,
        roi_shape=(100, 200),
    )
    assert frame.center_y == 49.5
    assert frame.center_x == 99.5
    assert frame.field_height == 100
    assert frame.field_width == pytest.approx(100 * 4 / 3)


def test_native_output_preserves_source_frame_and_pixel_sampling() -> None:
    output = PublicationOutput(
        size_mode="native",
        width=3000,
        height=3000,
    )
    frame = resolve_publication_frame(
        PublicationFrame(mode="full_roi", aspect_mode="crop"),
        output=output,
        current_frame=None,
        roi_shape=(100, 200),
    )

    assert frame.field_height == 100
    assert frame.field_width == 200
    assert resolve_publication_output_size(output, frame) == (200, 100)


def test_schema_v1_output_remains_custom_sized() -> None:
    output = PublicationOutput.model_validate({"width": 900, "height": 700})

    assert output.size_mode == "custom"
    assert resolve_publication_output_size(
        output,
        ResolvedPublicationFrame(50, 50, 100, 100),
    ) == (900, 700)


def test_render_geometry_changes_sampling_not_field_of_view() -> None:
    frame = ResolvedPublicationFrame(250, 250, 500, 500)
    small = publication_render_geometry(
        frame,
        output_width=500,
        output_height=500,
        supersampling=1,
        device_pixel_ratio=1.0,
    )
    large = publication_render_geometry(
        frame,
        output_width=1000,
        output_height=1000,
        supersampling=1,
        device_pixel_ratio=1.0,
    )

    assert small.zoom == 1.0
    assert large.zoom == 2.0
    assert small.logical_canvas_width / small.zoom == 500
    assert large.logical_canvas_width / large.zoom == 500


def test_render_geometry_accounts_for_qt_device_pixel_ratio() -> None:
    geometry = publication_render_geometry(
        ResolvedPublicationFrame(250, 250, 500, 500),
        output_width=1000,
        output_height=1000,
        supersampling=1,
        device_pixel_ratio=2.0,
    )

    assert geometry.logical_canvas_width == 500
    assert geometry.logical_canvas_height == 500
    assert geometry.zoom == 1.0


def test_fixed_frame_requires_all_coordinates() -> None:
    with pytest.raises(ValueError, match="requires centre"):
        PublicationFrame(mode="fixed", center_y=1, center_x=2)


def test_filename_always_contains_roi_and_channels() -> None:
    preset = _preset(
        output=PublicationOutput(
            width=800,
            height=600,
            filename_template="{recipe}",
        )
    )
    filename = build_publication_filename(preset, roi="Sample 1")
    assert "Sample-1" in filename
    assert "CD3-CD8" in filename
    assert filename.endswith(".png")


def test_filename_can_use_resolved_native_dimensions() -> None:
    preset = _preset(
        output=PublicationOutput(
            size_mode="native",
            width=2400,
            height=1800,
        )
    )

    filename = build_publication_filename(
        preset,
        roi="Sample 1",
        output_size=(1000, 750),
    )

    assert "1000x750" in filename


def test_scale_bar_requires_confirmed_calibration() -> None:
    frame = ResolvedPublicationFrame(50, 50, 100, 100)
    settings = PublicationScaleBar(visible=True)
    with pytest.raises(ValueError, match="Confirm"):
        resolve_scale_bar_length(
            settings,
            calibration=PixelCalibration(confirmed=False),
            frame=frame,
            output_width=1000,
        )
    physical, pixels = resolve_scale_bar_length(
        settings,
        calibration=PixelCalibration(confirmed=True, x_size=1.0),
        frame=frame,
        output_width=1000,
    )
    assert physical == 20
    assert pixels == 200


def test_native_scale_bar_uses_source_pixel_calibration() -> None:
    frame = ResolvedPublicationFrame(250, 250, 500, 500)
    output = PublicationOutput(size_mode="native")
    output_width, _output_height = resolve_publication_output_size(output, frame)

    physical, pixels = resolve_scale_bar_length(
        PublicationScaleBar(visible=True, length_mode="fixed", length=50),
        calibration=PixelCalibration(confirmed=True, x_size=1.0, unit="µm"),
        frame=frame,
        output_width=output_width,
    )

    assert physical == 50
    assert pixels == 50


def test_compositor_preserves_size_and_records_annotations() -> None:
    preset = _preset(
        calibration=PixelCalibration(confirmed=True, x_size=1, y_size=1),
        scale_bar=PublicationScaleBar(
            visible=True,
            length_mode="fixed",
            length=20,
            thickness=3,
            font_size=16,
            margin=10,
        ),
        annotations=PublicationAnnotations(
            show_roi=True,
            show_channels=True,
            font_size=16,
            margin=10,
        ),
    )
    image = np.zeros((600, 800, 4), dtype=np.uint8)
    image[..., 3] = 255
    composed, metadata = compose_publication_image(
        image,
        preset=preset,
        frame=ResolvedPublicationFrame(50, 50, 100, 100),
        roi="ROI-A",
    )
    assert composed.shape == (600, 800, 4)
    assert metadata["scale_bar_physical_length"] == 20
    assert metadata["scale_bar_output_pixels"] == 160
    assert metadata["annotation_text"] == ["ROI-A", "CD3 · CD8"]
    assert np.any(composed[..., :3] != 0)


def test_atomic_png_save(tmp_path) -> None:
    from PIL import Image

    image = np.zeros((20, 30, 4), dtype=np.uint8)
    image[..., 3] = 255
    destination = tmp_path / "figure.png"
    saved = save_publication_image(
        image,
        destination,
        dpi=300,
        metadata={"roi": "ROI-A"},
    )
    assert saved == destination.resolve()
    assert saved.is_file()
    assert not list(tmp_path.glob(".*.tmp.png"))
    with Image.open(saved) as opened:
        assert opened.info["dpi"][0] == pytest.approx(300, rel=1e-3)
        assert opened.info["dpi"][1] == pytest.approx(300, rel=1e-3)


def test_saved_publication_preset_keeps_frozen_recipe_snapshot() -> None:
    preset = _preset()
    state = PublicationExportState(
        presets={preset.preset_id: preset}, active_preset_id=preset.preset_id
    )
    restored = PublicationExportState.model_validate(
        state.model_dump(mode="json")
    )
    assert restored.presets[preset.preset_id].recipe.image_channels == ["CD3", "CD8"]
    assert restored.active_preset_id == preset.preset_id


def test_detects_calibrated_tiff_resolution_without_auto_confirming(tmp_path) -> None:
    from tifffile import imwrite

    path = tmp_path / "calibrated.tiff"
    # 10,000 pixels per centimetre is exactly 1 micrometre per pixel.
    imwrite(
        path,
        np.zeros((10, 10), dtype=np.uint8),
        resolution=(10_000, 10_000),
        resolutionunit="CENTIMETER",
    )
    detected = detect_tiff_pixel_calibration(path)
    assert detected is not None
    calibration, source = detected
    assert calibration.x_size == pytest.approx(1.0)
    assert calibration.y_size == pytest.approx(1.0)
    assert calibration.unit == "µm"
    assert calibration.confirmed is False
    assert "TIFF" in source
