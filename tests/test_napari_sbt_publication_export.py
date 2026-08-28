from __future__ import annotations

import numpy as np
import pytest

from SpatialBiologyToolkit.napari_sbt import publication_export
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
    publication_channel_colours,
    publication_render_geometry,
    publication_resolution_scale,
    resolve_publication_dpi,
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


def test_schema_v1_v2_output_remains_custom_sized() -> None:
    output = PublicationOutput.model_validate({"width": 900, "height": 700})

    assert output.resolution == "custom"
    assert output.size_mode == "custom"
    assert resolve_publication_output_size(
        output,
        ResolvedPublicationFrame(50, 50, 100, 100),
    ) == (900, 700)


def test_schema_v3_preset_receives_backward_compatible_annotation_defaults() -> None:
    payload = _preset().model_dump(mode="json")
    payload["schema_version"] = 3
    for key in (
        "show_label",
        "label_scale",
        "thickness_scale",
        "margin_scale",
        "box_padding_scale",
    ):
        payload["scale_bar"].pop(key, None)
    for key in (
        "title_scale",
        "roi_scale",
        "channel_scale",
        "margin_scale",
        "color_channels",
        "box_padding_scale",
    ):
        payload["annotations"].pop(key, None)

    restored = PublicationExportPreset.model_validate(payload)

    assert restored.scale_bar.show_label is True
    assert restored.scale_bar.label_scale == 1.0
    assert restored.scale_bar.thickness_scale == 1.0
    assert restored.annotations.title_scale == 1.0
    assert restored.annotations.roi_scale == 1.0
    assert restored.annotations.channel_scale == 1.0
    assert restored.annotations.color_channels is False


@pytest.mark.parametrize(
    ("resolution", "factor", "dpi"),
    [("low", 1, 150), ("medium", 2, 300), ("high", 4, 600)],
)
def test_simple_resolution_scales_pixels_annotations_and_dpi_together(
    resolution: str,
    factor: int,
    dpi: int,
) -> None:
    output = PublicationOutput(resolution=resolution)
    frame = ResolvedPublicationFrame(200, 250, 400, 500)
    width, height = resolve_publication_output_size(output, frame)

    assert (width, height) == (500 * factor, 400 * factor)
    assert publication_resolution_scale(output) == factor
    assert resolve_publication_dpi(output) == dpi
    assert width / dpi == pytest.approx(500 / 150)
    assert height / dpi == pytest.approx(400 / 150)


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


def test_scale_bar_can_render_without_physical_length_text() -> None:
    preset = _preset(
        calibration=PixelCalibration(confirmed=True, x_size=1, y_size=1),
        scale_bar=PublicationScaleBar(
            visible=True,
            length_mode="fixed",
            length=20,
            show_label=False,
        ),
    )
    image = np.zeros((600, 800, 4), dtype=np.uint8)
    image[..., 3] = 255

    _composed, metadata = compose_publication_image(
        image,
        preset=preset,
        frame=ResolvedPublicationFrame(50, 50, 100, 100),
        roi="ROI-A",
    )

    assert metadata["scale_bar_label_visible"] is False
    assert metadata["scale_bar_label"] is None
    assert metadata["scale_bar_rendered_font_size"] == 0
    assert metadata["scale_bar_rendered_text_width"] == 0
    assert metadata["scale_bar_rendered_text_height"] == 0


def test_relative_controls_multiply_automatic_annotation_sizes() -> None:
    preset = _preset(
        output=PublicationOutput(resolution="low"),
        calibration=PixelCalibration(confirmed=True, x_size=1, y_size=1),
        scale_bar=PublicationScaleBar(
            visible=True,
            length_mode="fixed",
            length=50,
            label_scale=1.5,
            thickness_scale=2.0,
            margin_scale=0.5,
        ),
        annotations=PublicationAnnotations(
            custom_title="Lymphocytes",
            show_roi=True,
            show_channels=True,
            title_scale=2.0,
            roi_scale=0.5,
            channel_scale=1.5,
            margin_scale=0.5,
        ),
    )
    image = np.zeros((600, 600, 4), dtype=np.uint8)
    image[..., 3] = 255

    _composed, metadata = compose_publication_image(
        image,
        preset=preset,
        frame=ResolvedPublicationFrame(300, 300, 600, 600),
        roi="ROI-A",
    )

    assert metadata["scale_bar_rendered_font_size"] == 24
    assert metadata["scale_bar_rendered_thickness"] == 6
    assert metadata["scale_bar_rendered_margin"] == 9
    assert metadata["annotation_rendered_font_size"] == 18
    assert metadata["annotation_rendered_font_sizes"] == {
        "title": 36,
        "roi": 9,
        "channels": 27,
    }
    assert metadata["annotation_rendered_margin"] == 9


def test_channel_name_colours_follow_frozen_recipe_colormaps() -> None:
    recipe = ExploreViewRecipe(
        image_mode="six_colour",
        image_channels=["CD3", "CD8", "CD20"],
        layer_colormaps={"image::CD8": "magenta"},
        layer_colormap_specs={
            "image::CD3": {
                "kind": "continuous",
                "name": "custom-CD3",
                "colours": [[0, 0, 0, 1], [0.1, 0.2, 0.3, 1]],
                "controls": [0, 1],
                "interpolation": "linear",
            }
        },
    )

    colours = publication_channel_colours(recipe)

    assert colours == {
        "CD3": "#1a334c",
        "CD8": "#ff00ff",
        "CD20": "#0000ff",
    }

    preset = _preset(
        recipe=recipe,
        annotations=PublicationAnnotations(
            show_channels=True,
            color_channels=True,
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
    assert metadata["annotation_channel_colours"] == colours
    assert np.any(np.all(composed[..., :3] == [26, 51, 76], axis=-1))
    assert np.any(np.all(composed[..., :3] == [255, 0, 255], axis=-1))


@pytest.mark.parametrize(
    (
        "resolution",
        "size",
        "scale_font",
        "annotation_font",
        "thickness",
        "margin",
    ),
    [
        ("low", 600, 16, 18, 3, 18),
        ("medium", 1200, 32, 36, 6, 36),
        ("high", 2400, 64, 72, 12, 72),
    ],
)
def test_simple_resolutions_choose_proportional_readable_annotation_styling(
    resolution: str,
    size: int,
    scale_font: int,
    annotation_font: int,
    thickness: int,
    margin: int,
) -> None:
    preset = _preset(
        output=PublicationOutput(resolution=resolution),
        calibration=PixelCalibration(confirmed=True, x_size=1, y_size=1),
        scale_bar=PublicationScaleBar(
            visible=True,
            length_mode="fixed",
            length=50,
            thickness=99,
            font_size=99,
            margin=99,
        ),
        annotations=PublicationAnnotations(
            show_roi=True,
            font_size=99,
            margin=99,
        ),
    )
    image = np.zeros((size, size, 4), dtype=np.uint8)
    image[..., 3] = 255

    _composed, metadata = compose_publication_image(
        image,
        preset=preset,
        frame=ResolvedPublicationFrame(300, 300, 600, 600),
        roi="ROI-A",
    )

    assert metadata["annotation_style_profile"] == "automatic"
    assert metadata["scale_bar_output_pixels"] == size // 12
    assert metadata["scale_bar_rendered_font_size"] == scale_font
    assert metadata["scale_bar_rendered_thickness"] == thickness
    assert metadata["scale_bar_rendered_margin"] == margin
    assert metadata["annotation_rendered_font_size"] == annotation_font
    assert metadata["annotation_rendered_margin"] == margin


def test_publication_font_fallback_honours_requested_pixel_size() -> None:
    small = publication_export._font(20)
    large = publication_export._font(40)
    small_box = small.getbbox("Scale bar 50 µm")
    large_box = large.getbbox("Scale bar 50 µm")
    small_width = small_box[2] - small_box[0]
    large_width = large_box[2] - large_box[0]
    small_height = small_box[3] - small_box[1]
    large_height = large_box[3] - large_box[1]

    assert large_width >= small_width * 1.8
    assert large_height >= small_height * 1.8


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
    preset = _preset(
        scale_bar=PublicationScaleBar(
            show_label=False,
            label_scale=1.4,
            thickness_scale=1.8,
        ),
        annotations=PublicationAnnotations(
            title_scale=1.6,
            roi_scale=0.8,
            channel_scale=1.2,
            color_channels=True,
        ),
    )
    state = PublicationExportState(
        presets={preset.preset_id: preset}, active_preset_id=preset.preset_id
    )
    restored = PublicationExportState.model_validate(state.model_dump(mode="json"))
    assert restored.presets[preset.preset_id].recipe.image_channels == ["CD3", "CD8"]
    assert restored.active_preset_id == preset.preset_id
    restored_preset = restored.presets[preset.preset_id]
    assert restored_preset.scale_bar.show_label is False
    assert restored_preset.scale_bar.label_scale == 1.4
    assert restored_preset.scale_bar.thickness_scale == 1.8
    assert restored_preset.annotations.title_scale == 1.6
    assert restored_preset.annotations.roi_scale == 0.8
    assert restored_preset.annotations.channel_scale == 1.2
    assert restored_preset.annotations.color_channels is True


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
