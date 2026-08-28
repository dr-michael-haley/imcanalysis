"""Typed, reproducible publication-image export helpers for NapariSBT."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .explore import SIX_COLOUR_COLORMAPS, ExploreViewRecipe

PUBLICATION_EXPORT_SCHEMA_VERSION = 4
DEFAULT_FILENAME_TEMPLATE = "{roi}__{recipe}__{channels}__{width}x{height}"
PUBLICATION_RESOLUTION_FACTORS = {"low": 1, "medium": 2, "high": 4}
PUBLICATION_RESOLUTION_DPI = {"low": 150, "medium": 300, "high": 600}
_HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$")
_FILENAME_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")
_NAMED_CHANNEL_COLOURS = {
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "cyan": "#00ffff",
    "yellow": "#ffff00",
    "magenta": "#ff00ff",
    "gray": "#ffffff",
    "grey": "#ffffff",
    "white": "#ffffff",
}


class PixelCalibration(BaseModel):
    """Physical size represented by one source-image pixel."""

    confirmed: bool = False
    x_size: float = Field(default=1.0, gt=0)
    y_size: float = Field(default=1.0, gt=0)
    unit: str = "µm"

    @field_validator("unit")
    @classmethod
    def _unit_is_not_empty(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("A physical calibration unit is required.")
        return text


class PublicationFrame(BaseModel):
    """A reproducible two-dimensional camera field of view."""

    mode: Literal["current_view", "full_roi", "fixed"] = "current_view"
    center_y: float | None = None
    center_x: float | None = None
    field_height: float | None = Field(default=None, gt=0)
    field_width: float | None = Field(default=None, gt=0)
    aspect_mode: Literal["crop", "pad"] = "crop"

    @model_validator(mode="after")
    def _fixed_frame_is_complete(self):
        if self.mode == "fixed" and any(
            value is None
            for value in (
                self.center_y,
                self.center_x,
                self.field_height,
                self.field_width,
            )
        ):
            raise ValueError(
                "A fixed publication frame requires centre Y/X and field height/width."
            )
        return self


class PublicationScaleBar(BaseModel):
    """Scale-bar rendering settings in final output pixels/physical units."""

    visible: bool = False
    length_mode: Literal["auto", "fixed"] = "auto"
    length: float = Field(default=50.0, gt=0)
    target_fraction: float = Field(default=0.2, gt=0.05, le=0.5)
    position: Literal["bottom_right", "bottom_left", "top_right", "top_left"] = (
        "bottom_right"
    )
    color: str = "#ffffff"
    thickness: int = Field(default=5, ge=1, le=100)
    font_size: int = Field(default=28, ge=6, le=300)
    margin: int = Field(default=30, ge=0, le=1000)
    show_label: bool = True
    label_scale: float = Field(default=1.0, ge=0.1, le=5.0)
    thickness_scale: float = Field(default=1.0, ge=0.1, le=5.0)
    margin_scale: float = Field(default=1.0, ge=0.1, le=5.0)
    ticks: bool = True
    box: bool = True
    box_color: str = "#000000a6"
    box_padding: int = Field(default=12, ge=0, le=500)
    box_padding_scale: float = Field(default=1.0, ge=0.1, le=5.0)

    @field_validator("color", "box_color")
    @classmethod
    def _valid_colour(cls, value: str) -> str:
        text = str(value).strip()
        if not _HEX_COLOUR.fullmatch(text):
            raise ValueError("Colours must use #RRGGBB or #RRGGBBAA notation.")
        return text.lower()


class PublicationAnnotations(BaseModel):
    """Optional text burned into the rendered image."""

    show_roi: bool = False
    show_channels: bool = False
    custom_title: str = ""
    position: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = (
        "top_left"
    )
    color: str = "#ffffff"
    font_size: int = Field(default=28, ge=6, le=300)
    margin: int = Field(default=30, ge=0, le=1000)
    title_scale: float = Field(default=1.0, ge=0.1, le=5.0)
    roi_scale: float = Field(default=1.0, ge=0.1, le=5.0)
    channel_scale: float = Field(default=1.0, ge=0.1, le=5.0)
    margin_scale: float = Field(default=1.0, ge=0.1, le=5.0)
    color_channels: bool = False
    box: bool = True
    box_color: str = "#000000a6"
    box_padding: int = Field(default=12, ge=0, le=500)
    box_padding_scale: float = Field(default=1.0, ge=0.1, le=5.0)

    @field_validator("color", "box_color")
    @classmethod
    def _valid_colour(cls, value: str) -> str:
        text = str(value).strip()
        if not _HEX_COLOUR.fullmatch(text):
            raise ValueError("Colours must use #RRGGBB or #RRGGBBAA notation.")
        return text.lower()


class PublicationOutput(BaseModel):
    """Raster-output and file-naming settings."""

    # Schema-v1/v2 presets had no simple resolution level, so ``custom`` keeps
    # their exact output dimensions, DPI, supersampling and fixed-pixel styling.
    resolution: Literal["low", "medium", "high", "custom"] = "custom"
    # ``custom`` is also the compatibility default for schema-v1 presets, which
    # did not persist a sizing mode. The GUI explicitly creates new presets at
    # its recommended coordinated ``medium`` resolution.
    size_mode: Literal["native", "custom"] = "custom"
    width: int = Field(default=2400, ge=128, le=30000)
    height: int = Field(default=1800, ge=128, le=30000)
    supersampling: Literal[1, 2, 4] = 1
    format: Literal["png", "tiff", "jpeg"] = "png"
    dpi: int = Field(default=300, ge=30, le=2400)
    filename_template: str = DEFAULT_FILENAME_TEMPLATE

    @field_validator("filename_template")
    @classmethod
    def _filename_template_is_safe(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("The filename template cannot be empty.")
        allowed = {"roi", "recipe", "channels", "width", "height", "fingerprint"}
        formatter = re.compile(r"\{([^{}]+)\}")
        unknown = sorted(set(formatter.findall(text)) - allowed)
        if unknown:
            raise ValueError(
                "Unknown filename-template token(s): " + ", ".join(unknown)
            )
        return text

    @property
    def extension(self) -> str:
        return {"png": ".png", "tiff": ".tiff", "jpeg": ".jpg"}[self.format]


class PublicationExportPreset(BaseModel):
    """One frozen, portable publication-export definition."""

    schema_version: int = PUBLICATION_EXPORT_SCHEMA_VERSION
    preset_id: str
    name: str
    source_recipe_id: str | None = None
    source_recipe_name: str = "Current Explore view"
    recipe: ExploreViewRecipe
    frame: PublicationFrame = Field(default_factory=PublicationFrame)
    calibration: PixelCalibration = Field(default_factory=PixelCalibration)
    scale_bar: PublicationScaleBar = Field(default_factory=PublicationScaleBar)
    annotations: PublicationAnnotations = Field(default_factory=PublicationAnnotations)
    output: PublicationOutput = Field(default_factory=PublicationOutput)

    @field_validator("preset_id", "name")
    @classmethod
    def _identity_is_not_empty(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Publication preset IDs and names cannot be empty.")
        return text

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"preset_id", "name"})
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class PublicationExportState(BaseModel):
    """Small persisted catalogue kept outside the hot Explore review state."""

    schema_version: int = PUBLICATION_EXPORT_SCHEMA_VERSION
    presets: dict[str, PublicationExportPreset] = Field(default_factory=dict)
    active_preset_id: str | None = None

    @model_validator(mode="after")
    def _active_preset_exists(self):
        if self.active_preset_id not in self.presets:
            self.active_preset_id = None
        return self


@dataclass(frozen=True)
class ResolvedPublicationFrame:
    """Concrete field-of-view coordinates used for one rendered ROI."""

    center_y: float
    center_x: float
    field_height: float
    field_width: float

    def as_dict(self) -> dict[str, float]:
        return {
            "center_y": float(self.center_y),
            "center_x": float(self.center_x),
            "field_height": float(self.field_height),
            "field_width": float(self.field_width),
        }


@dataclass(frozen=True)
class PublicationRenderGeometry:
    """Physical output and logical Qt-canvas geometry for one render."""

    output_width: int
    output_height: int
    render_width: int
    render_height: int
    logical_canvas_width: int
    logical_canvas_height: int
    zoom: float


def resolve_publication_output_size(
    output: PublicationOutput,
    frame: ResolvedPublicationFrame,
) -> tuple[int, int]:
    """Resolve final raster dimensions without changing the field of view."""

    resolution_factor = PUBLICATION_RESOLUTION_FACTORS.get(output.resolution)
    if resolution_factor is not None:
        return (
            max(1, int(round(frame.field_width * resolution_factor))),
            max(1, int(round(frame.field_height * resolution_factor))),
        )
    if output.size_mode == "native":
        return (
            max(1, int(round(frame.field_width))),
            max(1, int(round(frame.field_height))),
        )
    return int(output.width), int(output.height)


def publication_resolution_scale(output: PublicationOutput) -> float:
    """Return the coordinated raster/annotation scale for an output preset."""

    return float(PUBLICATION_RESOLUTION_FACTORS.get(output.resolution, 1))


def automatic_publication_style(
    *, output_width: int, output_height: int
) -> dict[str, int]:
    """Choose readable annotation geometry from the final raster dimensions."""

    reference = max(1, min(int(output_width), int(output_height)))
    return {
        "scale_bar_font_size": max(10, int(round(reference * 0.026667))),
        "annotation_font_size": max(10, int(round(reference * 0.03))),
        "scale_bar_thickness": max(2, int(round(reference * 0.005))),
        "margin": max(8, int(round(reference * 0.03))),
        "scale_bar_box_padding": max(4, int(round(reference * 0.01))),
        "annotation_box_padding": max(4, int(round(reference * 0.013333))),
        "tick_extension": max(1, int(round(reference * 0.003333))),
        "line_spacing": max(2, int(round(reference * 0.006667))),
    }


def resolve_publication_dpi(output: PublicationOutput) -> int:
    """Return the print DPI represented by a simple or legacy output preset."""

    return int(PUBLICATION_RESOLUTION_DPI.get(output.resolution, output.dpi))


def publication_render_geometry(
    frame: ResolvedPublicationFrame,
    *,
    output_width: int,
    output_height: int,
    supersampling: int,
    device_pixel_ratio: float,
) -> PublicationRenderGeometry:
    """Return a DPR-aware canvas and zoom for an exact source-pixel frame.

    Napari defines zoom using the *current logical canvas size*.  Callers must
    therefore resize the canvas to these logical dimensions before applying
    ``zoom``; setting zoom before ``viewer.screenshot(size=...)`` couples the
    captured field of view to the previous on-screen canvas size.
    """

    if output_width <= 0 or output_height <= 0 or supersampling <= 0:
        raise ValueError("Publication output dimensions must be positive.")
    if device_pixel_ratio <= 0 or not math.isfinite(device_pixel_ratio):
        raise ValueError("The Qt device-pixel ratio must be positive.")
    render_width = int(output_width) * int(supersampling)
    render_height = int(output_height) * int(supersampling)
    logical_width = max(1, int(round(render_width / device_pixel_ratio)))
    logical_height = max(1, int(round(render_height / device_pixel_ratio)))
    return PublicationRenderGeometry(
        output_width=int(output_width),
        output_height=int(output_height),
        render_width=render_width,
        render_height=render_height,
        logical_canvas_width=logical_width,
        logical_canvas_height=logical_height,
        zoom=min(
            logical_width / float(frame.field_width),
            logical_height / float(frame.field_height),
        ),
    )


def camera_frame_from_canvas(
    *, center: tuple[float, ...], zoom: float, canvas_width: float, canvas_height: float
) -> ResolvedPublicationFrame:
    """Convert a Napari 2-D camera into a canvas-size-independent field of view."""

    if zoom <= 0 or canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Camera zoom and canvas dimensions must be positive.")
    if len(center) < 2:
        raise ValueError("A two-dimensional camera centre is required.")
    return ResolvedPublicationFrame(
        center_y=float(center[-2]),
        center_x=float(center[-1]),
        field_height=float(canvas_height) / float(zoom),
        field_width=float(canvas_width) / float(zoom),
    )


def fit_frame_to_aspect(
    frame: ResolvedPublicationFrame,
    *,
    output_width: int,
    output_height: int,
    mode: Literal["crop", "pad"],
) -> ResolvedPublicationFrame:
    """Crop or expand a field of view to an exact output aspect ratio."""

    target_aspect = float(output_width) / float(output_height)
    frame_aspect = float(frame.field_width) / float(frame.field_height)
    height = float(frame.field_height)
    width = float(frame.field_width)
    if mode == "crop":
        if frame_aspect > target_aspect:
            width = height * target_aspect
        else:
            height = width / target_aspect
    elif mode == "pad":
        if frame_aspect > target_aspect:
            height = width / target_aspect
        else:
            width = height * target_aspect
    else:  # pragma: no cover - Pydantic prevents this for normal callers
        raise ValueError(f"Unsupported aspect mode: {mode}")
    return ResolvedPublicationFrame(
        center_y=frame.center_y,
        center_x=frame.center_x,
        field_height=height,
        field_width=width,
    )


def resolve_publication_frame(
    setting: PublicationFrame,
    *,
    output: PublicationOutput,
    current_frame: ResolvedPublicationFrame | None,
    roi_shape: tuple[int, int],
) -> ResolvedPublicationFrame:
    """Resolve current/full/fixed settings and enforce the requested aspect."""

    height, width = (int(roi_shape[0]), int(roi_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("The current ROI has no renderable two-dimensional extent.")
    if setting.mode == "full_roi":
        frame = ResolvedPublicationFrame(
            center_y=(height - 1) / 2.0,
            center_x=(width - 1) / 2.0,
            field_height=float(height),
            field_width=float(width),
        )
    elif setting.mode == "current_view":
        if current_frame is None:
            raise ValueError("Capture the current Napari view before exporting it.")
        frame = current_frame
    else:
        frame = ResolvedPublicationFrame(
            center_y=float(setting.center_y),
            center_x=float(setting.center_x),
            field_height=float(setting.field_height),
            field_width=float(setting.field_width),
        )
    if output.resolution != "custom" or output.size_mode == "native":
        return frame
    return fit_frame_to_aspect(
        frame,
        output_width=output.width,
        output_height=output.height,
        mode=setting.aspect_mode,
    )


def _slug(value: str, *, fallback: str = "item") -> str:
    text = _FILENAME_TOKEN.sub("-", str(value).strip()).strip("-._")
    return text or fallback


def detect_tiff_pixel_calibration(
    path: str | Path,
) -> tuple[PixelCalibration, str] | None:
    """Read trustworthy OME or TIFF-resolution calibration without guessing."""

    from xml.etree import ElementTree

    from tifffile import TiffFile

    source = Path(path).expanduser().resolve(strict=False)
    with TiffFile(source) as tif:
        ome = tif.ome_metadata
        if ome:
            try:
                root = ElementTree.fromstring(ome)
                pixels = next(
                    element
                    for element in root.iter()
                    if element.tag.rsplit("}", 1)[-1] == "Pixels"
                )
                x_value = pixels.attrib.get("PhysicalSizeX")
                y_value = pixels.attrib.get("PhysicalSizeY") or x_value
                x_unit = pixels.attrib.get("PhysicalSizeXUnit", "µm")
                y_unit = pixels.attrib.get("PhysicalSizeYUnit", x_unit)
                if x_value and y_value and x_unit == y_unit:
                    return (
                        PixelCalibration(
                            confirmed=False,
                            x_size=float(x_value),
                            y_size=float(y_value),
                            unit=x_unit,
                        ),
                        "OME PhysicalSizeX/PhysicalSizeY metadata",
                    )
            except (StopIteration, ValueError, ElementTree.ParseError):
                pass

        page = tif.pages[0]
        x_tag = page.tags.get("XResolution")
        y_tag = page.tags.get("YResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        if x_tag is None or y_tag is None or unit_tag is None:
            return None

        def resolution_value(value: Any) -> float:
            if isinstance(value, tuple) and len(value) == 2:
                return float(value[0]) / float(value[1])
            return float(value)

        x_resolution = resolution_value(x_tag.value)
        y_resolution = resolution_value(y_tag.value)
        unit_value = unit_tag.value
        unit_number = int(getattr(unit_value, "value", unit_value))
        micrometres_per_unit = {2: 25_400.0, 3: 10_000.0}.get(unit_number)
        if micrometres_per_unit is None or x_resolution <= 0 or y_resolution <= 0:
            return None
        return (
            PixelCalibration(
                confirmed=False,
                x_size=micrometres_per_unit / x_resolution,
                y_size=micrometres_per_unit / y_resolution,
                unit="µm",
            ),
            "TIFF XResolution/YResolution and ResolutionUnit tags",
        )


def channel_filename_token(channels: list[str], *, maximum: int = 96) -> str:
    """Return an informative channel token with a stable suffix when shortened."""

    if not channels:
        return "no-channels"
    token = "-".join(_slug(channel, fallback="channel") for channel in channels)
    if len(token) <= maximum:
        return token
    digest = hashlib.sha256("\0".join(channels).encode("utf-8")).hexdigest()[:10]
    return token[: maximum - 12].rstrip("-._") + "--" + digest


def build_publication_filename(
    preset: PublicationExportPreset,
    *,
    roi: str,
    output_size: tuple[int, int] | None = None,
) -> str:
    """Build a safe filename that always retains the ROI and channel identity."""

    width, height = output_size or (preset.output.width, preset.output.height)
    values = {
        "roi": _slug(roi, fallback="roi"),
        "recipe": _slug(preset.source_recipe_name, fallback="recipe"),
        "channels": channel_filename_token(list(preset.recipe.image_channels)),
        "width": str(int(width)),
        "height": str(int(height)),
        "fingerprint": preset.fingerprint[:10],
    }
    try:
        stem = preset.output.filename_template.format(**values)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid publication filename template: {exc}") from exc
    stem = _slug(stem, fallback="napari_sbt_export")[:220].rstrip("-._")
    # Even a custom template cannot accidentally remove the two basic identities.
    if values["roi"].casefold() not in stem.casefold():
        stem = f"{values['roi']}__{stem}"
    if values["channels"].casefold() not in stem.casefold():
        stem = f"{stem}__{values['channels']}"
    return stem + preset.output.extension


def _nice_scale_length(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        raise ValueError("The automatic scale-bar target must be positive.")
    exponent = math.floor(math.log10(value))
    scaled = value / (10**exponent)
    factor = 1.0
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if candidate <= scaled:
            factor = candidate
    return factor * (10**exponent)


def resolve_scale_bar_length(
    settings: PublicationScaleBar,
    *,
    calibration: PixelCalibration,
    frame: ResolvedPublicationFrame,
    output_width: int,
) -> tuple[float, int]:
    """Return scale length in physical units and final-output pixels."""

    if not calibration.confirmed:
        raise ValueError(
            "Confirm the source-image pixel calibration before drawing a scale bar."
        )
    physical_width = frame.field_width * calibration.x_size
    physical_length = (
        _nice_scale_length(physical_width * settings.target_fraction)
        if settings.length_mode == "auto"
        else float(settings.length)
    )
    pixels = int(round(physical_length / physical_width * int(output_width)))
    return float(physical_length), pixels


def _rgba_values_to_hex(value: Any) -> str | None:
    """Convert one serialized RGB(A) value into an opaque text colour."""

    try:
        rgba = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if rgba.size not in {3, 4} or not np.isfinite(rgba).all():
        return None
    if rgba.size == 4 and rgba[3] <= 0:
        return None
    rgb = np.clip(np.rint(rgba[:3] * 255), 0, 255).astype(np.uint8)
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _representative_colormap_colour(spec: dict[str, Any] | None) -> str | None:
    """Choose the visible high-value colour from a frozen continuous colormap."""

    if not spec or spec.get("kind") != "continuous":
        return None
    colours = spec.get("colours", [])
    if not isinstance(colours, list):
        return None
    for value in reversed(colours):
        colour = _rgba_values_to_hex(value)
        if colour is not None:
            return colour
    return None


def publication_channel_colours(recipe: ExploreViewRecipe) -> dict[str, str]:
    """Resolve channel-name colours from the exact frozen Explore recipe.

    Scalar layers prefer their serialized continuous-colormap endpoint, then a
    saved named colormap, and finally the colour implied by the image mode. RGB
    composites use their red/green/blue component roles. The mapping therefore
    remains reproducible during bulk export and does not depend on the live
    viewer having the same ROI open.
    """

    resolved: dict[str, str] = {}
    for index, channel in enumerate(recipe.image_channels):
        layer_name = f"image::{channel}"
        colour = _representative_colormap_colour(
            recipe.layer_colormap_specs.get(layer_name)
        )
        named = recipe.layer_colormaps.get(layer_name, "").strip().lower()
        if colour is None:
            if _HEX_COLOUR.fullmatch(named):
                colour = named[:7]
            else:
                colour = _NAMED_CHANNEL_COLOURS.get(named)
        if colour is None and recipe.image_mode == "rgb":
            colour = ("#ff0000", "#00ff00", "#0000ff")[index % 3]
        if colour is None and recipe.image_mode == "six_colour":
            default_name = SIX_COLOUR_COLORMAPS[index % len(SIX_COLOUR_COLORMAPS)]
            colour = _NAMED_CHANNEL_COLOURS[default_name]
        if colour is None and recipe.image_mode == "grayscale":
            colour = "#ffffff"
        resolved[str(channel)] = colour or "#ffffff"
    return resolved


def _rgba(value: str) -> tuple[int, int, int, int]:
    text = value.lstrip("#")
    if len(text) == 6:
        text += "ff"
    return tuple(int(text[index : index + 2], 16) for index in range(0, 8, 2))


def _font(size: int):
    from PIL import ImageFont

    requested_size = max(1, int(size))
    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = (
        Path("DejaVuSans.ttf"),
        Path("Arial.ttf"),
        windows_root / "Fonts" / "arial.ttf",
        windows_root / "Fonts" / "segoeui.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), requested_size)
        except OSError:
            continue
    # Pillow's scalable embedded Aileron font is available through ``size``.
    # Calling load_default() without it returns a fixed-size bitmap font, which
    # made high-resolution exports retain tiny scale-bar and annotation text.
    return ImageFont.load_default(size=requested_size)


def compose_publication_image(
    image: np.ndarray,
    *,
    preset: PublicationExportPreset,
    frame: ResolvedPublicationFrame,
    roi: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw deterministic publication annotations onto an RGBA screenshot."""

    from PIL import Image, ImageDraw

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("Publication screenshots must be RGB or RGBA arrays.")
    canvas = Image.fromarray(array.astype(np.uint8)).convert("RGBA")
    width, height = canvas.size
    style_scale = publication_resolution_scale(preset.output)
    automatic_style = preset.output.resolution in PUBLICATION_RESOLUTION_FACTORS
    style = automatic_publication_style(
        output_width=width,
        output_height=height,
    )

    def scaled(value: int, *, minimum: int = 0) -> int:
        return max(minimum, int(round(int(value) * style_scale)))

    def relative(value: int, factor: float, *, minimum: int = 0) -> int:
        return max(minimum, int(round(int(value) * float(factor))))

    metadata: dict[str, Any] = {
        "annotation_style_scale": style_scale,
        "annotation_style_profile": "automatic" if automatic_style else "custom",
    }

    scale_bar = preset.scale_bar
    if scale_bar.visible:
        physical_length, bar_width = resolve_scale_bar_length(
            scale_bar,
            calibration=preset.calibration,
            frame=frame,
            output_width=width,
        )
        bar_width = max(1, bar_width)
        base_margin = style["margin"] if automatic_style else scaled(scale_bar.margin)
        base_thickness = (
            style["scale_bar_thickness"]
            if automatic_style
            else scaled(scale_bar.thickness, minimum=1)
        )
        base_font_size = (
            style["scale_bar_font_size"]
            if automatic_style
            else scaled(scale_bar.font_size, minimum=1)
        )
        base_box_padding = (
            style["scale_bar_box_padding"]
            if automatic_style
            else scaled(scale_bar.box_padding)
        )
        margin = relative(base_margin, scale_bar.margin_scale)
        thickness = relative(base_thickness, scale_bar.thickness_scale, minimum=1)
        font_size = relative(base_font_size, scale_bar.label_scale, minimum=1)
        box_padding = relative(base_box_padding, scale_bar.box_padding_scale)
        if bar_width >= width - 2 * margin:
            raise ValueError(
                "The requested scale bar is wider than the exported field of view."
            )
        label = f"{physical_length:g} {preset.calibration.unit}"
        draw = ImageDraw.Draw(canvas, "RGBA")
        font = _font(font_size) if scale_bar.show_label else None
        if scale_bar.show_label:
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
        else:
            text_width = 0
            text_height = 0
        content_width = max(bar_width, text_width)
        label_gap = box_padding if scale_bar.show_label else 0
        content_height = text_height + label_gap + thickness
        left = scale_bar.position.endswith("left")
        top = scale_bar.position.startswith("top")
        x0 = margin if left else width - margin - content_width
        y0 = margin if top else height - margin - content_height
        if scale_bar.box:
            pad = box_padding
            draw.rounded_rectangle(
                (
                    x0 - pad,
                    y0 - pad,
                    x0 + content_width + pad,
                    y0 + content_height + pad,
                ),
                radius=max(2, pad // 2),
                fill=_rgba(scale_bar.box_color),
            )
        text_x = x0 + (content_width - text_width) / 2
        line_x = x0 + (content_width - bar_width) / 2
        if top:
            line_y = y0
            text_y = y0 + thickness + label_gap
        else:
            text_y = y0
            line_y = y0 + text_height + label_gap
        colour = _rgba(scale_bar.color)
        draw.rectangle(
            (
                line_x,
                line_y,
                line_x + bar_width,
                line_y + thickness,
            ),
            fill=colour,
        )
        if scale_bar.ticks:
            base_tick_extension = (
                style["tick_extension"] if automatic_style else scaled(2, minimum=1)
            )
            tick_extension = relative(
                base_tick_extension, scale_bar.thickness_scale, minimum=1
            )
            tick = max(thickness * 3, thickness + tick_extension)
            for tick_x in (line_x, line_x + bar_width):
                draw.rectangle(
                    (
                        tick_x - thickness / 2,
                        line_y - tick / 2,
                        tick_x + thickness / 2,
                        line_y + thickness + tick / 2,
                    ),
                    fill=colour,
                )
        if scale_bar.show_label:
            draw.text((text_x, text_y), label, font=font, fill=colour)
        metadata["scale_bar_physical_length"] = physical_length
        metadata["scale_bar_output_pixels"] = bar_width
        metadata["scale_bar_label_visible"] = scale_bar.show_label
        metadata["scale_bar_label"] = label if scale_bar.show_label else None
        metadata["scale_bar_rendered_font_size"] = (
            font_size if scale_bar.show_label else 0
        )
        metadata["scale_bar_rendered_text_width"] = text_width
        metadata["scale_bar_rendered_text_height"] = text_height
        metadata["scale_bar_rendered_thickness"] = thickness
        metadata["scale_bar_rendered_margin"] = margin

    annotations = preset.annotations
    annotation_text: list[str] = []
    annotation_lines: list[dict[str, Any]] = []
    base_font_size = (
        style["annotation_font_size"]
        if automatic_style
        else scaled(annotations.font_size, minimum=1)
    )
    line_definitions: list[tuple[str, str, float]] = []
    if annotations.custom_title.strip():
        line_definitions.append(
            ("title", annotations.custom_title.strip(), annotations.title_scale)
        )
    if annotations.show_roi:
        line_definitions.append(("roi", str(roi), annotations.roi_scale))
    if annotations.show_channels and preset.recipe.image_channels:
        line_definitions.append(
            (
                "channels",
                " · ".join(preset.recipe.image_channels),
                annotations.channel_scale,
            )
        )
    if line_definitions:
        draw = ImageDraw.Draw(canvas, "RGBA")
        default_colour = annotations.color
        channel_colours = (
            publication_channel_colours(preset.recipe)
            if annotations.color_channels
            else {}
        )
        rendered_font_sizes: dict[str, int] = {}
        for line_kind, text, font_scale in line_definitions:
            font_size = relative(base_font_size, font_scale, minimum=1)
            rendered_font_sizes[line_kind] = font_size
            font = _font(font_size)
            if line_kind == "channels" and annotations.color_channels:
                segments = []
                for index, channel in enumerate(preset.recipe.image_channels):
                    if index:
                        segments.append((" · ", default_colour, font))
                    segments.append((str(channel), channel_colours[str(channel)], font))
            else:
                segments = [(text, default_colour, font)]
            measured_segments = []
            line_width = 0
            line_height = 0
            for segment_text, segment_colour, segment_font in segments:
                bounds = draw.textbbox((0, 0), segment_text, font=segment_font)
                segment_width = bounds[2] - bounds[0]
                segment_height = bounds[3] - bounds[1]
                measured_segments.append(
                    {
                        "text": segment_text,
                        "colour": segment_colour,
                        "font": segment_font,
                        "bounds": bounds,
                        "width": segment_width,
                    }
                )
                line_width += segment_width
                line_height = max(line_height, segment_height)
            annotation_text.append(text)
            annotation_lines.append(
                {
                    "segments": measured_segments,
                    "width": line_width,
                    "height": line_height,
                }
            )

        base_margin = style["margin"] if automatic_style else scaled(annotations.margin)
        base_box_padding = (
            style["annotation_box_padding"]
            if automatic_style
            else scaled(annotations.box_padding)
        )
        margin = relative(base_margin, annotations.margin_scale)
        box_padding = relative(base_box_padding, annotations.box_padding_scale)
        spacing = style["line_spacing"] if automatic_style else scaled(4, minimum=1)
        text_width = max(line["width"] for line in annotation_lines)
        text_height = sum(line["height"] for line in annotation_lines) + spacing * (
            len(annotation_lines) - 1
        )
        left = annotations.position.endswith("left")
        top = annotations.position.startswith("top")
        x = margin if left else width - margin - text_width
        y = margin if top else height - margin - text_height
        if annotations.box:
            pad = box_padding
            draw.rounded_rectangle(
                (x - pad, y - pad, x + text_width + pad, y + text_height + pad),
                radius=max(2, pad // 2),
                fill=_rgba(annotations.box_color),
            )
        line_y = y
        for line in annotation_lines:
            cursor_x = x if left else x + text_width - line["width"]
            for segment in line["segments"]:
                bounds = segment["bounds"]
                draw.text(
                    (cursor_x - bounds[0], line_y - bounds[1]),
                    segment["text"],
                    font=segment["font"],
                    fill=_rgba(segment["colour"]),
                )
                cursor_x += segment["width"]
            line_y += line["height"] + spacing
        metadata["annotation_text"] = annotation_text
        metadata["annotation_rendered_font_size"] = base_font_size
        metadata["annotation_rendered_font_sizes"] = rendered_font_sizes
        metadata["annotation_channel_colours"] = channel_colours
        metadata["annotation_rendered_text_width"] = text_width
        metadata["annotation_rendered_text_height"] = text_height
        metadata["annotation_rendered_margin"] = margin

    return np.asarray(canvas), metadata


def downsample_publication_image(
    image: np.ndarray, *, width: int, height: int
) -> np.ndarray:
    """Resize one rendered image to its final dimensions with Lanczos filtering."""

    from PIL import Image

    canvas = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGBA")
    if canvas.size != (int(width), int(height)):
        canvas = canvas.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    return np.asarray(canvas)


def save_publication_image(
    image: np.ndarray,
    destination: str | Path,
    *,
    dpi: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save PNG, TIFF, or JPEG without exposing partial files."""

    from PIL import Image, PngImagePlugin

    destination = Path(destination).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    canvas = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGBA")
    suffix = destination.suffix.lower()
    save_kwargs: dict[str, Any] = {"dpi": (int(dpi), int(dpi))}
    if suffix == ".png":
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("napari_sbt", json.dumps(metadata or {}, sort_keys=True))
        save_kwargs["pnginfo"] = png_info
        format_name = "PNG"
    elif suffix in {".tif", ".tiff"}:
        save_kwargs["compression"] = "tiff_deflate"
        format_name = "TIFF"
    elif suffix in {".jpg", ".jpeg"}:
        canvas = canvas.convert("RGB")
        save_kwargs.update({"quality": 95, "subsampling": 0})
        format_name = "JPEG"
    else:
        raise ValueError(f"Unsupported publication image suffix: {suffix}")
    try:
        canvas.save(temporary, format=format_name, **save_kwargs)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "DEFAULT_FILENAME_TEMPLATE",
    "PUBLICATION_EXPORT_SCHEMA_VERSION",
    "PUBLICATION_RESOLUTION_DPI",
    "PUBLICATION_RESOLUTION_FACTORS",
    "PixelCalibration",
    "PublicationAnnotations",
    "PublicationExportPreset",
    "PublicationExportState",
    "PublicationFrame",
    "PublicationOutput",
    "PublicationRenderGeometry",
    "PublicationScaleBar",
    "ResolvedPublicationFrame",
    "automatic_publication_style",
    "build_publication_filename",
    "camera_frame_from_canvas",
    "channel_filename_token",
    "compose_publication_image",
    "downsample_publication_image",
    "detect_tiff_pixel_calibration",
    "fit_frame_to_aspect",
    "publication_render_geometry",
    "publication_channel_colours",
    "publication_resolution_scale",
    "resolve_publication_dpi",
    "resolve_publication_frame",
    "resolve_publication_output_size",
    "resolve_scale_bar_length",
    "save_publication_image",
]
