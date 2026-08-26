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

from .explore import ExploreViewRecipe

PUBLICATION_EXPORT_SCHEMA_VERSION = 2
DEFAULT_FILENAME_TEMPLATE = "{roi}__{recipe}__{channels}__{width}x{height}"
_HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$")
_FILENAME_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


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
    position: Literal[
        "bottom_right", "bottom_left", "top_right", "top_left"
    ] = "bottom_right"
    color: str = "#ffffff"
    thickness: int = Field(default=5, ge=1, le=100)
    font_size: int = Field(default=28, ge=6, le=300)
    margin: int = Field(default=30, ge=0, le=1000)
    ticks: bool = True
    box: bool = True
    box_color: str = "#000000a6"
    box_padding: int = Field(default=12, ge=0, le=500)

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
    box: bool = True
    box_color: str = "#000000a6"
    box_padding: int = Field(default=12, ge=0, le=500)

    @field_validator("color", "box_color")
    @classmethod
    def _valid_colour(cls, value: str) -> str:
        text = str(value).strip()
        if not _HEX_COLOUR.fullmatch(text):
            raise ValueError("Colours must use #RRGGBB or #RRGGBBAA notation.")
        return text.lower()


class PublicationOutput(BaseModel):
    """Raster-output and file-naming settings."""

    # ``custom`` is the compatibility default for schema-v1 presets, which did
    # not persist a sizing mode.  The GUI defaults new, unsaved presets to
    # ``native`` so one output pixel represents one source-image pixel.
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
    annotations: PublicationAnnotations = Field(
        default_factory=PublicationAnnotations
    )
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

    if output.size_mode == "native":
        return (
            max(1, int(round(frame.field_width))),
            max(1, int(round(frame.field_height))),
        )
    return int(output.width), int(output.height)


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
    if output.size_mode == "native":
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
        if (
            micrometres_per_unit is None
            or x_resolution <= 0
            or y_resolution <= 0
        ):
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


def _rgba(value: str) -> tuple[int, int, int, int]:
    text = value.lstrip("#")
    if len(text) == 6:
        text += "ff"
    return tuple(int(text[index : index + 2], 16) for index in range(0, 8, 2))


def _font(size: int):
    from PIL import ImageFont

    for candidate in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, int(size))
        except OSError:
            continue
    return ImageFont.load_default()


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
    metadata: dict[str, Any] = {}

    scale_bar = preset.scale_bar
    if scale_bar.visible:
        physical_length, bar_width = resolve_scale_bar_length(
            scale_bar,
            calibration=preset.calibration,
            frame=frame,
            output_width=width,
        )
        bar_width = max(1, bar_width)
        if bar_width >= width - 2 * scale_bar.margin:
            raise ValueError(
                "The requested scale bar is wider than the exported field of view."
            )
        label = f"{physical_length:g} {preset.calibration.unit}"
        font = _font(scale_bar.font_size)
        draw = ImageDraw.Draw(canvas, "RGBA")
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        content_width = max(bar_width, text_width)
        content_height = text_height + scale_bar.box_padding + scale_bar.thickness
        left = scale_bar.position.endswith("left")
        top = scale_bar.position.startswith("top")
        x0 = scale_bar.margin if left else width - scale_bar.margin - content_width
        y0 = scale_bar.margin if top else height - scale_bar.margin - content_height
        if scale_bar.box:
            pad = scale_bar.box_padding
            draw.rounded_rectangle(
                (x0 - pad, y0 - pad, x0 + content_width + pad, y0 + content_height + pad),
                radius=max(2, pad // 2),
                fill=_rgba(scale_bar.box_color),
            )
        text_x = x0 + (content_width - text_width) / 2
        line_x = x0 + (content_width - bar_width) / 2
        if top:
            line_y = y0
            text_y = y0 + scale_bar.thickness + scale_bar.box_padding
        else:
            text_y = y0
            line_y = y0 + text_height + scale_bar.box_padding
        colour = _rgba(scale_bar.color)
        draw.rectangle(
            (
                line_x,
                line_y,
                line_x + bar_width,
                line_y + scale_bar.thickness,
            ),
            fill=colour,
        )
        if scale_bar.ticks:
            tick = max(scale_bar.thickness * 3, scale_bar.thickness + 2)
            for tick_x in (line_x, line_x + bar_width):
                draw.rectangle(
                    (
                        tick_x - scale_bar.thickness / 2,
                        line_y - tick / 2,
                        tick_x + scale_bar.thickness / 2,
                        line_y + scale_bar.thickness + tick / 2,
                    ),
                    fill=colour,
                )
        draw.text((text_x, text_y), label, font=font, fill=colour)
        metadata["scale_bar_physical_length"] = physical_length
        metadata["scale_bar_output_pixels"] = bar_width

    annotations = preset.annotations
    lines = []
    if annotations.custom_title.strip():
        lines.append(annotations.custom_title.strip())
    if annotations.show_roi:
        lines.append(str(roi))
    if annotations.show_channels and preset.recipe.image_channels:
        lines.append(" · ".join(preset.recipe.image_channels))
    if lines:
        text = "\n".join(lines)
        draw = ImageDraw.Draw(canvas, "RGBA")
        font = _font(annotations.font_size)
        text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        left = annotations.position.endswith("left")
        top = annotations.position.startswith("top")
        x = annotations.margin if left else width - annotations.margin - text_width
        y = annotations.margin if top else height - annotations.margin - text_height
        if annotations.box:
            pad = annotations.box_padding
            draw.rounded_rectangle(
                (x - pad, y - pad, x + text_width + pad, y + text_height + pad),
                radius=max(2, pad // 2),
                fill=_rgba(annotations.box_color),
            )
        draw.multiline_text(
            (x, y),
            text,
            font=font,
            fill=_rgba(annotations.color),
            spacing=4,
            align="left" if left else "right",
        )
        metadata["annotation_text"] = lines

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
    "PixelCalibration",
    "PublicationAnnotations",
    "PublicationExportPreset",
    "PublicationExportState",
    "PublicationFrame",
    "PublicationOutput",
    "PublicationRenderGeometry",
    "PublicationScaleBar",
    "ResolvedPublicationFrame",
    "build_publication_filename",
    "camera_frame_from_canvas",
    "channel_filename_token",
    "compose_publication_image",
    "downsample_publication_image",
    "detect_tiff_pixel_calibration",
    "fit_frame_to_aspect",
    "publication_render_geometry",
    "resolve_publication_frame",
    "resolve_publication_output_size",
    "resolve_scale_bar_length",
    "save_publication_image",
]
