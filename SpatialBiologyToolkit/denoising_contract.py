"""Small compatibility boundary shared by IMC-Denoise integrations."""

from __future__ import annotations


DEFAULT_WEIGHTS_NAME_TEMPLATE = "weights_{channel}.weights.h5"
SUPPORTED_LOSS_FUNCTIONS = ("I_divergence", "mse", "mse_relu")
SUPPORTED_NETWORK_SIZES = ("small", "normal")


def resolve_weights_name(template: str, channel_name: str) -> str:
    """Render one DeepSNiF checkpoint name without sharing models between channels."""
    if "{channel}" not in template:
        raise ValueError(
            "denoising.weights_name_template must include the '{channel}' placeholder."
        )
    try:
        return template.format(channel=channel_name)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            "denoising.weights_name_template must be a valid Python format string "
            "using only the '{channel}' placeholder."
        ) from exc


def validate_deepsnif_options(loss_function: str, network_size: str) -> None:
    """Fail before patch generation when a config is outside IMC-Denoise's API."""
    if loss_function not in SUPPORTED_LOSS_FUNCTIONS:
        raise ValueError(
            "loss_function must be one of "
            f"{SUPPORTED_LOSS_FUNCTIONS}; got {loss_function!r}."
        )
    if network_size not in SUPPORTED_NETWORK_SIZES:
        raise ValueError(
            "network_size must be one of "
            f"{SUPPORTED_NETWORK_SIZES}; got {network_size!r}."
        )
