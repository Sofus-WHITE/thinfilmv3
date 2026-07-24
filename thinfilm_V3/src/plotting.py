"""Lightweight plotting utilities kept separate from optical physics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from .colorimetry import perceived_color_from_result
from .results import SimulationResult
from .angle_sweep import AngleSweepResult
from .thickness_sweep import ThicknessSweep1DResult, ThicknessSweep2DResult


def plot_reflectance(
    result: SimulationResult,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot reflectance and perceived color, and optionally save the figure."""

    perceived_color = perceived_color_from_result(result)

    fig, (swatch_ax, ax) = plt.subplots(
        2,
        1,
        figsize=(7, 5.4),
        gridspec_kw={"height_ratios": [1.35, 2.0], "hspace": 0.32},
    )

    fig.suptitle(
        f"{result.stack_summary}\nReflected spectrum under D65 illumination",
        fontsize=12,
        y=0.98,
    )
    swatch_ax.set_facecolor(perceived_color.srgb)
    swatch_ax.set_xticks([])
    swatch_ax.set_yticks([])
    swatch_ax.set_frame_on(True)
    for spine in swatch_ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")
    swatch_ax.text(
        0.5,
        0.5,
        f"Perceived color: {perceived_color.hex}  RGB{perceived_color.srgb_255}",
        ha="center",
        va="center",
        color=_readable_text_color(perceived_color.srgb),
        fontsize=10,
        transform=swatch_ax.transAxes,
    )

    _fill_under_curve_with_wavelength_colors(ax, result)
    ax.plot(result.wavelengths_nm, result.reflectance, color="black", linewidth=2.2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_xlim(float(result.wavelengths_nm.min()), float(result.wavelengths_nm.max()))
    ymax = max(float(result.reflectance.max()) * 1.06, 0.05)
    ax.set_ylim(0, min(ymax, 1.05))
    ax.grid(False)
    fig.subplots_adjust(top=0.86)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    return fig, ax


def _readable_text_color(srgb: tuple[float, float, float]) -> str:
    """Choose black or white text for contrast on a color swatch."""

    r, g, b = srgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.55 else "white"


def _fill_under_curve_with_wavelength_colors(
    ax,
    result: SimulationResult,
) -> None:
    """Fill the reflectance curve with approximate visible wavelength colors."""

    wavelengths = np.asarray(result.wavelengths_nm, dtype=float)
    reflectance = np.asarray(result.reflectance, dtype=float)

    for left, right, y_left, y_right in zip(
        wavelengths[:-1],
        wavelengths[1:],
        reflectance[:-1],
        reflectance[1:],
    ):
        center_wavelength = 0.5 * (left + right)
        polygon = Polygon(
            [(left, 0.0), (left, y_left), (right, y_right), (right, 0.0)],
            closed=True,
            facecolor=_wavelength_to_rgb(center_wavelength),
            edgecolor="none",
        )
        ax.add_patch(polygon)


def _wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    """Approximate display RGB for a visible wavelength."""

    wavelength = float(wavelength_nm)
    if wavelength < 380 or wavelength > 780:
        return (0.0, 0.0, 0.0)

    if wavelength < 440:
        red = -(wavelength - 440) / (440 - 380)
        green = 0.0
        blue = 1.0
    elif wavelength < 490:
        red = 0.0
        green = (wavelength - 440) / (490 - 440)
        blue = 1.0
    elif wavelength < 510:
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510) / (510 - 490)
    elif wavelength < 580:
        red = (wavelength - 510) / (580 - 510)
        green = 1.0
        blue = 0.0
    elif wavelength < 645:
        red = 1.0
        green = -(wavelength - 645) / (645 - 580)
        blue = 0.0
    else:
        red = 1.0
        green = 0.0
        blue = 0.0

    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength <= 700:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)

    gamma = 0.8
    return tuple((max(channel, 0.0) * factor) ** gamma for channel in (red, green, blue))


def plot_thickness_sweep_1d(
    result: ThicknessSweep1DResult,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot a 1D thickness sweep as a predicted-colour strip."""

    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    image = result.rgb_values[np.newaxis, :, :]
    ax.imshow(
        image,
        aspect="auto",
        extent=[
            float(result.thickness_values_nm[0]),
            float(result.thickness_values_nm[-1]),
            0.0,
            1.0,
        ],
        origin="lower",
    )
    ax.set_yticks([])
    ax.set_xlabel(f"{result.layer_name} thickness (nm)")
    ax.set_title(
        f"Predicted colour vs {result.layer_name} thickness\n{result.stack_label}",
        fontsize=15,
        fontweight="semibold",
    )
    ax.xaxis.label.set_size(14)
    ax.tick_params(axis="both", labelsize=11.5, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.67)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    return fig, ax


def plot_thickness_sweep_2d(
    result: ThicknessSweep2DResult,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot a 2D thickness sweep as a predicted-colour map."""

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.imshow(
        result.rgb_grid,
        aspect="auto",
        origin="lower",
        extent=[
            float(result.thickness_values_1_nm[0]),
            float(result.thickness_values_1_nm[-1]),
            float(result.thickness_values_2_nm[0]),
            float(result.thickness_values_2_nm[-1]),
        ],
    )
    ax.set_xlabel(f"{result.layer_name_1} thickness (nm)")
    ax.set_ylabel(f"{result.layer_name_2} thickness (nm)")
    ax.set_title(
        f"Predicted colour map: {result.layer_name_1} vs {result.layer_name_2}\n"
        f"{result.stack_label}",
        fontsize=15,
        fontweight="semibold",
    )
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis="both", labelsize=11.5, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.12, top=0.82)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    return fig, ax


def plot_angle_sweep(
    result: AngleSweepResult,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot an angle sweep as a predicted-colour strip."""

    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    ax.imshow(
        result.rgb_values[np.newaxis, :, :],
        aspect="auto",
        extent=[
            float(result.angle_values_deg[0]),
            float(result.angle_values_deg[-1]),
            0.0,
            1.0,
        ],
        origin="lower",
    )
    ax.set_yticks([])
    ax.set_xlabel("Angle of incidence (deg)")
    ax.set_title(f"Predicted colour vs angle\n{result.stack_label}", fontsize=15, fontweight="semibold")
    ax.xaxis.label.set_size(14)
    ax.tick_params(axis="both", labelsize=11.5, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.67)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    return fig, ax
