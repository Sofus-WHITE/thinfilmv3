"""Efficient 1D and 2D thickness sweeps using prepared TMM stacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .color import ColorConversionCache, prepare_color_conversion, reflectance_to_srgb
from .stack import ThinFilmStack
from .tmm_model import PreparedTMMStack, TMMModel


QUALITY_MODES = {
    "fast": {"wavelength_step_nm": 10.0, "points_1d": 60, "points_2d": 30},
    "normal": {"wavelength_step_nm": 5.0, "points_1d": 100, "points_2d": 40},
    "high_quality": {"wavelength_step_nm": 1.0, "points_1d": 200, "points_2d": 50},
}


@dataclass(frozen=True)
class ThicknessSweep1DResult:
    """Colour result from varying one selected finite layer thickness."""

    thickness_values_nm: NDArray[np.float64]
    rgb_values: NDArray[np.float64]
    layer_name: str
    layer_index: int
    stack_label: str
    wavelengths_nm: NDArray[np.float64]
    reflectance_spectra: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ThicknessSweep2DResult:
    """Colour result from varying two selected finite layer thicknesses."""

    thickness_values_1_nm: NDArray[np.float64]
    thickness_values_2_nm: NDArray[np.float64]
    rgb_grid: NDArray[np.float64]
    layer_name_1: str
    layer_name_2: str
    layer_index_1: int
    layer_index_2: int
    stack_label: str
    wavelengths_nm: NDArray[np.float64]
    reflectance_data: NDArray[np.float64] | None = None


def wavelength_grid_for_quality(quality: str = "normal") -> NDArray[np.float64]:
    """Return a visible wavelength grid for a named quality mode."""

    settings = _quality_settings(quality)
    return np.arange(400.0, 700.0 + settings["wavelength_step_nm"], settings["wavelength_step_nm"])


def run_thickness_sweep_1d(
    stack: ThinFilmStack,
    model: TMMModel,
    layer: int | str,
    thickness_min_nm: float,
    thickness_max_nm: float,
    angle_deg: float,
    wavelengths_nm: ArrayLike | None = None,
    num_points: int | None = None,
    quality: str = "normal",
    layer_occurrence: int = 0,
    save_reflectance: bool = False,
) -> ThicknessSweep1DResult:
    """Vary one layer thickness and return predicted colour at each sweep point."""

    wavelengths = _resolve_wavelengths(wavelengths_nm, quality)
    point_count = num_points or int(_quality_settings(quality)["points_1d"])
    thickness_values = np.linspace(thickness_min_nm, thickness_max_nm, point_count, dtype=float)
    prepared = model.prepare_stack(stack, wavelengths)
    color_cache = prepare_color_conversion(wavelengths)
    layer_index = prepared.finite_layer_index(layer, occurrence=layer_occurrence)
    layer_name = prepared.layer_names[layer_index]

    rgb_values = np.empty((point_count, 3), dtype=float)
    reflectance_spectra = (
        np.empty((point_count, wavelengths.size), dtype=float) if save_reflectance else None
    )
    d_list = prepared.base_d_list.copy()

    for i, thickness_nm in enumerate(thickness_values):
        d_list[layer_index] = thickness_nm
        reflectance = model.reflectance_from_prepared(prepared, d_list, angle_deg)
        rgb_values[i] = reflectance_to_srgb(reflectance, cache=color_cache)
        if reflectance_spectra is not None:
            reflectance_spectra[i] = reflectance

    return ThicknessSweep1DResult(
        thickness_values_nm=thickness_values,
        rgb_values=rgb_values,
        layer_name=layer_name,
        layer_index=layer_index,
        stack_label=_format_variable_stack_label(prepared, {layer_index: "x"}),
        wavelengths_nm=wavelengths,
        reflectance_spectra=reflectance_spectra,
    )


def run_thickness_sweep_2d(
    stack: ThinFilmStack,
    model: TMMModel,
    layer_1: int | str,
    layer_2: int | str,
    thickness_1_min_nm: float,
    thickness_1_max_nm: float,
    thickness_2_min_nm: float,
    thickness_2_max_nm: float,
    angle_deg: float,
    wavelengths_nm: ArrayLike | None = None,
    num_points_1: int | None = None,
    num_points_2: int | None = None,
    quality: str = "normal",
    layer_1_occurrence: int = 0,
    layer_2_occurrence: int = 0,
    save_reflectance: bool = False,
) -> ThicknessSweep2DResult:
    """Vary two layer thicknesses and return a predicted-colour image grid."""

    wavelengths = _resolve_wavelengths(wavelengths_nm, quality)
    settings = _quality_settings(quality)
    count_1 = num_points_1 or int(settings["points_2d"])
    count_2 = num_points_2 or int(settings["points_2d"])
    thickness_values_1 = np.linspace(thickness_1_min_nm, thickness_1_max_nm, count_1, dtype=float)
    thickness_values_2 = np.linspace(thickness_2_min_nm, thickness_2_max_nm, count_2, dtype=float)

    prepared = model.prepare_stack(stack, wavelengths)
    color_cache = prepare_color_conversion(wavelengths)
    layer_index_1 = prepared.finite_layer_index(layer_1, occurrence=layer_1_occurrence)
    layer_index_2 = prepared.finite_layer_index(layer_2, occurrence=layer_2_occurrence)
    if layer_index_1 == layer_index_2:
        raise ValueError("2D sweep requires two different finite layers.")

    rgb_grid = np.empty((count_2, count_1, 3), dtype=float)
    reflectance_data = (
        np.empty((count_2, count_1, wavelengths.size), dtype=float) if save_reflectance else None
    )
    d_list = prepared.base_d_list.copy()

    for y_index, thickness_2_nm in enumerate(thickness_values_2):
        d_list[layer_index_2] = thickness_2_nm
        for x_index, thickness_1_nm in enumerate(thickness_values_1):
            d_list[layer_index_1] = thickness_1_nm
            reflectance = model.reflectance_from_prepared(prepared, d_list, angle_deg)
            rgb_grid[y_index, x_index] = reflectance_to_srgb(reflectance, cache=color_cache)
            if reflectance_data is not None:
                reflectance_data[y_index, x_index] = reflectance

    return ThicknessSweep2DResult(
        thickness_values_1_nm=thickness_values_1,
        thickness_values_2_nm=thickness_values_2,
        rgb_grid=rgb_grid,
        layer_name_1=prepared.layer_names[layer_index_1],
        layer_name_2=prepared.layer_names[layer_index_2],
        layer_index_1=layer_index_1,
        layer_index_2=layer_index_2,
        stack_label=_format_variable_stack_label(
            prepared,
            {layer_index_1: "x", layer_index_2: "y"},
        ),
        wavelengths_nm=wavelengths,
        reflectance_data=reflectance_data,
    )


def prepare_sweep_stack(
    stack: ThinFilmStack,
    model: TMMModel,
    wavelengths_nm: ArrayLike,
) -> PreparedTMMStack:
    """Expose the prepared-stack workflow for future optimization code."""

    return model.prepare_stack(stack, wavelengths_nm)


def _resolve_wavelengths(
    wavelengths_nm: ArrayLike | None,
    quality: str,
) -> NDArray[np.float64]:
    """Use explicit wavelengths or the grid implied by a quality mode."""

    if wavelengths_nm is not None:
        return np.asarray(wavelengths_nm, dtype=float)
    return wavelength_grid_for_quality(quality)


def _quality_settings(quality: str) -> dict[str, float]:
    """Return settings for a quality mode."""

    try:
        return QUALITY_MODES[quality]
    except KeyError as exc:
        available = ", ".join(QUALITY_MODES)
        raise ValueError(f"Unknown quality mode {quality!r}. Use one of: {available}") from exc


def _format_variable_stack_label(
    prepared: PreparedTMMStack,
    variable_layers: dict[int, str],
) -> str:
    """Write the optical stack with selected layer thicknesses marked as variables."""

    parts: list[str] = []
    for index, name in enumerate(prepared.layer_names):
        if index not in prepared.display_layer_indices and index not in (0, len(prepared.layer_names) - 1):
            continue

        if index == 0 or index == len(prepared.layer_names) - 1:
            parts.append(name)
            continue

        variable = variable_layers.get(index)
        if variable is None:
            thickness = prepared.base_d_list[index]
            parts.append(f"{thickness:g} nm {name}")
        else:
            parts.append(f"{variable} nm {name}")
    return " | ".join(parts)
