"""Colour-target search for thin-film stack thicknesses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .color import prepare_color_conversion, reflectance_to_srgb, reflectance_to_xyz
from .experiments import COLOUR_METRIC_CIE76, delta_e_colour, normalise_colour_metric, xyz_to_lab
from .stack import ThinFilmStack
from .tmm_model import TMMModel


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class TargetSearchCandidate:
    """One simulated stack candidate from a colour-target thickness search."""

    thicknesses_nm: tuple[float, ...]
    layer_names: tuple[str, ...]
    reflectance: NDArray[np.float64]
    xyz: tuple[float, float, float]
    lab: tuple[float, float, float]
    srgb: tuple[float, float, float]
    delta_e: float
    score: float
    whiteness_index: float = float("nan")

    @property
    def srgb_255(self) -> tuple[int, int, int]:
        """Return display RGB as integer channels."""

        return tuple(int(round(channel * 255.0)) for channel in self.srgb)


@dataclass(frozen=True)
class TargetSearchResult:
    """Ranked output from a colour-target thickness search."""

    target_xyz: tuple[float, float, float]
    target_lab: tuple[float, float, float]
    wavelengths_nm: NDArray[np.float64]
    stack_label: str
    candidates: tuple[TargetSearchCandidate, ...]
    evaluated_count: int
    iterations: int
    colour_metric: str = COLOUR_METRIC_CIE76
    score_mode: str = "delta_e"

    @property
    def best(self) -> TargetSearchCandidate:
        """Return the best candidate according to the whiteness-aware score."""

        if not self.candidates:
            raise ValueError("No target-search candidates are available.")
        return self.candidates[0]


def search_thicknesses_for_target_colour(
    *,
    stack: ThinFilmStack,
    model: TMMModel,
    wavelengths_nm: ArrayLike,
    angle_deg: float,
    layer_bounds_nm: list[tuple[int, str, float, float]],
    target_xyz: tuple[float, float, float],
    iterations: int = 4,
    points_per_layer: int = 31,
    min_lightness: float = 92.0,
    brightness_weight: float = 0.35,
    strategy: str = "coordinate",
    top_n: int = 24,
    colour_metric: str = COLOUR_METRIC_CIE76,
    score_mode: str = "delta_e",
    progress_callback: ProgressCallback | None = None,
) -> TargetSearchResult:
    """Search selected deposited-layer thicknesses for a bright target colour.

    The stack is prepared once. The loop then edits only the numeric TMM
    thickness array, which keeps the search compatible with later optimizers.
    """

    if not layer_bounds_nm:
        raise ValueError("Select at least one layer to vary.")
    if points_per_layer < 2:
        raise ValueError("points_per_layer must be at least 2.")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    strategy_key = strategy.strip().lower().replace(" ", "_")
    if strategy_key not in {"coordinate", "full_grid"}:
        raise ValueError("Search strategy must be 'coordinate' or 'full_grid'.")
    score_key = score_mode.strip().lower().replace("-", "_").replace(" ", "_")
    if score_key not in {"delta_e", "wid_2016"}:
        raise ValueError("Search score mode must be 'delta_e' or 'wid_2016'.")

    metric = normalise_colour_metric(colour_metric)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    prepared = model.prepare_stack(stack, wavelengths)
    color_cache = prepare_color_conversion(wavelengths)
    target_lab_array = xyz_to_lab(target_xyz)
    target_lab = tuple(float(value) for value in target_lab_array)

    display_indices = prepared.display_layer_indices
    if not display_indices:
        raise ValueError("The stack has no deposited display layers to vary.")

    search_layers: list[tuple[int, str, float, float]] = []
    for display_index, layer_name, low, high in layer_bounds_nm:
        if display_index < 0 or display_index >= len(display_indices):
            raise ValueError(f"Layer index {display_index} is outside the deposited stack.")
        low_value = float(low)
        high_value = float(high)
        if high_value < low_value:
            low_value, high_value = high_value, low_value
        search_layers.append((display_indices[display_index], layer_name, low_value, high_value))

    d_list = np.asarray(prepared.base_d_list, dtype=float).copy()
    current = d_list.copy()
    layer_names = tuple(layer_name for _, layer_name, _, _ in search_layers)
    best_candidate = _evaluate_candidate(
        model=model,
        prepared=prepared,
        d_list=current,
        search_layers=search_layers,
        layer_names=layer_names,
        target_xyz=target_xyz,
        target_lab=target_lab_array,
        color_cache=color_cache,
        angle_deg=angle_deg,
        min_lightness=min_lightness,
        brightness_weight=brightness_weight,
        colour_metric=metric,
        score_mode=score_key,
    )
    candidates: list[TargetSearchCandidate] = [best_candidate]
    seen = {_thickness_key(best_candidate.thicknesses_nm)}
    evaluated = 1
    if strategy_key == "full_grid":
        value_axes = [np.linspace(low, high, points_per_layer) for _, _, low, high in search_layers]
        total = int(np.prod([axis.size for axis in value_axes]))
        candidates = []
        seen = set()
        evaluated = 0
        for values in product(*value_axes):
            trial = d_list.copy()
            for value, (tmm_index, _, _, _) in zip(values, search_layers):
                trial[tmm_index] = float(value)
            candidate = _evaluate_candidate(
                model=model,
                prepared=prepared,
                d_list=trial,
                search_layers=search_layers,
                layer_names=layer_names,
                target_xyz=target_xyz,
                target_lab=target_lab_array,
                color_cache=color_cache,
                angle_deg=angle_deg,
                min_lightness=min_lightness,
                brightness_weight=brightness_weight,
                colour_metric=metric,
                score_mode=score_key,
            )
            candidates.append(candidate)
            evaluated += 1
            if progress_callback is not None:
                progress_callback(evaluated, total, f"full grid point {evaluated:,}/{total:,}")
        ranked = tuple(sorted(candidates, key=lambda item: item.score)[:top_n])
        return TargetSearchResult(
            target_xyz=tuple(float(value) for value in target_xyz),
            target_lab=target_lab,
            wavelengths_nm=wavelengths,
            stack_label=stack.display_summary(),
            candidates=ranked,
            evaluated_count=evaluated,
            iterations=1,
            colour_metric=metric,
            score_mode=score_key,
        )

    total = 1 + iterations * len(search_layers) * points_per_layer

    for iteration in range(iterations):
        improved = False
        for _layer_number, (tmm_index, layer_name, low, high) in enumerate(search_layers, start=1):
            values = np.linspace(low, high, points_per_layer)
            local_best = best_candidate
            for value in values:
                trial = current.copy()
                trial[tmm_index] = float(value)
                candidate = _evaluate_candidate(
                    model=model,
                    prepared=prepared,
                    d_list=trial,
                    search_layers=search_layers,
                    layer_names=layer_names,
                    target_xyz=target_xyz,
                    target_lab=target_lab_array,
                    color_cache=color_cache,
                    angle_deg=angle_deg,
                    min_lightness=min_lightness,
                    brightness_weight=brightness_weight,
                    colour_metric=metric,
                    score_mode=score_key,
                )
                key = _thickness_key(candidate.thicknesses_nm)
                if key not in seen:
                    candidates.append(candidate)
                    seen.add(key)
                if candidate.score < local_best.score:
                    local_best = candidate
                evaluated += 1
                if progress_callback is not None:
                    progress_callback(
                        evaluated,
                        total,
                        f"iteration {iteration + 1}/{iterations}, {layer_name} point {evaluated}/{total}",
                    )
            if local_best.score < best_candidate.score:
                best_candidate = local_best
                current = _d_list_with_search_thicknesses(current, search_layers, best_candidate.thicknesses_nm)
                improved = True
        if not improved:
            break

    ranked = tuple(sorted(candidates, key=lambda item: item.score)[:top_n])
    return TargetSearchResult(
        target_xyz=tuple(float(value) for value in target_xyz),
        target_lab=target_lab,
        wavelengths_nm=wavelengths,
        stack_label=stack.display_summary(),
        candidates=ranked,
        evaluated_count=evaluated,
        iterations=iteration + 1,
        colour_metric=metric,
        score_mode=score_key,
    )


def xyz_from_srgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert display sRGB to CIE XYZ using the standard D65 matrix."""

    srgb = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=float,
    )
    xyz = 100.0 * matrix @ linear
    return tuple(float(value) for value in xyz)


def xyz_from_lab(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert CIELAB to XYZ with a D65 white point."""

    l_value, a_value, b_value = (float(value) for value in lab)
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    fy = (l_value + 16.0) / 116.0
    fx = fy + a_value / 500.0
    fz = fy - b_value / 200.0

    def inverse_f(value: float) -> float:
        delta = 6.0 / 29.0
        if value > delta:
            return value**3
        return 3.0 * delta**2 * (value - 4.0 / 29.0)

    xyz = white * np.array([inverse_f(fx), inverse_f(fy), inverse_f(fz)], dtype=float)
    return tuple(float(value) for value in xyz)


def wid2016_from_lab(lab: tuple[float, float, float] | NDArray[np.float64]) -> float:
    """Return the CIELAB-based dental whiteness index from the 2016 WID formula."""

    l_value, a_value, b_value = (float(value) for value in lab)
    return float(0.511 * l_value - 2.324 * a_value - 1.100 * b_value)


def _evaluate_candidate(
    *,
    model: TMMModel,
    prepared,
    d_list: NDArray[np.float64],
    search_layers: list[tuple[int, str, float, float]],
    layer_names: tuple[str, ...],
    target_xyz: tuple[float, float, float],
    target_lab: NDArray[np.float64],
    color_cache,
    angle_deg: float,
    min_lightness: float,
    brightness_weight: float,
    colour_metric: str = COLOUR_METRIC_CIE76,
    score_mode: str = "delta_e",
) -> TargetSearchCandidate:
    reflectance = model.reflectance_from_prepared(prepared, d_list, angle_deg)
    xyz = reflectance_to_xyz(reflectance, cache=color_cache)
    lab = xyz_to_lab(xyz)
    srgb = reflectance_to_srgb(reflectance, cache=color_cache)
    delta_e = float(delta_e_colour(target_xyz, xyz, metric=colour_metric))
    whiteness_index = wid2016_from_lab(lab)
    lightness_shortfall = max(0.0, float(min_lightness) - float(lab[0]))
    target_lightness_shortfall = max(0.0, float(target_lab[0]) - float(lab[0]))
    chroma = float(np.hypot(lab[1], lab[2]))
    if score_mode == "wid_2016":
        score = -whiteness_index + float(brightness_weight) * lightness_shortfall
    else:
        score = (
            delta_e
            + float(brightness_weight) * lightness_shortfall
            + 0.08 * target_lightness_shortfall
            + 0.03 * chroma
        )
    return TargetSearchCandidate(
        thicknesses_nm=tuple(float(d_list[tmm_index]) for tmm_index, _, _, _ in search_layers),
        layer_names=layer_names,
        reflectance=reflectance,
        xyz=tuple(float(value) for value in xyz),
        lab=tuple(float(value) for value in lab),
        srgb=tuple(float(value) for value in srgb),
        delta_e=delta_e,
        score=float(score),
        whiteness_index=float(whiteness_index),
    )


def _d_list_with_search_thicknesses(
    d_list: NDArray[np.float64],
    search_layers: list[tuple[int, str, float, float]],
    thicknesses_nm: tuple[float, ...],
) -> NDArray[np.float64]:
    updated = d_list.copy()
    for value, (tmm_index, _, _, _) in zip(thicknesses_nm, search_layers):
        updated[tmm_index] = float(value)
    return updated


def _thickness_key(thicknesses_nm: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for value in thicknesses_nm)
