"""Experiment thickness optimization with reusable cached TMM evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re
from typing import Any
import csv

import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure

from .color import prepare_color_conversion, reflectance_to_srgb, reflectance_to_xyz
from .colorimetry import PerceivedColor
from .experiments import (
    ExperimentDataStore,
    ExperimentLayerEstimate,
    build_stack_from_estimates,
    COLOUR_METRIC_CIE76,
    delta_e_colour,
    load_reflectance_csv,
    normalise_colour_metric,
    xyz_to_xy,
)
from .materials import Material
from .optical_model import OpticalModel
from .stack import NativeOxide


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ThicknessOptimizationLayerResult:
    """Thickness and deposition-rate change for one optimized experimental layer."""

    material_name: str
    base_thickness_nm: float
    optimized_thickness_nm: float
    percent_change: float
    base_rate_nm_per_min: float | None = None
    optimized_rate_nm_per_min: float | None = None
    deposition_time_min: float | None = None


@dataclass(frozen=True)
class ThicknessOptimizationResult:
    """Best cached thickness fit for one sample/measurement pair."""

    sample_name: str
    measurement_description: str
    stack_label: str
    wavelengths_nm: FloatArray
    measured_reflectance: FloatArray
    base_reflectance: FloatArray
    optimized_reflectance: FloatArray
    measured_color: PerceivedColor
    base_color: PerceivedColor
    optimized_color: PerceivedColor
    base_delta_e: float
    optimized_delta_e: float
    layer_results: tuple[ThicknessOptimizationLayerResult, ...]
    cache_path: Path
    evaluated_count: int
    reused_count: int
    new_count: int
    colour_metric: str = COLOUR_METRIC_CIE76
    reflectance_scale: float = 1.0
    scale_fit_enabled: bool = False

    @property
    def measured_xy(self) -> tuple[float, float]:
        """Measured chromaticity coordinate for plotting."""

        return xyz_to_xy(self.measured_color.xyz)

    @property
    def base_xy(self) -> tuple[float, float]:
        """Original thickness-estimate chromaticity coordinate for plotting."""

        return xyz_to_xy(self.base_color.xyz)

    @property
    def optimized_xy(self) -> tuple[float, float]:
        """Optimized-thickness chromaticity coordinate for plotting."""

        return xyz_to_xy(self.optimized_color.xyz)


def default_thickness_optimization_cache_dir(project_root: str | Path) -> Path:
    """Return the standard folder for cached experiment thickness optimizations."""

    return Path(project_root) / "outputs" / "thickness_optimization_cache"


def optimize_experiment_thicknesses(
    store: ExperimentDataStore,
    sample_name: str,
    measurement_index: int,
    materials: dict[str, Material],
    model: OpticalModel,
    wavelengths_nm: NDArray[np.float64],
    angle_deg: float,
    substrate_name: str,
    native_oxide: NativeOxide | None,
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
    range_percent: float = 5.0,
    step_percent: float = 1.0,
    cache_dir: str | Path | None = None,
    profile_name: str = "current",
    model_label: str = "tmm",
    model_settings: dict[str, Any] | None = None,
    group_by_material: bool = False,
    fixed_metal_threshold_nm: float = 50.0,
    colour_metric: str = COLOUR_METRIC_CIE76,
    fit_reflectance_scale: bool = True,
    reflectance_scale_min: float = 0.70,
    reflectance_scale_max: float = 1.08,
    progress_callback: Any | None = None,
) -> ThicknessOptimizationResult:
    """Fit estimated layer thicknesses by sweeping shared sputter-rate percentage errors.

    The search varies each deposited layer by percentage, not by a fixed nm amount.
    That mirrors a deposition-rate uncertainty: a 5 percent rate error moves a
    200 nm film more than a 50 nm film.
    """

    if range_percent < 0:
        raise ValueError("range_percent must be non-negative.")
    if step_percent <= 0:
        raise ValueError("step_percent must be positive.")
    if reflectance_scale_min <= 0 or reflectance_scale_max <= 0:
        raise ValueError("reflectance scale limits must be positive.")
    if reflectance_scale_min > reflectance_scale_max:
        raise ValueError("reflectance_scale_min cannot be larger than reflectance_scale_max.")
    if not hasattr(model, "prepare_stack") or not hasattr(model, "reflectance_from_prepared"):
        raise TypeError("Thickness optimization currently requires a prepared TMM-style model.")

    metric = normalise_colour_metric(colour_metric)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    sample = store.load_sample(sample_name)
    if not sample.layer_estimates:
        raise ValueError(f"Sample {sample_name} has no estimated layer thicknesses.")
    if measurement_index < 0 or measurement_index >= len(sample.measurements):
        raise ValueError("Measurement index is out of range.")

    measurement = sample.measurements[measurement_index]
    substrate = measurement.substrate_hint or substrate_name
    stack = build_stack_from_estimates(
        sample,
        materials=materials,
        substrate_name=substrate,
        native_oxide=native_oxide,
        use_effective_interfaces=use_effective_interfaces,
        interface_thickness_nm=interface_thickness_nm,
        interface_fraction=interface_fraction,
    )
    prepared = model.prepare_stack(stack, wavelengths)  # type: ignore[attr-defined]
    display_layer_indices = prepared.display_layer_indices
    if len(display_layer_indices) != len(sample.layer_estimates):
        raise ValueError("Could not map all experimental layers into the prepared TMM stack.")
    variable_labels, layer_to_variable = _optimization_variables(
        sample.layer_estimates,
        group_by_material=group_by_material,
        fixed_metal_threshold_nm=fixed_metal_threshold_nm,
    )

    measured_wavelengths, measured_raw = load_reflectance_csv(measurement.csv_path)
    measured_reflectance = np.interp(wavelengths, measured_wavelengths, measured_raw)
    color_cache = prepare_color_conversion(wavelengths)
    measured_color = _perceived_color(wavelengths, measured_reflectance, color_cache)

    base_d_list = np.asarray(prepared.base_d_list, dtype=float)
    base_reflectance = model.reflectance_from_prepared(  # type: ignore[attr-defined]
        prepared,
        base_d_list,
        angle_deg,
    )
    base_scale = _best_reflectance_scale(
        measured_reflectance,
        base_reflectance,
        enabled=fit_reflectance_scale,
        scale_min=reflectance_scale_min,
        scale_max=reflectance_scale_max,
    )
    scaled_base_reflectance = _apply_reflectance_scale(base_reflectance, base_scale)
    base_color = _perceived_color(wavelengths, scaled_base_reflectance, color_cache)
    base_delta = delta_e_colour(measured_color.xyz, base_color.xyz, metric=metric)

    cache_root = (
        default_thickness_optimization_cache_dir(Path.cwd())
        if cache_dir is None
        else Path(cache_dir)
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _cache_filename(
        sample_name=sample_name,
        measurement_description=measurement.description,
        profile_name=profile_name,
        model_label=model_label,
        substrate_name=substrate,
        angle_deg=angle_deg,
        wavelengths_nm=wavelengths,
        layer_estimates=sample.layer_estimates,
        use_effective_interfaces=use_effective_interfaces,
        interface_thickness_nm=interface_thickness_nm,
        interface_fraction=interface_fraction,
        native_oxide=native_oxide,
        model_settings=model_settings,
        group_by_material=group_by_material,
        fixed_metal_threshold_nm=fixed_metal_threshold_nm,
        colour_metric=metric,
        fit_reflectance_scale=fit_reflectance_scale,
        reflectance_scale_min=reflectance_scale_min,
        reflectance_scale_max=reflectance_scale_max,
    )
    cache = _load_cache(cache_path)
    evaluations: dict[str, dict[str, Any]] = cache.setdefault("evaluations", {})

    offsets = _percentage_offsets(range_percent, step_percent)
    all_offset_sets = list(itertools.product(offsets, repeat=len(variable_labels)))
    if not all_offset_sets:
        all_offset_sets = [tuple()]
    reused = 0
    added = 0
    best_key = ""
    best_delta = float("inf")

    total_trials = len(all_offset_sets)
    for trial_number, offset_set in enumerate(all_offset_sets, start=1):
        key = _offset_key(offset_set)
        existing = evaluations.get(key)
        if existing is not None:
            reused += 1
            delta_e = float(existing["delta_e"])
        else:
            trial_d_list = base_d_list.copy()
            for layer_number, tmm_index in enumerate(display_layer_indices):
                variable_index = layer_to_variable[layer_number]
                percent = 0.0 if variable_index is None else offset_set[variable_index]
                trial_d_list[tmm_index] = base_d_list[tmm_index] * (1.0 + percent / 100.0)
            reflectance = model.reflectance_from_prepared(  # type: ignore[attr-defined]
                prepared,
                trial_d_list,
                angle_deg,
            )
            scale = _best_reflectance_scale(
                measured_reflectance,
                reflectance,
                enabled=fit_reflectance_scale,
                scale_min=reflectance_scale_min,
                scale_max=reflectance_scale_max,
            )
            scaled_reflectance = _apply_reflectance_scale(reflectance, scale)
            trial_xyz = reflectance_to_xyz(scaled_reflectance, cache=color_cache)
            trial_rgb = reflectance_to_srgb(scaled_reflectance, cache=color_cache)
            delta_e = delta_e_colour(measured_color.xyz, trial_xyz, metric=metric)
            evaluations[key] = {
                "offsets_percent": [float(value) for value in offset_set],
                "variable_labels": list(variable_labels),
                "delta_e": float(delta_e),
                "reflectance_scale": float(scale),
                "rgb": [float(value) for value in np.asarray(trial_rgb, dtype=float)],
                "xyz": [float(value) for value in np.asarray(trial_xyz, dtype=float)],
            }
            added += 1
            if added % 25 == 0:
                cache["best_key"] = best_key
                _save_cache(cache_path, cache)

        if delta_e < best_delta:
            best_delta = delta_e
            best_key = key
        if progress_callback is not None and (trial_number == 1 or trial_number % 10 == 0 or trial_number == total_trials):
            progress_callback(trial_number, total_trials)

    cache["metadata"] = {
        "sample_name": sample_name,
        "measurement_description": measurement.description,
        "profile_name": profile_name,
        "model_label": model_label,
        "substrate_name": substrate,
        "angle_deg": float(angle_deg),
        "use_effective_interfaces": bool(use_effective_interfaces),
        "interface_thickness_nm": float(interface_thickness_nm),
        "interface_fraction": float(interface_fraction),
        "native_oxide": None
        if native_oxide is None
        else [native_oxide.material.name, float(native_oxide.thickness_nm)],
        "range_percent_last_run": float(range_percent),
        "step_percent": float(step_percent),
        "layer_names": [layer.material_name for layer in sample.layer_estimates],
        "base_thicknesses_nm": [float(layer.thickness_nm) for layer in sample.layer_estimates],
        "optimization_mode": "material_rate" if group_by_material else "layer",
        "variable_labels": list(variable_labels),
        "fixed_metal_threshold_nm": float(fixed_metal_threshold_nm),
        "model_settings": model_settings or {},
        "colour_metric": metric,
        "fit_reflectance_scale": bool(fit_reflectance_scale),
        "reflectance_scale_min": float(reflectance_scale_min),
        "reflectance_scale_max": float(reflectance_scale_max),
        "base_reflectance_scale": float(base_scale),
    }
    cache["best_key"] = best_key
    _save_cache(cache_path, cache)

    best_offsets = [float(value) for value in evaluations[best_key]["offsets_percent"]]
    optimized_d_list = base_d_list.copy()
    layer_percents = _layer_percents(best_offsets, layer_to_variable)
    for tmm_index, percent in zip(display_layer_indices, layer_percents):
        optimized_d_list[tmm_index] = base_d_list[tmm_index] * (1.0 + percent / 100.0)
    optimized_reflectance = model.reflectance_from_prepared(  # type: ignore[attr-defined]
        prepared,
        optimized_d_list,
        angle_deg,
    )
    best_scale = float(evaluations[best_key].get("reflectance_scale", 1.0))
    optimized_reflectance = _apply_reflectance_scale(optimized_reflectance, best_scale)
    optimized_color = _perceived_color(wavelengths, optimized_reflectance, color_cache)

    return ThicknessOptimizationResult(
        sample_name=sample_name,
        measurement_description=measurement.description,
        stack_label=prepared.display_summary,
        wavelengths_nm=wavelengths,
        measured_reflectance=measured_reflectance,
        base_reflectance=scaled_base_reflectance,
        optimized_reflectance=optimized_reflectance,
        measured_color=measured_color,
        base_color=base_color,
        optimized_color=optimized_color,
        base_delta_e=float(base_delta),
        optimized_delta_e=float(best_delta),
        layer_results=tuple(
            _layer_result(layer, percent) for layer, percent in zip(sample.layer_estimates, layer_percents)
        ),
        cache_path=cache_path,
        evaluated_count=len(all_offset_sets),
        reused_count=reused,
        new_count=added,
        colour_metric=metric,
        reflectance_scale=best_scale,
        scale_fit_enabled=bool(fit_reflectance_scale),
    )


def save_optimization_summary_outputs(
    results: list[ThicknessOptimizationResult],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save CSV and plots summarizing cached thickness/rate optimization results."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "thickness_optimization_summary.csv"
    rows: list[dict[str, Any]] = []
    for result in results:
        for layer_index, layer in enumerate(result.layer_results, start=1):
            rows.append(
                {
                    "sample_name": result.sample_name,
                    "measurement_description": result.measurement_description,
                    "layer_index": layer_index,
                    "material_name": layer.material_name,
                    "base_thickness_nm": layer.base_thickness_nm,
                    "optimized_thickness_nm": layer.optimized_thickness_nm,
                    "percent_change": layer.percent_change,
                    "base_rate_nm_per_min": layer.base_rate_nm_per_min,
                    "optimized_rate_nm_per_min": layer.optimized_rate_nm_per_min,
                    "deposition_time_min": layer.deposition_time_min,
                    "base_delta_e": result.base_delta_e,
                    "optimized_delta_e": result.optimized_delta_e,
                    "reflectance_scale": result.reflectance_scale,
                    "scale_fit_enabled": result.scale_fit_enabled,
                    "colour_metric": result.colour_metric,
                }
            )
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    delta_plot = out / "delta_e_before_after.png"
    rate_sample_plot = out / "sputter_rate_change_by_sample.png"
    rate_time_plot = out / "sputter_rate_change_by_time.png"
    _plot_delta_e(results, delta_plot)
    _plot_rate_changes(rows, "sample_name", rate_sample_plot)
    _plot_rate_changes(rows, "deposition_time_min", rate_time_plot)
    return {
        "summary_csv": csv_path,
        "delta_e_plot": delta_plot,
        "rate_by_sample_plot": rate_sample_plot,
        "rate_by_time_plot": rate_time_plot,
    }


def _layer_result(
    layer: ExperimentLayerEstimate,
    percent_change: float,
) -> ThicknessOptimizationLayerResult:
    optimized_thickness = layer.thickness_nm * (1.0 + percent_change / 100.0)
    optimized_rate = None
    if layer.time_min is not None and layer.time_min > 0:
        optimized_rate = optimized_thickness / layer.time_min
    elif layer.rate_nm_per_min is not None:
        optimized_rate = layer.rate_nm_per_min * (1.0 + percent_change / 100.0)
    return ThicknessOptimizationLayerResult(
        material_name=layer.material_name,
        base_thickness_nm=float(layer.thickness_nm),
        optimized_thickness_nm=float(optimized_thickness),
        percent_change=float(percent_change),
        base_rate_nm_per_min=layer.rate_nm_per_min,
        optimized_rate_nm_per_min=optimized_rate,
        deposition_time_min=layer.time_min,
    )


def _optimization_variables(
    layers: tuple[ExperimentLayerEstimate, ...],
    group_by_material: bool,
    fixed_metal_threshold_nm: float,
) -> tuple[tuple[str, ...], tuple[int | None, ...]]:
    labels: list[str] = []
    layer_to_variable: list[int | None] = []
    material_to_variable: dict[str, int] = {}
    for index, layer in enumerate(layers, start=1):
        if _is_fixed_metal(layer, fixed_metal_threshold_nm):
            layer_to_variable.append(None)
            continue
        if group_by_material:
            variable_index = material_to_variable.get(layer.material_name)
            if variable_index is None:
                variable_index = len(labels)
                material_to_variable[layer.material_name] = variable_index
                labels.append(layer.material_name)
            layer_to_variable.append(variable_index)
        else:
            layer_to_variable.append(len(labels))
            labels.append(f"{layer.material_name} #{index}")
    return tuple(labels), tuple(layer_to_variable)


def _is_fixed_metal(layer: ExperimentLayerEstimate, fixed_metal_threshold_nm: float) -> bool:
    return layer.material_name in {"Ag", "Au"} and layer.thickness_nm >= fixed_metal_threshold_nm


def _layer_percents(
    best_offsets: list[float],
    layer_to_variable: tuple[int | None, ...],
) -> list[float]:
    percents: list[float] = []
    for variable_index in layer_to_variable:
        percents.append(0.0 if variable_index is None else best_offsets[variable_index])
    return percents


def _best_reflectance_scale(
    measured_reflectance: FloatArray,
    simulated_reflectance: FloatArray,
    *,
    enabled: bool,
    scale_min: float,
    scale_max: float,
) -> float:
    if not enabled:
        return 1.0
    measured = np.asarray(measured_reflectance, dtype=float)
    simulated = np.asarray(simulated_reflectance, dtype=float)
    denominator = float(np.dot(simulated, simulated))
    if denominator <= 0.0:
        return 1.0
    scale = float(np.dot(measured, simulated) / denominator)
    return float(np.clip(scale, scale_min, scale_max))


def _apply_reflectance_scale(reflectance: FloatArray, scale: float) -> FloatArray:
    return np.clip(np.asarray(reflectance, dtype=float) * float(scale), 0.0, 1.0)


def _plot_delta_e(results: list[ThicknessOptimizationResult], path: Path) -> None:
    if not results:
        return
    x = np.arange(len(results))
    fig = Figure(figsize=(10, 4.5), dpi=160)
    ax = fig.subplots()
    ax.plot(x, [r.base_delta_e for r in results], marker="o", linewidth=1.4, label="Before")
    ax.plot(x, [r.optimized_delta_e for r in results], marker="o", linewidth=1.4, label="After")
    ax.set_ylabel(r"$\Delta E^*_{Lab}$")
    ax.set_xlabel("Optimized measurement")
    ax.set_title("Colour distance before and after thickness optimization")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def _plot_rate_changes(rows: list[dict[str, Any]], x_key: str, path: Path) -> None:
    plot_rows = [
        row for row in rows
        if row["percent_change"] is not None and (x_key != "deposition_time_min" or row[x_key] not in {None, ""})
    ]
    if not plot_rows:
        return
    materials = sorted({str(row["material_name"]) for row in plot_rows})
    fig = Figure(figsize=(10, 4.8), dpi=160)
    ax = fig.subplots()
    for material in materials:
        material_rows = [row for row in plot_rows if row["material_name"] == material]
        if x_key == "sample_name":
            x = np.arange(len(material_rows))
            labels = [str(row["sample_name"]) for row in material_rows]
            ax.scatter(x, [float(row["percent_change"]) for row in material_rows], label=material, s=26)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        else:
            ax.scatter(
                [float(row[x_key]) for row in material_rows],
                [float(row["percent_change"]) for row in material_rows],
                label=material,
                s=28,
            )
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_ylabel("Optimized sputter-rate change (%)")
    ax.set_xlabel("Sample" if x_key == "sample_name" else "Deposition time (min)")
    ax.set_title(
        "Sputter-rate change by sample"
        if x_key == "sample_name"
        else "Sputter-rate change vs deposition time"
    )
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def _perceived_color(
    wavelengths_nm: FloatArray,
    reflectance: FloatArray,
    color_cache,
) -> PerceivedColor:
    xyz = reflectance_to_xyz(reflectance, cache=color_cache)
    srgb = reflectance_to_srgb(reflectance, cache=color_cache)
    srgb_255 = tuple(int(round(channel * 255)) for channel in srgb)
    return PerceivedColor(
        srgb=tuple(float(channel) for channel in srgb),
        srgb_255=srgb_255,
        xyz=tuple(float(channel) for channel in xyz),
    )


def _percentage_offsets(range_percent: float, step_percent: float) -> NDArray[np.float64]:
    count = int(np.floor((2.0 * range_percent) / step_percent + 0.5)) + 1
    values = -range_percent + step_percent * np.arange(count, dtype=float)
    values = values[(values >= -range_percent - 1e-9) & (values <= range_percent + 1e-9)]
    if not np.any(np.isclose(values, 0.0)):
        values = np.sort(np.append(values, 0.0))
    return np.round(values, 6)


def _offset_key(offset_set: tuple[float, ...] | list[float]) -> str:
    return ",".join(f"{float(value):.6f}" for value in offset_set)


def _cache_filename(
    sample_name: str,
    measurement_description: str,
    profile_name: str,
    model_label: str,
    substrate_name: str,
    angle_deg: float,
    wavelengths_nm: FloatArray,
    layer_estimates: tuple[ExperimentLayerEstimate, ...],
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
    native_oxide: NativeOxide | None,
    model_settings: dict[str, Any] | None = None,
    group_by_material: bool = False,
    fixed_metal_threshold_nm: float = 50.0,
    colour_metric: str = COLOUR_METRIC_CIE76,
    fit_reflectance_scale: bool = True,
    reflectance_scale_min: float = 0.70,
    reflectance_scale_max: float = 1.08,
) -> str:
    metric = normalise_colour_metric(colour_metric)
    signature = {
        "sample_name": sample_name,
        "measurement_description": measurement_description,
        "profile_name": profile_name,
        "model_label": model_label,
        "substrate_name": substrate_name,
        "angle_deg": round(float(angle_deg), 6),
        "wavelength_start": round(float(wavelengths_nm[0]), 6),
        "wavelength_stop": round(float(wavelengths_nm[-1]), 6),
        "wavelength_count": int(wavelengths_nm.size),
        "layers": [
            [layer.material_name, round(float(layer.thickness_nm), 6)]
            for layer in layer_estimates
        ],
        "use_effective_interfaces": bool(use_effective_interfaces),
        "interface_thickness_nm": round(float(interface_thickness_nm), 6),
        "interface_fraction": round(float(interface_fraction), 6),
        "native_oxide": None
        if native_oxide is None
        else [native_oxide.material.name, round(float(native_oxide.thickness_nm), 6)],
        "model_settings": model_settings or {},
        "optimization_mode": "material_rate" if group_by_material else "layer",
        "fixed_metal_threshold_nm": round(float(fixed_metal_threshold_nm), 6),
        "fit_reflectance_scale": bool(fit_reflectance_scale),
        "reflectance_scale_min": round(float(reflectance_scale_min), 6),
        "reflectance_scale_max": round(float(reflectance_scale_max), 6),
    }
    if metric != COLOUR_METRIC_CIE76:
        signature["colour_metric"] = metric
    digest = sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    readable = _safe_filename(f"{sample_name}_{profile_name}_{model_label}")[:80]
    return f"{readable}_{digest}.json"


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "thickness_optimization"


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"evaluations": {}}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {"evaluations": {}}
    data.setdefault("evaluations", {})
    return data


def _save_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
