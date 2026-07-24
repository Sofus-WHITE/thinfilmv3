"""Fit shared roughness redistribution parameters for experiment groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from .color import prepare_color_conversion, reflectance_to_xyz
from .diffuse_redistribution_model import DiffuseRedistributionSettings, TMMWithDiffuseRedistributionModel
from .experiments import (
    COLOUR_METRIC_CIE76,
    ExperimentDataStore,
    build_stack_from_estimates,
    delta_e_colour,
    load_reflectance_csv,
    normalise_colour_metric,
)
from .materials import Material
from .stack import NativeOxide
from .tmm_model import TMMModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoughnessFitResult:
    """Best shared diffuse-redistribution parameters for one experiment group."""

    group_label: str
    sample_count: int
    spectrum_count: int
    rms_roughness_nm: float
    scatter_scale: float
    scatter_exponent: float
    max_scatter_fraction: float
    mean_delta_e: float
    mean_rmse: float
    output_path: Path
    colour_metric: str = COLOUR_METRIC_CIE76


@dataclass(frozen=True)
class _PreparedRoughnessMeasurement:
    sample_name: str
    measurement_description: str
    measured_reflectance: FloatArray
    measured_xyz: tuple[float, float, float]
    specular_reflectance: FloatArray
    diffuse_proxy_reflectance: FloatArray
    interface_count: int


def default_roughness_fit_output_dir(project_root: str | Path) -> Path:
    """Return the standard folder for saved roughness group fits."""

    return Path(project_root) / "outputs" / "roughness_fits"


def roughness_fit_profile_path(project_root: str | Path, group_label: str) -> Path:
    """Return the JSON path for a saved roughness group fit."""

    return default_roughness_fit_output_dir(project_root) / f"roughness_fit_{group_label}.json"


def fit_roughness_redistribution_parameters(
    project_root: str | Path,
    store: ExperimentDataStore,
    materials: dict[str, Material],
    wavelengths_nm: NDArray[np.float64],
    angle_deg: float,
    substrate_default: str,
    native_oxide_factory: Any,
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
    group_label: str,
    substrate_filter: str | None = None,
    surface_filter: str | None = "rough",
    measurement_kind_filter: str | None = None,
    max_scatter_fraction: float = 0.85,
    initial_rms_roughness_nm: float = 1.0,
    fit_rms: bool = True,
    colour_metric: str = COLOUR_METRIC_CIE76,
    progress_callback: Any | None = None,
) -> RoughnessFitResult:
    """Fit shared roughness redistribution parameters for the active experiment group."""

    metric = normalise_colour_metric(colour_metric)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    prepared_measurements = _prepare_group_measurements(
        store=store,
        materials=materials,
        wavelengths_nm=wavelengths,
        angle_deg=angle_deg,
        substrate_default=substrate_default,
        native_oxide_factory=native_oxide_factory,
        use_effective_interfaces=use_effective_interfaces,
        interface_thickness_nm=interface_thickness_nm,
        interface_fraction=interface_fraction,
        substrate_filter=substrate_filter,
        surface_filter=surface_filter,
        measurement_kind_filter=measurement_kind_filter,
    )
    if not prepared_measurements:
        raise ValueError("No spectra matched the selected roughness-fit filters.")

    color_cache = prepare_color_conversion(wavelengths)
    eval_count = 0

    def unpack(params: NDArray[np.float64]) -> tuple[float, float, float]:
        if fit_rms:
            return float(params[0]), float(params[1]), float(params[2])
        return float(initial_rms_roughness_nm), float(params[0]), float(params[1])

    def objective(params: NDArray[np.float64]) -> float:
        nonlocal eval_count
        rms, scatter_scale, exponent = unpack(params)
        delta_values: list[float] = []
        rmse_values: list[float] = []
        for item in prepared_measurements:
            scatter = _scatter_fraction(
                wavelengths,
                rms_roughness_nm=rms,
                scatter_scale=scatter_scale,
                scatter_exponent=exponent,
                max_scatter_fraction=max_scatter_fraction,
                angle_deg=angle_deg,
                interface_count=item.interface_count,
            )
            reflectance = (1.0 - scatter) * item.specular_reflectance + scatter * item.diffuse_proxy_reflectance
            xyz = reflectance_to_xyz(reflectance, cache=color_cache)
            delta_values.append(delta_e_colour(item.measured_xyz, xyz, metric=metric))
            rmse_values.append(float(np.sqrt(np.mean(np.square(reflectance - item.measured_reflectance)))))
        eval_count += 1
        if progress_callback is not None and eval_count == 1 or (progress_callback is not None and eval_count % 10 == 0):
            progress_callback(eval_count, f"roughness fit evaluation {eval_count}")
        return float(np.mean(delta_values) + 20.0 * np.mean(rmse_values))

    bounds = [(0.0, 50.0), (0.0, 10.0), (-2.0, 5.0)] if fit_rms else [(0.0, 10.0), (-2.0, 5.0)]
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=18,
        popsize=7,
        tol=0.01,
        polish=True,
        seed=11,
        updating="immediate",
        workers=1,
    )
    best_rms, best_scale, best_exponent = unpack(np.asarray(result.x, dtype=float))
    mean_delta, mean_rmse = _score_parameters(
        prepared_measurements,
        wavelengths,
        angle_deg,
        rms_roughness_nm=best_rms,
        scatter_scale=best_scale,
        scatter_exponent=best_exponent,
        max_scatter_fraction=max_scatter_fraction,
        color_cache=color_cache,
        colour_metric=metric,
    )
    output_path = roughness_fit_profile_path(project_root, group_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fit_result = RoughnessFitResult(
        group_label=group_label,
        sample_count=len({item.sample_name for item in prepared_measurements}),
        spectrum_count=len(prepared_measurements),
        rms_roughness_nm=float(best_rms),
        scatter_scale=float(best_scale),
        scatter_exponent=float(best_exponent),
        max_scatter_fraction=float(max_scatter_fraction),
        mean_delta_e=float(mean_delta),
        mean_rmse=float(mean_rmse),
        output_path=output_path,
        colour_metric=metric,
    )
    output_path.write_text(json.dumps(asdict(fit_result) | {"output_path": str(output_path)}, indent=2), encoding="utf-8")
    return fit_result


def _prepare_group_measurements(
    store: ExperimentDataStore,
    materials: dict[str, Material],
    wavelengths_nm: FloatArray,
    angle_deg: float,
    substrate_default: str,
    native_oxide_factory: Any,
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
    substrate_filter: str | None,
    surface_filter: str | None,
    measurement_kind_filter: str | None,
) -> list[_PreparedRoughnessMeasurement]:
    base_model = TMMModel()
    diffuse_model = TMMWithDiffuseRedistributionModel(
        DiffuseRedistributionSettings(
            rms_roughness_nm=1.0,
            scatter_scale=0.0,
            diffuse_angle_samples=17,
        )
    )
    color_cache = prepare_color_conversion(wavelengths_nm)
    prepared_items: list[_PreparedRoughnessMeasurement] = []
    for sample_name in store.sample_names(require_spectra=True):
        sample = store.load_sample(sample_name)
        if not sample.layer_estimates:
            continue
        for measurement in sample.measurements:
            substrate_name = measurement.substrate_hint or substrate_default
            substrate_group = measurement.substrate_group or substrate_name
            if substrate_filter and substrate_group != substrate_filter:
                continue
            if surface_filter and measurement.surface_class != surface_filter:
                continue
            if measurement_kind_filter and measurement.measurement_kind != measurement_kind_filter:
                continue
            try:
                stack = build_stack_from_estimates(
                    sample,
                    materials=materials,
                    substrate_name=substrate_name,
                    native_oxide=native_oxide_factory(substrate_name),
                    use_effective_interfaces=use_effective_interfaces,
                    interface_thickness_nm=interface_thickness_nm,
                    interface_fraction=interface_fraction,
                )
                prepared = base_model.prepare_stack(stack, wavelengths_nm)
                measured_wl, measured_raw = load_reflectance_csv(measurement.csv_path)
                measured = np.interp(wavelengths_nm, measured_wl, measured_raw)
                specular = base_model.reflectance_from_prepared(prepared, prepared.base_d_list, angle_deg)
                diffuse_proxy = diffuse_model.diffuse_proxy_from_prepared(prepared, prepared.base_d_list)
                measured_xyz = reflectance_to_xyz(measured, cache=color_cache)
                prepared_items.append(
                    _PreparedRoughnessMeasurement(
                        sample_name=sample_name,
                        measurement_description=measurement.description,
                        measured_reflectance=measured,
                        measured_xyz=tuple(float(value) for value in measured_xyz),
                        specular_reflectance=specular,
                        diffuse_proxy_reflectance=diffuse_proxy,
                        interface_count=len(prepared.display_layer_indices) + 1,
                    )
                )
            except Exception:
                continue
    return prepared_items


def _score_parameters(
    items: list[_PreparedRoughnessMeasurement],
    wavelengths_nm: FloatArray,
    angle_deg: float,
    rms_roughness_nm: float,
    scatter_scale: float,
    scatter_exponent: float,
    max_scatter_fraction: float,
    color_cache,
    colour_metric: str = COLOUR_METRIC_CIE76,
) -> tuple[float, float]:
    metric = normalise_colour_metric(colour_metric)
    delta_values: list[float] = []
    rmse_values: list[float] = []
    for item in items:
        scatter = _scatter_fraction(
            wavelengths_nm,
            rms_roughness_nm=rms_roughness_nm,
            scatter_scale=scatter_scale,
            scatter_exponent=scatter_exponent,
            max_scatter_fraction=max_scatter_fraction,
            angle_deg=angle_deg,
            interface_count=item.interface_count,
        )
        reflectance = (1.0 - scatter) * item.specular_reflectance + scatter * item.diffuse_proxy_reflectance
        xyz = reflectance_to_xyz(reflectance, cache=color_cache)
        delta_values.append(delta_e_colour(item.measured_xyz, xyz, metric=metric))
        rmse_values.append(float(np.sqrt(np.mean(np.square(reflectance - item.measured_reflectance)))))
    return float(np.mean(delta_values)), float(np.mean(rmse_values))


def _scatter_fraction(
    wavelengths_nm: FloatArray,
    rms_roughness_nm: float,
    scatter_scale: float,
    scatter_exponent: float,
    max_scatter_fraction: float,
    angle_deg: float,
    interface_count: int,
) -> FloatArray:
    sigma = max(float(rms_roughness_nm), 0.0)
    if sigma == 0.0 or scatter_scale <= 0.0:
        return np.zeros_like(wavelengths_nm, dtype=float)
    cos_theta = np.cos(np.deg2rad(angle_deg))
    base = 1.0 - np.exp(-interface_count * (4.0 * np.pi * sigma * cos_theta / wavelengths_nm) ** 2)
    wavelength_weight = (550.0 / wavelengths_nm) ** float(scatter_exponent)
    return np.clip(float(scatter_scale) * base * wavelength_weight, 0.0, max(float(max_scatter_fraction), 0.0))
