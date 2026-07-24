"""Empirical equipment-calibrated refractive-index fitting.

This module deliberately separates predictive/effective calibration from the
physics-constrained material constants.  The fitted values are meant to describe
one sputter tool and processing workflow, not literal material constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .color import prepare_color_conversion, reflectance_to_xyz
from .experiments import (
    COLOUR_METRIC_CIE76,
    ExperimentDataStore,
    delta_e_colour,
    load_reflectance_csv,
    normalize_substrate_name,
    xyz_to_lab,
)
from .materials import Material, make_tabulated_material
from .optical_model import OpticalModel
from .stack import Layer, NativeOxide, make_stack, make_stack_with_interfaces, native_oxide_for_substrate


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EmpiricalFitParameter:
    """One fitted empirical correction parameter."""

    material_name: str
    parameter: str
    value: float
    lower_bound: float
    upper_bound: float
    unit: str


@dataclass(frozen=True)
class EmpiricalFitPrediction:
    """Before/after prediction quality for one measured spectrum."""

    split: str
    sample_name: str
    measurement_index: int
    measurement_description: str
    substrate_name: str
    surface_class: str
    measurement_kind: str
    stack_label: str
    baseline_delta_e: float
    fitted_delta_e: float
    baseline_rms_reflectance: float
    fitted_rms_reflectance: float


@dataclass(frozen=True)
class EmpiricalFitResult:
    """Saved empirical calibration result."""

    parameters: tuple[EmpiricalFitParameter, ...]
    predictions: tuple[EmpiricalFitPrediction, ...]
    material_names: tuple[str, ...]
    fit_k: bool
    use_thickness_dependence: bool
    use_time_dependence: bool
    validation_fraction: float
    lab_weight: float
    training_count: int
    validation_count: int
    mean_train_delta_e_before: float
    mean_train_delta_e_after: float
    mean_validation_delta_e_before: float | None
    mean_validation_delta_e_after: float | None
    rms_residual: float
    model: str
    output_dir: str


@dataclass(frozen=True)
class _Observation:
    sample_name: str
    measurement_index: int
    sample: object
    measurement: object
    measured_reflectance: FloatArray
    measured_lab: FloatArray
    substrate_name: str
    split: str = "train"


@dataclass(frozen=True)
class _ParameterSpec:
    material_name: str
    parameter: str
    lower: float
    upper: float
    unit: str


def fit_empirical_refractive_index_model(
    *,
    store: ExperimentDataStore,
    measurement_pairs: Sequence[tuple[str, int]],
    base_materials: dict[str, Material],
    model: OpticalModel,
    wavelengths_nm: FloatArray,
    material_names: Sequence[str],
    angle_deg: float,
    substrate_name: str = "Si",
    native_oxide_enabled: bool = True,
    native_oxide_thickness_nm: float = 2.0,
    use_effective_interfaces: bool = False,
    interface_thickness_nm: float = 1.0,
    interface_fraction: float = 0.5,
    fit_k: bool = True,
    use_thickness_dependence: bool = True,
    use_time_dependence: bool = False,
    validation_fraction: float = 0.2,
    lab_weight: float = 0.02,
    max_nfev: int = 120,
    colour_metric: str = COLOUR_METRIC_CIE76,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> EmpiricalFitResult:
    """Fit loose effective n/k corrections for selected experiment rows."""

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    color_cache = prepare_color_conversion(wavelengths)
    selected_materials = tuple(
        name.strip() for name in material_names if name.strip() and name.strip() in base_materials
    )
    if not selected_materials:
        raise ValueError("Choose at least one material with available base constants.")

    observations = _load_observations(
        store=store,
        measurement_pairs=measurement_pairs,
        wavelengths_nm=wavelengths,
        color_cache=color_cache,
        fitted_materials=set(selected_materials),
        default_substrate=substrate_name,
    )
    if len(observations) < 2:
        raise ValueError("At least two measured spectra containing the selected materials are required.")

    observations = _assign_validation_split(observations, validation_fraction)
    train_observations = [observation for observation in observations if observation.split == "train"]
    if not train_observations:
        raise ValueError("The validation fraction left no training spectra.")

    stats = _material_feature_stats(train_observations, selected_materials)
    specs = _parameter_specs(
        selected_materials,
        fit_k=fit_k,
        use_thickness_dependence=use_thickness_dependence,
        use_time_dependence=use_time_dependence,
        stats=stats,
    )
    if not specs:
        raise ValueError("No empirical parameters were enabled.")

    x0 = np.zeros(len(specs), dtype=float)
    lower = np.asarray([spec.lower for spec in specs], dtype=float)
    upper = np.asarray([spec.upper for spec in specs], dtype=float)

    def report(done: int, total: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total, message)

    residual_calls = 0

    def residual(params: FloatArray) -> FloatArray:
        nonlocal residual_calls
        residual_calls += 1
        if residual_calls == 1 or residual_calls % 5 == 0:
            report(
                min(residual_calls, max_nfev),
                max_nfev,
                f"empirical n/k least-squares call {residual_calls} (nominal max {max_nfev})",
            )
        parts: list[FloatArray] = []
        parameter_map = _parameter_map(specs, params)
        for observation in train_observations:
            simulated = _simulate_observation(
                observation=observation,
                base_materials=base_materials,
                model=model,
                wavelengths_nm=wavelengths,
                parameter_map=parameter_map,
                fitted_materials=set(selected_materials),
                stats=stats,
                angle_deg=angle_deg,
                native_oxide_enabled=native_oxide_enabled,
                native_oxide_thickness_nm=native_oxide_thickness_nm,
                use_effective_interfaces=use_effective_interfaces,
                interface_thickness_nm=interface_thickness_nm,
                interface_fraction=interface_fraction,
            )
            parts.append(simulated - observation.measured_reflectance)
            if lab_weight > 0:
                simulated_lab = xyz_to_lab(reflectance_to_xyz(simulated, cache=color_cache))
                parts.append((simulated_lab - observation.measured_lab) * float(lab_weight))
        return np.concatenate(parts)

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        max_nfev=max(int(max_nfev), 1),
        x_scale="jac",
    )
    parameter_map = _parameter_map(specs, result.x)
    parameter_rows = tuple(
        EmpiricalFitParameter(
            material_name=spec.material_name,
            parameter=spec.parameter,
            value=float(value),
            lower_bound=float(spec.lower),
            upper_bound=float(spec.upper),
            unit=spec.unit,
        )
        for spec, value in zip(specs, result.x)
    )

    predictions = _evaluate_predictions(
        observations=observations,
        base_materials=base_materials,
        model=model,
        wavelengths_nm=wavelengths,
        color_cache=color_cache,
        parameter_map=parameter_map,
        fitted_materials=set(selected_materials),
        stats=stats,
        angle_deg=angle_deg,
        native_oxide_enabled=native_oxide_enabled,
        native_oxide_thickness_nm=native_oxide_thickness_nm,
        use_effective_interfaces=use_effective_interfaces,
        interface_thickness_nm=interface_thickness_nm,
        interface_fraction=interface_fraction,
        colour_metric=colour_metric,
    )

    root = Path(output_dir) if output_dir is not None else Path("outputs") / "empirical_fit"
    root.mkdir(parents=True, exist_ok=True)
    _save_result(root, parameter_rows, predictions, result)

    train_after = _mean_delta(predictions, "train", after=True)
    train_before = _mean_delta(predictions, "train", after=False)
    validation_after = _mean_delta(predictions, "validation", after=True)
    validation_before = _mean_delta(predictions, "validation", after=False)
    return EmpiricalFitResult(
        parameters=parameter_rows,
        predictions=tuple(predictions),
        material_names=selected_materials,
        fit_k=bool(fit_k),
        use_thickness_dependence=bool(use_thickness_dependence),
        use_time_dependence=bool(use_time_dependence),
        validation_fraction=float(validation_fraction),
        lab_weight=float(lab_weight),
        training_count=sum(prediction.split == "train" for prediction in predictions),
        validation_count=sum(prediction.split == "validation" for prediction in predictions),
        mean_train_delta_e_before=train_before,
        mean_train_delta_e_after=train_after,
        mean_validation_delta_e_before=validation_before,
        mean_validation_delta_e_after=validation_after,
        rms_residual=float(np.sqrt(np.mean(result.fun**2))) if result.fun.size else float("nan"),
        model="empirical thickness/time-dependent effective n/k correction",
        output_dir=str(root),
    )


def _load_observations(
    *,
    store: ExperimentDataStore,
    measurement_pairs: Sequence[tuple[str, int]],
    wavelengths_nm: FloatArray,
    color_cache,
    fitted_materials: set[str],
    default_substrate: str,
) -> list[_Observation]:
    observations: list[_Observation] = []
    for sample_name, measurement_index in measurement_pairs:
        sample = store.load_sample(sample_name)
        if not any(layer.material_name in fitted_materials for layer in sample.layer_estimates):
            continue
        if measurement_index < 0 or measurement_index >= len(sample.measurements):
            continue
        measurement = sample.measurements[measurement_index]
        measured_wl, measured_reflectance = load_reflectance_csv(measurement.csv_path)
        measured_grid = np.interp(wavelengths_nm, measured_wl, measured_reflectance)
        measured_lab = xyz_to_lab(reflectance_to_xyz(measured_grid, cache=color_cache))
        substrate = (
            measurement.substrate_hint
            or normalize_substrate_name(measurement.substrate_group)
            or default_substrate
        )
        observations.append(
            _Observation(
                sample_name=sample_name,
                measurement_index=int(measurement_index),
                sample=sample,
                measurement=measurement,
                measured_reflectance=measured_grid,
                measured_lab=measured_lab,
                substrate_name=substrate,
            )
        )
    return observations


def _assign_validation_split(
    observations: list[_Observation],
    validation_fraction: float,
) -> list[_Observation]:
    fraction = min(max(float(validation_fraction), 0.0), 0.8)
    sample_names = sorted({observation.sample_name for observation in observations})
    validation_count = int(round(len(sample_names) * fraction))
    if validation_count <= 0 or validation_count >= len(sample_names):
        return observations
    ranked = sorted(
        sample_names,
        key=lambda name: hashlib.sha1(name.encode("utf-8")).hexdigest(),
    )
    validation_samples = set(ranked[:validation_count])
    return [
        _Observation(
            **{
                **observation.__dict__,
                "split": "validation" if observation.sample_name in validation_samples else "train",
            }
        )
        for observation in observations
    ]


def _material_feature_stats(observations: Sequence[_Observation], material_names: Sequence[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for material_name in material_names:
        thicknesses: list[float] = []
        times: list[float] = []
        for observation in observations:
            for layer in observation.sample.layer_estimates:
                if layer.material_name != material_name:
                    continue
                thicknesses.append(float(layer.thickness_nm))
                if layer.time_min is not None and np.isfinite(float(layer.time_min)):
                    times.append(float(layer.time_min))
        thickness_array = np.asarray(thicknesses or [0.0], dtype=float)
        time_array = np.asarray(times or [0.0], dtype=float)
        stats[material_name] = {
            "thickness_center": float(np.mean(thickness_array)),
            "thickness_scale": float(max(np.std(thickness_array), 25.0)),
            "time_center": float(np.mean(time_array)),
            "time_scale": float(max(np.std(time_array), 5.0)),
            "has_time": bool(times),
        }
    return stats


def _parameter_specs(
    material_names: Sequence[str],
    *,
    fit_k: bool,
    use_thickness_dependence: bool,
    use_time_dependence: bool,
    stats: dict[str, dict[str, float]],
) -> list[_ParameterSpec]:
    specs: list[_ParameterSpec] = []
    for material_name in material_names:
        specs.append(_ParameterSpec(material_name, "dn0", -3.0, 3.0, "refractive-index offset"))
        if fit_k:
            specs.append(_ParameterSpec(material_name, "dk0", -8.0, 8.0, "extinction-index offset"))
        if use_thickness_dependence:
            specs.append(_ParameterSpec(material_name, "dn_dthickness", -2.0, 2.0, "per normalized thickness"))
            if fit_k:
                specs.append(_ParameterSpec(material_name, "dk_dthickness", -4.0, 4.0, "per normalized thickness"))
        if use_time_dependence and stats.get(material_name, {}).get("has_time", False):
            specs.append(_ParameterSpec(material_name, "dn_dtime", -2.0, 2.0, "per normalized sputter time"))
            if fit_k:
                specs.append(_ParameterSpec(material_name, "dk_dtime", -4.0, 4.0, "per normalized sputter time"))
    return specs


def _parameter_map(
    specs: Sequence[_ParameterSpec],
    values: FloatArray,
) -> dict[str, dict[str, float]]:
    mapped: dict[str, dict[str, float]] = {}
    for spec, value in zip(specs, values):
        mapped.setdefault(spec.material_name, {})[spec.parameter] = float(value)
    return mapped


def _simulate_observation(
    *,
    observation: _Observation,
    base_materials: dict[str, Material],
    model: OpticalModel,
    wavelengths_nm: FloatArray,
    parameter_map: dict[str, dict[str, float]],
    fitted_materials: set[str],
    stats: dict[str, dict[str, float]],
    angle_deg: float,
    native_oxide_enabled: bool,
    native_oxide_thickness_nm: float,
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
) -> FloatArray:
    stack = _build_empirical_stack(
        observation=observation,
        base_materials=base_materials,
        wavelengths_nm=wavelengths_nm,
        parameter_map=parameter_map,
        fitted_materials=fitted_materials,
        stats=stats,
        native_oxide_enabled=native_oxide_enabled,
        native_oxide_thickness_nm=native_oxide_thickness_nm,
        use_effective_interfaces=use_effective_interfaces,
        interface_thickness_nm=interface_thickness_nm,
        interface_fraction=interface_fraction,
    )
    return model.simulate(stack, wavelengths_nm, angle_deg).reflectance


def _build_empirical_stack(
    *,
    observation: _Observation,
    base_materials: dict[str, Material],
    wavelengths_nm: FloatArray,
    parameter_map: dict[str, dict[str, float]],
    fitted_materials: set[str],
    stats: dict[str, dict[str, float]],
    native_oxide_enabled: bool,
    native_oxide_thickness_nm: float,
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
):
    substrate_name = observation.substrate_name
    if substrate_name not in base_materials:
        substrate_name = "Si" if "Si" in base_materials else substrate_name
    if substrate_name not in base_materials:
        raise ValueError(f"Missing substrate constants for {observation.substrate_name!r}.")

    deposited_layers: list[Layer] = []
    for layer in observation.sample.layer_estimates:
        if layer.material_name not in base_materials:
            raise ValueError(f"Missing material constants for {layer.material_name!r}.")
        if layer.material_name in fitted_materials:
            material = _effective_layer_material(
                base_materials[layer.material_name],
                wavelengths_nm,
                layer,
                parameter_map.get(layer.material_name, {}),
                stats.get(layer.material_name, {}),
            )
        else:
            material = base_materials[layer.material_name]
        deposited_layers.append(Layer(material, float(layer.thickness_nm)))

    native_oxide = _native_oxide(
        base_materials,
        substrate_name=substrate_name,
        enabled=native_oxide_enabled,
        thickness_nm=native_oxide_thickness_nm,
    )
    if use_effective_interfaces:
        return make_stack_with_interfaces(
            incident_medium=base_materials["air"],
            deposited_layers=deposited_layers,
            substrate=base_materials[substrate_name],
            native_oxide=native_oxide,
            interface_thickness_nm=interface_thickness_nm,
            interface_fraction=interface_fraction,
            name=f"{observation.sample_name} empirical calibrated",
        )

    optical_layers = list(deposited_layers)
    if native_oxide is not None and native_oxide.thickness_nm > 0:
        optical_layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))
    return make_stack(
        incident_medium=base_materials["air"],
        substrate=base_materials[substrate_name],
        layers=optical_layers,
        name=f"{observation.sample_name} empirical calibrated",
        display_layers=deposited_layers,
    )


def _effective_layer_material(
    base_material: Material,
    wavelengths_nm: FloatArray,
    layer,
    params: dict[str, float],
    stats: dict[str, float],
) -> Material:
    base_nk = base_material.refractive_index(wavelengths_nm)
    thickness_feature = (
        (float(layer.thickness_nm) - float(stats.get("thickness_center", layer.thickness_nm)))
        / float(stats.get("thickness_scale", 25.0) or 25.0)
    )
    if layer.time_min is None or not np.isfinite(float(layer.time_min)):
        time_feature = 0.0
    else:
        time_feature = (
            (float(layer.time_min) - float(stats.get("time_center", layer.time_min)))
            / float(stats.get("time_scale", 5.0) or 5.0)
        )
    dn = (
        float(params.get("dn0", 0.0))
        + float(params.get("dn_dthickness", 0.0)) * thickness_feature
        + float(params.get("dn_dtime", 0.0)) * time_feature
    )
    dk = (
        float(params.get("dk0", 0.0))
        + float(params.get("dk_dthickness", 0.0)) * thickness_feature
        + float(params.get("dk_dtime", 0.0)) * time_feature
    )
    n_values = np.clip(base_nk.real + dn, 0.02, 8.0)
    k_values = np.clip(base_nk.imag + dk, 0.0, 12.0)
    return make_tabulated_material(base_material.name, wavelengths_nm, n_values, k_values)


def _native_oxide(
    materials: dict[str, Material],
    substrate_name: str,
    enabled: bool,
    thickness_nm: float,
) -> NativeOxide | None:
    if not enabled:
        return None
    oxide = native_oxide_for_substrate(materials, substrate_name)
    if oxide is None:
        return None
    return NativeOxide(oxide.material, float(thickness_nm))


def _evaluate_predictions(
    *,
    observations: Sequence[_Observation],
    base_materials: dict[str, Material],
    model: OpticalModel,
    wavelengths_nm: FloatArray,
    color_cache,
    parameter_map: dict[str, dict[str, float]],
    fitted_materials: set[str],
    stats: dict[str, dict[str, float]],
    angle_deg: float,
    native_oxide_enabled: bool,
    native_oxide_thickness_nm: float,
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
    colour_metric: str,
) -> list[EmpiricalFitPrediction]:
    rows: list[EmpiricalFitPrediction] = []
    empty_map = {name: {} for name in fitted_materials}
    for observation in observations:
        baseline = _simulate_observation(
            observation=observation,
            base_materials=base_materials,
            model=model,
            wavelengths_nm=wavelengths_nm,
            parameter_map=empty_map,
            fitted_materials=fitted_materials,
            stats=stats,
            angle_deg=angle_deg,
            native_oxide_enabled=native_oxide_enabled,
            native_oxide_thickness_nm=native_oxide_thickness_nm,
            use_effective_interfaces=use_effective_interfaces,
            interface_thickness_nm=interface_thickness_nm,
            interface_fraction=interface_fraction,
        )
        fitted = _simulate_observation(
            observation=observation,
            base_materials=base_materials,
            model=model,
            wavelengths_nm=wavelengths_nm,
            parameter_map=parameter_map,
            fitted_materials=fitted_materials,
            stats=stats,
            angle_deg=angle_deg,
            native_oxide_enabled=native_oxide_enabled,
            native_oxide_thickness_nm=native_oxide_thickness_nm,
            use_effective_interfaces=use_effective_interfaces,
            interface_thickness_nm=interface_thickness_nm,
            interface_fraction=interface_fraction,
        )
        measured_xyz = reflectance_to_xyz(observation.measured_reflectance, cache=color_cache)
        baseline_xyz = reflectance_to_xyz(baseline, cache=color_cache)
        fitted_xyz = reflectance_to_xyz(fitted, cache=color_cache)
        measurement = observation.measurement
        rows.append(
            EmpiricalFitPrediction(
                split=observation.split,
                sample_name=observation.sample_name,
                measurement_index=int(observation.measurement_index),
                measurement_description=str(measurement.description),
                substrate_name=observation.substrate_name,
                surface_class=str(measurement.surface_class),
                measurement_kind=str(measurement.measurement_kind),
                stack_label=observation.sample.stack_label,
                baseline_delta_e=float(delta_e_colour(measured_xyz, baseline_xyz, metric=colour_metric)),
                fitted_delta_e=float(delta_e_colour(measured_xyz, fitted_xyz, metric=colour_metric)),
                baseline_rms_reflectance=float(
                    np.sqrt(np.mean((baseline - observation.measured_reflectance) ** 2))
                ),
                fitted_rms_reflectance=float(
                    np.sqrt(np.mean((fitted - observation.measured_reflectance) ** 2))
                ),
            )
        )
    return rows


def _mean_delta(
    predictions: Sequence[EmpiricalFitPrediction],
    split: str,
    *,
    after: bool,
) -> float | None:
    values = [
        prediction.fitted_delta_e if after else prediction.baseline_delta_e
        for prediction in predictions
        if prediction.split == split
    ]
    if not values:
        return None
    return float(np.mean(values))


def _save_result(
    output_dir: Path,
    parameters: Sequence[EmpiricalFitParameter],
    predictions: Sequence[EmpiricalFitPrediction],
    scipy_result,
) -> None:
    pd.DataFrame([asdict(parameter) for parameter in parameters]).to_csv(
        output_dir / "empirical_parameters.csv",
        index=False,
    )
    pd.DataFrame([asdict(prediction) for prediction in predictions]).to_csv(
        output_dir / "empirical_predictions.csv",
        index=False,
    )
    summary = {
        "parameters": [asdict(parameter) for parameter in parameters],
        "mean_delta_e": {
            "train_before": _mean_delta(predictions, "train", after=False),
            "train_after": _mean_delta(predictions, "train", after=True),
            "validation_before": _mean_delta(predictions, "validation", after=False),
            "validation_after": _mean_delta(predictions, "validation", after=True),
        },
        "scipy": {
            "cost": float(scipy_result.cost),
            "optimality": float(scipy_result.optimality),
            "nfev": int(scipy_result.nfev),
            "success": bool(scipy_result.success),
            "message": str(scipy_result.message),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
