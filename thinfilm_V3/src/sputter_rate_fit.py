"""Fit shared sputter rates from measured colours grouped by deposition settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import csv

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .color import prepare_color_conversion
from .colorimetry import perceived_color_from_result
from .experiments import (
    COLOUR_METRIC_CIE76,
    ExperimentDataStore,
    ExperimentLayerEstimate,
    ExperimentSample,
    build_stack_from_estimates,
    delta_e_colour,
    load_reflectance_csv,
    normalise_colour_metric,
)
from .materials import Material
from .optical_model import OpticalModel
from .stack import NativeOxide


@dataclass(frozen=True)
class SputterRateGroupFit:
    """Best colour-fitted deposition rate for one shared sputter setting."""

    material_name: str
    target: str
    pressure_mbar: float | None
    sccm: float | None
    base_rate_nm_per_min: float
    fitted_rate_nm_per_min: float
    percent_change: float
    mean_delta_e_before: float
    mean_delta_e_after: float
    measurement_count: int
    sample_count: int
    sample_names: tuple[str, ...]
    colour_metric: str = COLOUR_METRIC_CIE76

    @property
    def group_label(self) -> str:
        pressure = "unknown" if self.pressure_mbar is None else f"{self.pressure_mbar:g} mbar"
        flow = "unknown" if self.sccm is None else f"{self.sccm:g} sccm"
        return f"{self.material_name} {self.target or 'T?'} {pressure} {flow}"


def fit_sputter_rates_from_colour(
    store: ExperimentDataStore,
    materials: dict[str, Material],
    model: OpticalModel,
    wavelengths_nm: NDArray[np.float64],
    angle_deg: float,
    substrate_default: str,
    native_oxide_factory: Any | None,
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
    range_percent: float = 50.0,
    num_points: int = 81,
    selected_group_keys: set[tuple[str, str, float | None, float | None]] | None = None,
    sample_filter: Any | None = None,
    colour_metric: str = COLOUR_METRIC_CIE76,
    progress_callback: Any | None = None,
) -> list[SputterRateGroupFit]:
    """Fit one shared rate per material/target/pressure/sccm group from colour.

    Other layers stay at their current calibrated estimates while the selected
    group is varied as ``thickness = trial_rate * deposition_time``.
    """

    metric = normalise_colour_metric(colour_metric)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    color_cache = prepare_color_conversion(wavelengths)
    groups = _collect_rate_groups(store, sample_filter=sample_filter)
    if selected_group_keys:
        groups = {
            key: refs
            for key, refs in groups.items()
            if key in selected_group_keys
        }
    results: list[SputterRateGroupFit] = []
    total_trials = max(len(groups) * int(num_points), 1)
    completed_trials = 0

    for group_index, (group_key, layer_refs) in enumerate(groups.items(), start=1):
        material_name, target, pressure_mbar, sccm = group_key
        base_rates = [ref["layer"].rate_nm_per_min for ref in layer_refs if ref["layer"].rate_nm_per_min]
        if not base_rates:
            continue
        base_rate = float(np.median(np.asarray(base_rates, dtype=float)))
        factors = np.linspace(1.0 - range_percent / 100.0, 1.0 + range_percent / 100.0, int(num_points))
        trial_rates = np.maximum(base_rate * factors, 0.0)

        before_values, used_measurements, sample_names = _evaluate_group_rate(
            store=store,
            materials=materials,
            model=model,
            wavelengths=wavelengths,
            color_cache=color_cache,
            angle_deg=angle_deg,
            substrate_default=substrate_default,
            native_oxide_factory=native_oxide_factory,
            use_effective_interfaces=use_effective_interfaces,
            interface_thickness_nm=interface_thickness_nm,
            interface_fraction=interface_fraction,
            group_key=group_key,
            layer_refs=layer_refs,
            trial_rate=base_rate,
            colour_metric=metric,
        )
        if not before_values:
            continue
        best_rate = base_rate
        best_mean = float(np.mean(before_values))
        for trial_number, trial_rate in enumerate(trial_rates, start=1):
            delta_values, _used, _samples = _evaluate_group_rate(
                store=store,
                materials=materials,
                model=model,
                wavelengths=wavelengths,
                color_cache=color_cache,
                angle_deg=angle_deg,
                substrate_default=substrate_default,
                native_oxide_factory=native_oxide_factory,
                use_effective_interfaces=use_effective_interfaces,
                interface_thickness_nm=interface_thickness_nm,
                interface_fraction=interface_fraction,
                group_key=group_key,
                layer_refs=layer_refs,
                trial_rate=float(trial_rate),
                colour_metric=metric,
            )
            completed_trials += 1
            if progress_callback is not None:
                progress_callback(
                    completed_trials,
                    total_trials,
                    f"{material_name} {target} trial {trial_number}/{len(trial_rates)}",
                )
            if not delta_values:
                continue
            mean_delta = float(np.mean(delta_values))
            if mean_delta < best_mean:
                best_mean = mean_delta
                best_rate = float(trial_rate)

        if not np.isfinite(best_mean):
            continue
        results.append(
            SputterRateGroupFit(
                material_name=material_name,
                target=target,
                pressure_mbar=pressure_mbar,
                sccm=sccm,
                base_rate_nm_per_min=base_rate,
                fitted_rate_nm_per_min=best_rate,
                percent_change=100.0 * (best_rate / base_rate - 1.0) if base_rate else 0.0,
                mean_delta_e_before=float(np.mean(before_values)),
                mean_delta_e_after=best_mean,
                measurement_count=used_measurements,
                sample_count=len(sample_names),
                sample_names=tuple(sorted(sample_names)),
                colour_metric=metric,
            )
        )

    return sorted(results, key=lambda item: (item.material_name, item.target, item.pressure_mbar or 0.0))


def save_sputter_rate_fit_outputs(
    results: list[SputterRateGroupFit],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save colour-fitted rate table and summary plots."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "colour_fitted_sputter_rates.csv"
    rows = [_result_row(result) for result in results]
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    rate_plot = out / "colour_fitted_rate_change.png"
    delta_plot = out / "colour_fitted_delta_e_before_after.png"
    _plot_rate_change(results, rate_plot)
    _plot_delta_change(results, delta_plot)
    return {"csv": csv_path, "rate_plot": rate_plot, "delta_plot": delta_plot}


def _collect_rate_groups(
    store: ExperimentDataStore,
    sample_filter: Any | None = None,
) -> dict[tuple[str, str, float | None, float | None], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, float | None, float | None], list[dict[str, Any]]] = {}
    for sample_name in store.sample_names(require_spectra=True):
        sample = store.load_sample(sample_name)
        if not sample.measurements:
            continue
        if sample_filter is not None and not sample_filter(sample):
            continue
        for layer in sample.layer_estimates:
            if layer.time_min is None or layer.time_min <= 0:
                continue
            if not layer.target or layer.rate_nm_per_min is None:
                continue
            key = (
                layer.material_name,
                layer.target,
                None if layer.pressure_mbar is None else round(float(layer.pressure_mbar), 7),
                None if layer.sccm is None else round(float(layer.sccm), 3),
            )
            groups.setdefault(key, []).append({"sample_name": sample_name, "layer": layer})
    return groups


def _evaluate_group_rate(
    store: ExperimentDataStore,
    materials: dict[str, Material],
    model: OpticalModel,
    wavelengths: NDArray[np.float64],
    color_cache,
    angle_deg: float,
    substrate_default: str,
    native_oxide_factory: Any | None,
    use_effective_interfaces: bool,
    interface_thickness_nm: float,
    interface_fraction: float,
    group_key: tuple[str, str, float | None, float | None],
    layer_refs: list[dict[str, Any]],
    trial_rate: float,
    colour_metric: str = COLOUR_METRIC_CIE76,
) -> tuple[list[float], int, set[str]]:
    metric = normalise_colour_metric(colour_metric)
    delta_values: list[float] = []
    sample_names: set[str] = set()
    seen_measurements: set[tuple[str, str]] = set()
    for ref in layer_refs:
        sample = store.load_sample(ref["sample_name"])
        sample_names.add(sample.sample_name)
        modified = _sample_with_trial_rate(sample, group_key, float(trial_rate))
        for measurement in sample.measurements:
            measurement_key = (sample.sample_name, measurement.description)
            if measurement_key in seen_measurements:
                continue
            seen_measurements.add(measurement_key)
            substrate = measurement.substrate_hint or substrate_default
            native_oxide = (
                native_oxide_factory(substrate)
                if native_oxide_factory is not None
                else None
            )
            try:
                stack = build_stack_from_estimates(
                    modified,
                    materials=materials,
                    substrate_name=substrate,
                    native_oxide=native_oxide,
                    use_effective_interfaces=use_effective_interfaces,
                    interface_thickness_nm=interface_thickness_nm,
                    interface_fraction=interface_fraction,
                )
                measured_wl, measured_raw = load_reflectance_csv(measurement.csv_path)
                measured_reflectance = np.interp(wavelengths, measured_wl, measured_raw)
                measured_xyz = _xyz_from_reflectance(measured_reflectance, color_cache)
                simulated = model.simulate(stack, wavelengths, angle_deg)
                simulated_color = perceived_color_from_result(simulated)
            except Exception:
                continue
            delta_values.append(delta_e_colour(measured_xyz, simulated_color.xyz, metric=metric))
    return delta_values, len(delta_values), sample_names


def _sample_with_trial_rate(
    sample: ExperimentSample,
    group_key: tuple[str, str, float | None, float | None],
    trial_rate_nm_per_min: float,
) -> ExperimentSample:
    material_name, target, pressure_mbar, sccm = group_key
    layers: list[ExperimentLayerEstimate] = []
    for layer in sample.layer_estimates:
        pressure_match = (
            pressure_mbar is None
            if layer.pressure_mbar is None
            else pressure_mbar is not None and np.isclose(layer.pressure_mbar, pressure_mbar)
        )
        sccm_match = (
            sccm is None
            if layer.sccm is None
            else sccm is not None and np.isclose(layer.sccm, sccm)
        )
        if (
            layer.material_name == material_name
            and layer.target == target
            and pressure_match
            and sccm_match
            and layer.time_min is not None
        ):
            layers.append(
                replace(
                    layer,
                    thickness_nm=float(trial_rate_nm_per_min) * float(layer.time_min),
                    rate_nm_per_min=float(trial_rate_nm_per_min),
                    rate_source="colour_fitted_rate_trial",
                )
            )
        else:
            layers.append(layer)
    return ExperimentSample(sample.sample_name, tuple(layers), sample.measurements)


def _xyz_from_reflectance(reflectance: NDArray[np.float64], color_cache) -> tuple[float, float, float]:
    from .color import reflectance_to_xyz

    xyz = reflectance_to_xyz(reflectance, cache=color_cache)
    return tuple(float(value) for value in xyz)


def _result_row(result: SputterRateGroupFit) -> dict[str, object]:
    return {
        "material": result.material_name,
        "target": result.target,
        "pressure_mbar": result.pressure_mbar,
        "sccm": result.sccm,
        "base_rate_nm_per_min": result.base_rate_nm_per_min,
        "colour_fitted_rate_nm_per_min": result.fitted_rate_nm_per_min,
        "percent_change": result.percent_change,
        "mean_delta_e_before": result.mean_delta_e_before,
        "mean_delta_e_after": result.mean_delta_e_after,
        "colour_metric": result.colour_metric,
        "measurement_count": result.measurement_count,
        "sample_count": result.sample_count,
        "sample_names": ", ".join(result.sample_names),
    }


def _plot_rate_change(results: list[SputterRateGroupFit], path: Path) -> None:
    if not results:
        return
    fig = Figure(figsize=(10.5, 4.8), dpi=170)
    ax = fig.subplots()
    x = np.arange(len(results))
    labels = [result.group_label for result in results]
    ax.bar(x, [result.percent_change for result in results], color="#2f6f9f")
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_ylabel("Colour-fitted rate change (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_title("Sputter-rate correction fitted from measured colour")
    ax.grid(True, axis="y", alpha=0.28)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.88)
    fig.savefig(path)


def _plot_delta_change(results: list[SputterRateGroupFit], path: Path) -> None:
    if not results:
        return
    fig = Figure(figsize=(10.5, 4.8), dpi=170)
    ax = fig.subplots()
    x = np.arange(len(results))
    labels = [result.group_label for result in results]
    ax.plot(x, [result.mean_delta_e_before for result in results], marker="o", label="Before")
    ax.plot(x, [result.mean_delta_e_after for result in results], marker="o", label="After")
    ax.set_ylabel(r"Mean $\Delta E^*_{Lab}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_title("Mean colour distance before/after grouped rate fit")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.88)
    fig.savefig(path)
