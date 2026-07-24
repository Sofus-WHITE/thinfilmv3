"""Experimental sample loading and simulated/measured colour comparison."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .color import reflectance_to_srgb, reflectance_to_xyz, xyz_to_srgb
from .colorimetry import PerceivedColor, perceived_color_from_result
from .materials import Material
from .optical_model import OpticalModel
from .results import SimulationResult
from .stack import Layer, NativeOxide, ThinFilmStack, make_stack, make_stack_with_interfaces


FloatArray = NDArray[np.float64]
COLOUR_METRIC_CIE76 = "cie76"
COLOUR_METRIC_CIEDE2000 = "ciede2000"
COLOUR_METRIC_LABELS = {
    COLOUR_METRIC_CIE76: "CIE76 (Delta E*ab)",
    COLOUR_METRIC_CIEDE2000: "CIEDE2000",
}


@dataclass(frozen=True)
class ExperimentLayerEstimate:
    """One estimated deposited layer for an experimental sample."""

    material_name: str
    thickness_nm: float
    layer_order: int
    time_min: float | None = None
    rate_nm_per_min: float | None = None
    confidence: str = ""
    source: str = ""
    target: str = ""
    pressure_mbar: float | None = None
    sccm: float | None = None
    calibration_sample: str = ""
    rate_source: str = ""
    deposition_date: str = ""


@dataclass(frozen=True)
class LatestSputterRate:
    """Current sputter-rate estimate for planning a new deposited layer."""

    material_name: str
    rate_nm_per_min: float
    error_nm_per_min: float | None = None
    target: str = ""
    pressure_mbar: float | None = None
    pressure_label: str = ""
    sccm: float | None = None
    source: str = ""
    period: str = ""
    confidence: str = ""
    basis: str = ""
    sample_name: str = ""
    deposition_date: str = ""
    row_index: int = 0

    @property
    def settings_label(self) -> str:
        parts: list[str] = []
        if self.target:
            parts.append(self.target)
        if self.pressure_label:
            parts.append(self.pressure_label)
        elif self.pressure_mbar is not None:
            parts.append(f"{self.pressure_mbar:.3g} mbar")
        if self.period:
            parts.append(self.period)
        return ", ".join(parts)


@dataclass(frozen=True)
class ExperimentMeasurement:
    """One measured reflectance spectrum linked to a sample."""

    description: str
    csv_path: Path
    source_system: str = ""
    instrument_sample_id: str = ""
    measurement_angle_deg: float | None = None
    substrate_hint: str | None = None
    substrate_group: str = ""
    measurement_mode: str = ""
    accessory: str = ""
    surface_class: str = "smooth"
    measurement_kind: str = "unknown"
    sample_condition_note: str = ""


@dataclass(frozen=True)
class ExperimentSample:
    """Sample metadata needed to compare estimated and measured colour."""

    sample_name: str
    layer_estimates: tuple[ExperimentLayerEstimate, ...]
    measurements: tuple[ExperimentMeasurement, ...]

    @property
    def sample_series(self) -> str:
        """Leading sample-series label, such as A, B, C, S, or Au."""

        return sample_series_from_name(self.sample_name)

    @property
    def stack_label(self) -> str:
        layers = " | ".join(
            f"{layer.thickness_nm:g} nm {layer.material_name}" for layer in self.layer_estimates
        )
        return layers if layers else "no layer estimates"


@dataclass(frozen=True)
class ExperimentComparisonResult:
    """Measured and simulated spectra/colours for one selected sample measurement."""

    sample_name: str
    stack: ThinFilmStack
    measurement: ExperimentMeasurement
    measured_wavelengths_nm: FloatArray
    measured_reflectance: FloatArray
    simulated_result: SimulationResult
    measured_color: PerceivedColor
    simulated_color: PerceivedColor
    layer_estimates: tuple[ExperimentLayerEstimate, ...]


@dataclass(frozen=True)
class CachedExperimentResults:
    """Saved experiment comparisons that can be redrawn without rerunning TMM."""

    wavelengths_nm: FloatArray
    sample_names: NDArray[np.str_]
    measurement_descriptions: NDArray[np.str_]
    sample_series: NDArray[np.str_]
    substrate_classes: NDArray[np.str_]
    surface_classes: NDArray[np.str_]
    measurement_kinds: NDArray[np.str_]
    source_systems: NDArray[np.str_]
    stack_labels: NDArray[np.str_]
    measured_reflectance: FloatArray
    simulated_reflectance: FloatArray
    measured_rgb: FloatArray
    simulated_rgb: FloatArray
    measured_xyz: FloatArray
    simulated_xyz: FloatArray
    measured_xy: FloatArray
    simulated_xy: FloatArray
    delta_e: FloatArray
    colour_metric: str = COLOUR_METRIC_CIE76

    @property
    def count(self) -> int:
        """Number of cached sample/measurement comparisons."""

        return int(self.sample_names.size)


class ExperimentDataStore:
    """Read the exported sample_data folder without changing the raw experiment files."""

    def __init__(self, sample_data_root: str | Path) -> None:
        self.sample_data_root = Path(sample_data_root)
        self.reflectivity_root = self.sample_data_root.parent
        self._sample_index = _read_csv(self.sample_data_root / "sample_index.csv")
        self._thickness = _read_csv(self.sample_data_root / "thickness_estimates.csv")
        self._measurements = _read_csv(self.sample_data_root / "measurement_index.csv")
        self._sample_condition_notes = self._build_sample_condition_notes()
        self._intended_thickness = _read_intended_thickness_workbook(self.sample_data_root)
        self._rate_calibrations = self._build_single_film_rate_calibrations()
        self._latest_sputter_rates_cache: dict[str, LatestSputterRate] | None = None

    def sample_names(self, require_spectra: bool = True) -> list[str]:
        """Return sample names, optionally limited to samples with linked spectra."""

        if self._sample_index.empty:
            return []
        data = self._sample_index.copy()
        if require_spectra and "spectra" in data.columns:
            data = data[pd.to_numeric(data["spectra"], errors="coerce").fillna(0) > 0]
        return sorted(data["sample_name"].dropna().astype(str).unique())

    def latest_sputter_rates(self) -> dict[str, LatestSputterRate]:
        """Return the latest available sputter-rate estimate for each material."""

        if self._latest_sputter_rates_cache is None:
            self._latest_sputter_rates_cache = self._build_latest_sputter_rates()
        return dict(self._latest_sputter_rates_cache)

    def latest_sputter_rate(self, material_name: str) -> LatestSputterRate | None:
        """Return the latest sputter-rate estimate for one material, if known."""

        return self.latest_sputter_rates().get(_clean_text(material_name))

    def substrate_group_for_measurement(
        self,
        sample_name: str,
        measurement_description: str = "",
        fallback: str = "",
    ) -> str:
        """Return the browsing/filter group for a cached measurement."""

        sample_note = self._sample_condition_notes.get(sample_name, "")
        fallback_group = normalize_substrate_group_label(fallback) or substrate_group_label(
            _substrate_from_description(measurement_description) or "Si",
            sample_note,
            measurement_description,
        )
        for measurement in self._measurements_for_sample(sample_name):
            if measurement.description == measurement_description:
                return measurement.substrate_group or fallback_group
        return substrate_group_label(fallback_group, sample_note, measurement_description)

    def _build_sample_condition_notes(self) -> dict[str, str]:
        """Collect sample-level notes such as double-polished Si from indexes/manifests."""

        notes_by_sample: dict[str, list[str]] = {}

        def add_note(sample_name: object, note: object) -> None:
            name = _clean_text(sample_name)
            text = _clean_text(note)
            if not name or not text:
                return
            notes_by_sample.setdefault(name, []).append(text)

        if "sample_condition_note" in self._measurements.columns:
            for _, row in self._measurements.iterrows():
                add_note(row.get("sample_name"), row.get("sample_condition_note"))

        manifest_root = self.sample_data_root / "by_sample"
        if manifest_root.exists():
            for manifest_path in manifest_root.glob("*/manifest.json"):
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                sample_name = manifest_path.parent.name
                for note in _iter_manifest_condition_notes(payload):
                    add_note(sample_name, note)

        return {sample: _join_unique_texts(notes) for sample, notes in notes_by_sample.items()}

    def _build_latest_sputter_rates(self) -> dict[str, LatestSputterRate]:
        """Build date-aware latest rates, with curated/direct values as fallbacks."""

        latest = self._latest_assigned_sputter_rates()
        for material, record in self._latest_direct_sputter_rates().items():
            latest.setdefault(material, record)
        for material, record in self._latest_thickness_row_sputter_rates().items():
            latest.setdefault(material, record)
        return dict(sorted(latest.items(), key=lambda item: item[0].lower()))

    def _latest_assigned_sputter_rates(self) -> dict[str, LatestSputterRate]:
        path = self.reflectivity_root / "Thickness" / "all_sputtering_assigned_rates.csv"
        data = _read_csv(path)
        if data.empty:
            return {}

        latest: dict[str, LatestSputterRate] = {}
        sort_keys: dict[str, tuple[tuple[int, int, int], int]] = {}
        for row_index, row in data.iterrows():
            material = _clean_text(row.get("material"))
            rate = _optional_float(row.get("assigned_rate_nm_min"))
            if not material or rate is None or rate <= 0:
                continue
            sort_key = (_date_sort_key(row.get("date")), int(row_index))
            if material in latest and sort_key <= sort_keys[material]:
                continue
            pressure_text = _clean_text(row.get("pressure"))
            latest[material] = LatestSputterRate(
                material_name=material,
                rate_nm_per_min=float(rate),
                error_nm_per_min=_optional_float(row.get("error_nm_min")),
                target=_clean_text(row.get("target")),
                pressure_mbar=_optional_pressure_mbar(pressure_text),
                pressure_label=pressure_text,
                source="Thickness/all_sputtering_assigned_rates.csv",
                period=_clean_text(row.get("rate_series")),
                basis=_clean_text(row.get("assignment_basis")),
                sample_name=_clean_text(row.get("entry")),
                deposition_date=_clean_text(row.get("date")),
                row_index=int(row_index),
            )
            sort_keys[material] = sort_key
        return latest

    def _latest_direct_sputter_rates(self) -> dict[str, LatestSputterRate]:
        path = self.reflectivity_root / "Thickness" / "sputtering_rate_direct_numbers.csv"
        data = _read_csv(path)
        if data.empty:
            return {}

        latest: dict[str, LatestSputterRate] = {}
        for row_index, row in data.iterrows():
            material = _clean_text(row.get("material"))
            rate = _optional_float(row.get("recommended_rate_nm_min"))
            if not material or rate is None or rate <= 0:
                continue
            pressure_text = _clean_text(row.get("pressure_mbar"))
            latest[material] = LatestSputterRate(
                material_name=material,
                rate_nm_per_min=float(rate),
                error_nm_per_min=_optional_float(row.get("error_estimate_nm_min")),
                target=_clean_text(row.get("target")),
                pressure_mbar=_optional_pressure_mbar(pressure_text),
                pressure_label=pressure_text,
                source="Thickness/sputtering_rate_direct_numbers.csv",
                period=_clean_text(row.get("series_or_period")),
                confidence=_clean_text(row.get("confidence")),
                basis=_clean_text(row.get("basis")),
                row_index=int(row_index),
            )
        return latest

    def _latest_thickness_row_sputter_rates(self) -> dict[str, LatestSputterRate]:
        if self._thickness.empty:
            return {}

        latest: dict[str, LatestSputterRate] = {}
        sort_keys: dict[str, tuple[tuple[int, int, int], int]] = {}
        for row_index, row in self._thickness.iterrows():
            material = _clean_text(row.get("material"))
            rate = _optional_float(row.get("rate_nm_per_min"))
            if not material or rate is None or rate <= 0:
                continue
            metadata = _parse_semicolon_metadata(row.get("notes"))
            date_text = _clean_text(
                metadata.get("calibration_date") or metadata.get("deposition_date")
            )
            sort_key = (_date_sort_key(date_text), int(row_index))
            if material in latest and sort_key <= sort_keys[material]:
                continue
            pressure_text = _clean_text(metadata.get("pressure"))
            source = _clean_text(row.get("source")) or "sample_data/thickness_estimates.csv"
            basis = _clean_text(row.get("method")) or _clean_text(row.get("measurement_method"))
            latest[material] = LatestSputterRate(
                material_name=material,
                rate_nm_per_min=float(rate),
                target=_clean_text(metadata.get("target")),
                pressure_mbar=_optional_pressure_mbar(pressure_text),
                pressure_label=pressure_text,
                sccm=_optional_float(metadata.get("sccm")),
                source=source,
                confidence=_clean_text(row.get("confidence")),
                basis=basis,
                sample_name=_clean_text(row.get("sample_name")),
                deposition_date=date_text,
                row_index=int(row_index),
            )
            sort_keys[material] = sort_key
        return latest

    def _build_single_film_rate_calibrations(self) -> dict[str, dict[tuple, Any]]:
        """Build sputter-rate lookups from single-deposited calibration films."""

        if self._thickness.empty:
            return {"by_sample": {}, "by_setting": {}}

        data = self._thickness.copy()
        data["numeric_layer_order"] = pd.to_numeric(data.get("layer_order"), errors="coerce")
        numeric = data[data["numeric_layer_order"].notna()].copy()
        if numeric.empty:
            return {"by_sample": {}, "by_setting": {}}

        single_samples = {
            str(sample_name)
            for sample_name, group in numeric.groupby(numeric["sample_name"].astype(str))
            if group["material"].dropna().astype(str).nunique() == 1
        }

        by_sample: dict[tuple[str, str], dict[str, Any]] = {}
        setting_rates: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
        for _, row in numeric.iterrows():
            sample_name = str(row.get("sample_name", "") or "")
            if sample_name not in single_samples:
                continue
            material = str(row.get("material", "") or "").strip()
            rate = _optional_float(row.get("rate_nm_per_min"))
            time_min = _optional_float(row.get("time_min"))
            if not material or rate is None or time_min is None or time_min <= 0:
                continue
            metadata = _parse_semicolon_metadata(row.get("notes"))
            target = str(metadata.get("target", "") or "").strip()
            pressure = _optional_float(metadata.get("pressure"))
            record = {
                "rate_nm_per_min": float(rate),
                "sample_name": sample_name,
                "material": material,
                "target": target,
                "pressure_mbar": pressure,
                "sccm": _optional_float(metadata.get("sccm")),
                "source": str(row.get("source", "") or ""),
            }
            by_sample[(sample_name, material)] = record
            key = _rate_setting_key(material, target, pressure)
            if key is not None:
                setting_rates.setdefault(key, []).append(record)

        by_setting: dict[tuple[str, str, float], dict[str, Any]] = {}
        for key, records in setting_rates.items():
            rates = np.asarray([record["rate_nm_per_min"] for record in records], dtype=float)
            representative = dict(records[int(np.argsort(np.abs(rates - np.median(rates)))[0])])
            representative["rate_nm_per_min"] = float(np.median(rates))
            representative["calibration_count"] = len(records)
            representative["calibration_samples"] = ",".join(
                sorted({str(record["sample_name"]) for record in records})
            )
            by_setting[key] = representative

        return {"by_sample": by_sample, "by_setting": by_setting}

    def _calibrated_rate_for_layer(
        self,
        material: str,
        metadata: dict[str, str],
    ) -> dict[str, Any] | None:
        calibration_sample = str(metadata.get("calibration_sample", "") or "").strip()
        if calibration_sample:
            record = self._rate_calibrations["by_sample"].get((calibration_sample, material))
            if record is not None:
                return dict(record)
        key = _rate_setting_key(
            material,
            str(metadata.get("target", "") or "").strip(),
            _optional_float(metadata.get("pressure")),
        )
        if key is None:
            return None
        record = self._rate_calibrations["by_setting"].get(key)
        return dict(record) if record is not None else None

    def load_sample(self, sample_name: str) -> ExperimentSample:
        """Load thickness estimates and available spectra for one sample."""

        layer_estimates = tuple(self._layer_estimates(sample_name))
        measurements = tuple(self._measurements_for_sample(sample_name))
        return ExperimentSample(
            sample_name=sample_name,
            layer_estimates=layer_estimates,
            measurements=measurements,
        )

    def compare_sample(
        self,
        sample_name: str,
        measurement_index: int,
        materials: dict[str, Material],
        model: OpticalModel,
        wavelengths_nm: NDArray[np.float64],
        angle_deg: float,
        substrate_name: str = "Si",
        native_oxide: NativeOxide | None = None,
        use_effective_interfaces: bool = False,
        interface_thickness_nm: float = 1.0,
        interface_fraction: float = 0.5,
    ) -> ExperimentComparisonResult:
        """Build an estimated stack and compare it to one measured spectrum."""

        sample = self.load_sample(sample_name)
        if not sample.measurements:
            raise ValueError(f"Sample {sample_name} has no linked reflectance spectra.")
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
        measured_wavelengths, measured_reflectance = load_reflectance_csv(measurement.csv_path)
        measured_on_grid = np.interp(wavelengths_nm, measured_wavelengths, measured_reflectance)

        simulated_result = model.simulate(stack, wavelengths_nm, angle_deg)
        measured_color = _perceived_color_from_arrays(wavelengths_nm, measured_on_grid)
        simulated_color = perceived_color_from_result(simulated_result)

        return ExperimentComparisonResult(
            sample_name=sample.sample_name,
            stack=stack,
            measurement=measurement,
            measured_wavelengths_nm=measured_wavelengths,
            measured_reflectance=measured_reflectance,
            simulated_result=simulated_result,
            measured_color=measured_color,
            simulated_color=simulated_color,
            layer_estimates=sample.layer_estimates,
        )

    def build_cached_results(
        self,
        materials: dict[str, Material],
        model: OpticalModel,
        wavelengths_nm: NDArray[np.float64],
        angle_deg: float,
        substrate_name: str = "Si",
        native_oxide_factory: Any | None = None,
        use_effective_interfaces: bool = False,
        interface_thickness_nm: float = 1.0,
        interface_fraction: float = 0.5,
        max_measurements_per_sample: int | None = None,
        colour_metric: str = COLOUR_METRIC_CIE76,
    ) -> CachedExperimentResults:
        """Calculate all sample comparisons once for later instant browsing."""

        metric = normalise_colour_metric(colour_metric)
        rows: list[ExperimentComparisonResult] = []
        for sample_name in self.sample_names(require_spectra=True):
            sample = self.load_sample(sample_name)
            if not sample.layer_estimates or not sample.measurements:
                continue
            selected_measurements = (
                sample.measurements
                if max_measurements_per_sample is None
                else sample.measurements[:max_measurements_per_sample]
            )
            for measurement_index, measurement in enumerate(selected_measurements):
                substrate = measurement.substrate_hint or substrate_name
                native_oxide = (
                    native_oxide_factory(substrate)
                    if native_oxide_factory is not None
                    else None
                )
                try:
                    rows.append(
                        self.compare_sample(
                            sample_name=sample_name,
                            measurement_index=measurement_index,
                            materials=materials,
                            model=model,
                            wavelengths_nm=wavelengths_nm,
                            angle_deg=angle_deg,
                            substrate_name=substrate,
                            native_oxide=native_oxide,
                            use_effective_interfaces=use_effective_interfaces,
                            interface_thickness_nm=interface_thickness_nm,
                            interface_fraction=interface_fraction,
                        )
                    )
                except Exception:
                    continue

        if not rows:
            raise ValueError("No experiment comparisons could be calculated.")

        measured_reflectance = []
        simulated_reflectance = []
        measured_rgb = []
        simulated_rgb = []
        measured_xyz = []
        simulated_xyz = []
        measured_xy = []
        simulated_xy = []
        delta_e = []
        for row in rows:
            measured_on_grid = np.interp(
                wavelengths_nm,
                row.measured_wavelengths_nm,
                row.measured_reflectance,
            )
            measured_reflectance.append(measured_on_grid)
            simulated_reflectance.append(row.simulated_result.reflectance)
            measured_rgb.append(row.measured_color.srgb)
            simulated_rgb.append(row.simulated_color.srgb)
            measured_xyz.append(row.measured_color.xyz)
            simulated_xyz.append(row.simulated_color.xyz)
            measured_xy.append(xyz_to_xy(row.measured_color.xyz))
            simulated_xy.append(xyz_to_xy(row.simulated_color.xyz))
            delta_e.append(delta_e_colour(row.measured_color.xyz, row.simulated_color.xyz, metric=metric))

        return CachedExperimentResults(
            wavelengths_nm=np.asarray(wavelengths_nm, dtype=float),
            sample_names=np.asarray([row.sample_name for row in rows], dtype=str),
            measurement_descriptions=np.asarray([row.measurement.description for row in rows], dtype=str),
            sample_series=np.asarray([sample_series_from_name(row.sample_name) for row in rows], dtype=str),
            substrate_classes=np.asarray(
                [row.measurement.substrate_group or row.measurement.substrate_hint or substrate_name for row in rows],
                dtype=str,
            ),
            surface_classes=np.asarray([row.measurement.surface_class for row in rows], dtype=str),
            measurement_kinds=np.asarray([row.measurement.measurement_kind for row in rows], dtype=str),
            source_systems=np.asarray([row.measurement.source_system for row in rows], dtype=str),
            stack_labels=np.asarray([row.stack.display_summary() for row in rows], dtype=str),
            measured_reflectance=np.asarray(measured_reflectance, dtype=float),
            simulated_reflectance=np.asarray(simulated_reflectance, dtype=float),
            measured_rgb=np.asarray(measured_rgb, dtype=float),
            simulated_rgb=np.asarray(simulated_rgb, dtype=float),
            measured_xyz=np.asarray(measured_xyz, dtype=float),
            simulated_xyz=np.asarray(simulated_xyz, dtype=float),
            measured_xy=np.asarray(measured_xy, dtype=float),
            simulated_xy=np.asarray(simulated_xy, dtype=float),
            delta_e=np.asarray(delta_e, dtype=float),
            colour_metric=metric,
        )

    def _layer_estimates(self, sample_name: str) -> list[ExperimentLayerEstimate]:
        intended = self._layer_estimates_from_intended_workbook(sample_name)
        if intended:
            return intended

        rows = self._thickness[self._thickness["sample_name"].astype(str) == sample_name]
        if rows.empty:
            return []

        rows = rows.copy()
        rows["numeric_layer_order"] = pd.to_numeric(rows["layer_order"], errors="coerce")
        numeric_rows = rows[rows["numeric_layer_order"].notna()].copy()
        top_rows = rows[rows["layer_order"].astype(str).str.lower().eq("top")].copy()
        if not numeric_rows.empty:
            numeric_rows["optical_order"] = -numeric_rows["numeric_layer_order"]
            if not top_rows.empty:
                top_rows["numeric_layer_order"] = -1_000_000
                top_rows["optical_order"] = -1_000_000
                ordered = pd.concat([top_rows, numeric_rows], ignore_index=True)
            else:
                ordered = numeric_rows
        else:
            ordered = rows.copy()
            order_text = ordered["layer_order"].astype(str).str.lower()
            top_mask = order_text.eq("top")
            ordered["numeric_layer_order"] = range(1, len(ordered) + 1)
            ordered["optical_order"] = ordered["numeric_layer_order"]
            ordered.loc[top_mask, "optical_order"] = -1_000_000
        ordered["thickness_nm_estimate"] = pd.to_numeric(
            ordered["thickness_nm_estimate"], errors="coerce"
        )
        ordered = ordered.dropna(subset=["material", "thickness_nm_estimate", "numeric_layer_order"])
        ordered = ordered.sort_values(["optical_order", "material"])

        estimates: list[ExperimentLayerEstimate] = []
        seen_orders: set[float] = set()
        for _, row in ordered.iterrows():
            layer_order = float(row["numeric_layer_order"])
            if layer_order in seen_orders:
                continue
            seen_orders.add(layer_order)
            material = str(row["material"])
            notes_metadata = _parse_semicolon_metadata(row.get("notes"))
            time_min = _optional_float(row.get("time_min"))
            rate_nm_per_min = _optional_float(row.get("rate_nm_per_min"))
            thickness_nm = float(row["thickness_nm_estimate"])
            calibration = self._calibrated_rate_for_layer(material, notes_metadata)
            rate_source = str(notes_metadata.get("rate_source", "") or "")
            if calibration is not None and time_min is not None and time_min > 0:
                rate_nm_per_min = float(calibration["rate_nm_per_min"])
                thickness_nm = rate_nm_per_min * time_min
                calibration_sample = str(calibration.get("sample_name", "") or "")
                rate_source = f"single_film_calibration:{calibration_sample}"
            else:
                calibration_sample = str(notes_metadata.get("calibration_sample", "") or "")
            estimates.append(
                ExperimentLayerEstimate(
                    material_name=material,
                    thickness_nm=thickness_nm,
                    layer_order=int(layer_order),
                    time_min=time_min,
                    rate_nm_per_min=rate_nm_per_min,
                    confidence=str(row.get("confidence", "") or ""),
                    source=str(row.get("source", "") or ""),
                    target=str(notes_metadata.get("target", "") or ""),
                    pressure_mbar=_optional_float(notes_metadata.get("pressure")),
                    sccm=_optional_float(notes_metadata.get("sccm")),
                    calibration_sample=calibration_sample,
                    rate_source=rate_source,
                    deposition_date=str(notes_metadata.get("calibration_date", "") or ""),
                )
            )
        return estimates

    def _layer_estimates_from_intended_workbook(self, sample_name: str) -> list[ExperimentLayerEstimate]:
        """Use the final intended-thickness workbook as the preferred sample thickness source."""

        if self._intended_thickness.empty:
            return []
        rows = self._intended_thickness[self._intended_thickness["entry"].astype(str) == sample_name].copy()
        if rows.empty:
            return []

        fallback = self._fallback_layer_estimates_by_material(sample_name)
        estimates: list[ExperimentLayerEstimate] = []
        for deposition_index, (_, row) in enumerate(rows.iterrows(), start=1):
            material = str(row.get("material", "") or "").strip()
            if not material:
                continue
            thickness_nm = _optional_float(row.get("intended_thickness_nm"))
            rate_nm_per_min = _optional_float(row.get("assigned_rate_nm_min"))
            time_min = _optional_float(row.get("sputter_time_min"))
            fallback_layer = fallback.get(material)
            if thickness_nm is None:
                if fallback_layer is None:
                    continue
                thickness_nm = fallback_layer.thickness_nm
            if rate_nm_per_min is None and fallback_layer is not None:
                rate_nm_per_min = fallback_layer.rate_nm_per_min
            if time_min is None and fallback_layer is not None:
                time_min = fallback_layer.time_min
            pressure = _optional_pressure_mbar(row.get("pressure"))
            estimates.append(
                ExperimentLayerEstimate(
                    material_name=material,
                    thickness_nm=float(thickness_nm),
                    layer_order=deposition_index,
                    time_min=time_min,
                    rate_nm_per_min=rate_nm_per_min,
                    confidence="final_intended" if _optional_float(row.get("intended_thickness_nm")) is not None else "fallback",
                    source="intended_thickness_and_sputter_rates.xlsx",
                    target=_clean_text(row.get("target")),
                    pressure_mbar=pressure,
                    sccm=fallback_layer.sccm if fallback_layer is not None else None,
                    calibration_sample="",
                    rate_source=_clean_text(row.get("rate_series")),
                    deposition_date=_clean_text(row.get("date")),
                )
            )
        return list(reversed(estimates))

    def _fallback_layer_estimates_by_material(self, sample_name: str) -> dict[str, ExperimentLayerEstimate]:
        rows = self._thickness[self._thickness["sample_name"].astype(str) == sample_name]
        if rows.empty:
            return {}
        rows = rows.copy()
        rows["numeric_layer_order"] = pd.to_numeric(rows["layer_order"], errors="coerce")
        rows["thickness_nm_estimate"] = pd.to_numeric(rows["thickness_nm_estimate"], errors="coerce")
        rows = rows.dropna(subset=["material", "thickness_nm_estimate"])
        numeric_rows = rows[rows["numeric_layer_order"].notna()].copy()
        if not numeric_rows.empty:
            rows = numeric_rows
        fallback: dict[str, ExperimentLayerEstimate] = {}
        for index, row in rows.iterrows():
            material = str(row.get("material", "") or "").strip()
            if not material or material in fallback:
                continue
            notes_metadata = _parse_semicolon_metadata(row.get("notes"))
            layer_order = _optional_float(row.get("numeric_layer_order"))
            fallback[material] = ExperimentLayerEstimate(
                material_name=material,
                thickness_nm=float(row["thickness_nm_estimate"]),
                layer_order=int(layer_order if layer_order is not None else index + 1),
                time_min=_optional_float(row.get("time_min")),
                rate_nm_per_min=_optional_float(row.get("rate_nm_per_min")),
                confidence=str(row.get("confidence", "") or ""),
                source=str(row.get("source", "") or ""),
                target=str(notes_metadata.get("target", "") or ""),
                pressure_mbar=_optional_float(notes_metadata.get("pressure")),
                sccm=_optional_float(notes_metadata.get("sccm")),
                calibration_sample=str(notes_metadata.get("calibration_sample", "") or ""),
                rate_source=str(notes_metadata.get("rate_source", "") or ""),
                deposition_date=str(notes_metadata.get("calibration_date", "") or ""),
            )
        return fallback

    def _measurements_for_sample(self, sample_name: str) -> list[ExperimentMeasurement]:
        rows = self._measurements[self._measurements["sample_name"].astype(str) == sample_name]
        measurements: list[ExperimentMeasurement] = []
        for _, row in rows.iterrows():
            csv_path_text = str(row.get("csv_path", "") or "").strip()
            if not csv_path_text:
                continue
            csv_path = self.reflectivity_root / csv_path_text
            if not csv_path.exists():
                continue
            description = str(row.get("description", "") or csv_path.name)
            source_system = str(row.get("source_system", "") or "")
            measurement_mode = str(row.get("measurement_mode", "") or "")
            accessory = str(row.get("accessory", "") or "")
            source_folder = str(row.get("source_folder", "") or "")
            side_note = str(row.get("side_note", "") or "")
            row_condition_note = str(row.get("sample_condition_note", "") or "")
            sample_condition_note = _join_unique_texts(
                [row_condition_note, self._sample_condition_notes.get(sample_name, "")]
            )
            substrate_hint = normalize_substrate_name(row.get("substrate")) or _substrate_from_description(
                description
            )
            substrate_group = substrate_group_label(
                substrate_hint,
                sample_condition_note,
                description,
                source_folder,
                side_note,
                csv_path_text,
            )
            surface_class = str(row.get("surface", "") or "").strip()
            if not surface_class:
                surface_class = classify_surface(description, source_folder, side_note)
            if substrate_hint == "Ti":
                surface_class = "rough"
            if "rough_surface" in csv_path_text.lower().replace("\\", "/"):
                surface_class = "rough"
            label_parts = [description]
            if source_system and source_system not in description:
                label_parts.append(source_system)
            if surface_class and surface_class not in description.lower():
                label_parts.append(surface_class)
            measurements.append(
                ExperimentMeasurement(
                    description="; ".join(label_parts),
                    csv_path=csv_path,
                    source_system=source_system,
                    instrument_sample_id=str(row.get("instrument_sample_id", "") or ""),
                    measurement_angle_deg=_optional_float(row.get("measurement_angle_deg")),
                    substrate_hint=substrate_hint,
                    substrate_group=substrate_group,
                    measurement_mode=measurement_mode,
                    accessory=accessory,
                    surface_class=surface_class,
                    measurement_kind=classify_measurement_kind(
                        source_system=source_system,
                        description=description,
                        measurement_mode=measurement_mode,
                        accessory=accessory,
                        measurement_angle_deg=_optional_float(row.get("measurement_angle_deg")),
                        source_folder=source_folder,
                    ),
                    sample_condition_note=sample_condition_note,
                )
            )
        return measurements


def build_stack_from_estimates(
    sample: ExperimentSample,
    materials: dict[str, Material],
    substrate_name: str = "Si",
    native_oxide: NativeOxide | None = None,
    use_effective_interfaces: bool = False,
    interface_thickness_nm: float = 1.0,
    interface_fraction: float = 0.5,
) -> ThinFilmStack:
    """Create a ThinFilmStack from experimental thickness estimates."""

    missing = [layer.material_name for layer in sample.layer_estimates if layer.material_name not in materials]
    if substrate_name not in materials:
        missing.append(substrate_name)
    if missing:
        raise ValueError(f"Missing material constants for: {', '.join(sorted(set(missing)))}")

    deposited_layers = [
        Layer(materials[layer.material_name], layer.thickness_nm) for layer in sample.layer_estimates
    ]
    if use_effective_interfaces:
        return make_stack_with_interfaces(
            incident_medium=materials["air"],
            deposited_layers=deposited_layers,
            substrate=materials[substrate_name],
            native_oxide=native_oxide,
            interface_thickness_nm=interface_thickness_nm,
            interface_fraction=interface_fraction,
            name=f"{sample.sample_name} estimated stack",
        )

    optical_layers = list(deposited_layers)
    if native_oxide is not None and native_oxide.thickness_nm > 0:
        optical_layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))
    return make_stack(
        incident_medium=materials["air"],
        substrate=materials[substrate_name],
        layers=optical_layers,
        name=f"{sample.sample_name} estimated stack",
        display_layers=deposited_layers,
    )


def load_reflectance_csv(csv_path: str | Path) -> tuple[FloatArray, FloatArray]:
    """Load an instrument reflectance CSV as wavelength nm and 0-1 reflectance arrays."""

    data = pd.read_csv(csv_path)
    if data.shape[1] < 2:
        raise ValueError(f"Reflectance file has too few columns: {csv_path}")
    wavelengths = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    reflectance = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(wavelengths) & np.isfinite(reflectance)
    wavelengths = wavelengths[mask]
    reflectance = reflectance[mask]
    if wavelengths.size < 2:
        raise ValueError(f"Reflectance file has too few numeric rows: {csv_path}")

    if np.nanmax(reflectance) > 1.5:
        reflectance = reflectance / 100.0
    order = np.argsort(wavelengths)
    return wavelengths[order], np.clip(reflectance[order], 0.0, 1.0)


def save_cached_results(results: CachedExperimentResults, output_path: str | Path) -> Path:
    """Save cached experiment comparisons as NPZ plus a human-readable CSV summary."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wavelengths_nm=results.wavelengths_nm,
        sample_names=results.sample_names,
        measurement_descriptions=results.measurement_descriptions,
        sample_series=results.sample_series,
        substrate_classes=results.substrate_classes,
        surface_classes=results.surface_classes,
        measurement_kinds=results.measurement_kinds,
        source_systems=results.source_systems,
        stack_labels=results.stack_labels,
        measured_reflectance=results.measured_reflectance,
        simulated_reflectance=results.simulated_reflectance,
        measured_rgb=results.measured_rgb,
        simulated_rgb=results.simulated_rgb,
        measured_xyz=results.measured_xyz,
        simulated_xyz=results.simulated_xyz,
        measured_xy=results.measured_xy,
        simulated_xy=results.simulated_xy,
        delta_e=results.delta_e,
        colour_metric=np.asarray([normalise_colour_metric(results.colour_metric)], dtype=str),
    )
    _cached_results_summary(results).to_csv(path.with_suffix(".csv"), index=False)
    return path


def load_cached_results(input_path: str | Path) -> CachedExperimentResults:
    """Load saved experiment comparisons from an NPZ cache."""

    data = np.load(input_path, allow_pickle=False)
    sample_names = data["sample_names"].astype(str)
    measurement_descriptions = data["measurement_descriptions"].astype(str)
    colour_metric = (
        normalise_colour_metric(str(data["colour_metric"][0]))
        if "colour_metric" in data.files
        else COLOUR_METRIC_CIE76
    )
    substrate_classes = np.asarray(
        [
            normalize_substrate_group_label(value) or str(value)
            for value in _cached_str_array(
                data, "substrate_classes", sample_names, lambda _text: "Si"
            )
        ],
        dtype=str,
    )
    surface_classes = _cached_str_array(
        data,
        "surface_classes",
        measurement_descriptions,
        lambda text: classify_surface(text),
    )
    surface_classes = np.asarray(
        [
            "rough" if normalize_substrate_name(substrate) == "Ti" else surface
            for substrate, surface in zip(substrate_classes, surface_classes)
        ],
        dtype=str,
    )
    return CachedExperimentResults(
        wavelengths_nm=data["wavelengths_nm"].astype(float),
        sample_names=sample_names,
        measurement_descriptions=measurement_descriptions,
        sample_series=_cached_str_array(data, "sample_series", sample_names, sample_series_from_name),
        substrate_classes=substrate_classes,
        surface_classes=surface_classes,
        measurement_kinds=_cached_str_array(
            data,
            "measurement_kinds",
            measurement_descriptions,
            lambda text: classify_measurement_kind(description=text),
        ),
        source_systems=_cached_str_array(data, "source_systems", sample_names, lambda _text: ""),
        stack_labels=data["stack_labels"].astype(str),
        measured_reflectance=data["measured_reflectance"].astype(float),
        simulated_reflectance=data["simulated_reflectance"].astype(float),
        measured_rgb=data["measured_rgb"].astype(float),
        simulated_rgb=data["simulated_rgb"].astype(float),
        measured_xyz=data["measured_xyz"].astype(float),
        simulated_xyz=data["simulated_xyz"].astype(float),
        measured_xy=data["measured_xy"].astype(float),
        simulated_xy=data["simulated_xy"].astype(float),
        delta_e=data["delta_e"].astype(float),
        colour_metric=colour_metric,
    )


def default_experiment_cache_path(project_root: str | Path) -> Path:
    """Return the standard cache location for experiment comparison results."""

    return Path(project_root) / "outputs" / "experiment_cache" / "experiment_results.npz"


@lru_cache(maxsize=4)
def cie_xy_background(
    resolution: int = 320,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return a cached CIE 1931 xy colour image and spectral locus."""

    x_values = np.linspace(0.0, 0.8, resolution)
    y_values = np.linspace(0.0, 0.9, resolution)
    xx, yy = np.meshgrid(x_values, y_values)
    locus = cie_1931_spectral_locus()
    valid = (yy > 1e-4) & ((xx + yy) <= 1.0)
    xyz = np.zeros((*xx.shape, 3), dtype=float)
    xyz[..., 1] = 60.0
    xyz[..., 0][valid] = xx[valid] * xyz[..., 1][valid] / yy[valid]
    xyz[..., 2][valid] = (1.0 - xx[valid] - yy[valid]) * xyz[..., 1][valid] / yy[valid]
    rgb = xyz_to_srgb(xyz)
    rgb[~valid] = 1.0
    return x_values, y_values, rgb, locus


def cie_1931_spectral_locus() -> FloatArray:
    """Return CIE 1931 2-degree spectral locus xy coordinates."""

    try:
        import colour

        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(
            colour.SpectralShape(380.0, 780.0, 1.0)
        )
        xyz = np.asarray(cmfs.values, dtype=float)
        total = np.sum(xyz, axis=1)
        xy = xyz[:, :2] / total[:, None]
        return xy.astype(float)
    except Exception:
        return np.array(
            [
                [0.174, 0.005],
                [0.160, 0.060],
                [0.130, 0.200],
                [0.170, 0.600],
                [0.420, 0.520],
                [0.700, 0.300],
                [0.735, 0.265],
            ],
            dtype=float,
        )


def _points_inside_locus(
    xx: FloatArray,
    yy: FloatArray,
    locus: FloatArray,
    edge_padding: float = 0.0,
) -> NDArray[np.bool_]:
    """Mask points inside the CIE spectral locus, optionally padded past the border."""

    try:
        from matplotlib.path import Path as MplPath

        polygon = np.vstack([locus, locus[0]])
        points = np.column_stack([xx.ravel(), yy.ravel()])
        return MplPath(polygon).contains_points(points, radius=edge_padding).reshape(xx.shape)
    except Exception:
        return (yy > 0.0) & (xx > 0.0) & ((xx + yy) <= 1.0)


def xyz_to_xy(xyz: tuple[float, float, float] | NDArray[np.float64]) -> tuple[float, float]:
    """Convert CIE XYZ to chromaticity x,y coordinates."""

    values = np.asarray(xyz, dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        return (0.0, 0.0)
    return (float(values[0] / total), float(values[1] / total))


def delta_e_cie76(
    xyz_1: tuple[float, float, float] | NDArray[np.float64],
    xyz_2: tuple[float, float, float] | NDArray[np.float64],
) -> float:
    """Return approximate Delta E as Euclidean distance in CIE Lab space."""

    lab_1 = xyz_to_lab(xyz_1)
    lab_2 = xyz_to_lab(xyz_2)
    return float(np.linalg.norm(lab_1 - lab_2))


def delta_e_ciede2000(
    xyz_1: tuple[float, float, float] | NDArray[np.float64],
    xyz_2: tuple[float, float, float] | NDArray[np.float64],
) -> float:
    """Return CIEDE2000 colour difference from XYZ inputs."""

    lab_1 = xyz_to_lab(xyz_1)
    lab_2 = xyz_to_lab(xyz_2)
    l1, a1, b1 = (float(value) for value in lab_1)
    l2, a2, b2 = (float(value) for value in lab_2)

    c1 = float(np.hypot(a1, b1))
    c2 = float(np.hypot(a2, b2))
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7))) if c_bar > 0 else 0.0

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = float(np.hypot(a1_prime, b1))
    c2_prime = float(np.hypot(a2_prime, b2))

    h1_prime = _lab_hue_angle_deg(a1_prime, b1, c1_prime)
    h2_prime = _lab_hue_angle_deg(a2_prime, b2, c2_prime)

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    delta_h_prime = _delta_hue_deg(h1_prime, h2_prime, c1_prime, c2_prime)
    delta_h_big_prime = (
        2.0
        * np.sqrt(c1_prime * c2_prime)
        * np.sin(np.deg2rad(delta_h_prime) / 2.0)
    )

    l_bar_prime = 0.5 * (l1 + l2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)
    h_bar_prime = _mean_hue_deg(h1_prime, h2_prime, c1_prime, c2_prime)

    t = (
        1.0
        - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.deg2rad(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.deg2rad((3.0 * h_bar_prime) + 6.0))
        - 0.20 * np.cos(np.deg2rad((4.0 * h_bar_prime) - 63.0))
    )
    delta_theta = 30.0 * np.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))
    c_bar_prime7 = c_bar_prime**7
    r_c = 2.0 * np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25.0**7)) if c_bar_prime > 0 else 0.0
    s_l = 1.0 + (
        (0.015 * ((l_bar_prime - 50.0) ** 2))
        / np.sqrt(20.0 + ((l_bar_prime - 50.0) ** 2))
    )
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(np.deg2rad(2.0 * delta_theta)) * r_c

    l_term = delta_l_prime / s_l
    c_term = delta_c_prime / s_c
    h_term = delta_h_big_prime / s_h
    return float(np.sqrt((l_term**2) + (c_term**2) + (h_term**2) + (r_t * c_term * h_term)))


def delta_e_colour(
    xyz_1: tuple[float, float, float] | NDArray[np.float64],
    xyz_2: tuple[float, float, float] | NDArray[np.float64],
    metric: str = COLOUR_METRIC_CIE76,
) -> float:
    """Return colour difference using the selected colour metric."""

    metric_key = normalise_colour_metric(metric)
    if metric_key == COLOUR_METRIC_CIEDE2000:
        return delta_e_ciede2000(xyz_1, xyz_2)
    return delta_e_cie76(xyz_1, xyz_2)


def normalise_colour_metric(value: object) -> str:
    """Normalize GUI/cache metric labels to stable keys."""

    text = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if compact in {"ciede2000", "de2000", "deltae2000", "deltae00", "cie2000"}:
        return COLOUR_METRIC_CIEDE2000
    return COLOUR_METRIC_CIE76


def colour_metric_label(metric: str) -> str:
    """Return the display label for a metric key or label."""

    return COLOUR_METRIC_LABELS[normalise_colour_metric(metric)]


def _lab_hue_angle_deg(a_value: float, b_value: float, chroma: float) -> float:
    if chroma <= 0.0:
        return 0.0
    hue = float(np.rad2deg(np.arctan2(b_value, a_value)))
    return hue + 360.0 if hue < 0.0 else hue


def _delta_hue_deg(hue_1: float, hue_2: float, chroma_1: float, chroma_2: float) -> float:
    if chroma_1 * chroma_2 == 0.0:
        return 0.0
    delta = hue_2 - hue_1
    if abs(delta) <= 180.0:
        return delta
    return delta - 360.0 if delta > 180.0 else delta + 360.0


def _mean_hue_deg(hue_1: float, hue_2: float, chroma_1: float, chroma_2: float) -> float:
    if chroma_1 * chroma_2 == 0.0:
        return hue_1 + hue_2
    if abs(hue_1 - hue_2) <= 180.0:
        return 0.5 * (hue_1 + hue_2)
    if hue_1 + hue_2 < 360.0:
        return 0.5 * (hue_1 + hue_2 + 360.0)
    return 0.5 * (hue_1 + hue_2 - 360.0)


def xyz_to_lab(xyz: tuple[float, float, float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert XYZ to CIE Lab using a D65 reference white."""

    values = np.asarray(xyz, dtype=float)
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    scaled = values / white
    f = np.where(scaled > 0.008856, np.cbrt(scaled), (7.787 * scaled) + (16.0 / 116.0))
    return np.array(
        [
            (116.0 * f[1]) - 16.0,
            500.0 * (f[0] - f[1]),
            200.0 * (f[1] - f[2]),
        ],
        dtype=float,
    )


def _cached_results_summary(results: CachedExperimentResults) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_name": results.sample_names,
            "measurement_description": results.measurement_descriptions,
            "sample_series": results.sample_series,
            "substrate": results.substrate_classes,
            "surface_class": results.surface_classes,
            "measurement_kind": results.measurement_kinds,
            "source_system": results.source_systems,
            "stack": results.stack_labels,
            "measured_x": results.measured_xy[:, 0],
            "measured_y": results.measured_xy[:, 1],
            "simulated_x": results.simulated_xy[:, 0],
            "simulated_y": results.simulated_xy[:, 1],
            "colour_metric": results.colour_metric,
            "delta_e": results.delta_e,
            "delta_e_cie76": results.delta_e if results.colour_metric == COLOUR_METRIC_CIE76 else np.nan,
            "measured_rgb": [_rgb_to_hex(rgb) for rgb in results.measured_rgb],
            "simulated_rgb": [_rgb_to_hex(rgb) for rgb in results.simulated_rgb],
        }
    )


def _rgb_to_hex(rgb: NDArray[np.float64]) -> str:
    values = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    channels = [int(round(channel * 255.0)) for channel in values]
    return "#{:02x}{:02x}{:02x}".format(*channels)


def _perceived_color_from_arrays(
    wavelengths_nm: NDArray[np.float64],
    reflectance: NDArray[np.float64],
) -> PerceivedColor:
    xyz = reflectance_to_xyz(reflectance, wavelengths_nm=wavelengths_nm)
    srgb = reflectance_to_srgb(reflectance, wavelengths_nm=wavelengths_nm)
    srgb_255 = tuple(int(round(channel * 255)) for channel in srgb)
    return PerceivedColor(
        srgb=tuple(float(channel) for channel in srgb),
        srgb_255=srgb_255,
        xyz=tuple(float(channel) for channel in xyz),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_intended_thickness_workbook(sample_data_root: Path) -> pd.DataFrame:
    path = (
        sample_data_root.parent
        / "Thickness"
        / "final_outputs"
        / "intended_thickness_and_sputter_rates.xlsx"
    )
    if not path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_excel(path, sheet_name="Layer thickness")
    except Exception:
        return pd.DataFrame()
    required = {"entry", "material", "sputter_time_min", "assigned_rate_nm_min", "intended_thickness_nm"}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    data = data.copy()
    data["entry"] = data["entry"].astype(str).str.strip()
    data["material"] = data["material"].astype(str).str.strip()
    return data[data["entry"].ne("") & data["material"].ne("")]


def _optional_float(value) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _date_sort_key(value) -> tuple[int, int, int]:
    text = _clean_text(value)
    if not text:
        return (0, 0, 0)
    for fmt in ("%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return (parsed.year, parsed.month, parsed.day)
    match = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", text)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = re.search(r"(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})", text)
    if not match:
        return (0, 0, 0)
    year = int(match.group(3))
    if year < 100:
        year += 2000
    return (year, int(match.group(2)), int(match.group(1)))


def _optional_pressure_mbar(value) -> float | None:
    direct = _optional_float(value)
    if direct is not None:
        return direct
    text = str(value or "").strip().lower()
    if not text:
        return None
    match = re.search(r"([-+]?\d+(?:[.,]\d+)?)\s*(?:\*|x|×)\s*10\^?\s*([-+]?\d+)", text)
    if match:
        return float(match.group(1).replace(",", ".")) * (10.0 ** float(match.group(2)))
    match = re.search(r"([-+]?\d+(?:[.,]\d+)?)\s*e\s*([-+]?\d+)", text)
    if match:
        return float(match.group(1).replace(",", ".")) * (10.0 ** float(match.group(2)))
    match = re.search(r"([-+]?\d+(?:[.,]\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def _parse_semicolon_metadata(value) -> dict[str, str]:
    """Parse semicolon-separated key=value notes from imported experiment CSVs."""

    text = str(value or "")
    metadata: dict[str, str] = {}
    for part in text.split(";"):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", maxsplit=1)
        key = key.strip().lower()
        if not key:
            continue
        metadata[key] = raw_value.strip()
    return metadata


def _rate_setting_key(
    material: str,
    target: str,
    pressure_mbar: float | None,
) -> tuple[str, str, float] | None:
    """Key for matching layers to single-film sputter-rate calibrations."""

    material_key = str(material or "").strip()
    target_key = str(target or "").strip().upper()
    if not material_key or not target_key or pressure_mbar is None:
        return None
    return (material_key, target_key, round(float(pressure_mbar), 7))


def sample_series_from_name(sample_name: str) -> str:
    """Return the leading experiment family label used for browsing, such as A, B, C, S, or Au."""

    text = str(sample_name or "").strip()
    match = re.match(r"([A-Za-z]+)", text)
    return match.group(1) if match else "Other"


def classify_surface(
    description: str,
    source_folder: str = "",
    side_note: str = "",
) -> str:
    """Classify a measurement/sample as smooth or rough from exported metadata."""

    text = " ".join([description, source_folder, side_note]).lower()
    if "rough" in text:
        return "rough"
    if "smooth" in text:
        return "smooth"
    return "smooth"


def classify_measurement_kind(
    source_system: str = "",
    description: str = "",
    measurement_mode: str = "",
    accessory: str = "",
    measurement_angle_deg: float | None = None,
    source_folder: str = "",
) -> str:
    """Classify spectra by optical collection geometry for filtering and later fitting."""

    text = " ".join(
        [source_system, description, measurement_mode, accessory, source_folder]
    ).lower()
    if "diffuse spectrophotometer" in text or "diffuse" in text or "sci" in text:
        return "diffuse_sphere"
    if (
        "sofus twodetectors" in text
        or "twodetectors" in text
        or "integrating sphere" in text
        or "sphere" in text
        or "downward" in text
    ):
        return "integrating_sphere"
    if measurement_angle_deg is not None or "universal reflectance" in text or "reflectance" in text:
        return "specular"
    return "unknown"


def normalize_substrate_name(value: object) -> str | None:
    """Normalize imported substrate labels to material keys used by the simulator."""

    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "unknown"}:
        return None
    if text in {"si", "silicon", "silicium"}:
        return "Si"
    if text in {"ti", "titanium"}:
        return "Ti"
    if text in {"glass", "substrate"}:
        return "substrate"
    return None


def normalize_substrate_group_label(value: object) -> str | None:
    """Normalize imported substrate labels to browsing groups."""

    text = _clean_text(value)
    if not text:
        return None
    substrate = normalize_substrate_name(text)
    text_lower = text.lower()
    if is_double_polished_note(text) and (
        substrate == "Si" or "si" in text_lower or "silicon" in text_lower
    ):
        return "Si double polished"
    return substrate or text


def substrate_group_label(substrate_hint: object, *condition_texts: object) -> str:
    """Return a filter label that separates double-polished Si from ordinary Si."""

    substrate = normalize_substrate_name(substrate_hint)
    text = _clean_text(substrate_hint)
    text_lower = text.lower()
    if substrate is None:
        if "silicon" in text_lower or re.search(r"\bsi\b", text_lower):
            substrate = "Si"
        elif "titanium" in text_lower or re.search(r"\bti\b", text_lower):
            substrate = "Ti"
        elif text:
            substrate = text
        else:
            substrate = ""
    if substrate == "Si" and is_double_polished_note(text, *condition_texts):
        return "Si double polished"
    return substrate


def is_double_polished_note(*values: object) -> bool:
    """True when any text value marks a sample as double polished."""

    text = " ".join(_clean_text(value).lower() for value in values)
    return re.search(r"\bdouble[\s_-]*polish(?:ed)?\b", text) is not None


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _join_unique_texts(values: Any) -> str:
    texts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.lower()
        if text and key not in seen:
            texts.append(text)
            seen.add(key)
    return "; ".join(texts)


def _iter_manifest_condition_notes(payload: Any):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "sample_condition_note":
                yield value
            else:
                yield from _iter_manifest_condition_notes(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_manifest_condition_notes(value)


def _cached_str_array(
    data: np.lib.npyio.NpzFile,
    key: str,
    fallback_source: NDArray[np.str_],
    fallback,
) -> NDArray[np.str_]:
    """Load a cached text array, deriving values for older caches that lack new metadata."""

    if key in data.files:
        return data[key].astype(str)
    return np.asarray([fallback(str(value)) for value in fallback_source], dtype=str)


def _substrate_from_description(description: str) -> str | None:
    match = re.search(r"\bon\s+(Si|Ti|substrate)\b", description, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1)
    if value.lower() == "si":
        return "Si"
    if value.lower() == "ti":
        return "Ti"
    return "substrate"
