"""Fit simple refractive-index constants from single-film reflectance samples."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .experiments import ExperimentDataStore, load_reflectance_csv
from .materials import ConstantMaterial, Material, make_tabulated_material
from .stack import Layer, make_stack, native_oxide_for_substrate
from .tmm_model import TMMModel


VISIBLE_WAVELENGTHS_NM = np.array([400, 450, 500, 550, 600, 650, 700], dtype=float)


@dataclass(frozen=True)
class FittedMaterial:
    """One fitted material constant derived from simple calibration samples."""

    material_name: str
    n: float
    k: float
    sample_count: int
    spectrum_count: int
    rms_error: float
    notes: str


@dataclass(frozen=True)
class FittedConstantsResult:
    """Saved fitted constants and metadata."""

    materials: tuple[FittedMaterial, ...]
    wavelength_grid_nm: tuple[float, ...]
    model: str


def default_fitted_constants_path(project_root: str | Path) -> Path:
    """Return the standard fitted-constants JSON path."""

    return Path(project_root) / "outputs" / "fitted_constants" / "single_film_constants.json"


def fit_single_film_constants(
    sample_data_root: str | Path,
    base_materials: dict[str, Material],
    wavelengths_nm: NDArray[np.float64] | None = None,
    material_names: tuple[str, ...] = ("TiO2", "SiO2", "ZrO2", "Ag"),
    angle_deg: float = 8.0,
) -> FittedConstantsResult:
    """Fit simple constant optical constants from single-film samples on Si."""

    wavelengths = (
        np.linspace(400.0, 700.0, 61)
        if wavelengths_nm is None
        else np.asarray(wavelengths_nm, dtype=float)
    )
    store = ExperimentDataStore(sample_data_root)
    fitted: list[FittedMaterial] = []

    for material_name in material_names:
        candidates = _single_film_candidates(store, material_name)
        if not candidates:
            continue

        fit_k = material_name in {"Ag"}

        def residual(params: NDArray[np.float64]) -> NDArray[np.float64]:
            n_value = float(params[0])
            k_value = float(params[1]) if fit_k else 0.0
            trial_materials = dict(base_materials)
            trial_materials[material_name] = ConstantMaterial(material_name, n_value + 1j * k_value)
            model = TMMModel()
            parts: list[NDArray[np.float64]] = []
            for _sample_name, thickness_nm, measurement_path in candidates:
                measured_wl, measured_r = load_reflectance_csv(measurement_path)
                measured_grid = np.interp(wavelengths, measured_wl, measured_r)
                stack = _single_layer_stack(
                    trial_materials,
                    material_name=material_name,
                    thickness_nm=thickness_nm,
                )
                simulated = model.simulate(stack, wavelengths, angle_deg).reflectance
                parts.append(simulated - measured_grid)
            return np.concatenate(parts)

        initial = _initial_n(base_materials, material_name)
        if fit_k:
            initial_k = _initial_k(base_materials, material_name)
            result = least_squares(
                residual,
                x0=np.array([max(0.02, min(initial, 2.5)), max(0.5, min(initial_k, 8.0))]),
                bounds=([0.02, 0.2], [2.5, 10.0]),
                max_nfev=60,
            )
            fitted_k = float(result.x[1])
            note = "Constant n,k metal fit, thickness fixed, Si substrate only."
        else:
            result = least_squares(
                residual,
                x0=np.array([initial]),
                bounds=(1.05, 3.8),
                max_nfev=40,
            )
            fitted_k = 0.0
            note = "Constant n fit, k fixed to 0, thickness fixed, Si substrate only."
        errors = residual(result.x)
        fitted.append(
            FittedMaterial(
                material_name=material_name,
                n=float(result.x[0]),
                k=fitted_k,
                sample_count=len({sample_name for sample_name, _, _path in candidates}),
                spectrum_count=len(candidates),
                rms_error=float(np.sqrt(np.mean(errors**2))),
                notes=note,
            )
        )

    return FittedConstantsResult(
        materials=tuple(fitted),
        wavelength_grid_nm=tuple(float(value) for value in VISIBLE_WAVELENGTHS_NM),
        model="single-film constant-n/k least-squares fit",
    )


def save_fitted_constants(result: FittedConstantsResult, path: str | Path) -> Path:
    """Save fitted constants as JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": result.model,
        "wavelength_grid_nm": list(result.wavelength_grid_nm),
        "materials": [asdict(material) for material in result.materials],
    }
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output


def load_fitted_materials(
    base_materials: dict[str, Material],
    path: str | Path,
) -> dict[str, Material]:
    """Return base materials updated with saved fitted constants."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    materials = dict(base_materials)
    wavelengths = np.asarray(data.get("wavelength_grid_nm", VISIBLE_WAVELENGTHS_NM), dtype=float)
    for entry in data.get("materials", []):
        name = str(entry["material_name"])
        n_value = float(entry["n"])
        k_value = float(entry.get("k", 0.0))
        materials[name] = make_tabulated_material(
            name,
            wavelengths,
            np.full_like(wavelengths, n_value, dtype=float),
            np.full_like(wavelengths, k_value, dtype=float),
        )
    return materials


def _single_film_candidates(
    store: ExperimentDataStore,
    material_name: str,
) -> list[tuple[str, float, Path]]:
    candidates: list[tuple[str, float, Path]] = []
    for sample_name in store.sample_names(require_spectra=True):
        sample = store.load_sample(sample_name)
        if len(sample.layer_estimates) != 1:
            continue
        layer = sample.layer_estimates[0]
        if layer.material_name != material_name:
            continue
        for measurement in sample.measurements:
            description = measurement.description.lower()
            if "rough" in description or "glass" in description:
                continue
            if measurement.substrate_hint not in {"Si", None}:
                continue
            candidates.append((sample_name, layer.thickness_nm, measurement.csv_path))
    return candidates


def _single_layer_stack(
    materials: dict[str, Material],
    material_name: str,
    thickness_nm: float,
):
    layer = Layer(materials[material_name], thickness_nm)
    oxide = native_oxide_for_substrate(materials, "Si")
    layers = [layer]
    if oxide is not None:
        layers.append(Layer(oxide.material, oxide.thickness_nm))
    return make_stack(
        incident_medium=materials["air"],
        substrate=materials["Si"],
        layers=layers,
        name=f"fit {material_name}",
        display_layers=[layer],
    )


def _initial_n(materials: dict[str, Material], material_name: str) -> float:
    material = materials[material_name]
    return float(np.mean(material.refractive_index(VISIBLE_WAVELENGTHS_NM).real))


def _initial_k(materials: dict[str, Material], material_name: str) -> float:
    material = materials[material_name]
    return float(np.mean(material.refractive_index(VISIBLE_WAVELENGTHS_NM).imag))
