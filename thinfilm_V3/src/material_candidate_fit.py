"""Rank optical-constant candidates against single-layer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray

from .color import prepare_color_conversion, reflectance_to_xyz
from .experiments import (
    COLOUR_METRIC_CIE76,
    ExperimentDataStore,
    delta_e_colour,
    load_reflectance_csv,
    normalise_colour_metric,
)
from .materials import Material, built_in_materials, make_tabulated_material, visible_material_table
from .refractiveindex_db import (
    default_candidate_config_path,
    default_candidate_data_dir,
    download_candidate_records,
    load_candidate_config,
    load_local_candidate_materials,
    safe_candidate_name,
)
from .stack import Layer, make_stack, native_oxide_for_substrate
from .tmm_model import TMMModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MaterialCandidateFitRow:
    """Fit score for one material candidate."""

    material_name: str
    candidate_label: str
    sample_count: int
    spectrum_count: int
    reflectance_rmse: float
    mean_delta_e: float
    combined_score: float


@dataclass(frozen=True)
class MaterialCandidateFitResult:
    """All candidate rankings and winning material choices."""

    rows: tuple[MaterialCandidateFitRow, ...]
    best_by_material: dict[str, MaterialCandidateFitRow]
    output_dir: Path
    best_profile_path: Path


def default_candidate_fit_output_dir(project_root: str | Path) -> Path:
    """Return the standard output folder for candidate-fit reports."""

    return Path(project_root) / "outputs" / "material_candidate_fits"


def default_best_candidate_profile_path(project_root: str | Path) -> Path:
    """Return the standard JSON file used by the GUI candidate profile."""

    return default_candidate_fit_output_dir(project_root) / "best_refractiveindex_candidates.json"


def grouped_best_candidate_profile_path(project_root: str | Path, group_label: str) -> Path:
    """Return the saved best-candidate profile path for one experiment group."""

    safe_label = _safe_group_label(group_label)
    return default_candidate_fit_output_dir(project_root) / f"best_refractiveindex_candidates_{safe_label}.json"


def fit_refractiveindex_candidates(
    project_root: str | Path,
    sample_data_root: str | Path,
    angle_deg: float = 8.0,
    wavelengths_nm: NDArray[np.float64] | None = None,
    material_names: tuple[str, ...] = ("Si", "Ti", "Ag", "TiO2", "SiO2", "ZrO2", "As2S3"),
    download_missing: bool = True,
    surface_class_filter: str | None = None,
    measurement_kind_filter: str | None = None,
    substrate_filter: str | None = None,
    model: TMMModel | None = None,
    fit_group_label: str | None = None,
    colour_metric: str = COLOUR_METRIC_CIE76,
) -> MaterialCandidateFitResult:
    """Rank current, legacy, and refractiveindex.info candidates on single films."""

    metric = normalise_colour_metric(colour_metric)
    root = Path(project_root)
    wavelengths = (
        np.linspace(400.0, 700.0, 151)
        if wavelengths_nm is None
        else np.asarray(wavelengths_nm, dtype=float)
    )
    output_dir = default_candidate_fit_output_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_group_label = _safe_group_label(fit_group_label) if fit_group_label else ""

    config_path = default_candidate_config_path(root)
    data_dir = default_candidate_data_dir(root)
    configured = load_candidate_config(config_path)
    local_candidates = (
        download_candidate_records(configured, data_dir)
        if download_missing
        else tuple(
            candidate.__class__(
                candidate.material_name,
                candidate.source_name,
                candidate.url,
                data_dir
                / safe_candidate_name(candidate.material_name)
                / f"{safe_candidate_name(candidate.source_name)}.yml",
            )
            for candidate in configured
        )
    )
    downloaded_materials = load_local_candidate_materials(local_candidates)

    store = ExperimentDataStore(sample_data_root)
    base_materials = built_in_materials("current")
    optical_model = model or TMMModel()
    rows: list[MaterialCandidateFitRow] = []
    best_materials = dict(base_materials)
    best_rows: dict[str, MaterialCandidateFitRow] = {}

    for material_name in material_names:
        experiments = _candidate_experiments(
            store,
            material_name,
            surface_class_filter=surface_class_filter,
            measurement_kind_filter=measurement_kind_filter,
            substrate_filter=substrate_filter,
        )
        if not experiments:
            continue
        candidates = _built_in_candidates(material_name)
        candidates.extend(downloaded_materials.get(material_name, []))
        if not candidates:
            continue

        scored: list[tuple[MaterialCandidateFitRow, Material]] = []
        for label, candidate_material in candidates:
            row = _score_candidate(
                material_name=material_name,
                candidate_label=label,
                candidate_material=candidate_material,
                experiments=experiments,
                base_materials=base_materials,
                wavelengths_nm=wavelengths,
                angle_deg=angle_deg,
                model=optical_model,
                colour_metric=metric,
            )
            rows.append(row)
            scored.append((row, candidate_material))

        best_row, best_material = min(scored, key=lambda item: item[0].combined_score)
        best_rows[material_name] = best_row
        best_materials[material_name] = best_material

    best_profile_path = (
        grouped_best_candidate_profile_path(root, safe_group_label)
        if safe_group_label
        else default_best_candidate_profile_path(root)
    )
    _save_best_profile(
        best_rows,
        best_materials,
        best_profile_path,
        group_label=safe_group_label or None,
        filters={
            "substrate": substrate_filter or "All",
            "surface": surface_class_filter or "All",
            "measurement": measurement_kind_filter or "All",
            "colour_metric": metric,
        },
    )
    suffix = f"_{safe_group_label}" if safe_group_label else ""
    _save_rankings(rows, output_dir / f"candidate_rankings{suffix}.csv")
    _save_summary(rows, best_rows, best_profile_path, output_dir / f"candidate_fit_summary{suffix}.json")
    return MaterialCandidateFitResult(
        rows=tuple(rows),
        best_by_material=best_rows,
        output_dir=output_dir,
        best_profile_path=best_profile_path,
    )


def load_best_candidate_materials(
    base_materials: dict[str, Material],
    profile_path: str | Path,
) -> dict[str, Material]:
    """Load the saved best-candidate profile into a material dictionary."""

    data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    materials = dict(base_materials)
    for entry in data.get("materials", []):
        name = str(entry["material_name"])
        wavelengths = np.asarray(entry["wavelengths_nm"], dtype=float)
        n_values = np.asarray(entry["n"], dtype=float)
        k_values = np.asarray(entry["k"], dtype=float)
        materials[name] = make_tabulated_material(name, wavelengths, n_values, k_values)
    return materials


def _candidate_experiments(
    store: ExperimentDataStore,
    material_name: str,
    surface_class_filter: str | None = None,
    measurement_kind_filter: str | None = None,
    substrate_filter: str | None = None,
) -> list[tuple[str, str, str, float, Path, str]]:
    experiments: list[tuple[str, str, str, float, Path, str]] = []
    for sample_name in store.sample_names(require_spectra=True):
        sample = store.load_sample(sample_name)
        if len(sample.layer_estimates) != 1:
            continue
        layer = sample.layer_estimates[0]
        is_substrate_material = material_name in {"Si", "Ti"}
        if not is_substrate_material and layer.material_name != material_name:
            continue
        for measurement in sample.measurements:
            substrate_name = measurement.substrate_hint or "Si"
            substrate_group = measurement.substrate_group or substrate_name
            if is_substrate_material and substrate_name != material_name:
                continue
            if substrate_filter and substrate_group != substrate_filter:
                continue
            if surface_class_filter and measurement.surface_class != surface_class_filter:
                continue
            if measurement_kind_filter and measurement.measurement_kind != measurement_kind_filter:
                continue
            description = measurement.description.lower()
            if surface_class_filter != "rough" and "rough" in description:
                continue
            if "glass" in description:
                continue
            experiments.append(
                (
                    sample_name,
                    layer.material_name,
                    measurement.description,
                    layer.thickness_nm,
                    measurement.csv_path,
                    substrate_name,
                )
            )
    return experiments


def _built_in_candidates(material_name: str) -> list[tuple[str, Material]]:
    candidates: list[tuple[str, Material]] = []
    for profile in ("current", "legacy_ideal", "legacy_wip"):
        materials = built_in_materials(profile)
        if material_name in materials:
            candidates.append((profile, materials[material_name]))
    return candidates


def _score_candidate(
    material_name: str,
    candidate_label: str,
    candidate_material: Material,
    experiments: list[tuple[str, str, str, float, Path, str]],
    base_materials: dict[str, Material],
    wavelengths_nm: FloatArray,
    angle_deg: float,
    model: TMMModel,
    colour_metric: str = COLOUR_METRIC_CIE76,
) -> MaterialCandidateFitRow:
    metric = normalise_colour_metric(colour_metric)
    trial_materials = dict(base_materials)
    trial_materials[material_name] = candidate_material
    color_cache = prepare_color_conversion(wavelengths_nm)
    squared_errors: list[float] = []
    delta_e_values: list[float] = []
    sample_names: set[str] = set()

    for sample_name, layer_material_name, _description, thickness_nm, csv_path, substrate_name in experiments:
        sample_names.add(sample_name)
        measured_wl, measured_r = load_reflectance_csv(csv_path)
        measured_grid = np.interp(wavelengths_nm, measured_wl, measured_r)
        stack = _single_layer_stack(
            trial_materials,
            material_name=layer_material_name,
            thickness_nm=thickness_nm,
            substrate_name=substrate_name,
        )
        simulated = model.simulate(stack, wavelengths_nm, angle_deg).reflectance
        squared_errors.extend(np.square(simulated - measured_grid).tolist())
        measured_xyz = reflectance_to_xyz(measured_grid, cache=color_cache)
        simulated_xyz = reflectance_to_xyz(simulated, cache=color_cache)
        delta_e_values.append(delta_e_colour(measured_xyz, simulated_xyz, metric=metric))

    rmse = float(np.sqrt(np.mean(squared_errors)))
    mean_delta_e = float(np.mean(delta_e_values))
    return MaterialCandidateFitRow(
        material_name=material_name,
        candidate_label=candidate_label,
        sample_count=len(sample_names),
        spectrum_count=len(experiments),
        reflectance_rmse=rmse,
        mean_delta_e=mean_delta_e,
        combined_score=rmse + 0.01 * mean_delta_e,
    )


def _single_layer_stack(
    materials: dict[str, Material],
    material_name: str,
    thickness_nm: float,
    substrate_name: str = "Si",
):
    layer = Layer(materials[material_name], thickness_nm)
    oxide = native_oxide_for_substrate(materials, substrate_name)
    layers = [layer]
    if oxide is not None:
        layers.append(Layer(oxide.material, oxide.thickness_nm))
    return make_stack(
        incident_medium=materials["air"],
        substrate=materials[substrate_name],
        layers=layers,
        name=f"candidate fit {material_name}",
        display_layers=[layer],
    )


def _save_best_profile(
    best_rows: dict[str, MaterialCandidateFitRow],
    materials: dict[str, Material],
    path: Path,
    group_label: str | None = None,
    filters: dict[str, str] | None = None,
) -> None:
    visible = np.array([400, 450, 500, 550, 600, 650, 700], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for material_name, row in sorted(best_rows.items()):
        wavelengths, n_values, k_values = visible_material_table(materials[material_name], visible)
        entries.append(
            {
                "material_name": material_name,
                "candidate_label": row.candidate_label,
                "reflectance_rmse": row.reflectance_rmse,
                "mean_delta_e": row.mean_delta_e,
                "wavelengths_nm": wavelengths.tolist(),
                "n": n_values.tolist(),
                "k": k_values.tolist(),
            }
        )
    path.write_text(
        json.dumps(
            {
                "model": "best refractiveindex.info/current/legacy candidates from single-layer experiments",
                "group_label": group_label or "global",
                "filters": filters or {},
                "materials": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _save_rankings(rows: list[MaterialCandidateFitRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in sorted(rows, key=lambda item: (item.material_name, item.combined_score)):
                writer.writerow(asdict(row))


def _save_summary(
    rows: list[MaterialCandidateFitRow],
    best_rows: dict[str, MaterialCandidateFitRow],
    best_profile_path: Path,
    path: Path,
) -> None:
    payload = {
        "best_profile_path": str(best_profile_path),
        "best_by_material": {name: asdict(row) for name, row in best_rows.items()},
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_group_label(label: str | None) -> str:
    """Return a stable filename/profile suffix for one fit group."""

    text = str(label or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return cleaned.strip("_") or "group"
