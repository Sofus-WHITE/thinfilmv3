"""Utilities for local refractiveindex.info YAML candidate records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np
import yaml

from .materials import Material, make_tabulated_material


@dataclass(frozen=True)
class RefractiveIndexCandidate:
    """One optical-constant candidate record."""

    material_name: str
    source_name: str
    url: str
    local_path: Path | None = None

    @property
    def label(self) -> str:
        """Human-readable label used in reports and plots."""

        return f"refractiveindex.info:{self.source_name}"


def default_candidate_config_path(project_root: str | Path) -> Path:
    """Return the standard candidate URL config path."""

    return Path(project_root) / "config" / "refractiveindex_candidates.yaml"


def default_candidate_data_dir(project_root: str | Path) -> Path:
    """Return the local folder where downloaded YAML records are cached."""

    return Path(project_root) / "data" / "refractiveindex_candidates"


def load_candidate_config(path: str | Path) -> tuple[RefractiveIndexCandidate, ...]:
    """Load material/URL candidate definitions from YAML."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    candidates: list[RefractiveIndexCandidate] = []
    for entry in data.get("candidates", []):
        url = str(entry["url"])
        material = str(entry["material"])
        candidates.append(
            RefractiveIndexCandidate(
                material_name=material,
                source_name=_source_name_from_url(url),
                url=url,
            )
        )
    return tuple(candidates)


def download_candidate_records(
    candidates: tuple[RefractiveIndexCandidate, ...],
    output_dir: str | Path,
    timeout_s: float = 30.0,
) -> tuple[RefractiveIndexCandidate, ...]:
    """Download missing YAML files and return candidates with local paths."""

    root = Path(output_dir)
    downloaded: list[RefractiveIndexCandidate] = []
    for candidate in candidates:
        material_dir = root / _safe_name(candidate.material_name)
        material_dir.mkdir(parents=True, exist_ok=True)
        local_path = material_dir / f"{_safe_name(candidate.source_name)}.yml"
        if not local_path.exists():
            with urlopen(candidate.url, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8")
            local_path.write_text(raw, encoding="utf-8")
        downloaded.append(
            RefractiveIndexCandidate(
                material_name=candidate.material_name,
                source_name=candidate.source_name,
                url=candidate.url,
                local_path=local_path,
            )
        )
    return tuple(downloaded)


def load_local_candidate_materials(
    candidates: tuple[RefractiveIndexCandidate, ...],
) -> dict[str, list[tuple[str, Material]]]:
    """Parse downloaded candidates into tabulated Material objects grouped by material."""

    grouped: dict[str, list[tuple[str, Material]]] = {}
    for candidate in candidates:
        if candidate.local_path is None:
            raise ValueError(f"Candidate {candidate.label} has no local YAML path.")
        material = material_from_refractiveindex_yaml(
            material_name=candidate.material_name,
            yaml_path=candidate.local_path,
        )
        grouped.setdefault(candidate.material_name, []).append((candidate.label, material))
    return grouped


def material_from_refractiveindex_yaml(
    material_name: str,
    yaml_path: str | Path,
) -> Material:
    """Create a tabulated material from a refractiveindex.info YAML record."""

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    wavelengths_um, n_values, k_values = _extract_tabulated_nk(data)
    wavelengths_nm = wavelengths_um * 1000.0
    order = np.argsort(wavelengths_nm)
    return make_tabulated_material(
        material_name,
        wavelengths_nm[order],
        n_values[order],
        k_values[order],
    )


def _extract_tabulated_nk(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for block in data.get("DATA", []):
        block_type = str(block.get("type", "")).lower()
        if block_type.startswith("tabulated"):
            rows = _parse_numeric_rows(str(block.get("data", "")))
            if rows.size == 0:
                continue
            if "nk" in block_type and rows.shape[1] >= 3:
                return rows[:, 0], rows[:, 1], rows[:, 2]
            if block_type.endswith(" n") and rows.shape[1] >= 2:
                return rows[:, 0], rows[:, 1], np.zeros(rows.shape[0], dtype=float)
            if block_type.endswith(" k") and rows.shape[1] >= 2:
                raise ValueError("k-only records are not enough to build a material candidate.")
        if block_type == "formula 1":
            return _formula_1_nk(block)
    raise ValueError("No supported tabulated n/nk data block was found.")


def _formula_1_nk(block: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength_range = [float(value) for value in str(block["wavelength_range"]).split()]
    coefficients = [float(value) for value in str(block["coefficients"]).split()]
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], 401)
    lambda_sq = wavelengths**2
    n_sq = np.ones_like(wavelengths) + coefficients[0]
    for index in range(1, len(coefficients) - 1, 2):
        strength = coefficients[index]
        resonance = coefficients[index + 1]
        n_sq += strength * lambda_sq / (lambda_sq - resonance**2)
    n_values = np.sqrt(np.clip(n_sq, 0.0, None))
    return wavelengths, n_values, np.zeros_like(wavelengths)


def _parse_numeric_rows(text: str) -> np.ndarray:
    rows: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = [float(part) for part in stripped.split()]
        rows.append(values)
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = min(len(row) for row in rows)
    return np.asarray([row[:width] for row in rows], dtype=float)


def _source_name_from_url(url: str) -> str:
    name = url.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if name.lower().endswith((".yml", ".yaml")):
        name = name.rsplit(".", maxsplit=1)[0]
    return name.replace("%20", " ")


def safe_candidate_name(text: str) -> str:
    """Return a filesystem-safe candidate/material name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "candidate"


_safe_name = safe_candidate_name
