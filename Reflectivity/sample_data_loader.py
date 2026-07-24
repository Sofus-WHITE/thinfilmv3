from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "sample_data"


def _normalize_sample_name(sample_name: str) -> str:
    prefix, number = sample_name.strip().replace(" ", "").split("-", 1)
    if prefix.upper() == "AU":
        prefix = "Au"
    else:
        prefix = prefix.upper()
    return f"{prefix}-{int(number)}"


def load_samples() -> dict[str, dict[str, Any]]:
    """Load every indexed sample keyed by names like S-11, D-12, or B-2."""
    with (DATA_DIR / "samples.json").open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return {record["sample_name"]: record for record in records}


def get_sample(sample_name: str) -> dict[str, Any]:
    """Return sputtering, reflectance spectrum, and color records for one sample."""
    samples = load_samples()
    key = _normalize_sample_name(sample_name)
    if key not in samples:
        raise KeyError(f"{key} is not present in {DATA_DIR / 'samples.json'}")
    return samples[key]


def spectrum_paths(sample_name: str) -> list[Path]:
    """Return CSV spectrum paths for a sample, resolving paths against this workspace."""
    sample = get_sample(sample_name)
    return [ROOT / item["csv_path"] for item in sample["spectra"] if item.get("csv_path")]


def load_spectrum(sample_name: str, index: int = 0) -> pd.DataFrame:
    """Load one reflectance spectrum as a DataFrame with columns such as nm and %R."""
    paths = spectrum_paths(sample_name)
    if not paths:
        raise FileNotFoundError(f"No reflectance CSV spectra indexed for {sample_name}")
    frame = pd.read_csv(paths[index], skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_all_spectra(sample_name: str) -> list[pd.DataFrame]:
    """Load all reflectance spectra indexed for a sample."""
    frames = []
    for path in spectrum_paths(sample_name):
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [str(column).strip() for column in frame.columns]
        frames.append(frame)
    return frames


def spectrum_records(sample_name: str) -> list[dict[str, Any]]:
    """Return indexed spectrum records, including source system and metadata."""
    return get_sample(sample_name).get("spectra", [])


def load_spectra_with_metadata(sample_name: str) -> list[dict[str, Any]]:
    """Load every spectrum and keep the metadata beside each DataFrame."""
    loaded = []
    for record in spectrum_records(sample_name):
        csv_path = record.get("csv_path")
        if not csv_path:
            continue
        frame = pd.read_csv(ROOT / csv_path, skipinitialspace=True)
        frame.columns = [str(column).strip() for column in frame.columns]
        loaded.append({"metadata": record, "data": frame})
    return loaded


def colors(sample_name: str) -> list[dict[str, Any]]:
    """Return calculated color records, when available, for the sample."""
    return get_sample(sample_name).get("colors", [])


def thickness_estimates(sample_name: str) -> list[dict[str, Any]]:
    """Return nominal/measured and rate-derived layer thickness estimates."""
    return get_sample(sample_name).get("thickness_estimates", [])


def measurement_metadata(sample_name: str) -> list[dict[str, Any]]:
    """Return acquisition metadata for each indexed reflectance spectrum."""
    keys = [
        "source_system",
        "measurement_uid",
        "instrument_sample_id",
        "measurement_number",
        "description",
        "csv_path",
        "surface",
        "substrate",
        "measurement_angle_deg",
        "measurement_angle_note",
        "measurement_width_mm",
        "measurement_length_mm",
        "illuminant",
        "std_observer_deg",
        "instrument",
        "instrument_software",
        "accessory",
        "detector",
        "created_at_raw",
        "method_guid",
        "task_id",
        "metadata_source",
    ]
    return [{key: item.get(key, "") for key in keys} for item in get_sample(sample_name).get("spectra", [])]


def sample_attributes(sample_name: str) -> dict[str, Any]:
    """Return sample-level surface/substrate labels and observed values."""
    sample = get_sample(sample_name)
    return {
        "sample_name": sample["sample_name"],
        "surface": sample.get("surface", "smooth"),
        "substrate": sample.get("substrate", "Silicon"),
        "surface_values_observed": sample.get("surface_values_observed", ["smooth"]),
        "substrate_values_observed": sample.get("substrate_values_observed", ["Silicon"]),
        "base_sample_names": sample.get("base_sample_names", []),
    }


def load_measurement_comparisons() -> pd.DataFrame:
    """Load the flat per-measurement export without collapsing duplicate sample names."""
    path = DATA_DIR / "measurement_comparisons.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run tools/export_measurement_comparisons.py")
    return pd.read_csv(path, keep_default_na=False)


def measurement_comparisons(sample_name: str | None = None) -> pd.DataFrame:
    """Return flat rows for all measurements, or only one normalized sample name."""
    frame = load_measurement_comparisons()
    if sample_name is None:
        return frame
    key = _normalize_sample_name(sample_name)
    return frame[frame["sample_name"] == key].reset_index(drop=True)
