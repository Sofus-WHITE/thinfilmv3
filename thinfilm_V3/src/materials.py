"""Material definitions and refractive-index interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class Material(Protocol):
    """Protocol for any object that can return complex refractive index vs wavelength."""

    name: str

    def refractive_index(self, wavelengths_nm: ArrayLike) -> ComplexArray:
        """Return complex refractive index at the requested wavelengths in nm."""


@dataclass(frozen=True)
class ConstantMaterial:
    """Material with wavelength-independent complex refractive index."""

    name: str
    n: complex

    def refractive_index(self, wavelengths_nm: ArrayLike) -> ComplexArray:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        return np.full(wavelengths.shape, self.n, dtype=np.complex128)


@dataclass(frozen=True)
class TabulatedMaterial:
    """Material with tabulated complex refractive index interpolated by wavelength."""

    name: str
    wavelengths_nm: FloatArray
    n_complex: ComplexArray

    def __post_init__(self) -> None:
        wavelengths = np.asarray(self.wavelengths_nm, dtype=float)
        n_complex = np.asarray(self.n_complex, dtype=np.complex128)

        if wavelengths.ndim != 1 or n_complex.ndim != 1:
            raise ValueError("Tabulated material data must be one-dimensional.")
        if wavelengths.size != n_complex.size:
            raise ValueError("Wavelength and refractive-index arrays must have equal length.")
        if wavelengths.size < 2:
            raise ValueError("At least two tabulated wavelengths are required.")
        if np.any(np.diff(wavelengths) <= 0):
            raise ValueError("Tabulated wavelengths must be strictly increasing.")

        object.__setattr__(self, "wavelengths_nm", wavelengths)
        object.__setattr__(self, "n_complex", n_complex)

    def refractive_index(self, wavelengths_nm: ArrayLike) -> ComplexArray:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        real = np.interp(wavelengths, self.wavelengths_nm, self.n_complex.real)
        imag = np.interp(wavelengths, self.wavelengths_nm, self.n_complex.imag)
        return (real + 1j * imag).astype(np.complex128)


@dataclass(frozen=True)
class MaxwellGarnettMaterial:
    """Effective medium for a thin intermixed interface between two materials."""

    name: str
    matrix: Material
    inclusion: Material
    inclusion_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.inclusion_fraction <= 1.0:
            raise ValueError("inclusion_fraction must be between 0 and 1.")

    def refractive_index(self, wavelengths_nm: ArrayLike) -> ComplexArray:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        n_matrix = self.matrix.refractive_index(wavelengths)
        n_inclusion = self.inclusion.refractive_index(wavelengths)
        fraction = self.inclusion_fraction

        eps_matrix = n_matrix**2
        eps_inclusion = n_inclusion**2
        eps_effective = eps_matrix * (
            (eps_inclusion + 2 * eps_matrix + 2 * fraction * (eps_inclusion - eps_matrix))
            / (eps_inclusion + 2 * eps_matrix - fraction * (eps_inclusion - eps_matrix))
        )
        return np.sqrt(eps_effective).astype(np.complex128)


def make_tabulated_material(
    name: str,
    wavelengths_nm: ArrayLike,
    n: ArrayLike,
    k: ArrayLike | None = None,
) -> TabulatedMaterial:
    """Build a tabulated material from n/k arrays or an already complex n array."""

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    n_values = np.asarray(n, dtype=np.complex128)
    if k is not None:
        n_values = np.asarray(n, dtype=float) + 1j * np.asarray(k, dtype=float)
    return TabulatedMaterial(name=name, wavelengths_nm=wavelengths, n_complex=n_values)


def built_in_materials(profile: str = "current") -> dict[str, Material]:
    """Return approximate built-in materials for example simulations."""

    if profile == "current":
        return _current_materials()
    if profile == "legacy_ideal":
        return _legacy_ideal_materials()
    if profile == "legacy_wip":
        return _legacy_wip_materials()
    raise ValueError(f"Unknown material profile: {profile}")


def material_profile_names() -> tuple[str, ...]:
    """Return available built-in optical-constant profiles."""

    return (
        "current",
        "legacy_ideal",
        "legacy_wip",
        "fitted_single_films",
        "best_refractiveindex_candidates",
    )


def _base_materials() -> tuple[FloatArray, dict[str, Material]]:
    visible = np.array([400, 450, 500, 550, 600, 650, 700], dtype=float)
    materials: dict[str, Material] = {
        "air": ConstantMaterial("air", 1.0 + 0j),
        "substrate": ConstantMaterial("substrate", 1.52 + 0j),
        "Si": make_tabulated_material(
            "Si",
            visible,
            np.array([5.57, 4.67, 4.30, 4.08, 3.95, 3.85, 3.78]),
            np.array([0.387, 0.145, 0.0728, 0.0406, 0.0257, 0.0164, 0.0126]),
        ),
        "Ti": make_tabulated_material(
            "Ti",
            visible,
            np.array([2.09, 2.27, 2.37, 2.50, 2.64, 2.74, 2.85]),
            np.array([2.96, 3.04, 3.21, 3.43, 3.65, 3.81, 3.95]),
        ),
        "Ag": make_tabulated_material(
            "Ag",
            visible,
            np.array([0.19, 0.11, 0.08, 0.07, 0.07, 0.07, 0.08]),
            np.array([1.85, 2.30, 2.82, 3.25, 3.66, 4.07, 4.45]),
        ),
        "Au": make_tabulated_material(
            "Au",
            visible,
            np.array([1.67, 1.538, 0.848, 0.324, 0.189, 0.126, 0.0987]),
            np.array([1.97, 1.91, 1.83, 2.597, 3.24, 3.79, 4.31]),
        ),
    }
    return visible, materials


def _current_materials() -> dict[str, Material]:
    """Return the current clean-project default constants."""

    visible = np.array([400, 450, 500, 550, 600, 650, 700], dtype=float)

    return {
        "air": ConstantMaterial("air", 1.0 + 0j),
        "substrate": ConstantMaterial("substrate", 1.52 + 0j),
        "SiO2": make_tabulated_material(
            "SiO2",
            visible,
            np.array([1.48, 1.48, 1.48, 1.47, 1.47, 1.47, 1.47]),
        ),
        "TiO2": make_tabulated_material(
            "TiO2",
            visible,
            np.array([2.57, 2.45, 2.39, 2.35, 2.32, 2.30, 2.28]),
        ),
        "ZrO2": make_tabulated_material(
            "ZrO2",
            visible,
            np.array([2.18, 2.15, 2.12, 2.10, 2.08, 2.07, 2.06]),
        ),
        "Ag": make_tabulated_material(
            "Ag",
            visible,
            np.array([0.19, 0.11, 0.08, 0.07, 0.07, 0.07, 0.08]),
            np.array([1.85, 2.30, 2.82, 3.25, 3.66, 4.07, 4.45]),
        ),
        "Au": make_tabulated_material(
            "Au",
            visible,
            np.array([1.47, 0.92, 0.54, 0.43, 0.27, 0.16, 0.14]),
            np.array([1.95, 1.84, 2.23, 2.46, 2.93, 3.60, 4.10]),
        ),
        "Si": make_tabulated_material(
            "Si",
            visible,
            np.array([5.57, 4.67, 4.30, 4.08, 3.95, 3.85, 3.78]),
            np.array([0.387, 0.145, 0.0728, 0.0406, 0.0257, 0.0164, 0.0126]),
        ),
        "Ti": make_tabulated_material(
            "Ti",
            visible,
            np.array([2.09, 2.27, 2.37, 2.50, 2.64, 2.74, 2.85]),
            np.array([2.96, 3.04, 3.21, 3.43, 3.65, 3.81, 3.95]),
        ),
    }


def _legacy_ideal_materials() -> dict[str, Material]:
    """Return constants matching the cleaner legacy ideal stack builder."""

    visible, materials = _base_materials()
    materials.update(
        {
            "SiO2": ConstantMaterial("SiO2", 1.45 + 0j),
            "TiO2": make_tabulated_material(
                "TiO2",
                visible,
                np.array([2.57, 2.45, 2.39, 2.35, 2.32, 2.30, 2.28]),
            ),
            "ZrO2": ConstantMaterial("ZrO2", 2.15 + 0j),
        }
    )
    return materials


def _legacy_wip_materials() -> dict[str, Material]:
    """Return constants from the legacy WIP roughness stack builder."""

    visible, materials = _base_materials()
    materials.update(
        {
            "SiO2": make_tabulated_material(
                "SiO2",
                visible,
                np.array([1.48, 1.48, 1.48, 1.47, 1.47, 1.47, 1.47]),
            ),
            "TiO2": make_tabulated_material(
                "TiO2",
                visible,
                np.array([2.66, 2.55, 2.49, 2.45, 2.42, 2.40, 2.39]),
            ),
            "ZrO2": ConstantMaterial("ZrO2", 2.15 + 0j),
        }
    )
    return materials


def visible_material_table(
    material: Material,
    wavelengths_nm: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return wavelength, n, k arrays for editing or display."""

    wavelengths = (
        np.array([400, 450, 500, 550, 600, 650, 700], dtype=float)
        if wavelengths_nm is None
        else np.asarray(wavelengths_nm, dtype=float)
    )
    n_complex = material.refractive_index(wavelengths)
    return wavelengths, n_complex.real.astype(float), n_complex.imag.astype(float)
