"""Fast reflectance-to-colour conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ColorConversionCache:
    """Precomputed D65/CIE colour matching data for one wavelength grid."""

    wavelengths_nm: NDArray[np.float64]
    xyz_weights: NDArray[np.float64]
    uses_colour_science: bool = True


def prepare_color_conversion(wavelengths_nm: ArrayLike) -> ColorConversionCache:
    """Precompute D65/CIE weights used to convert many spectra efficiently."""

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths_nm must be one-dimensional.")

    try:
        import colour

        spacing = _uniform_spacing_nm(wavelengths)
        shape = colour.SpectralShape(
            float(wavelengths[0]),
            float(wavelengths[-1]),
            spacing,
        )
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
        illuminant = colour.SDS_ILLUMINANTS["D65"].copy().align(shape)
        cmf_values = np.asarray(cmfs.values, dtype=float)
        illuminant_values = np.asarray(illuminant.values, dtype=float)
        if cmf_values.shape[0] != wavelengths.size:
            raise ValueError("Colour matching data did not align to the wavelength grid.")

        normalizer = 100.0 / np.sum(illuminant_values * cmf_values[:, 1] * spacing)
        xyz_weights = normalizer * illuminant_values[:, None] * cmf_values * spacing
        return ColorConversionCache(wavelengths, xyz_weights, uses_colour_science=True)
    except Exception:
        xyz_weights = _fallback_xyz_weights(wavelengths)
        return ColorConversionCache(wavelengths, xyz_weights, uses_colour_science=False)


def reflectance_to_xyz(
    reflectance: ArrayLike,
    cache: ColorConversionCache | None = None,
    wavelengths_nm: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Convert one or many reflectance spectra to CIE XYZ under D65."""

    spectra = np.asarray(reflectance, dtype=float)
    if cache is None:
        if wavelengths_nm is None:
            raise ValueError("wavelengths_nm is required when cache is not supplied.")
        cache = prepare_color_conversion(wavelengths_nm)

    return np.tensordot(spectra, cache.xyz_weights, axes=([-1], [0]))


def xyz_to_srgb(xyz: ArrayLike) -> NDArray[np.float64]:
    """Convert XYZ values to clipped display sRGB values."""

    xyz_array = np.asarray(xyz, dtype=float)
    try:
        import colour

        srgb = colour.XYZ_to_sRGB(xyz_array / 100.0)
    except Exception:
        srgb = _fallback_xyz_to_srgb(xyz_array)
    return np.clip(np.asarray(srgb, dtype=float), 0.0, 1.0)


def reflectance_to_srgb(
    reflectance: ArrayLike,
    cache: ColorConversionCache | None = None,
    wavelengths_nm: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Convert one or many reflectance spectra to clipped display sRGB."""

    xyz = reflectance_to_xyz(reflectance, cache=cache, wavelengths_nm=wavelengths_nm)
    return xyz_to_srgb(xyz)


def _uniform_spacing_nm(wavelengths_nm: NDArray[np.float64]) -> float:
    """Return wavelength spacing and require a uniform grid for colour-science alignment."""

    if wavelengths_nm.size < 2:
        raise ValueError("At least two wavelengths are required.")
    differences = np.diff(wavelengths_nm)
    spacing = float(differences[0])
    if not np.allclose(differences, spacing):
        raise ValueError("Colour conversion currently requires a uniform wavelength grid.")
    return spacing


def _fallback_xyz_weights(wavelengths_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Crude Gaussian fallback when colour-science is unavailable."""

    spacing = _uniform_spacing_nm(wavelengths_nm)
    x_bar = np.exp(-0.5 * ((wavelengths_nm - 600.0) / 45.0) ** 2)
    y_bar = np.exp(-0.5 * ((wavelengths_nm - 550.0) / 38.0) ** 2)
    z_bar = np.exp(-0.5 * ((wavelengths_nm - 450.0) / 32.0) ** 2)
    illuminant = np.ones_like(wavelengths_nm)
    normalizer = 100.0 / np.sum(illuminant * y_bar * spacing)
    return normalizer * illuminant[:, None] * np.column_stack([x_bar, y_bar, z_bar]) * spacing


def _fallback_xyz_to_srgb(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Approximate XYZ-to-sRGB conversion used only when colour-science is unavailable."""

    xyz_scaled = xyz / 100.0
    matrix = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    rgb_linear = np.matmul(xyz_scaled, matrix.T)
    rgb_linear = np.clip(rgb_linear, 0.0, None)
    return np.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1 / 2.4) - 0.055,
    )
