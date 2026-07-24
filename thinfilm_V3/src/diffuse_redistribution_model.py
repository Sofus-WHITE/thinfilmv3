"""TMM model that redistributes rough-surface reflection into a diffuse proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .results import SimulationResult
from .stack import ThinFilmStack
from .tmm_model import PreparedTMMStack, TMMModel


@dataclass(frozen=True)
class DiffuseRedistributionSettings:
    """Settings for a roughness-based specular/diffuse appearance approximation."""

    rms_roughness_nm: float = 1.0
    scatter_scale: float = 1.0
    wavelength_exponent: float = 0.0
    max_scatter_fraction: float = 0.85
    diffuse_angle_min_deg: float = 0.0
    diffuse_angle_max_deg: float = 80.0
    diffuse_angle_samples: int = 17
    reference_wavelength_nm: float = 550.0
    interface_count: int | None = None


class TMMWithDiffuseRedistributionModel(TMMModel):
    """Blend 8-degree/specular TMM with an angle-averaged diffuse proxy."""

    def __init__(self, settings: DiffuseRedistributionSettings | None = None) -> None:
        self.settings = settings or DiffuseRedistributionSettings()

    def simulate(
        self,
        stack: ThinFilmStack,
        wavelengths_nm: ArrayLike,
        angle_deg: float,
    ) -> SimulationResult:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        prepared = self.prepare_stack(stack, wavelengths)
        reflectance = self.reflectance_from_prepared(prepared, prepared.base_d_list, angle_deg)
        return SimulationResult(
            wavelengths_nm=wavelengths,
            reflectance=reflectance,
            angle_deg=angle_deg,
            stack_name=stack.name,
            stack_summary=stack.display_summary(),
            metadata={
                "model": "coherent TMM + diffuse roughness redistribution",
                "polarization": "unpolarized",
                "rms_roughness_nm": self.settings.rms_roughness_nm,
                "scatter_scale": self.settings.scatter_scale,
                "wavelength_exponent": self.settings.wavelength_exponent,
                "max_scatter_fraction": self.settings.max_scatter_fraction,
                "diffuse_angle_min_deg": self.settings.diffuse_angle_min_deg,
                "diffuse_angle_max_deg": self.settings.diffuse_angle_max_deg,
                "diffuse_angle_samples": self.settings.diffuse_angle_samples,
                "roughness_interface_count": self._prepared_interface_count(prepared),
            },
        )

    def reflectance_from_prepared(
        self,
        prepared: PreparedTMMStack,
        d_list: ArrayLike,
        angle_deg: float,
    ) -> NDArray[np.float64]:
        """Return specular TMM blended with a multi-angle diffuse proxy."""

        specular = TMMModel.reflectance_from_prepared(self, prepared, d_list, angle_deg)
        diffuse_proxy = self.diffuse_proxy_from_prepared(prepared, d_list)
        scatter_fraction = self.scatter_fraction_from_prepared(prepared, angle_deg)
        reflected = (1.0 - scatter_fraction) * specular + scatter_fraction * diffuse_proxy
        return np.clip(reflected, 0.0, 1.0)

    def diffuse_proxy_from_prepared(
        self,
        prepared: PreparedTMMStack,
        d_list: ArrayLike,
    ) -> NDArray[np.float64]:
        """Approximate sphere-collected diffuse colour as an angle-averaged TMM spectrum."""

        settings = self.settings
        sample_count = max(int(settings.diffuse_angle_samples), 1)
        if sample_count == 1:
            angles = np.array([settings.diffuse_angle_min_deg], dtype=float)
        else:
            angles = np.linspace(
                settings.diffuse_angle_min_deg,
                settings.diffuse_angle_max_deg,
                sample_count,
                dtype=float,
            )
        angles = np.clip(angles, 0.0, 89.0)
        weights = np.cos(np.deg2rad(angles))
        weights = np.maximum(weights, 0.0)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones_like(angles)

        accumulated = np.zeros_like(prepared.wavelengths_nm, dtype=float)
        for angle, weight in zip(angles, weights):
            accumulated += float(weight) * TMMModel.reflectance_from_prepared(
                self,
                prepared,
                d_list,
                float(angle),
            )
        return accumulated / float(np.sum(weights))

    def scatter_fraction_from_prepared(
        self,
        prepared: PreparedTMMStack,
        angle_deg: float,
    ) -> NDArray[np.float64]:
        """Return wavelength-dependent fraction mixed into the diffuse proxy."""

        settings = self.settings
        sigma = max(float(settings.rms_roughness_nm), 0.0)
        if sigma == 0.0 or settings.scatter_scale <= 0.0:
            return np.zeros_like(prepared.wavelengths_nm, dtype=float)

        cos_theta = np.cos(np.deg2rad(angle_deg))
        interface_count = self._prepared_interface_count(prepared)
        base = 1.0 - np.exp(
            -interface_count
            * (4.0 * np.pi * sigma * cos_theta / prepared.wavelengths_nm) ** 2
        )
        wavelength_weight = (
            float(settings.reference_wavelength_nm) / prepared.wavelengths_nm
        ) ** float(settings.wavelength_exponent)
        scatter = float(settings.scatter_scale) * base * wavelength_weight
        return np.clip(scatter, 0.0, max(float(settings.max_scatter_fraction), 0.0))

    def _prepared_interface_count(self, prepared: PreparedTMMStack) -> int:
        if self.settings.interface_count is not None:
            return max(int(self.settings.interface_count), 0)
        return len(prepared.display_layer_indices) + 1
