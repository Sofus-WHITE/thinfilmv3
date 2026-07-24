"""TMM model with an approximate RMS roughness correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .results import SimulationResult
from .stack import ThinFilmStack
from .tmm_model import PreparedTMMStack, TMMModel


@dataclass(frozen=True)
class RoughnessCorrectionSettings:
    """Settings for specular attenuation from RMS interface roughness."""

    rms_roughness_nm: float = 1.0
    interface_count: int | None = None


class TMMWithRoughnessModel(TMMModel):
    """Coherent TMM followed by a Debye-Waller-style specular roughness correction."""

    def __init__(self, settings: RoughnessCorrectionSettings | None = None) -> None:
        self.settings = settings or RoughnessCorrectionSettings()

    def simulate(
        self,
        stack: ThinFilmStack,
        wavelengths_nm: ArrayLike,
        angle_deg: float,
    ) -> SimulationResult:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        prepared = self.prepare_stack(stack, wavelengths)
        smooth_reflectance = TMMModel.reflectance_from_prepared(
            self,
            prepared,
            prepared.base_d_list,
            angle_deg,
        )
        factor = self.roughness_factor_from_prepared(prepared, angle_deg)
        corrected = np.clip(smooth_reflectance * factor, 0.0, 1.0)
        return SimulationResult(
            wavelengths_nm=wavelengths,
            reflectance=corrected,
            angle_deg=angle_deg,
            stack_name=stack.name,
            stack_summary=stack.display_summary(),
            metadata={
                "model": "coherent TMM + RMS roughness correction",
                "polarization": "unpolarized",
                "rms_roughness_nm": self.settings.rms_roughness_nm,
                "roughness_interface_count": self._interface_count(stack),
            },
        )

    def roughness_factor(
        self,
        wavelengths_nm: ArrayLike,
        angle_deg: float,
        stack: ThinFilmStack,
    ) -> NDArray[np.float64]:
        """Return wavelength-dependent specular attenuation from RMS roughness."""

        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        sigma = max(float(self.settings.rms_roughness_nm), 0.0)
        if sigma == 0:
            return np.ones_like(wavelengths, dtype=float)

        cos_theta = np.cos(np.deg2rad(angle_deg))
        interface_count = self._interface_count(stack)
        exponent = -interface_count * (4.0 * np.pi * sigma * cos_theta / wavelengths) ** 2
        return np.exp(exponent)

    def reflectance_from_prepared(
        self,
        prepared: PreparedTMMStack,
        d_list: ArrayLike,
        angle_deg: float,
    ) -> NDArray[np.float64]:
        """Calculate prepared-stack reflectance and apply RMS roughness attenuation."""

        base_reflectance = super().reflectance_from_prepared(prepared, d_list, angle_deg)
        factor = self.roughness_factor_from_prepared(prepared, angle_deg)
        return np.clip(base_reflectance * factor, 0.0, 1.0)

    def roughness_factor_from_prepared(
        self,
        prepared: PreparedTMMStack,
        angle_deg: float,
    ) -> NDArray[np.float64]:
        """Return roughness attenuation for a prepared stack."""

        sigma = max(float(self.settings.rms_roughness_nm), 0.0)
        if sigma == 0:
            return np.ones_like(prepared.wavelengths_nm, dtype=float)

        cos_theta = np.cos(np.deg2rad(angle_deg))
        interface_count = self._prepared_interface_count(prepared)
        exponent = -interface_count * (4.0 * np.pi * sigma * cos_theta / prepared.wavelengths_nm) ** 2
        return np.exp(exponent)

    def _interface_count(self, stack: ThinFilmStack) -> int:
        if self.settings.interface_count is not None:
            return max(int(self.settings.interface_count), 0)
        display_layers = stack.display_layers if stack.display_layers is not None else stack.layers
        return len(display_layers) + 1

    def _prepared_interface_count(self, prepared: PreparedTMMStack) -> int:
        if self.settings.interface_count is not None:
            return max(int(self.settings.interface_count), 0)
        return len(prepared.display_layer_indices) + 1
