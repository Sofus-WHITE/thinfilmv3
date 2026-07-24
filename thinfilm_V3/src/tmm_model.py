"""Transfer-matrix-method optical model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tmm import coh_tmm

from .optical_model import OpticalModel
from .results import SimulationResult
from .stack import ThinFilmStack


@dataclass(frozen=True)
class PreparedTMMStack:
    """TMM-ready stack with refractive indices precomputed on a wavelength grid."""

    wavelengths_nm: NDArray[np.float64]
    n_matrix: NDArray[np.complex128]
    base_d_list: NDArray[np.float64]
    layer_names: tuple[str, ...]
    layer_indices: dict[str, tuple[int, ...]]
    display_layer_indices: tuple[int, ...]
    stack_name: str
    stack_summary: str
    display_summary: str

    def finite_layer_index(self, layer: int | str, occurrence: int = 0) -> int:
        """Resolve a selected finite layer to its TMM thickness-list index."""

        if isinstance(layer, int):
            if layer <= 0 or layer >= len(self.base_d_list) - 1:
                raise ValueError("Layer index must refer to a finite TMM layer.")
            return layer

        matches = self.layer_indices.get(layer)
        if not matches:
            available = ", ".join(sorted(self.layer_indices))
            raise ValueError(f"Unknown layer {layer!r}. Available layer names: {available}")
        if occurrence < 0 or occurrence >= len(matches):
            raise ValueError(f"Layer {layer!r} occurrence {occurrence} does not exist.")
        return matches[occurrence]


class TMMModel(OpticalModel):
    """Coherent TMM model that returns unpolarized reflectance spectra."""

    def simulate(
        self,
        stack: ThinFilmStack,
        wavelengths_nm: ArrayLike,
        angle_deg: float,
    ) -> SimulationResult:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths_nm must be a one-dimensional array.")
        if np.any(wavelengths <= 0):
            raise ValueError("Wavelengths must be positive.")

        prepared = self.prepare_stack(stack, wavelengths)
        reflectance = self.reflectance_from_prepared(prepared, prepared.base_d_list, angle_deg)

        return SimulationResult(
            wavelengths_nm=wavelengths,
            reflectance=reflectance,
            angle_deg=angle_deg,
            stack_name=stack.name,
            stack_summary=stack.display_summary(),
            metadata={"model": "coherent TMM", "polarization": "unpolarized"},
        )

    def prepare_stack(
        self,
        stack: ThinFilmStack,
        wavelengths_nm: ArrayLike,
    ) -> PreparedTMMStack:
        """Precompute wavelength-dependent refractive indices for repeated TMM calls."""

        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths_nm must be a one-dimensional array.")
        if np.any(wavelengths <= 0):
            raise ValueError("Wavelengths must be positive.")

        base_d_list = np.asarray(stack.thicknesses_for_tmm(), dtype=float)
        layer_names = tuple(material.name for material in stack.all_materials)
        layer_indices: dict[str, list[int]] = {}
        for index, name in enumerate(layer_names[1:-1], start=1):
            layer_indices.setdefault(name, []).append(index)

        return PreparedTMMStack(
            wavelengths_nm=wavelengths,
            n_matrix=stack.refractive_indices_for_tmm(wavelengths),
            base_d_list=base_d_list,
            layer_names=layer_names,
            layer_indices={name: tuple(indices) for name, indices in layer_indices.items()},
            display_layer_indices=_display_layer_indices(stack, layer_names),
            stack_name=stack.name,
            stack_summary=stack.layer_summary(),
            display_summary=stack.display_summary(),
        )

    def reflectance_from_prepared(
        self,
        prepared: PreparedTMMStack,
        d_list: ArrayLike,
        angle_deg: float,
    ) -> NDArray[np.float64]:
        """Calculate reflectance using precomputed indices and supplied thicknesses."""

        thicknesses = np.asarray(d_list, dtype=float)
        if thicknesses.shape != prepared.base_d_list.shape:
            raise ValueError("d_list shape does not match the prepared stack.")

        theta_rad = np.deg2rad(angle_deg)
        reflectance = np.empty_like(prepared.wavelengths_nm, dtype=float)
        d_list_for_tmm = thicknesses.tolist()
        for index, wavelength_nm in enumerate(prepared.wavelengths_nm):
            n_list = prepared.n_matrix[:, index].tolist()
            r_s = coh_tmm("s", n_list, d_list_for_tmm, theta_rad, wavelength_nm)["R"]
            r_p = coh_tmm("p", n_list, d_list_for_tmm, theta_rad, wavelength_nm)["R"]
            reflectance[index] = 0.5 * (r_s + r_p)
        return np.clip(reflectance, 0.0, 1.0)

    def simulate_from_prepared(
        self,
        prepared: PreparedTMMStack,
        d_list: ArrayLike,
        angle_deg: float,
    ) -> SimulationResult:
        """Return a standard result from an already prepared stack."""

        reflectance = self.reflectance_from_prepared(prepared, d_list, angle_deg)
        return SimulationResult(
            wavelengths_nm=prepared.wavelengths_nm,
            reflectance=reflectance,
            angle_deg=angle_deg,
            stack_name=prepared.stack_name,
            stack_summary=prepared.display_summary,
            metadata={"model": "coherent TMM", "polarization": "unpolarized", "prepared": True},
        )


def _display_layer_indices(
    stack: ThinFilmStack,
    layer_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Map user-facing deposited layers to their indices in the full optical stack."""

    display_layers = stack.display_layers if stack.display_layers is not None else stack.layers
    indices: list[int] = []
    search_start = 1
    for display_layer in display_layers:
        for index in range(search_start, len(layer_names) - 1):
            if layer_names[index] == display_layer.material.name:
                indices.append(index)
                search_start = index + 1
                break
    return tuple(indices)
