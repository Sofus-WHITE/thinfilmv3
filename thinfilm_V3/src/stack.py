"""Layer and stack objects for thin-film simulations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import Material, MaxwellGarnettMaterial


@dataclass(frozen=True)
class Layer:
    """A finite film layer with material and thickness in nanometers."""

    material: Material
    thickness_nm: float

    def __post_init__(self) -> None:
        if self.thickness_nm < 0:
            raise ValueError("Layer thickness must be non-negative.")


@dataclass(frozen=True)
class NativeOxide:
    """Finite oxide layer naturally present on top of a substrate."""

    material: Material
    thickness_nm: float

    def __post_init__(self) -> None:
        if self.thickness_nm < 0:
            raise ValueError("Native oxide thickness must be non-negative.")


@dataclass(frozen=True)
class ThinFilmStack:
    """Ordered thin-film stack with semi-infinite incident and substrate media."""

    incident_medium: Material
    layers: tuple[Layer, ...] = field(default_factory=tuple)
    substrate: Material | None = None
    name: str = "unnamed stack"
    display_layers: tuple[Layer, ...] | None = None

    def __post_init__(self) -> None:
        if self.substrate is None:
            raise ValueError("A substrate material is required.")
        object.__setattr__(self, "layers", tuple(self.layers))
        if self.display_layers is not None:
            object.__setattr__(self, "display_layers", tuple(self.display_layers))

    @property
    def all_materials(self) -> tuple[Material, ...]:
        """Materials in TMM order: incident medium, finite layers, substrate."""

        return (self.incident_medium, *(layer.material for layer in self.layers), self.substrate)

    def thicknesses_for_tmm(self) -> list[float]:
        """Return TMM thickness list with infinite incident and substrate media."""

        return [np.inf, *(layer.thickness_nm for layer in self.layers), np.inf]

    def refractive_indices_for_tmm(self, wavelengths_nm: ArrayLike) -> NDArray[np.complex128]:
        """Return a layer-by-wavelength complex index matrix for efficient simulation."""

        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        return np.vstack([material.refractive_index(wavelengths) for material in self.all_materials])

    def layer_summary(self) -> str:
        """Return a compact human-readable stack description."""

        finite = " / ".join(
            f"{layer.thickness_nm:g} nm {layer.material.name}" for layer in self.layers
        )
        return f"{self.incident_medium.name} / {finite} / {self.substrate.name}"

    def display_summary(self) -> str:
        """Return the user-facing stack without hidden interface/oxide details."""

        layers = self.display_layers if self.display_layers is not None else self.layers
        finite = " / ".join(f"{layer.thickness_nm:g} nm {layer.material.name}" for layer in layers)
        if finite:
            return f"{self.incident_medium.name} / {finite} / {self.substrate.name}"
        return f"{self.incident_medium.name} / {self.substrate.name}"


def make_stack(
    incident_medium: Material,
    substrate: Material,
    layers: list[Layer] | tuple[Layer, ...],
    name: str = "unnamed stack",
    display_layers: list[Layer] | tuple[Layer, ...] | None = None,
) -> ThinFilmStack:
    """Create a validated stack while keeping construction concise for scripts and GUIs."""

    return ThinFilmStack(
        incident_medium=incident_medium,
        layers=tuple(layers),
        substrate=substrate,
        name=name,
        display_layers=None if display_layers is None else tuple(display_layers),
    )


def make_stack_with_interfaces(
    incident_medium: Material,
    deposited_layers: list[Layer] | tuple[Layer, ...],
    substrate: Material,
    native_oxide: NativeOxide | None = None,
    interface_thickness_nm: float = 1.0,
    interface_fraction: float = 0.5,
    name: str = "stack with roughness interfaces",
) -> ThinFilmStack:
    """Build a stack with thin effective-medium roughness layers between media."""

    if interface_thickness_nm < 0:
        raise ValueError("Interface thickness must be non-negative.")

    optical_layers = list(deposited_layers)
    if native_oxide is not None and native_oxide.thickness_nm > 0:
        optical_layers.append(Layer(native_oxide.material, native_oxide.thickness_nm))

    layers_with_interfaces: list[Layer] = []
    previous_material = incident_medium
    for layer in optical_layers:
        if interface_thickness_nm > 0:
            layers_with_interfaces.append(
                _make_interface_layer(
                    previous_material,
                    layer.material,
                    interface_thickness_nm,
                    interface_fraction,
                )
            )
        layers_with_interfaces.append(layer)
        previous_material = layer.material

    if interface_thickness_nm > 0:
        layers_with_interfaces.append(
            _make_interface_layer(
                previous_material,
                substrate,
                interface_thickness_nm,
                interface_fraction,
            )
        )

    return make_stack(
        incident_medium=incident_medium,
        substrate=substrate,
        layers=layers_with_interfaces,
        name=name,
        display_layers=deposited_layers,
    )


def native_oxide_for_substrate(
    materials: dict[str, Material],
    substrate_name: str,
) -> NativeOxide | None:
    """Return a simple built-in native oxide layer for common substrates."""

    native_oxides = {
        "Si": ("SiO2", 2.0),
        "Ti": ("TiO2", 5.0),
    }
    oxide = native_oxides.get(substrate_name)
    if oxide is None:
        return None

    oxide_name, thickness_nm = oxide
    return NativeOxide(material=materials[oxide_name], thickness_nm=thickness_nm)


def _make_interface_layer(
    lower_material: Material,
    upper_material: Material,
    thickness_nm: float,
    inclusion_fraction: float,
) -> Layer:
    """Create one finite effective-medium interface layer."""

    material = MaxwellGarnettMaterial(
        name=f"mix({lower_material.name},{upper_material.name})",
        matrix=lower_material,
        inclusion=upper_material,
        inclusion_fraction=inclusion_fraction,
    )
    return Layer(material=material, thickness_nm=thickness_nm)
