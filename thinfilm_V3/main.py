"""Example simulation for the clean thin-film backend."""

from __future__ import annotations

from pathlib import Path

from src.colorimetry import perceived_color_from_result
from src.materials import built_in_materials
from src.plotting import plot_reflectance
from src.simulation import run_simulation
from src.stack import Layer, make_stack_with_interfaces, native_oxide_for_substrate
from src.tmm_model import TMMModel
from src.utils import wavelength_grid


def build_example_stack():
    """Construct air / TiO2 / SiO2 / Ag / Si with internal roughness and native oxide."""

    materials = built_in_materials()
    substrate_name = "Si"
    return make_stack_with_interfaces(
        incident_medium=materials["air"],
        deposited_layers=[
            Layer(materials["TiO2"], 80.0),
            Layer(materials["SiO2"], 120.0),
            Layer(materials["Ag"], 40.0),
        ],
        substrate=materials[substrate_name],
        native_oxide=native_oxide_for_substrate(materials, substrate_name),
        interface_thickness_nm=1.0,
        interface_fraction=0.5,
        name="air / TiO2 / SiO2 / Ag / Si",
    )


def main() -> None:
    """Run one reflectance simulation and save a plot."""

    stack = build_example_stack()
    wavelengths_nm = wavelength_grid(400.0, 700.0, 301)
    model = TMMModel()

    result = run_simulation(
        model=model,
        stack=stack,
        wavelengths_nm=wavelengths_nm,
        angle_deg=8.0,
    )

    output_path = Path("outputs") / "example_reflectance.png"
    plot_reflectance(result, save_path=output_path, show=False)
    perceived_color = perceived_color_from_result(result)

    print(result.stack_summary)
    print(f"Reflectance range: {result.reflectance.min():.3f} - {result.reflectance.max():.3f}")
    print(f"Perceived color under D65: {perceived_color.hex} RGB{perceived_color.srgb_255}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
