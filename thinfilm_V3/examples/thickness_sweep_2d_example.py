"""Example 2D thickness sweep for TiO2 and SiO2."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_example_stack
from src.plotting import plot_thickness_sweep_2d
from src.thickness_sweep import run_thickness_sweep_2d
from src.tmm_model import TMMModel


def main() -> None:
    stack = build_example_stack()
    model = TMMModel()
    result = run_thickness_sweep_2d(
        stack=stack,
        model=model,
        layer_1="TiO2",
        layer_2="SiO2",
        thickness_1_min_nm=20.0,
        thickness_1_max_nm=200.0,
        thickness_2_min_nm=20.0,
        thickness_2_max_nm=200.0,
        angle_deg=8.0,
        quality="normal",
    )
    output_path = PROJECT_ROOT / "outputs" / "thickness_sweep_2d.png"
    plot_thickness_sweep_2d(result, save_path=output_path, show=False)
    print(
        f"Swept {result.layer_name_1} x {result.layer_name_2} over "
        f"{result.rgb_grid.shape[1]} x {result.rgb_grid.shape[0]} points."
    )
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
