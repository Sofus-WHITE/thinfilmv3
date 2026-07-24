"""Example 1D thickness sweep for TiO2."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_example_stack
from src.plotting import plot_thickness_sweep_1d
from src.thickness_sweep import run_thickness_sweep_1d
from src.tmm_model import TMMModel


def main() -> None:
    stack = build_example_stack()
    model = TMMModel()
    result = run_thickness_sweep_1d(
        stack=stack,
        model=model,
        layer="TiO2",
        thickness_min_nm=20.0,
        thickness_max_nm=200.0,
        angle_deg=8.0,
        quality="normal",
    )
    output_path = PROJECT_ROOT / "outputs" / "thickness_sweep_1d.png"
    plot_thickness_sweep_1d(result, save_path=output_path, show=False)
    print(f"Swept {result.layer_name} over {result.thickness_values_nm.size} points.")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
