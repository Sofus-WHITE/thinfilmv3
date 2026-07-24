"""Example angle sweep for the current example stack."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_example_stack
from src.angle_sweep import run_angle_sweep
from src.plotting import plot_angle_sweep
from src.tmm_model import TMMModel


def main() -> None:
    stack = build_example_stack()
    model = TMMModel()
    result = run_angle_sweep(
        stack=stack,
        model=model,
        angle_min_deg=0.0,
        angle_max_deg=80.0,
        quality="normal",
        num_points=100,
    )
    output_path = PROJECT_ROOT / "outputs" / "angle_sweep.png"
    plot_angle_sweep(result, save_path=output_path, show=False)
    print(f"Swept angle over {result.angle_values_deg.size} points.")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
