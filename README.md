# ThinFilm V3 Demo Package

This folder is a GitHub-ready demo copy of `thinfilm_V3` with a small sample dataset.
It is intended to show the program interface, reflectance simulation, experiment loading,
colour comparison, sweep maps, and fitting workflow without publishing the full research archive.

## Included

- `thinfilm_V3/`: application source code, examples, config, and markdown docs.
- `Reflectivity/sample_data/`: a small generated index subset.
- Referenced reflectance CSV files for the demo samples only.
- Minimal sputter-rate CSV files for the latest-rate display.

## Demo Samples

The demo dataset contains seven samples:

- `S-10`: TiO2 single layer
- `S-11`: Ag single layer
- `C-5`: SiO2 single layer
- `D-12`: TiO2 / Ag two-layer
- `D-18`: TiO2 / Ag two-layer
- `B-2`: TiO2 / SiO2 two-layer
- `B-1`: TiO2 / SiO2 / Ag triple-layer

The full experimental dataset, Word deposition log, PDFs, and generated caches are intentionally omitted.

## Run

Create or activate a Python environment with the requirements installed, then run:

```powershell
cd thinfilm_V3
python gui.py
```

By default, the GUI looks for data at `../Reflectivity/sample_data`, which is included in this demo package.

## Notes

This dataset is small, so long fitting workflows are only illustrative. For serious calibration,
use the full private Reflectivity dataset outside the public repository.
