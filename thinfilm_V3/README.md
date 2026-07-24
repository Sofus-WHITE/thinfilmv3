# thinfilm_V3

Thinfilm V3 is a cleaned-up desktop workspace for thin-film reflectance simulation,
experiment comparison, cached fitting, and colour-distance review.

The physics backend is preserved from `thinfilm_v2`:

- coherent transfer-matrix reflectance;
- effective interface layers;
- RMS roughness attenuation;
- diffuse redistribution for rough integrating-sphere style measurements;
- D65/CIE/sRGB colour conversion;
- 1D/2D thickness sweeps and angle sweeps.

The main GUI change is workflow organization. V3 keeps `Reflectance` and `Sweep`,
omits the Machine Learning tab, and groups fitting actions into one
`Fit & Optimize` tab.

## Data Source

V3 reads the generated indexes in:

```text
..\Reflectivity\sample_data
```

Those files are built from the Reflectivity tools and `all sputtering.docx`.
Important inputs are:

- `samples.json`
- `sample_index.csv`
- `measurement_index.csv`
- `thickness_estimates.csv`
- `thickness_calibrations.csv`
- `measurement_comparisons.csv`

The `Reflectivity\sample_data_loader.py` helper and
`Reflectivity\sample_data_README.md` describe the same index contract. V3 uses
the generated lists as the source of truth rather than parsing raw spectra or
the Word document inside the GUI.

## Tabs

- `Reflectance`: preserved live stack simulation and colour swatch.
- `Sweep`: preserved 1D thickness, 2D thickness, and angle sweeps.
- `Experiments`: cleaner experiment browsing, cache loading, measured/simulated
  spectrum plots, CIE plots, and quick selected-row actions.
- `Fit & Optimize`: one workspace for filtered sample groups, thickness
  optimization, overnight cache generation, grouped sputter-rate fitting,
  roughness fitting, and model/constants calibration.
- `Search`: target-colour search using the currently selected model/constants.
  Use `Fit & Optimize -> Model calibration -> Use selected model` before search
  when you want to search with a calibrated recipe.
- `Colour Distance`: compares raw model-cache Delta E against best cached
  thickness fits, grouped by substrate/surface and sample.
- `Constants`: preserved constants inspection, editing, importing, and candidate
  fitting.

## Caches

Long calculations are designed to be reused:

```text
outputs\experiment_cache
outputs\thickness_optimization_cache
outputs\thickness_optimization_summary
outputs\sputter_rate_colour_fit
outputs\physical_calibration
outputs\roughness_fits
outputs\model_constant_benchmark
```

The overnight thickness cache respects the V3 filters:

- series;
- substrate;
- surface;
- measurement type;
- composition;
- optional sample limit.

Completed thickness trials are saved continuously and reused on later runs.

## Run

From this folder:

```powershell
.\.venv\Scripts\python.exe gui.py
```

or use the same Python environment used for `thinfilm_v2`.

## Machine Learning

Machine learning is intentionally omitted from the V3 GUI for now. The safer
near-term path is:

1. fit constants and optical model groups with the physics model;
2. cache thickness/rate/roughness fits;
3. use those cached, physically labelled results as training data later.

End-to-end ML can be added later as a surrogate or residual-correction model,
but it should not replace the TMM model until the dataset is large enough and
split by sample/date/group to avoid leakage.
