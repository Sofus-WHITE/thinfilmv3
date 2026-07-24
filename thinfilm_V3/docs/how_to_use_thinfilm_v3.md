# Thinfilm V3 How-To

Updated: 2026-06-30

Thinfilm V3 is organized around this workflow:

```text
Reflectivity sample_data indexes
-> material constants + optical model
-> experiment comparison cache
-> thickness/rate/roughness/model fitting
-> colour-distance review
-> target search
```

## 1. Load Experiment Data

Open the `Experiments` tab and check the `Reflectivity index overview`.
The default path points to:

```text
..\Reflectivity\sample_data
```

This folder is generated from `all sputtering.docx`, the reflectance exports,
and the Reflectivity build scripts. V3 reads these generated lists:

- `sample_index.csv`
- `measurement_index.csv`
- `thickness_estimates.csv`
- `thickness_calibrations.csv`
- `samples.json`

## 2. Choose Physics Settings

Use the left settings panel on `Experiments`, `Fit & Optimize`, and
`Colour Distance`:

- constants profile;
- optical model;
- angle;
- effective-interface thickness and mix fraction;
- RMS/diffuse redistribution settings;
- native substrate oxide.

Keep the distinction clear:

```text
constants profile = refractive-index data
optical model = how reflectance is calculated
```

## 3. Build Or Load The Model Cache

Use:

```text
Build model cache
```

or:

```text
Load saved results
```

The cache stores simulated-vs-measured spectra and colour distances for browsing
without rerunning TMM every time.

## 4. Fit And Optimize

Use the `Fit & Optimize` tab for all heavy fitting.

Filters:

- series;
- substrate;
- surface;
- measurement type;
- composition;
- optional sample limit.

Actions:

- `Optimize selected`: fit the selected experiment row.
- `Overnight thickness cache`: precompute filtered thickness fits and reuse saved
  trials later.
- `Fit selected rate groups`: fit selected material/target/pressure/flow groups.
- `Fit all visible rate groups`: fit every rate group currently visible under
  the filters.
- `Benchmark constants/models`: compare constants profiles and optical models.
- `Fit roughness group`: fit diffuse redistribution roughness parameters for the
  active experiment group.

## 5. Use A Calibrated Model For Search

After a model/constants benchmark, select a row in:

```text
Fit & Optimize -> Model calibration
```

Then press:

```text
Use selected model
```

The `Search` tab will then use that active constants profile and optical model.

## 6. Compare Fits

Open `Colour Distance`.

This compares:

- raw model-cache Delta E;
- best cached thickness-fit Delta E, when available.

Use it to find:

- groups that improve after thickness fitting;
- groups that still fail after fitting;
- samples worth rerunning with different constants or roughness settings.

## 7. Future Machine Learning

Machine learning is not active in V3. A practical future version can train on
cached physics results as either:

- a fast surrogate for expensive TMM searches;
- a residual correction from TMM-predicted colour to measured colour;
- a recommender for promising thickness/rate/model settings.

The safest route is not "fit everything blindly." Use the physics caches first,
then train with group-aware validation by date/sample family so the model does
not simply memorize neighbouring measurements.
