# Legacy Program Comparison Report

Date: 2026-05-26

## Short answer

The new program and the legacy program use the same underlying coherent transfer-matrix physics when they are reduced to the same layer list, refractive-index constants, polarization, and colour conversion assumptions. The core TMM math in the new code is correct.

They do not match exactly by default, because the legacy scripts are not one single well-defined model. Different legacy files use different constants, different roughness assumptions, and sometimes different colour-conversion paths. The biggest differences are:

1. The main legacy colour function uses the roughness stack builder, not the clean ideal builder.
2. The legacy roughness builder uses TiO2 constants that are higher than the new defaults.
3. The legacy roughness Maxwell-Garnett equation appears non-standard and can give nonphysical interface indices.
4. The legacy colour path uses `tmm.color.calc_reflectances()` with its default `pol='s'`, while the new program uses unpolarized reflectance.
5. The old `give_colour` path does not run cleanly in the current environment because `tmm.color` disables itself after its `colorpy` import check.
6. The simulator itself did not have a stack-order problem. The ambiguity was in the experiment importer's interpretation of exported `layer_order`: those rows appear to follow deposition order in the sputtering log, while the optical solver expects air-facing order.

## What Was Compared

Legacy reference files:

- `Legacy programming/Multilayersimulation - Copy/calculatendlists.py`
- `Legacy programming/Multilayersimulation - Copy/calculatendlists_WIProughness.py`
- `Legacy programming/Multilayersimulation - Copy/givecolourfrommultilayer.py`
- `Legacy programming/Multilayersimulation - Copy/givecolourfrommultilayerV2withXYZ.py`
- `Legacy programming/Multilayersimulation - Copy/xyYtoRGB.py`
- `Legacy programming/Multilayersimulation - Copy/previous versions/SpectrophotometerplotV7.py`

New reference files:

- `src/tmm_model.py`
- `src/materials.py`
- `src/stack.py`
- `src/color.py`
- `src/experiments.py`

## TMM Math

The new program uses `tmm.coh_tmm` for s and p polarizations and averages them:

```text
R_unpolarized = 0.5 * (R_s + R_p)
```

That is the right choice for ordinary unpolarized reflectance simulation.

The legacy `tmm.color.calc_reflectances()` internally calls `coh_tmm`, so the underlying transfer-matrix math is the same. However, its default is `pol='s'`. The old `give_colour(...)` function calls:

```python
reflectances = color.calc_reflectances(n_fn_list, d_list, th_0)
```

without specifying polarization. Therefore the legacy colour simulation is s-polarized unless another script explicitly overrides it.

At 8 degrees the s-vs-unpolarized difference is usually not the dominant source of mismatch, but it is still a real difference.

## Stack Ordering

The legacy stack builders expect `thickness_data` in optical order:

```text
air / thickness_data[0] / thickness_data[1] / ... / substrate
```

The experimental data table appears to use `layer_order` as deposition order. For sputtering, the first deposited layer is closest to the substrate, while the final deposited layer is closest to air. Therefore an importer that wants to infer the optical stack from deposition-order rows must reverse numeric `layer_order` before building the optical stack.

This was not a problem with the TMM stack object itself. If a user manually enters:

```text
air / TiO2 / SiO2 / Ag / Si
```

then the simulator uses exactly that order. The ambiguity only appeared when importing experiment rows from `thickness_estimates.csv`.

If those exported rows are interpreted literally as air-facing order, several samples become top-metal stacks. For instance, A-2 would be imported as:

```text
air / Ag / SiO2 / TiO2 / Si
```

instead of:

```text
air / TiO2 / SiO2 / Ag / Si
```

That caused several simulated colours to collapse toward near-white metal reflection.

### Concrete Examples

The table below shows examples where interpreting the exported row order literally gives a different optical stack from interpreting it as deposition order. Colour swatches are included because RGB codes alone are not visually useful.

| Sample | Literal exported-row order | Literal colour | Deposition-order interpretation | Optical colour |
|---|---|---|---|---|
| A-2 | `air / 779.9 nm Ag / 78 nm SiO2 / 137.875 nm TiO2 / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fdfdf6;border:1px solid #888"></span> `#fdfdf6` RGB(253,253,246) | `air / 137.875 nm TiO2 / 78 nm SiO2 / 779.9 nm Ag / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fafefa;border:1px solid #888"></span> `#fafefa` RGB(250,254,250) |
| A-3 | `air / 3899.5 nm Ag / 62.4 nm SiO2 / 110.3 nm TiO2 / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fdfdf6;border:1px solid #888"></span> `#fdfdf6` RGB(253,253,246) | `air / 110.3 nm TiO2 / 62.4 nm SiO2 / 3899.5 nm Ag / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fffafa;border:1px solid #888"></span> `#fffafa` RGB(255,250,250) |
| A-4 | `air / 779.9 nm Ag / 156 nm SiO2 / 68.9375 nm TiO2 / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fdfdf6;border:1px solid #888"></span> `#fdfdf6` RGB(253,253,246) | `air / 68.9375 nm TiO2 / 156 nm SiO2 / 779.9 nm Ag / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fbf5fb;border:1px solid #888"></span> `#fbf5fb` RGB(251,245,251) |
| B-1 | `air / 62.8253 nm Ag / 64.0638 nm SiO2 / 113.236 nm TiO2 / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fcfcf1;border:1px solid #888"></span> `#fcfcf1` RGB(252,252,241) | `air / 113.236 nm TiO2 / 64.0638 nm SiO2 / 62.8253 nm Ag / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fef6f9;border:1px solid #888"></span> `#fef6f9` RGB(254,246,249) |
| B-3 | `air / 51.9933 nm Ag / 75.8335 nm SiO2 / 223.408 nm TiO2 / 44.5662 nm SiO2 / 110.864 nm TiO2 / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fef8f0;border:1px solid #888"></span> `#fef8f0` RGB(254,248,240) | `air / 110.864 nm TiO2 / 44.5662 nm SiO2 / 223.408 nm TiO2 / 75.8335 nm SiO2 / 51.9933 nm Ag / Si` | <span style="display:inline-block;width:3em;height:1.2em;background:#fef6f5;border:1px solid #888"></span> `#fef6f5` RGB(254,246,245) |

These examples do not prove that one interpretation is experimentally correct for every sample. They show that the importer interpretation matters a lot, and it explains the repeated `#fdfdf6` result.

## Constants Comparison

### Ideal legacy builder vs new defaults

The clean legacy builder `calculatendlists.py` is close to the new default constants for several materials:

| Material | Legacy ideal | New program | Comment |
|---|---:|---:|---|
| Si | wavelength-dependent complex n | same values | matches |
| TiO2 | 2.57, 2.45, 2.39, 2.35, 2.32, 2.30, 2.28 | same values | matches |
| Ag | 0.19+1.85j to 0.08+4.45j | same values | matches |
| SiO2 | constant 1.45 in `calculatendlists.py` | tabulated 1.48 to 1.47 | small difference |
| ZrO2 | constant 2.15 | tabulated 2.18 to 2.06 | small/moderate difference |
| Au | tabulated in ideal legacy | now added in new program, but values differ slightly | should be aligned if exact legacy matching is desired |

### Roughness legacy builder vs new effective-interface model

The legacy colour function imports `calculatendlists_WIProughness.py`. That file differs more:

| Material | Legacy WIP roughness | New program | Effect |
|---|---:|---:|---|
| TiO2 | 2.66, 2.55, 2.49, 2.45, 2.42, 2.40, 2.39 | 2.57, 2.45, 2.39, 2.35, 2.32, 2.30, 2.28 | significant phase shift |
| SiO2 | 1.48 to 1.47 | same | matches |
| ZrO2 | constant 2.15 | 2.18 to 2.06 | small/moderate |
| Ag | same default Ag table | same | matches |
| Au | not present in WIP material map | present in new program | legacy WIP treats Au as unknown n=1 if used |

## Roughness / Interface Model Difference

Both systems have an “effective interface layer” idea, but the equations are not equivalent.

The new program uses the standard Maxwell-Garnett effective permittivity form:

```text
eps_eff = eps_matrix *
          (eps_inclusion + 2 eps_matrix + 2 f (eps_inclusion - eps_matrix))
        / (eps_inclusion + 2 eps_matrix - f (eps_inclusion - eps_matrix))
```

The legacy WIP roughness code uses:

```text
n_eff = n_matrix *
        sqrt(((1 - f)(n_inclusion^2 + 2 n_matrix^2) + f(n_inclusion^2 - n_matrix^2))
             /((n_inclusion^2 + 2 n_matrix^2) + f(n_inclusion^2 - n_matrix^2)))
```

For an air/TiO2-like interface with `n_matrix=1`, `n_inclusion=2.5`, `f=0.5`:

```text
legacy WIP formula: n_eff ≈ 0.788
new standard formula: n_eff ≈ 1.549
```

The legacy value below 1 for an air/high-index mixture is nonphysical. This strongly suggests the WIP roughness equation is not mathematically reliable. The new implementation should not be changed to reproduce that behavior unless the goal is strict historical reproduction rather than physical correctness.

## Colour Conversion

The new program uses:

- CIE 1931 2-degree observer
- D65 illumination
- XYZ integration on the simulation wavelength grid
- `colour.XYZ_to_sRGB(XYZ / 100)`
- clipping to `[0, 1]`

The legacy program has several colour paths:

- `givecolourfrommultilayer.py` uses `tmm.color.calc_color`, takes `xyY`, and converts through `xyY_to_RGB`.
- `givecolourfrommultilayerV2withXYZ.py` takes `color_dict["rgb"]` and then applies `linear_to_srgb`.
- measured-data scripts use `colour.sd_to_XYZ` and `XYZ_to_sRGB`.

The legacy `tmm.color` path normally evaluates 360-830 nm and extends 400-700 nm constants outside the visible narrow range. The new program currently simulates 400-700 nm for normal reflectance plots and experiment cache. That is reasonable for the visible thin-film workflow, but it is not bit-for-bit identical to `tmm.color`.

## Numerical Comparison

The table below compares reflectance spectra using the same wavelength grid, 400-700 nm, and the same colour conversion after reflectance. This isolates stack/constants/model differences rather than old GUI plotting behavior.

### Legacy WIP roughness vs new effective-interface mode

| Case | Legacy WIP RGB | New effective RGB | Mean abs reflectance difference | Max abs reflectance difference |
|---|---:|---:|---:|---:|
| TiO2 / SiO2 / Ag = 80 / 120 / 40 nm | <span style="display:inline-block;width:3em;height:1.2em;background:#ffdde8;border:1px solid #888"></span> `#ffdde8` | <span style="display:inline-block;width:3em;height:1.2em;background:#f6c7e8;border:1px solid #888"></span> `#f6c7e8` | 0.0576 | 0.5267 |
| TiO2 / SiO2 = 80 / 120 nm | <span style="display:inline-block;width:3em;height:1.2em;background:#e0986b;border:1px solid #888"></span> `#e0986b` | <span style="display:inline-block;width:3em;height:1.2em;background:#dc9f5c;border:1px solid #888"></span> `#dc9f5c` | 0.0230 | 0.0576 |
| TiO2 / ZrO2 = 102 / 118 nm | <span style="display:inline-block;width:3em;height:1.2em;background:#33a36a;border:1px solid #888"></span> `#33a36a` | <span style="display:inline-block;width:3em;height:1.2em;background:#009b7b;border:1px solid #888"></span> `#009b7b` | 0.0453 | 0.0888 |
| Experiment A-2 optical order | <span style="display:inline-block;width:3em;height:1.2em;background:#fbfef8;border:1px solid #888"></span> `#fbfef8` | <span style="display:inline-block;width:3em;height:1.2em;background:#f2eefb;border:1px solid #888"></span> `#f2eefb` | 0.0506 | 0.4400 |
| Experiment B-2 optical order | <span style="display:inline-block;width:3em;height:1.2em;background:#00a6da;border:1px solid #888"></span> `#00a6da` | <span style="display:inline-block;width:3em;height:1.2em;background:#0089db;border:1px solid #888"></span> `#0089db` | 0.0732 | 0.1566 |
| Au 500 nm | <span style="display:inline-block;width:3em;height:1.2em;background:#9ea3af;border:1px solid #888"></span> `#9ea3af` | <span style="display:inline-block;width:3em;height:1.2em;background:#fae0b4;border:1px solid #888"></span> `#fae0b4` | 0.3628 | 0.6294 |

The Au mismatch is expected: the WIP legacy builder does not include Au in its material map, so it falls back to `n=1`.

### Legacy ideal vs new ideal mode

| Case | Legacy ideal RGB | New ideal RGB | Mean abs reflectance difference |
|---|---:|---:|---:|
| TiO2 / SiO2 / Ag = 80 / 120 / 40 nm | <span style="display:inline-block;width:3em;height:1.2em;background:#ffe1df;border:1px solid #888"></span> `#ffe1df` | <span style="display:inline-block;width:3em;height:1.2em;background:#ffdee2;border:1px solid #888"></span> `#ffdee2` | 0.0122 |
| TiO2 / SiO2 = 80 / 120 nm | <span style="display:inline-block;width:3em;height:1.2em;background:#ddae50;border:1px solid #888"></span> `#ddae50` | <span style="display:inline-block;width:3em;height:1.2em;background:#dca752;border:1px solid #888"></span> `#dca752` | 0.0198 |
| TiO2 / ZrO2 = 102 / 118 nm | <span style="display:inline-block;width:3em;height:1.2em;background:#009b84;border:1px solid #888"></span> `#009b84` | <span style="display:inline-block;width:3em;height:1.2em;background:#009685;border:1px solid #888"></span> `#009685` | 0.0184 |
| Experiment A-2 optical order | <span style="display:inline-block;width:3em;height:1.2em;background:#fafdfa;border:1px solid #888"></span> `#fafdfa` | <span style="display:inline-block;width:3em;height:1.2em;background:#fafefa;border:1px solid #888"></span> `#fafefa` | 0.0032 |
| Experiment B-2 optical order | <span style="display:inline-block;width:3em;height:1.2em;background:#007fdb;border:1px solid #888"></span> `#007fdb` | <span style="display:inline-block;width:3em;height:1.2em;background:#007fda;border:1px solid #888"></span> `#007fda` | 0.0034 |
| Au 500 nm | <span style="display:inline-block;width:3em;height:1.2em;background:#ffe3a1;border:1px solid #888"></span> `#ffe3a1` | <span style="display:inline-block;width:3em;height:1.2em;background:#fae6b7;border:1px solid #888"></span> `#fae6b7` | 0.0624 |

This is the more meaningful comparison for the core TMM framework. It shows the new ideal model agrees closely with the clean legacy ideal model for the common non-Au cases. Remaining differences are mainly from SiO2/ZrO2/Au constants and colour-grid details.

## Do The Simulated Colours Match?

They approximately match the clean legacy ideal TMM model, especially for stacks using TiO2, SiO2, Ag, and Si.

They do not exactly match the legacy WIP roughness colour function, and they should not be forced to, because:

- the WIP roughness model uses a questionable effective-medium equation;
- the WIP TiO2 constants differ from the clean legacy constants;
- the WIP path does not contain Au;
- the legacy colour wrapper currently fails in this Python environment;
- the legacy colour calculation is s-polarized by default, not unpolarized.

## Measured Data Fit

I also compared simulated colours against the measured reflectance-derived colours in the experiment dataset using the three constants profiles now available in the GUI:

- `current`
- `legacy_ideal`
- `legacy_wip`

For each profile, I rebuilt the experiment cache and calculated Delta E in Lab space between the measured colour and the simulated colour. The full comparison table was saved here:

```text
outputs/experiment_cache/profile_fit_comparison.csv
```

### Overall Profile Fit

| Constants profile | Measurements | Mean Delta E | Median Delta E | Best Delta E | Worst Delta E |
|---|---:|---:|---:|---:|---:|
| `legacy_wip` | 312 | 18.63 | 16.24 | 0.87 | 88.60 |
| `legacy_ideal` | 312 | 19.29 | 14.33 | 0.87 | 102.28 |
| `current` | 312 | 19.29 | 15.03 | 0.77 | 102.17 |

By mean Delta E, `legacy_wip` is slightly best overall. By median Delta E, `legacy_ideal` is slightly best. The differences are small enough that this should not be treated as a final physical conclusion; it mostly says that the current simple constants are in the right ballpark, but none of the constant profiles fully explains every measured sample.

### Which Profile Wins Most Often?

For the 312 measurement rows, the lowest Delta E profile was:

| Winning profile | Number of measurement rows |
|---|---:|
| `current` | 152 |
| `legacy_wip` | 82 |
| `legacy_ideal` | 78 |

By sample-averaged Delta E, the lowest Delta E profile was:

| Winning profile | Number of samples |
|---|---:|
| `current` | 30 |
| `legacy_wip` | 25 |
| `legacy_ideal` | 25 |

This means no single constants profile dominates. Different samples prefer different constants, which is plausible for sputtered films where density, roughness, morphology, oxidation, and island growth can change the optical constants.

### Best-Fitting Measurements

| Sample | Best profile | Delta E | Measured colour | Simulated colour | Measurement |
|---|---|---:|---|---|---|
| S-4 | `current` | 0.77 | <span style="display:inline-block;width:3em;height:1.2em;background:#82755a;border:1px solid #888"></span> `#82755a` | <span style="display:inline-block;width:3em;height:1.2em;background:#807358;border:1px solid #888"></span> `#807358` | ZrO2 36 nm on Si (S-4)_1 |
| S-19 | `current` | 0.87 | <span style="display:inline-block;width:3em;height:1.2em;background:#fefef9;border:1px solid #888"></span> `#fefef9` | <span style="display:inline-block;width:3em;height:1.2em;background:#fdfdf6;border:1px solid #888"></span> `#fdfdf6` | Ag 6min (S-19) |
| D-13 | `legacy_wip` | 0.87 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdfef4;border:1px solid #888"></span> `#fdfef4` | <span style="display:inline-block;width:3em;height:1.2em;background:#fcfcf3;border:1px solid #888"></span> `#fcfcf3` | Ag 120 nm 5 nm TiO2 on Si (D-13)_1 |
| S-4 | `current` | 0.88 | <span style="display:inline-block;width:3em;height:1.2em;background:#82765a;border:1px solid #888"></span> `#82765a` | <span style="display:inline-block;width:3em;height:1.2em;background:#807358;border:1px solid #888"></span> `#807358` | ZrO2 36 nm on Si (S-4)_2 |
| S-21 | `legacy_ideal` | 1.03 | <span style="display:inline-block;width:3em;height:1.2em;background:#8c897f;border:1px solid #888"></span> `#8c897f` | <span style="display:inline-block;width:3em;height:1.2em;background:#8f8b80;border:1px solid #888"></span> `#8f8b80` | 20;S-21 |

### Worst Remaining Mismatches

| Sample | Best available profile | Delta E | Measured colour | Simulated colour | Measurement |
|---|---|---:|---|---|---|
| B-2 | `legacy_wip` | 88.60 | <span style="display:inline-block;width:3em;height:1.2em;background:#c9c473;border:1px solid #888"></span> `#c9c473` | <span style="display:inline-block;width:3em;height:1.2em;background:#009ed9;border:1px solid #888"></span> `#009ed9` | 90;B-2 |
| B-2 | `legacy_wip` | 86.75 | <span style="display:inline-block;width:3em;height:1.2em;background:#c7c476;border:1px solid #888"></span> `#c7c476` | <span style="display:inline-block;width:3em;height:1.2em;background:#009ed9;border:1px solid #888"></span> `#009ed9` | 112;B-2 |
| B-9 | `legacy_wip` | 71.87 | <span style="display:inline-block;width:3em;height:1.2em;background:#ce7c7e;border:1px solid #888"></span> `#ce7c7e` | <span style="display:inline-block;width:3em;height:1.2em;background:#b7cc58;border:1px solid #888"></span> `#b7cc58` | 98;B-9 |
| S-6 | `current` | 66.61 | <span style="display:inline-block;width:3em;height:1.2em;background:#b07872;border:1px solid #888"></span> `#b07872` | <span style="display:inline-block;width:3em;height:1.2em;background:#b751c6;border:1px solid #888"></span> `#b751c6` | TiO2 150 nm on Ti (S-6)_1 |
| S-6 | `current` | 66.03 | <span style="display:inline-block;width:3em;height:1.2em;background:#ad7b76;border:1px solid #888"></span> `#ad7b76` | <span style="display:inline-block;width:3em;height:1.2em;background:#b751c6;border:1px solid #888"></span> `#b751c6` | TiO2 150 nm on Ti (S-6)_3 |

These poor fits are probably not just a constants-profile problem. They may indicate one or more of:

- wrong or incomplete substrate interpretation,
- rough/diffuse samples being compared to a specular TMM model,
- thickness estimate errors,
- sputtered-film constants differing strongly from bulk/literature constants,
- oxidation, porosity, island growth, or discontinuous metal films,
- measurements taken with integrating-sphere geometry while the model predicts specular reflectance.

## Is The Math Right?

Yes, the new core TMM math is right:

- layer ordering is now corrected for experimental deposition data;
- coherent TMM uses `coh_tmm`;
- unpolarized reflectance is computed as average of s and p;
- material interpolation is separated from simulation;
- colour conversion uses standard CIE 1931 / D65 / sRGB conversion;
- Lab Delta E is calculated from XYZ-derived Lab values.

The strongest math issue found is not in the new code, but in the legacy WIP roughness Maxwell-Garnett formula.

## Recommendations

1. Keep the new ideal TMM model as the reference physical model.
2. Keep the new standard Maxwell-Garnett effective-interface model, not the legacy WIP formula.
3. If exact historical reproduction is needed, add a separate `LegacyConstants` or `LegacyCompatibilityModel`, but do not mix it with the physically cleaner default model.
4. Decide which optical constants are the project standard:
   - legacy ideal constants,
   - current new constants,
   - or refractiveindex.info / measured thin-film constants.
5. For experiments, prefer fitting or measuring the actual thin-film optical constants for sputtered Ag/Au/TiO2, because bulk constants can be very wrong for rough or island-like films.
6. Use the GUI constants-profile selector to compare `current`, `legacy_ideal`, and `legacy_wip` constants deliberately.
