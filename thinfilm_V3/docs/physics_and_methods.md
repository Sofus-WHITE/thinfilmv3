# Physics and Methods Used in `thinfilm_V3`

This document explains the main physical models and colour calculations used in the thin-film simulation program.

The program is built around a modular workflow:

```text
materials -> layer stack -> optical model -> reflectance spectrum -> colour calculation -> plotting / sweeps
```

The current optical models are:

- Ideal coherent transfer-matrix method, `TMMModel`.
- Effective-interface TMM, using thin mixed layers between films.
- RMS roughness corrected TMM, `TMMWithRoughnessModel`.
- Effective-interface plus RMS roughness corrected TMM.
- Diffuse redistribution TMM for approximate specular plus diffuse rough-sample measurements.
- Effective-interface plus diffuse redistribution TMM.

The main output is a reflectance spectrum:

```text
R(lambda)
```

where `lambda` is wavelength, usually from 400-700 nm.

---

## 1. Refractive Index

Each material is described by a complex refractive index:

```text
N(lambda) = n(lambda) + i k(lambda)
```

where:

- `n` is the ordinary refractive index.
- `k` is the extinction coefficient.
- `i` is the imaginary unit.
- `lambda` is wavelength.

The real part `n` controls phase velocity and refraction. The imaginary part `k` controls absorption.

For absorbing materials, the absorption coefficient is approximately:

```text
alpha(lambda) = 4 pi k(lambda) / lambda
```

where `lambda` and `alpha` must use consistent units.

In the program, materials can be:

- constant, such as `air: n = 1`;
- tabulated, such as wavelength-dependent `Ag`, `TiO2`, `Si`, etc.;
- effective media, such as mixed roughness-interface layers.

For tabulated materials, the program interpolates `n` and `k` onto the simulation wavelength grid.

---

## 2. Thin-Film Stack Representation

A stack is defined as:

```text
incident medium / finite layers / substrate
```

Example:

```text
air / TiO2 / SiO2 / Ag / Si
```

The TMM calculation internally represents the incident medium and substrate as semi-infinite:

```text
d_list = [inf, d1, d2, d3, inf]
```

For example:

```text
air / 80 nm TiO2 / 120 nm SiO2 / 40 nm Ag / Si
```

becomes:

```text
n_list = [N_air, N_TiO2, N_SiO2, N_Ag, N_Si]
d_list = [inf, 80, 120, 40, inf]
```

The display stack shown in the GUI intentionally hides internal details such as native oxide or roughness-interface layers, unless those are directly relevant to the user.

---

## 3. Transfer-Matrix Method

The transfer-matrix method, or TMM, is a coherent wave-optics method for calculating reflection and transmission through layered media.

It assumes:

- each layer is laterally uniform;
- each interface is flat and infinite;
- each finite layer has a well-defined thickness;
- the incident light is a plane wave;
- interference between reflections is coherent.

TMM is therefore very suitable for smooth thin films where interference matters.

### 3.1 Phase Accumulation

Inside a layer, light accumulates phase:

```text
delta_j = 2 pi N_j d_j cos(theta_j) / lambda
```

where:

- `N_j` is the complex refractive index of layer `j`;
- `d_j` is the layer thickness;
- `theta_j` is the propagation angle inside that layer;
- `lambda` is the free-space wavelength.

The angle inside each layer is related to the incident angle by Snell's law:

```text
N_0 sin(theta_0) = N_j sin(theta_j)
```

Because `N_j` may be complex, the internal angle can also be complex in absorbing layers.

### 3.2 Interface Reflection

At each interface, part of the wave reflects and part transmits. The Fresnel reflection coefficients differ for s and p polarization.

For s polarization:

```text
r_s = (N_i cos(theta_i) - N_j cos(theta_j))
      / (N_i cos(theta_i) + N_j cos(theta_j))
```

For p polarization:

```text
r_p = (N_j cos(theta_i) - N_i cos(theta_j))
      / (N_j cos(theta_i) + N_i cos(theta_j))
```

TMM combines all interface reflections and layer phases into a total reflected amplitude.

The reflectance is:

```text
R = |r|^2
```

where `r` is the total complex reflection amplitude.

### 3.3 Unpolarized Reflectance

The program calculates both s and p reflectance and averages them:

```text
R_unpolarized = (R_s + R_p) / 2
```

This is appropriate when the incident light is unpolarized.

---

## 4. Prepared TMM Stack

Thickness and angle sweeps require many repeated simulations.

Rebuilding materials and interpolating refractive indices every time would be inefficient, so the program uses a prepared stack representation.

The prepared stack contains:

```text
wavelengths_nm
n_matrix
base_d_list
layer_names
layer_indices
display_layer_indices
```

The `n_matrix` has shape:

```text
number of optical layers x number of wavelengths
```

This means all wavelength-dependent refractive indices are computed once.

During a sweep, the program only changes:

- selected thickness values in `d_list`, or
- angle of incidence.

It does not rebuild the full `ThinFilmStack`, `Layer`, or `Material` objects for each point.

This prepared-stack workflow is important for:

- fast thickness sweeps;
- angle sweeps;
- future fitting;
- future optimization;
- future GUI interaction.

---

## 5. Effective-Interface Roughness Approximation

Real thin films are rarely perfectly sharp at interfaces.

A simple way to approximate intermixing or rough interfaces is to insert a thin effective-medium layer between two materials:

```text
TiO2 / mix(TiO2, SiO2) / SiO2
```

The mixed layer has a small thickness, for example:

```text
1 nm
```

and an effective refractive index calculated from the two neighboring materials.

The program uses a Maxwell-Garnett-style effective medium approximation:

```text
epsilon_eff = epsilon_m *
    (epsilon_i + 2 epsilon_m + 2 f (epsilon_i - epsilon_m))
    / (epsilon_i + 2 epsilon_m - f (epsilon_i - epsilon_m))
```

where:

- `epsilon_eff` is the effective dielectric function;
- `epsilon_m` is the matrix dielectric function;
- `epsilon_i` is the inclusion dielectric function;
- `f` is the inclusion volume fraction.

The effective refractive index is:

```text
N_eff = sqrt(epsilon_eff)
```

This approximation is useful for:

- intermixing;
- gradual interfaces;
- native transition regions;
- small roughness compared with wavelength.

It does not explicitly predict diffuse scattering.

---

## 6. Native Oxide Layer

Some substrates naturally form a thin oxide layer.

For example:

```text
Si -> SiO2
Ti -> TiO2
```

The program can include a native oxide internally, such as:

```text
2 nm SiO2 on Si
```

or:

```text
5 nm TiO2 on Ti
```

This oxide is included in the optical calculation if enabled, but it is normally hidden from the user-facing stack label to keep the GUI readable.

---

## 7. RMS Roughness Correction

The effective-interface model changes the optical stack by adding mixed layers.

The RMS roughness correction does something different: it reduces the calculated specular reflectance to account for surface/interface roughness.

The current model uses a Debye-Waller-style attenuation:

```text
R_rough(lambda) = R_TMM(lambda) * exp[-N (4 pi sigma cos(theta) / lambda)^2]
```

where:

- `R_TMM` is the coherent TMM reflectance;
- `sigma` is RMS roughness in nm;
- `theta` is the incident angle;
- `lambda` is wavelength;
- `N` is the number of main interfaces.

This correction represents loss of specular reflected intensity caused by roughness.

It is useful for:

- measured specular reflectance;
- rough films where specular signal is reduced;
- fast fitting or sweeps.

It does not calculate the full diffuse scattering pattern.

This distinction is important:

```text
specular detector:
roughness can reduce the measured reflected intensity

integrating sphere:
roughness can redirect light into other angles, but much of it may still be collected
```

For full diffuse scattering, one would need additional information such as:

- RMS roughness;
- lateral correlation length;
- detector collection angle;
- whether the measurement is specular or integrating-sphere;
- particle/island morphology.

---

## 8. Difference Between Effective Interfaces and RMS Roughness

The GUI offers several optical model choices.

### Ideal TMM

This is the clean coherent model:

```text
main layers only
no effective interface
no roughness correction
```

Use this for ideal smooth films.

### Effective Interface TMM

This inserts thin mixed layers between adjacent materials:

```text
air / mix / TiO2 / mix / SiO2 / mix / Ag / mix / Si
```

Use this when the interface is not abrupt or there is intermixing.

### RMS Roughness TMM

This keeps the main stack but attenuates the specular reflectance:

```text
R_rough = R_TMM * roughness_factor
```

Use this when the measured specular reflectance is reduced by roughness.

### Effective Interface + RMS

This uses both:

- mixed interface layers;
- specular roughness attenuation.

This can be useful, but it should be used carefully because both mechanisms represent interface imperfection in different ways.

---

## 8.1 Diffuse Redistribution Model

The diffuse redistribution model is an approximate way to compare rough samples measured with an integrating sphere.

It does not try to calculate a full bidirectional scattering distribution. Instead, it blends the normal TMM spectrum at the selected measurement angle with an angle-averaged TMM spectrum:

```text
R_seen(lambda) =
    (1 - f_scatter(lambda)) R_TMM(lambda, theta_meas)
    + f_scatter(lambda) R_angle_average(lambda)
```

where:

- `R_TMM(lambda, theta_meas)` is the coherent reflectance at the selected angle, for example 8 degrees;
- `R_angle_average(lambda)` is an averaged reflectance over many angles, currently used as a practical proxy for redistributed/scattered light;
- `f_scatter(lambda)` is the fraction of reflected light treated as redistributed by roughness.

The GUI controls are:

- `RMS roughness nm`: the roughness height scale. This can be connected to AFM RMS roughness, but the model is still empirical.
- `Scatter scale`: converts RMS roughness into scattering strength. Larger values make the same RMS roughness produce more redistribution.
- `Scatter exponent`: controls how strongly scattering increases with roughness-to-wavelength ratio. A value near 2 follows the common idea that many roughness effects scale roughly with `(sigma / lambda)^2`.
- `Max scatter fraction`: caps the redistributed fraction so the model cannot turn the entire spectrum into the angle-averaged proxy.

Conceptually:

```text
smooth sample:
f_scatter near 0
-> mostly ordinary TMM at the selected angle

rough integrating-sphere sample:
f_scatter larger
-> mixture of selected-angle TMM and angle-averaged contribution
```

This model is most useful for:

- rough samples;
- integrating-sphere measurements;
- samples that still show thin-film colour, but with reduced angular purity or more matte appearance.

It is not a full Mie, Rayleigh-Rice, PSD, RCWA, or Monte Carlo model. It is a controlled intermediate model that keeps the coherent TMM film colour while adding a simple collection-geometry approximation.

---

## 9. Why Not Monte Carlo First?

Monte Carlo ray tracing is useful for diffuse scattering and multiple scattering.

However, ordinary Monte Carlo is not naturally coherent. It does not automatically reproduce thin-film interference fringes.

For thin films of about 50-200 nm, interference is usually important, so TMM should remain the primary model.

A more complete scattering workflow would be:

```text
TMM -> coherent specular reflection
roughness model -> interface mixing or specular loss
scattering model -> diffuse redistribution
detector model -> collected signal
```

The current diffuse redistribution model is a practical intermediate step before introducing a full diffuse scattering model.

---

## 9.1 Future Diffuse Scattering Models

Full microstructure-resolved diffuse scattering and Mie scattering are not currently active features in the program.

They are future development targets for rough, hazy, granular, porous, island-like, or particulate samples.

A future diffuse model should be designed around the experimental measurement geometry:

- specular detector or integrating sphere;
- detector angular acceptance;
- RMS roughness;
- lateral roughness correlation length;
- particle or island size distribution;
- film porosity or volume fraction;
- whether scattering happens mainly at interfaces, inside layers, or both.

Mie theory may be useful for particle-like or island-like scatterers, but it should be introduced with a careful parameter workflow and validation against measured diffuse reflectance data.

For now, roughness is handled by:

- effective-interface layers for intermixing and gradual optical transitions;
- RMS specular attenuation for specular-loss comparisons;
- diffuse redistribution for approximate integrating-sphere rough-sample comparisons.

---

## 10. Thickness Sweeps

A thickness sweep varies one or two selected layer thicknesses and calculates the predicted colour.

### 1D Thickness Sweep

One layer varies:

```text
air / x nm TiO2 / 120 nm SiO2 / 40 nm Ag / Si
```

The output is a colour strip:

```text
x-axis = thickness
colour = predicted perceived colour
```

### 2D Thickness Sweep

Two layers vary:

```text
air / x nm TiO2 / y nm SiO2 / 40 nm Ag / Si
```

The output is a colour map:

```text
x-axis = thickness of layer 1
y-axis = thickness of layer 2
pixel colour = predicted perceived colour
```

By default, the program stores RGB values only. It does not store every reflectance spectrum unless explicitly requested, because that can use significant memory.

---

## 11. Angle Sweeps

An angle sweep varies incident angle:

```text
theta = 0 to 80 degrees
```

At each angle, the program calculates:

```text
R(lambda, theta)
```

then converts the reflectance spectrum into a perceived colour.

The output is a colour strip:

```text
x-axis = angle of incidence
colour = predicted perceived colour
```

Angle sweeps are useful because thin-film colours often shift strongly with viewing angle.

---

## 12. Colour Theory

The program converts reflectance spectra into approximate perceived colours.

The basic idea is:

```text
reflectance spectrum
-> illuminated reflected spectrum
-> CIE XYZ
-> sRGB
```

### 12.1 Reflectance Spectrum

The optical model returns:

```text
R(lambda)
```

This is the fraction of incident light reflected at each wavelength.

### 12.2 Illumination

Colour depends on illumination. The program uses D65 illumination by default.

D65 approximates average daylight and is commonly used in colour science.

The reflected spectral power is:

```text
S_reflected(lambda) = R(lambda) * S_D65(lambda)
```

where:

- `R(lambda)` is reflectance;
- `S_D65(lambda)` is the D65 illuminant spectrum.

### 12.3 CIE 1931 Colour Matching Functions

Human colour perception is approximated using the CIE 1931 2-degree standard observer.

The reflected spectrum is integrated against colour matching functions:

```text
X = integral S_reflected(lambda) x_bar(lambda) d lambda
Y = integral S_reflected(lambda) y_bar(lambda) d lambda
Z = integral S_reflected(lambda) z_bar(lambda) d lambda
```

The result is a CIE XYZ colour.

`Y` roughly corresponds to luminance.

### 12.4 XYZ to sRGB

Most displays use sRGB, so the program converts:

```text
XYZ -> linear RGB -> gamma-corrected sRGB
```

The final RGB values are clipped to:

```text
0 <= R, G, B <= 1
```

This clipping is necessary because some physical spectra produce colours outside the displayable sRGB gamut.

### 12.5 Important Colour Limitations

The displayed colour is an approximation because:

- monitors differ;
- ambient lighting differs;
- sRGB cannot display every real colour;
- surface gloss and diffuse scattering are not fully represented;
- the calculation assumes a standard observer and D65 illumination.

Still, it is very useful for comparing trends between stacks.

---

## 13. Reflectance Plot Colour Fill

The reflectance plot is filled with approximate visible wavelength colours under the curve.

This is a visualization aid.

It helps show which wavelength regions contribute to the reflected spectrum:

- violet/blue around 400-480 nm;
- green around 500-560 nm;
- yellow/orange around 570-620 nm;
- red above about 620 nm.

The fill colour is not itself the perceived colour. The perceived colour comes from integrating the whole reflected spectrum under D65 illumination.

---

## 14. Which Model Should Be Used?

For clean, smooth films:

```text
Ideal TMM
```

For intermixing or gradual interfaces:

```text
Effective interface TMM
```

For specular measurements on rough films:

```text
RMS roughness TMM
```

For films with both gradual interfaces and reduced specular reflection:

```text
Effective interface + RMS
```

For rough samples measured with an integrating sphere:

```text
Effective interface + diffuse redistribution
```

This is usually the best first approximation when the measured signal is specular plus scattered reflected light, rather than only the specular beam.

---

## 15. Current Physical Limitations

The current program does not yet fully model:

- first-principles diffuse angular scattering;
- Mie scattering from particles or islands;
- roughness correlation length;
- polarization-resolved measurements beyond internal s/p averaging;
- spectrometer collection geometry;
- measured-data fitting for all parameters at once;
- microstructure-dependent optical constants.

These are future extension points.

The present framework is designed so these models can be added without rewriting the core TMM simulation.

---

## 16. Experiment Fitting Strategy

The experiment tools separate three different questions:

```text
Which refractive-index data should each material use?
What thicknesses best match the measured colour?
What roughness/scattering parameters best describe rough groups?
```

These should not all be fitted together at first, because many combinations can produce similar colours.

### Smooth Samples

Smooth samples on polished wafers are closest to the assumptions of ideal coherent TMM.

Recommended use:

```text
Substrate: Si or Ti
Surface: smooth
Measurement: specular or integrating sphere
Optical model: Ideal TMM or Effective interface TMM
```

Smooth single-layer samples are the best starting point for choosing refractive-index constants, because roughness and diffuse scattering should be small.

### Rough Samples

Rough samples should usually be treated separately from smooth samples.

Recommended use:

```text
Surface: rough
Measurement: integrating sphere, if available
Optical model: Effective interface + diffuse redistribution
```

For these samples, the refractive index may still come from smooth-film fits or from group-specific candidate fits, but the roughness/scattering parameters should be fitted as a group.

### Group-Specific Constants

The program can fit refractiveindex.info candidate records separately for groups such as:

```text
smooth Si
rough Si
rough Ti
```

This avoids mixing samples that likely have different substrate effects, roughness, microstructure, or measurement geometry.

### Thickness Optimization

Thickness optimization changes estimated layer thicknesses within a sputter-rate error range.

The error is percentage based rather than a fixed number of nanometres, because sputter-rate uncertainty scales with deposition time:

```text
longer deposition time -> larger absolute thickness uncertainty
```

For Ag layers at about 50 nm or thicker, the program can avoid optimizing Ag thickness because the layer is already optically very opaque. In that regime, changing Ag thickness often has little effect compared with changing transparent layer thicknesses.
