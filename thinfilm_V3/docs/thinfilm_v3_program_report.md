# ThinFilm V3 Program Report

## Purpose

ThinFilm V3 is a desktop program for designing, simulating, comparing, and fitting optical thin-film stacks. It was built to replace the older thin-film program with a cleaner workflow focused on reflectance, colour prediction, experimental comparison, and physically meaningful optimization. The main goal is to connect deposition data, measured reflectance spectra, optical constants, and transfer-matrix simulations so that future sputtering experiments can be predicted more accurately.

The program uses the sample lists and reflectance measurements from the Reflectivity data folder. These files describe each sample, its deposited layers, estimated thicknesses, substrate, surface class, measurement type, and linked measured spectra. ThinFilm V3 turns those records into a searchable experiment database. It can then compare the measured sample colour and reflectance spectrum with simulated thin-film models.

## Physical Model

The core physics is based on optical thin-film interference. A stack is represented as air, one or more deposited layers, optional native substrate oxide, and a substrate such as Si, Ti, or another substrate group. For each wavelength in the visible range, the program calculates reflected intensity using a transfer matrix method. The model uses complex refractive indices, where the real part controls phase velocity and interference, and the imaginary part controls absorption.

The program supports several optical model levels. The simplest is standard TMM, which treats each film as a smooth, uniform layer with sharp interfaces. More advanced options include effective interface layers, RMS roughness corrections, and diffuse redistribution models. The effective interface model approximates rough or mixed boundaries by inserting a thin mixed layer between materials. The diffuse redistribution model accounts for loss of specular reflectance due to scattering, controlled by RMS roughness, scatter scale, wavelength exponent, and maximum scatter fraction. These roughness models are important because many measured samples have the correct spectral shape in simulation but differ in absolute reflectance intensity.

The simulated reflectance is converted into perceived colour under D65 illumination. The program calculates CIE XYZ, sRGB colour, CIE 1931 xy position, and colour difference. Colour difference can be evaluated as standard CIELAB Delta E or CIEDE2000, which is usually more perceptually accurate. These metrics make it possible to compare fitted models numerically while still checking the visual colour result.

## Main Program Tabs

The Reflectance tab shows the predicted spectrum and colour for the currently selected layer stack. It also estimates sputtering time from the latest sputter-rate tables, so a planned stack can be translated into a practical deposition recipe. The Sweep tab calculates one-dimensional and two-dimensional thickness sweeps, plus angular sweeps, to show how colour changes as layer thickness or angle of incidence changes.

The Experiments tab loads measured sample data and compares it with the current optical model. It displays measured and simulated reflectance curves, colour swatches, CIE diagrams, and sample metadata. The tab supports filtering by series, substrate, surface class, measurement kind, and composition, so one can focus on a specific family of samples such as smooth Si with TiO2/Ag.

The Fit & Optimize tab contains the main fitting controls. It can fit individual layer thicknesses for a selected sample, run overnight cached thickness optimization for many samples, fit grouped sputter-rate corrections, benchmark constants and models, and fit roughness parameters for selected groups. The caching is important because thickness fits can be expensive; once a fit has been calculated, later plots can reuse the result.

The Plots tab provides visual comparison maps. It overlays measured sample colours onto simulated sweep maps, either before thickness fitting or after cached thickness fitting. The map dropdown now separates substrate and surface classes, for example smooth Si and rough Si, so the user can inspect different physical sample groups separately. It also includes Delta E distribution plots and fit-impact plots that show how each fitting stage changes the error.

The Configuration Fit tab is the newest workflow. It allows one configuration, such as TiO2/Ag on smooth Si, to be selected and optimized as a bounded staged search. It compares available refractive-index profiles, optical models, and selected roughness/model parameter variants. It can then run individual thickness optimization for that configuration. The final result is saved and plotted as a report figure showing the sweep map, measured colours, Delta E reduction through the workflow, and the relative impact of constants/model candidates.

## Fitting Strategy

The best fitting approach is staged rather than fitting every parameter freely at once. This is important because thin-film models have correlated parameters. A wrong refractive index can sometimes be hidden by a wrong thickness, and roughness can mimic a reflectance-scale change. If every parameter is allowed to float simultaneously, the fit may become numerically good but physically meaningless.

The recommended workflow is therefore:

1. Fit or select refractive-index constants first, mainly using single-layer samples.
2. Benchmark optical models and roughness settings for the relevant substrate and surface group.
3. Fit grouped sputter-rate corrections where samples share deposition conditions.
4. Fit individual thicknesses only as the final diagnostic and refinement.
5. Use plots and Delta E summaries to decide which step actually improved the model.

For refractive index, the program supports two approaches. The simple single-film fitter estimates constant n or n,k values from single-layer films, which is useful for diagnostics but too crude for strongly dispersive materials such as Ag. The better route is refractiveindex.info candidate fitting, where complete wavelength-dependent datasets are tested against the measured samples. This is especially important for Ag because a poor Ag dataset can make thick silver appear yellowish or otherwise non-white in simulation.

## Outputs and Caching

ThinFilm V3 writes results into the outputs folder. Experiment comparisons are saved as cache files and CSV summaries. Thickness optimization caches store trial results and best-fit metadata so they can be reused without recalculating. Sputter-rate fits, refractive-index candidate fits, roughness fits, model benchmarks, and configuration-fit reports are also saved. This makes the program suitable for overnight calculations and later review.

The Configuration Fit workflow saves a folder for each run containing candidate tables, stage summaries, a JSON summary, and a PNG report. These files document which constants profile, optical model, and fitting stage gave the best result for the selected configuration.

## Future Machine Learning

Machine learning could be useful, but it should be added after the physical workflow is stable. A good first ML step would not be a black-box predictor of everything. Instead, the saved fit tables could be used for regression, feature importance, or PCA to identify which parameters most strongly control Delta E. For example, PCA could reveal whether error is mainly connected to material choice, Ag thickness, TiO2 thickness, substrate class, roughness, or measurement type.

A future model could rank which physical correction is most significant for a configuration. It could also suggest which experiment should be made next to reduce uncertainty. However, refractive-index fitting should remain constrained by physical dispersion and known material datasets, rather than allowing arbitrary wavelength-by-wavelength values without regularization.

## Conclusion

ThinFilm V3 is a physics-focused analysis and design tool for sputtered optical thin films. It combines measured reflectance data, sputter-rate information, optical constants, transfer-matrix modelling, colour science, and cached optimization. Its main strength is that it makes the modelling process visible: the user can see spectra, colours, CIE positions, sweep maps, Delta E distributions, and the effect of each fitting stage. This makes it possible to improve prediction accuracy while still keeping the model tied to physically meaningful parameters.
