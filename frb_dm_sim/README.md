# FRB DM-distribution Monte Carlo: merged version

One pipeline, one parameter set, nine surveys.

```
params.py          shared tunables (SimParams): flags, luminosity function, SNR_threshold, fscale, ...
core.py            the shared z-loop / rejection sampling / rate weighting (run_survey)
beams.py           the only physics that really differs: gain + sampling geometry per telescope type
telescopes.py      per-telescope numbers (D, eta, Trec, BW, nu_cen, n_comp, n_chan, csv paths)
tsky.py            one frequency-dependent Tsky loader for every csv layout
cosmo_table.py     cosmology tables computed once and shared by all surveys (big speed-up)
run_all.py         command line:  python run_all.py pks fast --set SNR_threshold=9 --tag _snr9
DM_comparison_mc_all.ipynb   driver notebook: tune once, run everything, quick plots
tests/             compare_with_notebooks.py: old notebook vs new package, same inputs, same seed
```

Install: put the folder next to your notebooks (or anywhere on `PYTHONPATH`); needs numpy, scipy, pandas, astropy.
Data paths default to `/home/thsu/FRB_population`; override with `FRB_DATA_DIR` / `FRB_OUT_DIR`.

## Where each old-notebook difference went

| survey | beam model | what is specific |
|---|---|---|
| askap_fe | `GaussianDishBeam("single")` | one dish gain |
| askap_ics, meerkat_ib | `GaussianDishBeam("sqrt_n", n_dish)` | gain x sqrt(n_dish) |
| pks | `MultiBeamListDish` | on-axis gain drawn per FRB from the 13-beam list |
| fast, fast_plane | `FastMultiBeamDish` | beam drawn per FRB from FAST_beam_gain.csv; runs on the csv frequencies (n_comp = number of Gain_ columns) |
| dsa110 | `TiedRectBeam(primary="dish")` | dish primary x 2 tied beams |
| apertif | `TiedRectBeam(primary="apertif_empirical")` | empirical primary FWHM(nu) x 2 tied beams, area-fraction correction |
| meerkat_cb | `TiedTriangleBeam` | dish primary x 3 tied beams, tiling-circle area fraction |

Adding a survey = one entry in `telescopes.py`. A new geometry = one new class in `beams.py` with
`frequency_grid / prepare / draw / gain`.

## Reproducibility
`tests/compare_with_notebooks.py <key>` executes the original notebook and the new package on identical (synthetic)
csv inputs with seed 42 and compares the saved `.npy` arrays. On the real csv files run the same idea: run the new code
with `--tag _new`, then `np.allclose(np.load(old), np.load(new))`.
The random-number call order is kept identical to the notebooks, so results should match to floating-point precision.

## Things worth checking (kept as in the notebooks, not changed)
* `tsamp` units: the sampling width is `tsamp * 1e-3` s. DSA-110 is entered in ms (262.144e-3), but the others look like
  seconds (e.g. Parkes 0.064e-3, FAST 196.608e-6, Apertif 0.08192e-3), which makes their sampling term ~1000x too small.
  Only matters when `USE_SMEARING=True`.
* `fast_plane.ipynb` was byte-identical to `fast.ipynb`: set the plane-specific inputs in `telescopes.py`.
* Constant `L_BREAK_SCH = 8.952e15` and the 10 000-draw cap per z-bin silently truncate the high-z tail; both are now parameters.
