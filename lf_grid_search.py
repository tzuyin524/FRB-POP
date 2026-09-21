"""
Grid search over Schechter luminosity-function parameters
(L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH) to find combinations that give:

    n_frb(ASKAP fly's-eye "d0") / n_frb(FAST "25d0")  ~ 10
    n_frb(Parkes "5d0")         / n_frb(FAST "25d0")  ~ 1

This reuses the exact physics from DM_comparison_mc_fluence_DM.ipynb
(cells 0-10: telescope params, beam gain, Schechter sampler, per-z-bin
Monte Carlo detection loop, SFR-weighted volume -> n_total_population),
but:
  1. wraps it in a function of the 4 LF parameters,
  2. vectorizes the inner rejection-sampling loop (batches of trials at
     once instead of one Python-level sample at a time) so a grid of
     parameter combinations finishes in a reasonable time,
  3. lets you trade accuracy vs. speed via Z_STEP / TARGET_DETECTIONS /
     MAX_ITERATIONS / BATCH_SIZE.

Run with e.g.:  python3 lf_grid_search.py
Tune the CONFIG block below to widen/narrow the grid or speed it up.
"""

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
try:
    from scipy.integrate import cumtrapz
except ImportError:  # scipy >= 1.14 renamed this
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d

# ============================================================================
# CONFIG — tune these first
# ============================================================================

# --- Parameter grid to test (edit ranges / resolution here) ---------------
L_STAR_GRID     = np.logspace(13, 17, 5)      # Jy kpc^2   (5 values)
ALPHA_SCH_GRID  = np.linspace(-2.5, -0.5, 5)  # faint-end slope (5 values)
L_MIN_SCH_GRID  = np.logspace(9, 11, 3)       # Jy kpc^2   (3 values)
L_MAX_SCH_GRID  = np.logspace(15, 17, 3)      # Jy kpc^2   (3 values)
# -> 5*5*3*3 = 225 combinations by default. Shrink these first if it's slow.

# --- Speed vs. accuracy knobs ----------------------------------------------
Z_MIN, Z_MAX = 0.01, 5.0
Z_STEP       = 0.05      # notebook default is 0.01 (10x more z-bins, 10x slower)
TARGET_DETECTIONS = 5    # notebook default is 10
MAX_ITERATIONS    = 200_000
BATCH_SIZE        = 20_000   # vectorized trial batch size per rejection-sampling round

N_WORKERS = 1   # set >1 to parallelize across grid combos (uses multiprocessing)

OUTPUT_CSV = "lf_grid_results.csv"

# --- Targets you're aiming for ----------------------------------------------
TARGET_ASKAP_OVER_FAST = 10.0
TARGET_PKS_OVER_FAST = 1.0

# ============================================================================
# FIXED PHYSICS / TELESCOPE SETUP (copied from the notebook, unchanged)
# ============================================================================

TELESCOPE_PARAMS = {
    "d0": {   # ASKAP fly's-eye, 12 m
        "D": 12, "eta": 0.6, "Trec": 22, "Tsky": 3, "BW": 300e6,
        "nu_cen": 1.4e9, "n_comp": 16, "n_chan": 1024, "tsamp": 1.0e-3,
    },
    "5d0": {  # Parkes, 64 m
        "D": 5 * 12, "eta": 0.6, "Trec": 22, "Tsky": 3, "BW": 300e6,
        "nu_cen": 1.4e9, "n_comp": 16, "n_chan": 1024, "tsamp": 1.0e-3,
    },
    "25d0": {  # FAST, 300 m
        "D": 25 * 12, "eta": 0.6, "Trec": 22, "Tsky": 3, "BW": 300e6,
        "nu_cen": 1.4e9, "n_comp": 16, "n_chan": 1024, "tsamp": 1.0e-3,
    },
}
TEL_LABELS = {"d0": "ASKAP (12m)", "5d0": "Parkes (64m)", "25d0": "FAST (300m)"}

k = 1.38e-23
c = 3.0e8
Np = 2
SNR_threshold = 10
t_obs = 0.001
fscale = 0.2e-7
alpha_intrinsic = -1.5

# Effect flags — matches the notebook's "on" configuration for the DM/fluence
# comparison (cosmology + SFR on, smearing/spectral-index/beam off).
USE_COSMOLOGY = True
USE_SFR = True
USE_SMEARING = False
USE_SPECTRAL_INDEX = False
USE_BEAM = False

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
H0_km_s_Mpc = 70.0
DH_Mpc = 299792.458 / H0_km_s_Mpc

Z_ARRAY = np.arange(Z_MIN, Z_MAX, Z_STEP)


def schechter_function(L, L_star, alpha, phi_star=1.0):
    x = L / L_star
    return phi_star * x ** alpha * np.exp(-x)


def create_truncated_schechter_sampler(L_min, L_max, L_star, alpha):
    """Inverse-CDF sampler for the Schechter function truncated to [L_min, L_max]."""
    L_grid = np.logspace(np.log10(L_min), np.log10(L_max), 2000)
    phi_grid = schechter_function(L_grid, L_star, alpha)
    phi_grid = np.nan_to_num(phi_grid, nan=0.0, posinf=0.0, neginf=0.0)
    cdf = cumtrapz(phi_grid, L_grid, initial=0)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        # Degenerate LF over this range (extreme alpha/L_star combo underflowed
        # to all-zero) -- fall back to a point mass at L_min rather than NaN.
        def degenerate_sampler(u):
            return np.full_like(np.atleast_1d(u), L_min, dtype=float)
        return degenerate_sampler
    cdf = cdf / cdf[-1]
    inv_cdf = interp1d(cdf, L_grid, kind="linear",
                        bounds_error=False, fill_value=(L_min, L_max))
    return inv_cdf


def gaussian_beam_gain(nu, G0, theta, D, c):
    coeff = (theta ** 2 * D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c ** 2)
    return G0 * np.exp(-coeff * nu ** 2)


def simulate_detections_vectorized(dl, z_factor, wi_z, w_obs_factor,
                                    theta_max, theta_hp, D, G0, nu_array, nu0,
                                    nu_scaling, n_comp, S_SNR_CONST,
                                    inv_cdf_sampler,
                                    target_detections, max_iterations, batch_size):
    """
    Vectorized re-implementation of the notebook's per-z rejection-sampling
    while-loop: draws batches of trial FRBs at once, computes SNR for all of
    them together, and stops once target_detections are found (or the
    iteration budget runs out). Same physics/equations as the notebook,
    just batched instead of one sample per Python-loop iteration.
    """
    _inner_coeff = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c ** 2)
    n_generated = 0
    n_detected = 0

    while n_detected < target_detections and n_generated < max_iterations:
        n_try = min(batch_size, max_iterations - n_generated)

        theta_frb = theta_max * np.sqrt(np.random.random(n_try))
        theta_ok = theta_frb <= 4.0 * theta_hp

        lum = inv_cdf_sampler(np.random.uniform(0, 1, n_try))
        S0_frb = (lum / (dl ** 2 * z_factor)) * (wi_z / w_obs_factor)
        S_nu = S0_frb[:, None] * nu_scaling[None, :]

        if USE_BEAM:
            G_nu = G0 * np.exp(-(_inner_coeff * theta_frb[:, None] ** 2) * nu_array[None, :] ** 2)
        else:
            G_nu = np.full((n_try, nu_array.size), G0)

        snr_spectrum = S_nu * G_nu * S_SNR_CONST
        snr_coherent = np.sum(snr_spectrum, axis=1) / np.sqrt(n_comp)

        detected = theta_ok & (snr_coherent > SNR_threshold)
        n_detected += int(detected.sum())
        n_generated += n_try

    detection_rate = n_detected / n_generated if n_generated > 0 else 0.0
    return n_detected, n_generated, detection_rate


def run_telescope(tel_key, L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH):
    """Returns total n_frb (summed n_total_population over all z-bins) for one telescope."""
    params = TELESCOPE_PARAMS[tel_key]
    D = params["D"]; eta = params["eta"]; Trec = params["Trec"]; Tsky = params["Tsky"]
    BW = params["BW"]; n_comp = params["n_comp"]; BW_comp = BW / n_comp
    nu_cen = params["nu_cen"]; nu_min = nu_cen - BW / 2; nu_max = nu_cen + BW / 2
    nu_array = np.linspace(nu_min, nu_max, n_comp + 1)
    nu0 = nu_cen

    G0 = np.pi * D * D / 4 * eta / 2.0 / k * 1e-26
    theta_max = 2 * 1.22 * c / D / nu_min
    fwhm_rad = 1.22 * (c / nu_min) / D
    fwhm_deg = fwhm_rad * (180 / np.pi)
    r_deg = 2.0 * fwhm_deg
    area_tel = np.pi * r_deg ** 2

    np.random.seed(42)

    # --- L_min(z) ---
    L_min_by_z = {}
    for z in Z_ARRAY:
        dl = (cosmo.luminosity_distance(z).value * 1000.0) if USE_COSMOLOGY else (DH_Mpc * z * 1000.0)
        G_max = gaussian_beam_gain(nu0, G0, 0, D, c)
        S_min = SNR_threshold * (Trec + Tsky) / (G_max * np.sqrt(BW * t_obs * Np))
        z_factor = (1 + z) ** (-1 - alpha_intrinsic) if USE_SPECTRAL_INDEX else 1.0
        L_min = S_min * (dl ** 2) * z_factor
        L_min_by_z[z] = np.clip(L_min, L_MIN_SCH, L_MAX_SCH)

    sampler_cache = {}
    S_SNR_CONST = np.sqrt(BW_comp * t_obs * Np) / (Trec + Tsky)
    nu_scaling = (nu_array / nu0) ** alpha_intrinsic if USE_SPECTRAL_INDEX else np.ones_like(nu_array)
    _inner_coeff = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c ** 2)

    total_n_frb = 0.0

    for z in Z_ARRAY:
        zmin, zmax = z - 0.5 * Z_STEP, z + 0.5 * Z_STEP
        L_min_z = L_min_by_z[z]

        if L_min_z >= L_MAX_SCH:
            break

        if L_min_z not in sampler_cache:
            sampler_cache[L_min_z] = create_truncated_schechter_sampler(
                L_min_z, L_MAX_SCH, L_STAR, ALPHA_SCH
            )
        inv_cdf_sampler = sampler_cache[L_min_z]

        wi_z = 0.001 * (1 + z)
        w_obs_factor = wi_z  # USE_SMEARING is False -> other width terms are 0

        dl = (cosmo.luminosity_distance(z).value * 1000.0) if USE_COSMOLOGY else (DH_Mpc * z * 1000.0)
        z_factor = (1 + z) ** (-1 - alpha_intrinsic) if USE_SPECTRAL_INDEX else 1.0
        theta_hp = np.sqrt(np.log(2) / (_inner_coeff * nu0 ** 2))

        n_detected, n_generated, detection_rate = simulate_detections_vectorized(
            dl, z_factor, wi_z, w_obs_factor, theta_max, theta_hp, D, G0,
            nu_array, nu0, nu_scaling, n_comp, S_SNR_CONST, inv_cdf_sampler,
            TARGET_DETECTIONS, MAX_ITERATIONS, BATCH_SIZE,
        )

        if n_detected < TARGET_DETECTIONS:
            continue  # couldn't hit target within the iteration budget at this z

        L_grid_full = np.logspace(np.log10(L_MIN_SCH), np.log10(L_MAX_SCH), 5000)
        phi_grid_full = schechter_function(L_grid_full, L_STAR, ALPHA_SCH)
        phi_grid_full = np.nan_to_num(phi_grid_full, nan=0.0, posinf=0.0, neginf=0.0)
        cdf_full = cumtrapz(phi_grid_full, L_grid_full, initial=0)
        if not np.isfinite(cdf_full[-1]) or cdf_full[-1] <= 0:
            # Degenerate LF -- no well-defined N_ratio, treat as no contribution
            # from this z-bin rather than letting NaN poison the running total.
            continue
        cdf_full /= cdf_full[-1]
        cdf_interp = interp1d(L_grid_full, cdf_full, kind="linear",
                               bounds_error=False, fill_value=(0.0, 1.0))
        N_ratio = 1.0 - cdf_interp(L_min_z)

        if USE_COSMOLOGY:
            dt = (cosmo.age(zmin).value - cosmo.age(zmax).value) * 1e9
            dvol = (cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value) * (
                area_tel / (4.0 * np.pi * ((180.0 / np.pi) ** 2))
            )
        else:
            r_Mpc = DH_Mpc * z
            dr_Mpc = DH_Mpc * Z_STEP
            area_sr = area_tel * (np.pi / 180.0) ** 2
            dvol = (r_Mpc ** 2) * dr_Mpc * area_sr
            dr_m = dr_Mpc * 3.0857e22
            dt = dr_m / c / (3600.0 * 24.0 * 365.0)

        psi = 0.015 * ((1 + z) ** 2.7) / (1 + ((1 + z) / 2.9) ** 5.6) if USE_SFR else 0.015

        nfrb_sfr = psi * dvol * dt * fscale
        n_total_population = detection_rate * N_ratio * nfrb_sfr
        total_n_frb += float(n_total_population)

    return total_n_frb


def evaluate_combo(params_tuple):
    L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH = params_tuple
    if L_MIN_SCH >= L_MAX_SCH:
        return None
    n = {}
    for tel_key in ["d0", "5d0", "25d0"]:
        n[tel_key] = run_telescope(tel_key, L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH)

    n_fast = n["25d0"]
    if n_fast <= 0:
        ratio_askap_fast = np.nan
        ratio_pks_fast = np.nan
    else:
        ratio_askap_fast = n["d0"] / n_fast
        ratio_pks_fast = n["5d0"] / n_fast

    # score: distance in log-ratio space from the two targets (0 = perfect)
    if np.isfinite(ratio_askap_fast) and ratio_askap_fast > 0 and np.isfinite(ratio_pks_fast) and ratio_pks_fast > 0:
        score = (
            (np.log10(ratio_askap_fast) - np.log10(TARGET_ASKAP_OVER_FAST)) ** 2
            + (np.log10(ratio_pks_fast) - np.log10(TARGET_PKS_OVER_FAST)) ** 2
        )
    else:
        score = np.inf

    return {
        "L_STAR": L_STAR,
        "ALPHA_SCH": ALPHA_SCH,
        "L_MIN_SCH": L_MIN_SCH,
        "L_MAX_SCH": L_MAX_SCH,
        "n_askap": n["d0"],
        "n_pks": n["5d0"],
        "n_fast": n["25d0"],
        "ratio_askap_fast": ratio_askap_fast,
        "ratio_pks_fast": ratio_pks_fast,
        "score": score,
    }


def main():
    combos = list(itertools.product(L_STAR_GRID, ALPHA_SCH_GRID, L_MIN_SCH_GRID, L_MAX_SCH_GRID))
    print(f"Total combinations to test: {len(combos)}")
    print(f"z-bins per telescope: {len(Z_ARRAY)} (Z_STEP={Z_STEP})")

    t0 = time.time()
    rows = []

    if N_WORKERS > 1:
        from multiprocessing import Pool
        with Pool(N_WORKERS) as pool:
            for i, result in enumerate(pool.imap_unordered(evaluate_combo, combos)):
                if result is not None:
                    rows.append(result)
                if (i + 1) % max(1, len(combos) // 20) == 0:
                    elapsed = time.time() - t0
                    print(f"  {i+1}/{len(combos)} done ({elapsed:.0f}s elapsed)")
    else:
        for i, combo in enumerate(combos):
            result = evaluate_combo(combo)
            if result is not None:
                rows.append(result)
            if (i + 1) % max(1, len(combos) // 20) == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(combos)} done ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rows).sort_values("score")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone in {time.time()-t0:.0f}s. Saved {len(df)} rows to {OUTPUT_CSV}")
    print(f"\nTop 10 combinations closest to ratio_askap_fast~{TARGET_ASKAP_OVER_FAST}, ratio_pks_fast~{TARGET_PKS_OVER_FAST}:\n")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df.head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    main()
