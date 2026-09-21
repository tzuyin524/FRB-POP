"""
DM_grid_scan.py
================
Refactor of DM_comparison_mc_fluence_DM.ipynb (Cells 10 + 11 only -- the
3-telescope DM-population Monte Carlo and the DM histogram plot).

Instead of running once with a single (L_STAR, ALPHA_SCH, L_MIN_SCH,
L_MAX_SCH), this script loops over a parameter grid for the Schechter
luminosity function and saves one DM-distribution PNG per valid
combination into ./DM_ratio/.

Notes / design choices carried over from the notebook (unchanged physics):
  - Same telescope definitions (d0 / 5d0 / 25d0), same detection loop,
    same rejection-sampling logic, same SFR-weighted population estimate.
  - The fluence-vs-DM scatter plot (Cell 12) is NOT reproduced here --
    you only asked for the DM distribution plot. Say the word if you want
    that saved per-combo too (it roughly doubles runtime per combo).
  - One change from the notebook: the hardcoded early-break threshold
    `L_min_z > 8.952e15` is now `L_min_z > L_MAX_SCH` (i.e. tied to the
    grid value being tested, not a hardcoded constant), since otherwise
    L_MAX_SCH_GRID values above ~9e15 would never actually change
    behavior. If you want the literal notebook behavior back, hardcode
    it again in `simulate_population`.

IMPORTANT -- runtime:
  Your example grid is 5 x 5 x 3 x 3 = 225 combinations. Each combination
  reruns the full detection loop for 3 telescopes across ~500 redshift
  bins. That is roughly 225x the cost of running the notebook once.
  Set MAX_COMBOS below to a small number first to sanity-check timing
  before committing to the full grid.
"""

from pathlib import Path
import itertools
import time

import numpy as np
try:
    from scipy.integrate import cumtrapz
except ImportError:  # scipy >= 1.14 removed cumtrapz in favor of cumulative_trapezoid
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows too
import matplotlib.pyplot as plt

# ============================================================================
# OUTPUT
# ============================================================================
OUTDIR = Path('/home/thsu/FRB_population/DM_ratio')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Set to an int (e.g. 3) to test a handful of combos first, or None for the
# full grid.
MAX_COMBOS = None

# Skip a combo if its output PNG already exists (lets you resume after an
# interruption without redoing finished work).
SKIP_EXISTING = True

# ============================================================================
# YOUR PARAMETER GRID
# ============================================================================
L_STAR_GRID     = np.logspace(14, 15, 4)      # Jy kpc^2   (4 values)
ALPHA_SCH_GRID  = np.linspace(-2.5, -0.5, 5)  # faint-end slope (5 values)
L_MIN_SCH_GRID  = np.logspace(8, 9, 2)       # Jy kpc^2   (3 values)
L_MAX_SCH_GRID  = np.logspace(14, 16, 3)      # Jy kpc^2   (3 values)

# ============================================================================
# TELESCOPE DEFINITIONS (unchanged from the notebook)
# ============================================================================
TELESCOPE_PARAMS = {
    "d0": {
        "D": 12, "eta": 0.6, "Trec": 70, "Tsky": 3, "BW": 336e6,
        "nu_cen": 1.32e9, "n_comp": 16, "n_dishes": 36, "n_chan": 2048,
        "tsamp": 1.265e-3,
    },
    "5d0": {
        "D": 64, "eta": 0.6, "Trec": 28, "Tsky": 3, "BW": 340e6,
        "nu_cen": 1.382e9, "n_comp": 16, "n_dishes": 1,
        "n_chan": int(1024 * 340 / 400), "n_beams": 13, "tsamp": 0.064e-3,
    },
    "25d0": {
        "D": 300, "eta": 0.6, "Trec": 17, "Tsky": 3, "BW": 400e6,
        "nu_cen": 1.25e9, "n_chan": 2048, "n_comp": 8, "n_dishes": 1,
        "n_beams": 19, "tsamp": 196.608e-6,
    },
}
TEL_KEYS = ["d0", "5d0", "25d0"]
colors_tel = {"d0": "#1f77b4", "5d0": "#ff7f0e", "25d0": "#2ca02c"}
labels_tel = {"d0": "D0 (12m)", "5d0": "5D0 (64m)", "25d0": "25D0 (300m)"}

# Physical constants
k = 1.38e-23
c = 3.0e8
Np = 2

alpha_intrinsic = -1  # Fixed intrinsic spectral index

# EFFECT CONTROL FLAGS (unchanged from the notebook)
USE_COSMOLOGY = True
USE_SFR = True
USE_SMEARING = True
USE_SPECTRAL_INDEX = True
USE_BEAM = True
LUMINOSITY_FUNCTION = "schechter"

# Redshift grid
dz = 0.01
z_min = 0.01
z_max = 5.0
z_array_opt = np.arange(z_min, z_max, dz)

fscale = 0.2 * 1e-7
SNR_threshold = 10
t_obs = 0.001
TARGET_DETECTIONS = 10
BATCH_SIZE = 4096
MAX_ITERATIONS = 1_000_000

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
H0_km_s_Mpc = cosmo.H0.value
DH_Mpc = 299792.458 / H0_km_s_Mpc


# ============================================================================
# LUMINOSITY FUNCTION HELPERS
# ============================================================================
def schechter_function(L, L_star, alpha, phi_star=1.0):
    x = L / L_star
    return phi_star * x ** alpha * np.exp(-x)


def create_truncated_schechter_sampler(L_min, L_max, L_star, alpha):
    L_grid = np.logspace(np.log10(L_min), np.log10(L_max), 2000)
    phi_grid = schechter_function(L_grid, L_star, alpha)
    cdf = cumtrapz(phi_grid, L_grid, initial=0)
    cdf = cdf / cdf[-1]
    inv_cdf = interp1d(cdf, L_grid, kind="linear",
                        bounds_error=False, fill_value=(L_min, L_max))
    return inv_cdf


def std_sampler(L_candle):
    def sampler(u):
        return L_candle
    return sampler


def gaussian_beam_gain(nu, G0, theta, D, c_):
    coeff = (theta ** 2 * D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c_ ** 2)
    return G0 * np.exp(-coeff * nu ** 2)


# ============================================================================
# CORE SIMULATION (Cell 10, refactored into a function of the 4 LF params)
# ============================================================================
def simulate_population(L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH):
    """Run the 3-telescope DM-population Monte Carlo for one set of
    Schechter luminosity-function parameters. Returns all_results, a dict
    tel_key -> list of {z, DM, n_total_population}."""

    # Schechter CDF interpolator -- built once per combo, reused per telescope/z-bin.
    L_grid_full = np.logspace(np.log10(L_MIN_SCH), np.log10(L_MAX_SCH), 5000)
    phi_grid_full = schechter_function(L_grid_full, L_STAR, ALPHA_SCH)
    cdf_full = cumtrapz(phi_grid_full, L_grid_full, initial=0)
    cdf_full /= cdf_full[-1]
    cdf_interp = interp1d(L_grid_full, cdf_full, kind="linear",
                           bounds_error=False, fill_value=(0.0, 1.0))

    def n_ratio_for(L_min_z):
        if LUMINOSITY_FUNCTION.lower() == "std":
            return 1.0 if L_std >= L_min_z else 0.0
        return 1.0 - cdf_interp(L_min_z)

    all_results = {}

    for tel_key in TEL_KEYS:
        params = TELESCOPE_PARAMS[tel_key]
        D = params["D"]
        eta = params["eta"]
        G0 = np.pi * D * D / 4 * eta / 2.0 / k * 1e-26

        np.random.seed(42)

        nu0 = params["nu_cen"]
        nu_cen = params["nu_cen"]
        Trec = params["Trec"]
        Tsky = params["Tsky"]
        BW = params["BW"]
        n_comp = params["n_comp"]
        BW_comp = BW / n_comp
        nu_min = nu0 - BW / 2
        nu_max = nu0 + BW / 2
        n_chan = params["n_chan"]
        BW_chan = BW / n_chan
        nu_array = np.linspace(nu_min, nu_max, n_comp + 1)
        freq_powers_scatter = np.array(
            [(nu_min * BW_comp * (i + 1)) ** (-4) for i in range(n_comp)]
        )
        theta_max = 2 * 1.22 * c / D / nu_min

        fwhm_rad = 1.22 * (c / nu_min) / D
        fwhm_deg = fwhm_rad * (180 / np.pi)
        r_deg = 2.0 * fwhm_deg
        area_tel = np.pi * (r_deg) ** 2

        # L_min(z) for this telescope
        L_min_by_z = {}
        for z in z_array_opt:
            dl = (cosmo.luminosity_distance(z).value * 1000.0) if USE_COSMOLOGY \
                else (DH_Mpc * z * 1000.0)
            G_max = gaussian_beam_gain(nu0, G0, 0, D, 3.0e8)
            S_min = SNR_threshold * (Trec + Tsky) / (G_max * np.sqrt(BW * t_obs * Np))
            z_factor = (1 + z) ** (-1 - alpha_intrinsic) if USE_SPECTRAL_INDEX else 1.0
            L_min = S_min * (dl ** 2) * z_factor
            L_min_by_z[z] = np.clip(L_min, L_MIN_SCH, L_MAX_SCH)

        sampler_cache = {}
        results_tel = []

        INNER_COEFF_CONST = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c ** 2)
        S_SNR_CONST = np.sqrt(BW_comp * t_obs * Np) / (Trec + Tsky)
        nu_scaling = (nu_array / nu0) ** alpha_intrinsic if USE_SPECTRAL_INDEX \
            else np.ones_like(nu_array)

        for z_idx, z in enumerate(z_array_opt):
            zmin, zmax = z - 0.5 * dz, z + 0.5 * dz
            L_min_z = L_min_by_z[z]

            if LUMINOSITY_FUNCTION.lower() == "std":
                if L_min_z > L_std:
                    break
                truncated_sampler = std_sampler(L_std)
            else:
                if L_min_z > L_MAX_SCH:
                    break
                if L_min_z not in sampler_cache:
                    sampler_cache[L_min_z] = create_truncated_schechter_sampler(
                        L_min_z, L_MAX_SCH, L_STAR, ALPHA_SCH
                    )
                truncated_sampler = sampler_cache[L_min_z]

            DM = 1000 * z
            wi_z = 0.001 * (1 + z)
            BW_chan_MHz = BW_chan / 1e6
            nu_cen_GHz = nu_cen / 1e9

            if USE_SMEARING:
                w_DM = 8.3e-6 * DM * BW_chan_MHz / (nu_cen_GHz ** 3)
                w_scatter_comp = (
                    1.9e-7 * (DM ** 1.5) * (1 + (DM ** 3) * 3.55e-5)
                    * 3 * freq_powers_scatter * 1e-3
                )
                w_scatter = np.sqrt(np.sum(w_scatter_comp ** 2) / len(w_scatter_comp))
                w_sampling = params["tsamp"] * 1e-3
            else:
                w_DM = w_scatter = w_sampling = 0.0

            w_obs_factor = np.sqrt(wi_z ** 2 + w_DM ** 2 + w_scatter ** 2 + w_sampling ** 2)

            dl = (cosmo.luminosity_distance(z).value * 1000.0) if USE_COSMOLOGY \
                else (DH_Mpc * z * 1000.0)
            z_factor = (1 + z) ** (-1 - alpha_intrinsic) if USE_SPECTRAL_INDEX else 1.0
            theta_hp = np.sqrt(np.log(2) / (INNER_COEFF_CONST * nu0 ** 2))

            n_generated = 0
            detected = []

            while len(detected) < TARGET_DETECTIONS and n_generated < MAX_ITERATIONS:
                n = min(BATCH_SIZE, MAX_ITERATIONS - n_generated)

                theta_b = theta_max * np.sqrt(np.random.random(n))
                angle_ok = theta_b <= 4.0 * theta_hp

                lum_b = np.asarray(truncated_sampler(np.random.uniform(0, 1, n)), dtype=float)
                S0_b = (lum_b / (dl ** 2 * z_factor)) * (wi_z / w_obs_factor)

                S_nu_b = S0_b[:, None] * nu_scaling[None, :]
                theta_for_gain = np.where(angle_ok, theta_b, 0.0) if USE_BEAM else np.zeros(n)
                G_nu_b = G0 * np.exp(
                    -(INNER_COEFF_CONST * theta_for_gain[:, None] ** 2) * nu_array[None, :] ** 2
                )

                snr_spectrum_b = S_nu_b * G_nu_b * S_SNR_CONST
                snr_coherent_b = snr_spectrum_b.sum(axis=1) / np.sqrt(n_comp)

                hits = np.where(angle_ok & (snr_coherent_b > SNR_threshold))[0]

                need = TARGET_DETECTIONS - len(detected)
                if len(hits) >= need:
                    hits_used = hits[:need]
                    n_generated += int(hits_used[-1]) + 1
                else:
                    hits_used = hits
                    n_generated += n

                for i in hits_used:
                    fluence_Jyms = (lum_b[i] / (dl ** 2 * z_factor)) * (wi_z * 1000.0)
                    detected.append((lum_b[i], fluence_Jyms))

            n_detected = len(detected)

            if n_detected == TARGET_DETECTIONS:
                detection_rate = n_detected / n_generated
                N_ratio = n_ratio_for(L_min_z)

                if USE_COSMOLOGY:
                    dt = (cosmo.age(zmin).value - cosmo.age(zmax).value) * 1e9
                    dvol = (
                        cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value
                    ) * (area_tel / (4.0 * np.pi * ((180.0 / np.pi) ** 2)))
                else:
                    r_Mpc = DH_Mpc * z
                    dr_Mpc = DH_Mpc * dz
                    area_sr = area_tel * (np.pi / 180.0) ** 2
                    dvol = (r_Mpc ** 2) * dr_Mpc * area_sr
                    dr_m = dr_Mpc * 3.0857e22
                    dt = dr_m / c / (3600.0 * 24.0 * 365.0)

                psi = (0.015 * ((1 + z) ** 2.7) / (1 + ((1 + z) / 2.9) ** 5.6)) if USE_SFR else 0.015
                nfrb_sfr = psi * dvol * dt * fscale
                n_total_population = detection_rate * N_ratio * nfrb_sfr

                results_tel.append({"z": z, "DM": DM, "n_total_population": n_total_population})

        all_results[tel_key] = results_tel

    return all_results


# ============================================================================
# PLOTTING (Cell 11, refactored to save instead of plt.show())
# ============================================================================
def plot_dm_histogram(all_results, L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH, outpath):
    valid_keys = [k_ for k_ in TEL_KEYS if len(all_results.get(k_, [])) > 0]
    if not valid_keys:
        return False  # nothing detected for any telescope at this combo -- skip plot

    fig, ax = plt.subplots(figsize=(15, 10), dpi=130)

    dm_min = min(1000 * all_results[k_][0]["z"] for k_ in valid_keys)
    dm_max = max(1000 * all_results[k_][-1]["z"] for k_ in valid_keys)
    common_bins = np.linspace(dm_min, dm_max, 40)

    n_frb_arr = []
    for tel_key in TEL_KEYS:
        results = all_results.get(tel_key, [])
        if len(results) == 0:
            n_frb_arr.append(np.nan)
            continue
        DMs = np.array([r["DM"] for r in results])
        pops = np.array([r["n_total_population"] for r in results])
        nfrb = 1000 * pops.sum()
        ax.hist(DMs, bins=common_bins, weights=pops, density=True, alpha=0.6,
                label=f'{labels_tel[tel_key]}, n$_{{\\mathrm{{frb}}}}$={int(nfrb)}',
                color=colors_tel[tel_key], linewidth=1.5)
        n_frb_arr.append(nfrb)

    ax2 = ax.secondary_xaxis("top", functions=(lambda x: x / 1000, lambda x: x * 1000))
    ax2.set_xlabel("Redshift (z)", fontsize=23)
    ax2.tick_params(axis="x", labelsize=18)

    ax.set_xlabel("DM (pc/cm\u00b3)", fontsize=23)
    ax.set_ylabel(r"$P(\mathrm{DM}\,|\,\mathrm{detected})$ (pc$^{-1}$ cm$^3$)", fontsize=23)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.legend(fontsize=18, loc="upper right", framealpha=0.95)

    LF_text = (f"L*={L_STAR:.2e}, \u03b1={ALPHA_SCH:.2f}, "
               f"L_min={L_MIN_SCH:.2e}, L_max={L_MAX_SCH:.2e}")
    props = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="white", linewidth=2)
    ax.text(0.48, 0.8, LF_text, transform=ax.transAxes, fontsize=15, bbox=props)

    if all(not np.isnan(v) and v > 0 for v in n_frb_arr):
        ratio_text = (f"ASKAP/PKS: {n_frb_arr[0] / n_frb_arr[1]:.2f}\n"
                      f"PKS/FAST: {n_frb_arr[1] / n_frb_arr[2]:.2f}\n"
                      f"ASKAP/FAST: {n_frb_arr[0] / n_frb_arr[2]:.2f}\n")
        ax.text(0.48, 0.65, ratio_text, transform=ax.transAxes, fontsize=15, bbox=props)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def combo_filename(L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH):
    return (OUTDIR /
            f"DM_dist_Lstar{L_STAR:.2e}_alpha{ALPHA_SCH:.2f}"
            f"_Lmin{L_MIN_SCH:.2e}_Lmax{L_MAX_SCH:.2e}.png")


# ============================================================================
# MAIN GRID SCAN
# ============================================================================
def main():
    combos = [
        (L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH)
        for L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH
        in itertools.product(L_STAR_GRID, ALPHA_SCH_GRID, L_MIN_SCH_GRID, L_MAX_SCH_GRID)
        if L_MIN_SCH < L_MAX_SCH  # skip physically invalid combos
    ]
    if MAX_COMBOS is not None:
        combos = combos[:MAX_COMBOS]

    print(f"Total valid combinations to run: {len(combos)}")
    t_start = time.time()
    n_done = 0
    n_skipped = 0

    for i, (L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH) in enumerate(combos):
        outpath = combo_filename(L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH)

        if SKIP_EXISTING and outpath.exists():
            n_skipped += 1
            continue

        t0 = time.time()
        all_results = simulate_population(L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH)
        made_plot = plot_dm_histogram(all_results, L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH, outpath)
        dt = time.time() - t0
        n_done += 1

        elapsed = time.time() - t_start
        avg = elapsed / n_done
        remaining = avg * (len(combos) - n_done - n_skipped)

        status = "saved" if made_plot else "no detections (skipped plot)"
        print(f"[{i+1}/{len(combos)}] L*={L_STAR:.2e} alpha={ALPHA_SCH:.2f} "
              f"Lmin={L_MIN_SCH:.2e} Lmax={L_MAX_SCH:.2e} "
              f"-> {status} ({dt:.1f}s, ETA {remaining/60:.1f} min)")

    print(f"\nDone. {n_done} combos run, {n_skipped} skipped (already existed). "
          f"Total time {(time.time() - t_start)/60:.1f} min.")
    print(f"Plots saved in: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
