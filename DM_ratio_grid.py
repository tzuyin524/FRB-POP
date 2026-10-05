#!/usr/bin/env python3
"""
Grid search over (alpha_intrinsic, L_STAR, ALPHA_SCH) for the 3-telescope
FRB DM_obs population comparison, adapted from
DM_comparison_mc_fluence_DM_copy.ipynb (cells 10-11: the "3-telescope DM
population" comparison and the DM_obs histogram + ratio annotation).

The three "telescopes" are the dish sizes already defined in the notebook's
TELESCOPE_PARAMS: d0 (12 m, ASKAP-like), 5d0 (64 m, Parkes/PKS-like) and
25d0 (300 m, FAST-like). Only the physical simulation is kept here (the
fluence/DM scatter part of the notebook is dropped since it isn't needed
for the DM histogram + ratio CSV requested).

For every (alpha_intrinsic, L_STAR, ALPHA_SCH) combination this script:
  1. Runs the Monte Carlo detection simulation for all three telescopes.
  2. Saves the weighted DM_obs distribution plot (PNG) to OUTPUT_DIR.
  3. Appends a row to a summary CSV in OUTPUT_DIR with the parameter values
     and the ASKAP/PKS, PKS/FAST, ASKAP/FAST detected-FRB-count ratios.

The CSV is rewritten after every combination (not just at the end), so a
killed/interrupted job on a cluster still leaves usable partial results.

Usage:
    python DM_ratio_grid.py                     # full 5x5x3 = 75-point grid
    python DM_ratio_grid.py --test               # 1 quick combination, to
                                                   # sanity-check the pipeline
    python DM_ratio_grid.py --output-dir DIR      # override the output folder
"""

import argparse
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — we only ever save PNGs, never show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

try:  # cumtrapz was renamed/removed in newer scipy (>=1.14)
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz

# ============================================================================
# OUTPUT LOCATION
# ============================================================================
DEFAULT_OUTPUT_DIR = Path("/home/thsu/FRB_population/DM_ratio")

# CSV of the 19 FAST beams' gain (K/Jy) at several frequencies (Gain_<MHz> columns).
# Must sit next to this script, or point --fast-beam-csv at it.
DEFAULT_FAST_BEAM_CSV = Path(__file__).resolve().parent / "FAST_beam_gain.csv"

# ============================================================================
# PARAMETER GRID (edit these three lists to change what's swept)
# ============================================================================
ALPHA_INTRINSIC_GRID = [0, -0.5, -1.5, -2, -2.5]
L_STAR_GRID = [1e13, 3e13, 5e13, 8e13, 1e14]
ALPHA_SCH_GRID = [-1.2, -1.5, -1.8]  # faint-end slope

# ============================================================================
# FIXED TELESCOPE PARAMETERS (unchanged from the notebook)
# ============================================================================
TELESCOPE_PARAMS = {
    "d0": {
        "D": 12, "eta": 0.8, "Trec": 70, "Tsky": 3, "BW": 336e6,
        "nu_cen": 1.32e9, "n_comp": 16, "n_dishes": 36, "n_chan": 2048,
        "tsamp": 1.265e-3,
    },
    "5d0": {
        "D": 64, "eta": 0.7, "Trec": 28, "Tsky": 3, "BW": 340e6,
        "nu_cen": 1.382e9, "n_comp": 16, "n_dishes": 1,
        "n_chan": int(1024 * 340 / 400), "n_beams": 13, "tsamp": 0.064e-3,
    },
    "25d0": {
        "D": 300, "eta": 0.6, "Trec": 17, "Tsky": 3, "BW": 400e6,
        "nu_cen": 1.25e9, "n_chan": 2048, "n_comp": 8, "n_dishes": 1,
        "n_beams": 19, "tsamp": 196.608e-6,
    },
}
# Order matters: it defines ASKAP / PKS / FAST in the ratio columns below.
TEL_ORDER = ["d0", "5d0", "25d0"]
TEL_SHORT = {"d0": "ASKAP", "5d0": "PKS", "25d0": "FAST"}
COLORS_TEL = {"d0": "#1f77b4", "5d0": "#ff7f0e", "25d0": "#2ca02c"}
LABELS_TEL = {
    "d0": "ASKAP-like D0 (12m)",
    "5d0": "PKS-like 5D0 (64m)",
    "25d0": "FAST-like 25D0 (300m)",
}

# NOTE on fidelity: earlier versions of this script (and of the notebook it
# was adapted from) used the module-level `eta` from d0 (0.8) to compute G0
# for every telescope, rather than each dish's own eta. That bug is fixed
# below -- d0 now uses TELESCOPE_PARAMS["d0"]["eta"] via params["eta"].

k_B = 1.38e-23
c_light = 3.0e8
Np = 2

# ============================================================================
# PER-TELESCOPE GAIN INPUTS
#   d0   : single gain from the dish formula (unchanged)
#   5d0  : on-axis gain (K/Jy) of each of its 13 beams; one is drawn per FRB
#   25d0 : 19 FAST beams, each with a gain measured at several frequencies;
#          one beam is drawn per FRB, and its gain varies with frequency
# ============================================================================
G0_PKS = [0.735, 0.690, 0.690, 0.690, 0.690, 0.690, 0.690,
          0.581, 0.581, 0.581, 0.581, 0.581, 0.581]

_df_fast_beams = None
_fast_gain_columns = None
_fast_freqs_measured = None
_fast_gain_matrix = None


def load_fast_beam_data(csv_path=DEFAULT_FAST_BEAM_CSV):
    """Load the FAST beam-gain CSV once; safe to call repeatedly (cached)."""
    global _df_fast_beams, _fast_gain_columns, _fast_freqs_measured, _fast_gain_matrix
    if _df_fast_beams is not None:
        return
    _df_fast_beams = pd.read_csv(csv_path)
    _fast_gain_columns = [c for c in _df_fast_beams.columns if c.startswith("Gain_")]
    _fast_freqs_measured = np.array(
        [int(c.split("_")[1]) for c in _fast_gain_columns]
    ) * 1e6  # MHz -> Hz
    _fast_gain_matrix = _df_fast_beams[_fast_gain_columns].values.astype(float)  # (n_beams, n_freq)
    print(f"Loaded FAST beam data: {_fast_gain_matrix.shape[0]} beams, "
          f"gain columns: {_fast_gain_columns}")


def build_fast_beam_gain_cache(nu_array):
    """Gain of every FAST beam interpolated onto nu_array -> (n_beams, len(nu_array))."""
    interp = interp1d(_fast_freqs_measured, _fast_gain_matrix, kind="linear", axis=1,
                       bounds_error=False, fill_value="extrapolate")
    return np.clip(interp(np.atleast_1d(nu_array)), 0, _fast_gain_matrix.max(axis=1, keepdims=True))

# ============================================================================
# EFFECT CONTROL FLAGS (unchanged from the notebook)
# ============================================================================
USE_COSMOLOGY = True
USE_SFR = True
USE_SMEARING = True
USE_SPECTRAL_INDEX = True
USE_BEAM = True
LUMINOSITY_FUNCTION = "schechter"  # "schechter" or "std"

# ============================================================================
# REDSHIFT / SIMULATION PARAMETERS (unchanged from the notebook)
# ============================================================================
DZ = 0.001
Z_MIN = 0.01
Z_MAX = 5.0
L_MIN_SCH = 1e6
L_MAX_SCH = 1e16
L_STD = 1e12  # Jy.kpc^2, used only if LUMINOSITY_FUNCTION == "std"
FSCALE = 100 * 1e-7
SNR_THRESHOLD = 8
T_OBS = 0.001
TARGET_DETECTIONS = 10  # target detections per z-bin
BATCH_SIZE = 4096
MAX_ITERATIONS = 10000
SEED = 42

# DM_host model (notebook cell 11)
SIGMA_HOST = 0.8
MU_HOST = 100.0  # pc/cc, median of DM_host

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
DH_MPC = 299792.458 / COSMO.H0.value


def luminosity_distance_kpc(z):
    """Luminosity distance in kpc, cosmological or Euclidean per USE_COSMOLOGY."""
    if USE_COSMOLOGY:
        return COSMO.luminosity_distance(z).value * 1000.0
    return DH_MPC * z * 1000.0


# ============================================================================
# HELPER FUNCTIONS (unchanged physics from the notebook)
# ============================================================================
def gaussian_beam_gain(nu, G0, theta, D, c):
    """G(nu) = G0 * exp[-theta^2 nu^2 D^2 * 2.355^2 / (2 * 1.22^2 * c^2)]"""
    coeff = (theta ** 2 * D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c ** 2)
    return G0 * np.exp(-coeff * nu ** 2)


def schechter_function(L, L_star, alpha, phi_star=1.0):
    """phi(L) = phi* (L/L*)^alpha * exp(-L/L*)"""
    x = L / L_star
    return phi_star * x ** alpha * np.exp(-x)


def create_truncated_schechter_sampler(L_min, L_max, L_star, alpha):
    """Inverse-CDF sampler for the Schechter function truncated to [L_min, L_max]."""
    L_grid = np.logspace(np.log10(L_min), np.log10(L_max), 2000)
    phi_grid = schechter_function(L_grid, L_star, alpha)
    cdf = cumtrapz(phi_grid, L_grid, initial=0)
    cdf = cdf / cdf[-1]
    return interp1d(cdf, L_grid, kind="linear", bounds_error=False,
                     fill_value=(L_min, L_max))


def std_sampler(L_candle):
    """Fixed-luminosity sampler for the standard-candle model."""
    def sampler(u):
        return L_candle
    return sampler


# ============================================================================
# ONE FULL 3-TELESCOPE RUN FOR A GIVEN (alpha_intrinsic, L_STAR, ALPHA_SCH)
# ============================================================================
def run_one_combination(alpha_intrinsic, L_STAR, ALPHA_SCH, z_max=Z_MAX,
                         target_detections=TARGET_DETECTIONS,
                         max_iterations=MAX_ITERATIONS, verbose=False):
    """Runs the per-z-bin rejection-sampling detection loop for d0/5d0/25d0.

    Returns {tel_key: [{"z":..., "DM":..., "n_total_population":...}, ...]}
    """
    z_array = np.arange(Z_MIN, z_max, DZ)

    # Schechter CDF interpolator -- built once per parameter combination,
    # reused for every telescope/z-bin (mirrors the notebook's optimisation).
    L_grid_full = np.logspace(np.log10(L_MIN_SCH), np.log10(L_MAX_SCH), 5000)
    phi_grid_full = schechter_function(L_grid_full, L_STAR, ALPHA_SCH)
    cdf_full = cumtrapz(phi_grid_full, L_grid_full, initial=0)
    cdf_full /= cdf_full[-1]
    cdf_interp = interp1d(L_grid_full, cdf_full, kind="linear",
                           bounds_error=False, fill_value=(0.0, 1.0))

    def n_ratio_for(L_min_z):
        if LUMINOSITY_FUNCTION.lower() == "std":
            return 1.0 if L_STD >= L_min_z else 0.0
        return 1.0 - cdf_interp(L_min_z)

    all_results = {}

    for tel_key in TEL_ORDER:
        params = TELESCOPE_PARAMS[tel_key]
        D = params["D"]

        np.random.seed(SEED)  # same seed for every telescope, as in the notebook

        nu0 = params["nu_cen"]
        nu_cen = params["nu_cen"]
        Trec = params["Trec"]
        Tsky = params["Tsky"]
        BW = params["BW"]
        n_comp = params["n_comp"]
        if tel_key == "25d0":
            load_fast_beam_data()
            n_comp = len(_fast_freqs_measured)  # one frequency component per CSV gain column
        BW_comp = BW / n_comp
        nu_min = nu0 - BW / 2
        n_chan = params["n_chan"]
        BW_chan = BW / n_chan
        if tel_key == "25d0":
            nu_array = _fast_freqs_measured.copy()  # exactly the frequencies the CSV gains are given at
        else:
            nu_array = np.linspace(nu_min, nu0 + BW / 2, n_comp + 1)
        freq_powers_scatter = np.array(
            [(nu_min * BW_comp * (i + 1)) ** (-4) for i in range(n_comp)]
        )
        theta_max = 2 * 1.22 * c_light / D / nu_min

        fwhm_rad = 1.22 * (c_light / nu_min) / D
        fwhm_deg = fwhm_rad * (180 / np.pi)
        r_deg = 2.0 * fwhm_deg
        area_tel = np.pi * (r_deg) ** 2

        # ------------------------------------------------------------------
        # Telescope-specific on-axis gain G0 (K/Jy). G0_max is the best-case
        # gain; it sets L_min(z) so the luminosity sampler is truncated at
        # the faintest burst any beam could detect.
        # ------------------------------------------------------------------
        if tel_key == "5d0":
            G0_max = max(G0_PKS)
        elif tel_key == "25d0":
            fast_beam_cache = build_fast_beam_gain_cache(nu_array)          # (n_beams, len(nu_array))
            G0_max = build_fast_beam_gain_cache(np.array([nu0])).max()      # best beam at nu0
        else:
            G0 = np.pi * D * D / 4 * params["eta"] / 2.0 / k_B * 1e-26
            G0_max = G0

        if verbose:
            print(f"    [{tel_key}] D={D}m G0_max={G0_max:.4e} theta_max={theta_max:.6f} "
                  f"n_comp={n_comp} area={area_tel:.4f} deg^2")

        # --- L_min(z) for this telescope ---
        L_min_by_z = {}
        for z in z_array:
            dl = luminosity_distance_kpc(z)
            G_max = gaussian_beam_gain(nu0, G0_max, 0, D, c_light)
            S_min = SNR_THRESHOLD * (Trec + Tsky) / (G_max * np.sqrt(BW * T_OBS * Np))
            z_factor = (1 + z) ** (-1 - alpha_intrinsic) if USE_SPECTRAL_INDEX else 1.0
            L_min = S_min * (dl ** 2) * z_factor
            L_min_by_z[z] = np.clip(L_min, L_MIN_SCH, L_MAX_SCH)

        sampler_cache = {}
        results_tel = []

        INNER_COEFF_CONST = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c_light ** 2)
        S_SNR_CONST = np.sqrt(BW_comp * T_OBS * Np) / (Trec + Tsky)
        nu_scaling = (nu_array / nu0) ** alpha_intrinsic if USE_SPECTRAL_INDEX else np.ones_like(nu_array)

        for z_idx, z in enumerate(z_array):
            zmin, zmax = z - 0.5 * DZ, z + 0.5 * DZ
            L_min_z = L_min_by_z[z]

            if LUMINOSITY_FUNCTION.lower() == "std":
                if L_min_z > L_STD:
                    break
                truncated_sampler = std_sampler(L_STD)
            else:
                if L_min_z > 8.952e15:
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
                    1.9e-7 * (DM ** 1.5) * (1 + (DM ** 3) * 3.55e-5) * 3 * freq_powers_scatter * 1e-3
                )
                w_scatter = np.sqrt(np.sum(w_scatter_comp ** 2) / len(w_scatter_comp))
                w_sampling = params["tsamp"] * 1e-3
            else:
                w_DM = w_scatter = w_sampling = 0.0

            w_obs_factor = np.sqrt(wi_z ** 2 + w_DM ** 2 + w_scatter ** 2 + w_sampling ** 2)

            dl = luminosity_distance_kpc(z)
            z_factor = (1 + z) ** (-1 - alpha_intrinsic) if USE_SPECTRAL_INDEX else 1.0
            theta_hp = np.sqrt(np.log(2) / (INNER_COEFF_CONST * nu0 ** 2))

            # ---------------- detection loop (counts only) ----------------
            n_generated = 0
            n_detected = 0
            max_iter_local = max_iterations

            while n_detected < target_detections and n_generated < max_iter_local:
                n = min(BATCH_SIZE, max_iter_local - n_generated)

                theta_b = theta_max * np.sqrt(np.random.random(n))
                angle_ok = theta_b <= 4.0 * theta_hp

                lum_b = np.asarray(truncated_sampler(np.random.uniform(0, 1, n)), dtype=float)
                if LUMINOSITY_FUNCTION.lower() == "std":
                    lum_b = np.full(n, L_STD, dtype=float)
                S0_b = (lum_b / (dl ** 2 * z_factor)) * (wi_z / w_obs_factor)

                S_nu_b = S0_b[:, None] * nu_scaling[None, :]
                theta_for_gain = np.where(angle_ok, theta_b, 0.0) if USE_BEAM else np.zeros(n)
                # Per-FRB on-axis gain: scalar (d0), (n,1) (5d0) or (n, n_nu) (25d0)
                if tel_key == "5d0":
                    G0_b = np.random.choice(G0_PKS, size=n)[:, None]
                elif tel_key == "25d0":
                    G0_b = fast_beam_cache[np.random.randint(fast_beam_cache.shape[0], size=n)]
                else:
                    G0_b = G0
                G_nu_b = G0_b * np.exp(
                    -(INNER_COEFF_CONST * theta_for_gain[:, None] ** 2) * nu_array[None, :] ** 2
                )

                snr_spectrum_b = S_nu_b * G_nu_b * S_SNR_CONST
                snr_coherent_b = snr_spectrum_b.sum(axis=1) / np.sqrt(n_comp)

                hits = np.where(angle_ok & (snr_coherent_b > SNR_THRESHOLD))[0]

                need = target_detections - n_detected
                if len(hits) >= need:
                    hits_used = hits[:need]
                    n_generated += int(hits_used[-1]) + 1
                else:
                    hits_used = hits
                    n_generated += n

                n_detected += len(hits_used)

            if n_detected == target_detections:
                detection_rate = n_detected / n_generated
                N_ratio = n_ratio_for(L_min_z)

                if USE_COSMOLOGY:
                    dt = (COSMO.age(zmin).value - COSMO.age(zmax).value) * 1e9
                    dvol = (
                        COSMO.comoving_volume(zmax).value - COSMO.comoving_volume(zmin).value
                    ) * (area_tel / (4.0 * np.pi * ((180.0 / np.pi) ** 2)))
                else:
                    r_Mpc = DH_MPC * z
                    dr_Mpc = DH_MPC * DZ
                    area_sr = area_tel * (np.pi / 180.0) ** 2
                    dvol = (r_Mpc ** 2) * dr_Mpc * area_sr
                    dr_m = dr_Mpc * 3.0857e22
                    dt = dr_m / c_light / (3600.0 * 24.0 * 365.0)

                psi = (0.015 * ((1 + z) ** 2.7) / (1 + ((1 + z) / 2.9) ** 5.6)) if USE_SFR else 0.015
                nfrb_sfr = psi * dvol * dt * FSCALE
                n_total_population = detection_rate * N_ratio * nfrb_sfr

                results_tel.append({"z": z, "DM": DM, "n_total_population": n_total_population})

        all_results[tel_key] = results_tel

    return all_results


# ============================================================================
# DM_obs HISTOGRAM + RATIO COMPUTATION (notebook cell 11)
# ============================================================================
def _fmt(x):
    """Compact, filename-safe representation of a numeric parameter."""
    return f"{x:g}"


def compute_dm_obs_and_plot(all_results, alpha_intrinsic, L_STAR, ALPHA_SCH, output_dir, seed=SEED):
    rng = np.random.default_rng(seed)  # reproducible DM_host/MW draws per combination

    dmobs_dict = {}
    for tel_key in TEL_ORDER:
        results = all_results.get(tel_key, [])
        if len(results) == 0:
            continue
        z_arr = np.array([r["z"] for r in results])
        DM_cosmic = 1000 * z_arr

        DM_host = rng.lognormal(mean=np.log(MU_HOST), sigma=SIGMA_HOST, size=len(z_arr))
        DM_MWism = rng.uniform(0, 100, size=len(z_arr))
        DM_MWhalo = rng.normal(50, 20, size=len(z_arr))
        DM_obs = DM_cosmic + DM_host / (1 + z_arr) + DM_MWhalo + DM_MWism
        dmobs_dict[tel_key] = DM_obs

    if not dmobs_dict:
        return None  # no telescope reached target_detections for any z-bin

    dm_min = min(dmobs_dict[k].min() for k in dmobs_dict)
    dm_max = max(dmobs_dict[k].max() for k in dmobs_dict)
    common_bins = np.linspace(dm_min, dm_max, 40)

    fig, ax = plt.subplots(figsize=(15, 10), dpi=130)

    n_frb_arr = []  # in TEL_ORDER: [ASKAP(d0), PKS(5d0), FAST(25d0)]
    for tel_key in TEL_ORDER:
        if tel_key not in dmobs_dict:
            n_frb_arr.append(0.0)
            continue
        results = all_results[tel_key]
        pops = np.array([r["n_total_population"] for r in results])
        DM_obs = dmobs_dict[tel_key]
        nfrb = pops.sum()
        ax.hist(DM_obs, bins=common_bins, weights=pops, density=True, alpha=0.6,
                label=f"{LABELS_TEL[tel_key]}, n_frb={int(nfrb)}",
                color=COLORS_TEL[tel_key], linewidth=1.5)
        n_frb_arr.append(nfrb)

    ax.set_xlabel(r"DM$_{\mathrm{obs}}$ (pc/cm$^3$)", fontsize=23)
    ax.set_ylabel(r"$P(\mathrm{DM_{obs}}\,|\,\mathrm{detected})$ (pc$^{-1}$ cm$^3$)", fontsize=23)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.legend(fontsize=18, loc="upper right", framealpha=0.95)

    n_askap, n_pks, n_fast = n_frb_arr
    ratio_askap_pks = n_askap / n_pks if n_pks else np.nan
    ratio_pks_fast = n_pks / n_fast if n_fast else np.nan
    ratio_askap_fast = n_askap / n_fast if n_fast else np.nan

    config_text = (
        f"CONFIGURATION:\n"
        f"alpha_intrinsic: {alpha_intrinsic}\n"
        f"Luminosity Function: {LUMINOSITY_FUNCTION}\n"
        f"L*={L_STAR:.2e}, alpha_sch={ALPHA_SCH:.2f}\n"
    )
    ratio_text = (
        f"ASKAP/PKS: {ratio_askap_pks:.2f}\n"
        f"PKS/FAST: {ratio_pks_fast:.2f}\n"
        f"ASKAP/FAST: {ratio_askap_fast:.2f}\n"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="white", linewidth=2)
    ax.text(0.46, 0.78, config_text, transform=ax.transAxes, fontsize=13, bbox=props)
    ax.text(0.46, 0.58, ratio_text, transform=ax.transAxes, fontsize=15, bbox=props)

    plt.tight_layout()

    fname = (
        f"DM_dist_alpha{_fmt(alpha_intrinsic)}_Lstar{L_STAR:.0e}_alphaSch{_fmt(ALPHA_SCH)}.png"
    )
    fig_path = Path(output_dir) / fname
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_ASKAP": n_askap, "n_PKS": n_pks, "n_FAST": n_fast,
        "ASKAP_PKS_ratio": ratio_askap_pks,
        "PKS_FAST_ratio": ratio_pks_fast,
        "ASKAP_FAST_ratio": ratio_askap_fast,
        "plot_file": fname,
    }


# ============================================================================
# GRID DRIVER
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Grid search over alpha_intrinsic, L_STAR, ALPHA_SCH "
                     "for the 3-telescope FRB DM_obs comparison."
    )
    parser.add_argument("--test", action="store_true",
                         help="Run a single quick combination (reduced detections "
                              "and z-range) to sanity-check the pipeline before "
                              "launching the full grid.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                         help=f"Where to write plots + CSV (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--fast-beam-csv", default=str(DEFAULT_FAST_BEAM_CSV),
                         help=f"Path to the FAST beam-gain CSV (default: {DEFAULT_FAST_BEAM_CSV})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "DM_ratio_grid_results.csv"
    load_fast_beam_data(args.fast_beam_csv)  # loaded once up front; reused every combination

    if args.test:
        alpha_grid = [ALPHA_INTRINSIC_GRID[2]]  # -1.5, the notebook's original default
        lstar_grid = [L_STAR_GRID[0]]
        alphasch_grid = [ALPHA_SCH_GRID[1]]
        target_det = 3
        z_max = 1.0
        print("Running in --test mode: 1 combination, reduced detections/z-range.\n")
    else:
        alpha_grid = ALPHA_INTRINSIC_GRID
        lstar_grid = L_STAR_GRID
        alphasch_grid = ALPHA_SCH_GRID
        target_det = TARGET_DETECTIONS
        z_max = Z_MAX

    combos = list(product(alpha_grid, lstar_grid, alphasch_grid))
    print(f"Grid size: {len(combos)} combination(s) -> output dir: {output_dir}\n")

    rows = []
    t_start_all = time.time()
    for idx, (alpha_intrinsic, L_STAR, ALPHA_SCH) in enumerate(combos, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(combos)}] alpha_intrinsic={alpha_intrinsic}, "
              f"L_STAR={L_STAR:.1e}, ALPHA_SCH={ALPHA_SCH} ...", flush=True)

        try:
            all_results = run_one_combination(
                alpha_intrinsic, L_STAR, ALPHA_SCH,
                z_max=z_max, target_detections=target_det,
            )
            plot_result = compute_dm_obs_and_plot(
                all_results, alpha_intrinsic, L_STAR, ALPHA_SCH, output_dir
            )
        except Exception as exc:  # keep the grid going even if one point fails
            print(f"    FAILED: {exc!r}")
            plot_result = None

        row = {"alpha_intrinsic": alpha_intrinsic, "L_STAR": L_STAR, "ALPHA_SCH": ALPHA_SCH}
        if plot_result is None:
            row.update({
                "n_ASKAP": np.nan, "n_PKS": np.nan, "n_FAST": np.nan,
                "ASKAP_PKS_ratio": np.nan, "PKS_FAST_ratio": np.nan,
                "ASKAP_FAST_ratio": np.nan, "plot_file": "",
            })
        else:
            row.update(plot_result)
        rows.append(row)

        # Rewrite the CSV after every combination so partial results survive
        # an interrupted/killed job on the cluster.
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        dt = time.time() - t0
        print(f"    done in {dt:.1f}s -> "
              f"ASKAP/PKS={row['ASKAP_PKS_ratio']}, "
              f"PKS/FAST={row['PKS_FAST_ratio']}, "
              f"ASKAP/FAST={row['ASKAP_FAST_ratio']}")

    total_dt = time.time() - t_start_all
    print(f"\nAll {len(combos)} combination(s) done in {total_dt / 60:.1f} min.")
    print(f"CSV summary: {csv_path}")
    print(f"Plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
