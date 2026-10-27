#!/usr/bin/env python3
"""
Parameter scan of the 4-telescope FRB detection MC (ASKAP fly's eye = d0,
ASKAP ICS = d0_ics, Parkes = 5d0, FAST = 25d0).  T_sky is frequency dependent
(read from ASKAP_Tsky.csv / Parkes_Tsky.csv / FAST_Tsky.csv).

For every (alpha_intrinsic, L_STAR, ALPHA_SCH) combination it computes the
weighted number of detected FRBs per telescope and writes

    alpha_intrinsic,L_STAR,ALPHA_SCH,n_ASKAP_FE,n_ASKAP_ICS,n_PKS,n_FAST,
    ASKAP_FE_PKS_ratio,ASKAP_ICS_PKS_ratio,PKS_FAST_ratio,
    ASKAP_FE_FAST_ratio,ASKAP_ICS_FAST_ratio,ASKAP_ICS_FE_ratio

No plots are made; only the CSV is written.

Usage
    python scan_frb_params.py                # scan -> csv
    python scan_frb_params.py --dz 0.001     # same z grid as the notebook (slower)
"""
import argparse
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================================
# SCAN GRID
# ============================================================================
L_STAR_GRID = [1e13, 3e13, 5e13, 8e13, 1e14, 3e14, 5e14]
ALPHA_SCH_GRID = [-1.2, -1.5, -1.8]
ALPHA_INT_GRID = [0, -0.5, -1, -1.5, -2, -2.5, -3]

# ============================================================================
# FIXED SETTINGS (same as the notebook)
# ============================================================================
FAST_BEAM_CSV = "FAST_beam_gain.csv"   # <- put the full path here if needed
OUT_DIR = "/home/thsu/FRB_population/DM_ratio_ics"   # csv is written here
OUT_CSV = "frb_param_scan_ics.csv"

USE_COSMOLOGY = True
USE_SFR = True
USE_SMEARING = True
USE_SPECTRAL_INDEX = True
USE_BEAM = True

L_MIN_SCH, L_MAX_SCH = 1e6, 1e16
fscale = 100 * 1e-7
SNR_threshold = 8
t_obs = 0.001
Np = 2
TARGET_DETECTIONS = 10
MAX_ITERATIONS = 10000
BATCH_SIZE = 4096
SEED = 42

Z_MIN, Z_MAX = 0.01, 5.0
DZ_DEFAULT = 0.001          # same as the notebook; --dz 0.005 is ~5x faster (ratios agree to a few %)
DZ_REF = 0.001              # absolute n_* are rescaled to this dz (see note in prepare())

# Stop a telescope's z loop after this many consecutive z-bins in which fewer than
# TARGET_DETECTIONS FRBs were found (those bins are discarded anyway, as in the
# notebook).  Set to None to run every z-bin.
EARLY_STOP_FAILS = 40
# False = notebook behaviour (a z-bin with <10 detections is dropped).
# True  = keep it, using rate = n_detected / n_generated (removes the MC floor).
KEEP_PARTIAL = False

# Frequency-dependent T_sky.  False = old constant Tsky=3 K from TELESCOPE_PARAMS.
USE_TSKY_FILES = True
TSKY_DIR = "/home/thsu/FRB_population"
TSKY_FILES = {"d0": "ASKAP_Tsky.csv", "d0_ics": "ASKAP_Tsky.csv",
              "5d0": "Parkes_Tsky.csv", "25d0": "FAST_Tsky.csv"}

k_B = 1.38e-23
c = 3.0e8
TELESCOPE_PARAMS = {
    "d0":   dict(D=12,  eta=0.8, Trec=70, Tsky=3, BW=336e6, nu_cen=1.32e9,  n_comp=16,
                 n_chan=2048, tsamp=1.265e-3),
    # ASKAP ICS: same as fly's eye, but G0 is multiplied by sqrt(n_dishes) (incoherent sum)
    "d0_ics": dict(D=12,  eta=0.8, Trec=70, Tsky=3, BW=336e6, nu_cen=1.32e9,  n_comp=16,
                   n_chan=2048, tsamp=1.265e-3, n_dishes=25),
    "5d0":  dict(D=64,  eta=0.7, Trec=28, Tsky=3, BW=340e6, nu_cen=1.382e9, n_comp=16,
                 n_chan=int(1024 * 340 / 400), tsamp=0.064e-3),
    "25d0": dict(D=300, eta=0.6, Trec=17, Tsky=3, BW=400e6, nu_cen=1.25e9,  n_comp=8,
                 n_chan=2048, tsamp=196.608e-6),
}
G0_PKS = [0.735, 0.690, 0.690, 0.690, 0.690, 0.690, 0.690,
          0.581, 0.581, 0.581, 0.581, 0.581, 0.581]
TEL_ORDER = ["d0", "d0_ics", "5d0", "25d0"]
TEL_LABEL = {"d0": "ASKAP FE", "d0_ics": "ASKAP ICS", "5d0": "Parkes", "25d0": "FAST"}

# ============================================================================
# SHARED, PARAMETER-INDEPENDENT PRE-COMPUTATION (done once per process)
# ============================================================================
_STATE = {}


def _get_cosmo():
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=70, Om0=0.3)


def _cumtrapz():
    try:
        from scipy.integrate import cumulative_trapezoid
        return cumulative_trapezoid
    except ImportError:                       # old scipy
        from scipy.integrate import cumtrapz
        return cumtrapz


def _load_fast_beams():
    df = pd.read_csv(FAST_BEAM_CSV)
    cols = [col for col in df.columns if col.startswith("Gain_")]
    freqs = np.array([int(col.split("_")[1]) for col in cols]) * 1e6
    mat = df[cols].values.astype(float)
    return freqs, mat


def _load_tsky(path):
    """Read a Tsky(freq) csv -> (freq_Hz sorted, Tsky_K).  Column/unit auto-detected."""
    df = pd.read_csv(path)
    try:                                         # headerless file
        [float(c) for c in df.columns]
        df = pd.read_csv(path, header=None)
    except ValueError:
        pass
    low = {str(c).lower().strip(): c for c in df.columns}
    fcol = next((c for k, c in low.items() if "freq" in k or k in ("nu", "f", "mhz", "ghz")),
                df.columns[0])
    tcol = next((c for k, c in low.items() if "tsky" in k or "t_sky" in k or k in ("t", "temp")),
                next(c for c in df.columns if c != fcol))
    f = df[fcol].values.astype(float)
    T = df[tcol].values.astype(float)
    name = str(fcol).lower()
    if "ghz" in name:   f = f * 1e9
    elif "mhz" in name: f = f * 1e6
    elif "hz" in name:  pass
    elif f.max() < 50:  f = f * 1e9
    elif f.max() < 1e5: f = f * 1e6
    o = np.argsort(f)
    return f[o], T[o]


def _tsky_func(key):
    """Return Tsky(nu_Hz) for telescope `key` (np.interp, clamped at the file edges)."""
    if not USE_TSKY_FILES:
        const = TELESCOPE_PARAMS[key]["Tsky"]
        return lambda nu: np.full_like(np.asarray(nu, dtype=float), const)
    f, T = _load_tsky(os.path.join(TSKY_DIR, TSKY_FILES[key]))
    return lambda nu: np.interp(np.asarray(nu, dtype=float), f, T)


def prepare(dz):
    """Everything that does not depend on (alpha_int, L_STAR, ALPHA_SCH)."""
    if _STATE.get("dz") == dz:
        return _STATE
    _STATE.clear()
    cosmo = _get_cosmo()
    z = np.arange(Z_MIN, Z_MAX, dz)
    zmin, zmax = z - 0.5 * dz, z + 0.5 * dz
    DH_Mpc = 299792.458 / cosmo.H0.value

    if USE_COSMOLOGY:
        dl = cosmo.luminosity_distance(z).value * 1000.0            # kpc
        dt = (cosmo.age(zmin).value - cosmo.age(zmax).value) * 1e9  # yr
        dvol_full = cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value
    else:
        dl = DH_Mpc * z * 1000.0
        dr_Mpc = DH_Mpc * dz
        dt = dr_Mpc * 3.0857e22 / c / (3600.0 * 24.0 * 365.0)
        dvol_full = None
    psi = (0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)) if USE_SFR else np.full_like(z, 0.015)
    DM = 1000 * z
    wi_z = 0.001 * (1 + z)

    fast_freqs, fast_mat = _load_fast_beams()
    tel = {}
    for key in TEL_ORDER:
        p = TELESCOPE_PARAMS[key]
        D, BW, nu0, Trec = p["D"], p["BW"], p["nu_cen"], p["Trec"]
        n_comp = len(fast_freqs) if key == "25d0" else p["n_comp"]
        BW_comp = BW / n_comp
        nu_min, nu_max = nu0 - BW / 2, nu0 + BW / 2
        BW_chan = BW / p["n_chan"]
        nu_array = fast_freqs.copy() if key == "25d0" else np.linspace(nu_min, nu_max, n_comp + 1)
        freq_powers = np.array([(nu_min * BW_comp * (i + 1)) ** (-4) for i in range(n_comp)])
        tsky_fn = _tsky_func(key)
        Tsys_arr = Trec + tsky_fn(nu_array)                 # per frequency component
        Tsys_min = min(Tsys_arr.min(), Trec + float(tsky_fn(nu0)))   # best case for L_min
        theta_max = 2 * 1.22 * c / D / nu_min
        fwhm_deg = np.degrees(1.22 * (c / nu_min) / D)
        area_tel = np.pi * (2.0 * fwhm_deg) ** 2
        coeff = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c ** 2)
        theta_hp = np.sqrt(np.log(2) / (coeff * nu0 ** 2))

        if key == "5d0":
            G0_max = max(G0_PKS)
            G0 = None
            beam_cache = None
        elif key == "25d0":
            interp = interp1d(fast_freqs, fast_mat, kind="linear", axis=1,
                              bounds_error=False, fill_value="extrapolate")
            beam_cache = np.clip(interp(nu_array), 0, fast_mat.max(axis=1, keepdims=True))
            G0_max = np.clip(interp(np.array([nu0])), 0, fast_mat.max(axis=1, keepdims=True)).max()
            G0 = None
        else:
            G0 = np.pi * D * D / 4 * p["eta"] / 2.0 / k_B * 1e-26
            if key == "d0_ics":
                G0 = G0 * np.sqrt(p["n_dishes"])            # incoherent sum of 36 dishes
            G0_max = G0
            beam_cache = None

        # best-case gain on axis at nu0 -> minimum detectable flux density
        S_min = SNR_threshold * Tsys_min / (G0_max * np.sqrt(BW * t_obs * Np))
        L_min_base = S_min * dl ** 2                       # times z_factor later

        # observed-width factor per z-bin
        if USE_SMEARING:
            w_DM = 8.3e-6 * DM * (BW_chan / 1e6) / ((nu0 / 1e9) ** 3)
            comp = (1.9e-7 * DM[:, None] ** 1.5 * (1 + DM[:, None] ** 3 * 3.55e-5) * 3
                    * freq_powers[None, :] * 1e-3)
            w_scatter = np.sqrt(np.sum(comp ** 2, axis=1) / comp.shape[1])
            w_samp = p["tsamp"] * 1e-3
        else:
            w_DM = w_scatter = np.zeros_like(z)
            w_samp = 0.0
        w_obs = np.sqrt(wi_z ** 2 + w_DM ** 2 + w_scatter ** 2 + w_samp ** 2)

        if USE_COSMOLOGY:
            dvol = dvol_full * (area_tel / (4.0 * np.pi * (180.0 / np.pi) ** 2))
        else:
            area_sr = area_tel * (np.pi / 180.0) ** 2
            dvol = ((DH_Mpc * z) ** 2) * (DH_Mpc * dz) * area_sr
        # NOTE: dvol and dt are both proportional to dz, so the summed n_* scales with dz.
        # Rescale so the absolute counts match the notebook's dz=0.001 whatever --dz is used.
        nfrb_sfr = psi * dvol * dt * fscale * (DZ_REF / dz)

        tel[key] = dict(
            D=D, BW=BW, nu0=nu0, Trec=Trec, n_comp=n_comp, nu_array=nu_array,
            theta_max=theta_max, theta_hp=theta_hp, coeff=coeff, G0=G0, beam_cache=beam_cache,
            L_min_base=L_min_base, wi_over_wobs=wi_z / w_obs,
            S_SNR_CONST=np.sqrt(BW_comp * t_obs * Np) / Tsys_arr,   # array, one per component
            nfrb_sfr=nfrb_sfr,
        )
    _STATE.update(dz=dz, z=z, dl=dl, tel=tel)
    return _STATE


# ============================================================================
# ONE (telescope, parameter-set) SIMULATION
# ============================================================================
def schechter_tables(L_star, alpha):
    """Full Schechter CDF on a log grid (built once per L_STAR/ALPHA_SCH)."""
    L = np.logspace(np.log10(L_MIN_SCH), np.log10(L_MAX_SCH), 5000)
    phi = (L / L_star) ** alpha * np.exp(-L / L_star)
    cdf = _cumtrapz()(phi, L, initial=0)
    cdf /= cdf[-1]
    return np.log10(L), cdf


def n_detected_weighted(tel_key, alpha_int, L_star, alpha_sch, dz):
    st = prepare(dz)
    t = st["tel"][tel_key]
    z, dl = st["z"], st["dl"]
    logL, cdf = schechter_tables(L_star, alpha_sch)
    rng = np.random.default_rng(SEED)

    z_factor = (1 + z) ** (-1 - alpha_int) if USE_SPECTRAL_INDEX else np.ones_like(z)
    L_min_raw = t["L_min_base"] * z_factor
    L_min = np.clip(L_min_raw, L_MIN_SCH, L_MAX_SCH)
    c0 = np.interp(np.log10(L_min), logL, cdf)       # CDF at L_min(z)
    N_ratio = 1.0 - c0
    nu_array, nu0 = t["nu_array"], t["nu0"]
    nu_scaling = (nu_array / nu0) ** alpha_int if USE_SPECTRAL_INDEX else np.ones_like(nu_array)
    nu2 = nu_array[None, :] ** 2

    total = 0.0
    w_z = np.zeros(len(z))            # weighted FRB count per z-bin -> DM distribution
    n_fail_run = 0
    for i in range(len(z)):
        if L_min_raw[i] >= L_MAX_SCH or N_ratio[i] <= 0:      # nothing can be detected
            n_fail_run += 1
            if EARLY_STOP_FAILS and n_fail_run >= EARLY_STOP_FAILS:
                break
            continue
        norm = 1.0 / (dl[i] ** 2 * z_factor[i])
        n_det, n_gen = 0, 0
        while n_det < TARGET_DETECTIONS and n_gen < MAX_ITERATIONS:
            n = min(BATCH_SIZE, MAX_ITERATIONS - n_gen)
            theta_b = t["theta_max"] * np.sqrt(rng.random(n))
            angle_ok = theta_b <= 4.0 * t["theta_hp"]
            u = c0[i] + rng.random(n) * (1.0 - c0[i])           # truncated Schechter draw
            lum = 10 ** np.interp(u, cdf, logL)
            S0 = lum * norm * t["wi_over_wobs"][i]
            S_nu = S0[:, None] * nu_scaling[None, :]
            theta_g = np.where(angle_ok, theta_b, 0.0) if USE_BEAM else np.zeros(n)
            if tel_key == "5d0":
                G0_b = rng.choice(G0_PKS, size=n)[:, None]
            elif tel_key == "25d0":
                G0_b = t["beam_cache"][rng.integers(t["beam_cache"].shape[0], size=n)]
            else:
                G0_b = t["G0"]
            G_nu = G0_b * np.exp(-(t["coeff"] * theta_g[:, None] ** 2) * nu2)
            snr = (S_nu * G_nu * t["S_SNR_CONST"][None, :]).sum(axis=1) / np.sqrt(t["n_comp"])
            hits = np.where(angle_ok & (snr > SNR_threshold))[0]
            need = TARGET_DETECTIONS - n_det
            if len(hits) >= need:
                n_det += need
                n_gen += int(hits[need - 1]) + 1
            else:
                n_det += len(hits)
                n_gen += n
        ok = n_det == TARGET_DETECTIONS or (KEEP_PARTIAL and n_det > 0)
        if ok:
            n_fail_run = 0
            w_z[i] = (n_det / n_gen) * N_ratio[i] * t["nfrb_sfr"][i]
            total += w_z[i]
        else:
            n_fail_run += 1
            if EARLY_STOP_FAILS and n_fail_run >= EARLY_STOP_FAILS:
                break
    return total, w_z


COLS = ["alpha_intrinsic", "L_STAR", "ALPHA_SCH", "n_ASKAP_FE", "n_ASKAP_ICS", "n_PKS", "n_FAST",
        "ASKAP_FE_PKS_ratio", "ASKAP_ICS_PKS_ratio", "PKS_FAST_ratio",
        "ASKAP_FE_FAST_ratio", "ASKAP_ICS_FAST_ratio", "ASKAP_ICS_FE_ratio"]


def run_combo(args):
    alpha_int, L_star, alpha_sch, dz = args
    res = {key: n_detected_weighted(key, alpha_int, L_star, alpha_sch, dz) for key in TEL_ORDER}
    n = {key: res[key][0] for key in TEL_ORDER}
    weights = {key: res[key][1] for key in TEL_ORDER}
    nE, nI, nP, nF = n["d0"], n["d0_ics"], n["5d0"], n["25d0"]
    div = lambda a, b: a / b if b > 0 else np.nan
    row = dict(alpha_intrinsic=alpha_int, L_STAR=L_star, ALPHA_SCH=alpha_sch,
               n_ASKAP_FE=nE, n_ASKAP_ICS=nI, n_PKS=nP, n_FAST=nF,
               ASKAP_FE_PKS_ratio=div(nE, nP), ASKAP_ICS_PKS_ratio=div(nI, nP),
               PKS_FAST_ratio=div(nP, nF),
               ASKAP_FE_FAST_ratio=div(nE, nF), ASKAP_ICS_FAST_ratio=div(nI, nF),
               ASKAP_ICS_FE_ratio=div(nI, nE))
    return row, weights, prepare(dz)["z"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dz", type=float, default=DZ_DEFAULT)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, OUT_CSV)
    combos = [(ai, L, As, a.dz) for As, L, ai in
              itertools.product(ALPHA_SCH_GRID, L_STAR_GRID, ALPHA_INT_GRID)]
    prepare(a.dz)                                   # fail early if a Tsky file can't be read
    rows, t0 = [], time.time()
    with Pool(a.procs) as pool:
        for k, (row, weights, z) in enumerate(pool.imap(run_combo, combos), 1):
            rows.append(row)
            print(f"[{k}/{len(combos)}] a_int={row['alpha_intrinsic']:g} "
                  f"L*={row['L_STAR']:.0e} a_sch={row['ALPHA_SCH']:g}  "
                  f"nFE={row['n_ASKAP_FE']:.1f} nICS={row['n_ASKAP_ICS']:.1f} "
                  f"nP={row['n_PKS']:.1f} nF={row['n_FAST']:.1f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows, columns=COLS).to_csv(csv_path, index=False)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
