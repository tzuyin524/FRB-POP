#!/usr/bin/env python3
"""
DM percentiles (25%, median, 75%) of the detected FRB population vs dish diameter.

Ported from DM_comparison_mc_fluence_DM.ipynb (single-dish "d0" branch).
Only the dish diameter D changes (5, 6, ..., 300 m, step 1 m); every other
telescope parameter is fixed to the values below.

Output: DM_percentiles_vs_dish_size.csv
        columns: dish_size, DM_25, DM_median, DM_75
Run:    python dm_vs_dish_size.py            (uses all CPU cores)
"""
import os
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DIAMETERS = np.arange(5, 301, 1)          # 5, 6, ..., 300 m
OUT_CSV = "DM_percentiles_vs_dish_size.csv"

PARAMS = dict(eta=0.7, Trec=22, Tsky=3, BW=300e6, nu_cen=1.3e9,
              n_comp=16, n_chan=2048, tsamp=1e-4)
alpha_intrinsic = 0.0                     # flat spectrum
WI_INTRINSIC = 1e-3                       # intrinsic width 1 ms (times (1+z))

USE_COSMOLOGY = USE_SFR = USE_SMEARING = USE_SPECTRAL_INDEX = USE_BEAM = True

k_B = 1.38e-23
c = 3.0e8
Np = 2
SNR_threshold = 8
t_obs = 0.001
TARGET_DETECTIONS = 10
BATCH_SIZE = 4096
MAX_ITER = 10000
SEED = 42

# luminosity function (Schechter)
L_STAR, ALPHA_SCH = 3e13, -1.2
L_MIN_SCH, L_MAX_SCH = 1e6, 1e16

# redshift grid
dz, z_min, z_max = 0.001, 0.01, 5.0
z_array = np.arange(z_min, z_max, dz)

fscale = 100 * 1e-7
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ----------------------------------------------------------------------------
# Quantities independent of dish size (computed once)
# ----------------------------------------------------------------------------
def schechter(L, L_star, alpha):
    x = L / L_star
    return x ** alpha * np.exp(-x)

_Lg = np.logspace(np.log10(L_MIN_SCH), np.log10(L_MAX_SCH), 5000)
_cdf = cumtrapz(schechter(_Lg, L_STAR, ALPHA_SCH), _Lg, initial=0)
_cdf /= _cdf[-1]
_cdf_interp = interp1d(_Lg, _cdf, bounds_error=False, fill_value=(0.0, 1.0))

def truncated_sampler(L_min):
    Lg = np.logspace(np.log10(L_min), np.log10(L_MAX_SCH), 2000)
    cdf = cumtrapz(schechter(Lg, L_STAR, ALPHA_SCH), Lg, initial=0)
    cdf /= cdf[-1]
    return interp1d(cdf, Lg, bounds_error=False, fill_value=(L_min, L_MAX_SCH))

def _cosmo_tables():
    zmin_, zmax_ = z_array - 0.5 * dz, z_array + 0.5 * dz
    dl = cosmo.luminosity_distance(z_array).value * 1000.0               # kpc
    dt = (cosmo.age(zmin_).value - cosmo.age(zmax_).value) * 1e9          # yr
    vol = cosmo.comoving_volume(zmax_).value - cosmo.comoving_volume(zmin_).value  # Mpc^3 (all sky)
    psi = 0.015 * (1 + z_array) ** 2.7 / (1 + ((1 + z_array) / 2.9) ** 5.6)
    return dl, dt, vol, psi

DL, DT, VOL_ALLSKY, PSI = _cosmo_tables()

P = PARAMS
nu0 = P["nu_cen"]; BW = P["BW"]; n_comp = P["n_comp"]
Trec = P["Trec"]; Tsky = P["Tsky"]
BW_comp = BW / n_comp
nu_min = nu0 - BW / 2
nu_array = np.linspace(nu_min, nu0 + BW / 2, n_comp + 1)   # as in the notebook (n_comp+1 points)
BW_chan = BW / P["n_chan"]
freq_powers_scatter = np.array([(nu_min * BW_comp * (i + 1)) ** (-4) for i in range(n_comp)])
nu_scaling = (nu_array / nu0) ** alpha_intrinsic

# per-z observed-width factor (independent of D)
DMs = 1000.0 * z_array
wi_z = WI_INTRINSIC * (1 + z_array)
w_DM = 8.3e-6 * DMs * (BW_chan / 1e6) / (nu0 / 1e9) ** 3
w_scat = np.array([
    np.sqrt(np.mean((1.9e-7 * d ** 1.5 * (1 + d ** 3 * 3.55e-5) * 3 * freq_powers_scatter * 1e-3) ** 2))
    for d in DMs])
w_samp = P["tsamp"] * 1e-3          # same convention as the notebook
w_obs = np.sqrt(wi_z ** 2 + w_DM ** 2 + w_scat ** 2 + w_samp ** 2) if USE_SMEARING else wi_z
z_factor = (1 + z_array) ** (-1 - alpha_intrinsic)
S_SNR_CONST = np.sqrt(BW_comp * t_obs * Np) / (Trec + Tsky)

# ----------------------------------------------------------------------------
def weighted_percentiles(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(qs, cw, x)

def run_one(D):
    """Return (DM25, DM50, DM75) for dish diameter D [m]."""
    rng = np.random.RandomState(SEED)
    G0 = np.pi * D * D / 4 * P["eta"] / 2.0 / k_B * 1e-26
    coeff = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c ** 2)
    theta_max = 2 * 1.22 * c / D / nu_min
    theta_hp = np.sqrt(np.log(2) / (coeff * nu0 ** 2))
    area_deg2 = np.pi * (2.0 * np.degrees(1.22 * (c / nu_min) / D)) ** 2
    sky_frac = area_deg2 / (4.0 * np.pi * (180.0 / np.pi) ** 2)

    S_min = SNR_threshold * (Trec + Tsky) / (G0 * np.sqrt(BW * t_obs * Np))
    L_min_all = np.clip(S_min * DL ** 2 * z_factor, L_MIN_SCH, L_MAX_SCH)

    cache, dm_out, w_out = {}, [], []
    for i, z in enumerate(z_array):
        L_min = L_min_all[i]
        # if L_min > 8.952e15:
        #     break
        if L_min not in cache:
            cache[L_min] = truncated_sampler(L_min)
        sampler = cache[L_min]

        n_gen, n_det = 0, 0
        while n_det < TARGET_DETECTIONS and n_gen < MAX_ITER:
            n = min(BATCH_SIZE, MAX_ITER - n_gen)
            theta_b = theta_max * np.sqrt(rng.random_sample(n))
            angle_ok = theta_b <= 4.0 * theta_hp
            lum = np.asarray(sampler(rng.uniform(0, 1, n)), dtype=float)
            S0 = lum / (DL[i] ** 2 * z_factor[i]) * (wi_z[i] / w_obs[i])
            S_nu = S0[:, None] * nu_scaling[None, :]
            th = np.where(angle_ok, theta_b, 0.0) if USE_BEAM else np.zeros(n)
            G_nu = G0 * np.exp(-coeff * th[:, None] ** 2 * nu_array[None, :] ** 2)
            snr = (S_nu * G_nu * S_SNR_CONST).sum(axis=1) / np.sqrt(n_comp)
            hits = np.where(angle_ok & (snr > SNR_threshold))[0]
            need = TARGET_DETECTIONS - n_det
            if len(hits) >= need:
                n_gen += int(hits[need - 1]) + 1
                n_det = TARGET_DETECTIONS
            else:
                n_gen += n
                n_det += len(hits)

        if n_det == TARGET_DETECTIONS:
            rate = n_det / n_gen
            N_ratio = 1.0 - _cdf_interp(L_min)
            n_frb = PSI[i] * VOL_ALLSKY[i] * sky_frac * DT[i] * fscale
            dm_out.append(DMs[i])
            w_out.append(rate * N_ratio * n_frb)

    if not w_out or sum(w_out) <= 0:
        return D, np.nan, np.nan, np.nan
    q = weighted_percentiles(np.array(dm_out), np.array(w_out), [0.25, 0.5, 0.75])
    return D, q[0], q[1], q[2]

if __name__ == "__main__":
    t0 = time.time()
    nproc = os.cpu_count() or 1
    print(f"Running {len(DIAMETERS)} dish sizes on {nproc} processes ...")
    rows = []
    with Pool(nproc) as pool:
        for r in pool.imap(run_one, DIAMETERS):
            rows.append(r)
            if len(rows) % 10 == 0:
                print(f"  {len(rows)}/{len(DIAMETERS)} done  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["dish_size", "DM_25", "DM_median", "DM_75"])
    df["dish_size"] = df["dish_size"].astype(int)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} in {time.time()-t0:.0f}s")
