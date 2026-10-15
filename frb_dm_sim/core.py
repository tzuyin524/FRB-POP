"""The shared Monte-Carlo pipeline (identical for every telescope)."""
import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

try:                                    # scipy >= 1.6
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:                     # very old scipy
    from scipy.integrate import cumtrapz as _cumtrapz

from beams import C
from cosmo_table import get_zgrid
from params import SimParams


def schechter(L, L_star, alpha):
    x = L / L_star
    return x ** alpha * np.exp(-x)


def truncated_schechter_sampler(L_min, L_max, L_star, alpha):
    """Inverse-CDF sampler for the Schechter function truncated to [L_min, L_max]."""
    L_grid = np.logspace(np.log10(L_min), np.log10(L_max), 2000)
    cdf = _cumtrapz(schechter(L_grid, L_star, alpha), L_grid, initial=0)
    cdf = cdf / cdf[-1]
    return interp1d(cdf, L_grid, kind="linear", bounds_error=False, fill_value=(L_min, L_max))


@dataclass
class SurveyResult:
    key: str
    label: str
    out_name: str
    params: SimParams
    results: list = field(default_factory=list)      # {z, DM, n_total_population}
    events: list = field(default_factory=list)       # per detected FRB (fluence, gain_ratio, snr, ...)
    elapsed_s: float = 0.0

    @property
    def dm_distribution(self):
        """(N, 2) array: col 0 = DM [pc cm^-3], col 1 = weighted number of FRBs. Same as the old .npy."""
        return np.column_stack([[r["DM"] for r in self.results],
                               [r["n_total_population"] for r in self.results]])

    @property
    def n_frb(self):
        return float(sum(r["n_total_population"] for r in self.results))

    def save(self, out_dir, tag=""):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"DM_distribution_{self.out_name}{tag}.npy")
        np.save(path, self.dm_distribution)
        return path


def run_survey(tel, params=None, seed=42, verbose=True):
    """Simulate one survey. `tel` is a Telescope (telescopes.get_telescope("pks"))."""
    p = (params or SimParams()).with_overrides(**tel.overrides)
    zg = get_zgrid(p)
    z_arr = zg.z
    t0 = time.time()
    np.random.seed(seed)

    # ---- frequency set-up -----------------------------------------------------------------
    nu0 = tel.nu_cen
    nu_min, nu_max = nu0 - tel.BW / 2, nu0 + tel.BW / 2
    nu_array, n_comp = tel.beam.frequency_grid(tel)
    BW_comp = tel.BW / n_comp
    BW_chan = tel.BW / tel.n_chan
    tsky = tel.tsky_curve()
    Tsky_nu = tsky.at(nu_array)
    Tsky_0 = float(tsky.at(np.array([nu0]))[0])
    freq_powers_scatter = np.array([(nu_min * BW_comp * (i + 1)) ** (-4) for i in range(n_comp)])

    geo = tel.beam.prepare(tel, nu_array, nu0)
    area_tel, f_area, G0_max = geo["area_deg2"], geo["f_area"], geo["G0_max"]

    if verbose:
        print(f"[{tel.key}] D={tel.D} m, {tel.beam.desc}")
        print(f"  n_comp={n_comp}, nu: {nu_array[0]/1e6:.0f}-{nu_array[-1]/1e6:.0f} MHz ({len(nu_array)} pts), "
              f"sky area {area_tel:.4f} deg2, f_area {f_area:.4f}")
        print(f"  Trec={tel.Trec} K, Tsky {Tsky_nu.min():.1f}-{Tsky_nu.max():.1f} K "
              f"({Tsky_0:.1f} K at nu0), BW={tel.BW/1e6:.0f} MHz")
        tsky.check_against(nu_array)

    # ---- L_min(z): faintest burst detectable on the best-gain axis ----------------------------
    z_factor_all = (1 + z_arr) ** (-1 - p.alpha_intrinsic) if p.USE_SPECTRAL_INDEX else np.ones_like(z_arr)
    S_min = p.SNR_threshold * (tel.Trec + Tsky_0) / (G0_max * np.sqrt(tel.BW * p.t_obs * p.Np))
    L_min_all = np.clip(S_min * zg.dl_kpc ** 2 * z_factor_all, p.L_MIN_SCH, p.L_MAX_SCH)

    # ---- luminosity-function helpers ------------------------------------------------------------
    use_std = p.LUMINOSITY_FUNCTION.lower() == "std"
    L_grid = np.logspace(np.log10(p.L_MIN_SCH), np.log10(p.L_MAX_SCH), 5000)
    cdf = _cumtrapz(schechter(L_grid, p.L_STAR, p.ALPHA_SCH), L_grid, initial=0)
    cdf /= cdf[-1]
    cdf_interp = interp1d(L_grid, cdf, kind="linear", bounds_error=False, fill_value=(0.0, 1.0))

    def n_ratio_for(L_min_z):
        if use_std:
            return 1.0 if p.L_std >= L_min_z else 0.0
        return 1.0 - cdf_interp(L_min_z)

    S_SNR_CONST = np.sqrt(BW_comp * p.t_obs * p.Np) / (tel.Trec + Tsky_nu)       # (n_nu,)
    nu_scaling = (nu_array / nu0) ** p.alpha_intrinsic if p.USE_SPECTRAL_INDEX else np.ones_like(nu_array)
    sampler_cache = {}
    out = SurveyResult(tel.key, tel.label, tel.out_name, p)

    # ---- loop over redshift -------------------------------------------------------------------------
    for z_idx, z in enumerate(z_arr):
        L_min_z = L_min_all[z_idx]
        if use_std:
            if L_min_z > p.L_std:
                break
            sampler = lambda u: p.L_std
        else:
            if L_min_z > p.L_BREAK_SCH:
                break
            if L_min_z not in sampler_cache:
                sampler_cache[L_min_z] = truncated_schechter_sampler(L_min_z, p.L_MAX_SCH, p.L_STAR, p.ALPHA_SCH)
            sampler = sampler_cache[L_min_z]

        DM = 1000 * z
        wi_z = 0.001 * (1 + z)
        if p.USE_SMEARING:
            w_DM = 8.3e-6 * DM * (BW_chan / 1e6) / ((nu0 / 1e9) ** 3)
            w_sc = 1.9e-7 * (DM ** 1.5) * (1 + (DM ** 3) * 3.55e-5) * 3 * freq_powers_scatter * 1e-3
            w_scatter = np.sqrt(np.sum(w_sc ** 2) / len(w_sc))
            w_sampling = tel.tsamp * 1e-3
        else:
            w_DM = w_scatter = w_sampling = 0.0
        w_obs = np.sqrt(wi_z ** 2 + w_DM ** 2 + w_scatter ** 2 + w_sampling ** 2)

        dl = zg.dl_kpc[z_idx]
        z_factor = z_factor_all[z_idx]

        n_generated, detected = 0, []
        while len(detected) < p.TARGET_DETECTIONS and n_generated < p.MAX_DRAWS_PER_BIN:
            n = min(p.BATCH_SIZE, p.MAX_DRAWS_PER_BIN - n_generated)
            geom = tel.beam.draw(n)                                           # geometry first (RNG order!)
            lum = np.asarray(sampler(np.random.uniform(0, 1, n)), dtype=float)
            if use_std:
                lum = np.full(n, p.L_std, dtype=float)
            S0 = (lum / (dl ** 2 * z_factor)) * (wi_z / w_obs)
            S_nu = S0[:, None] * nu_scaling[None, :]
            G_nu, gain_ratio, ok = tel.beam.gain(geom, n, p.USE_BEAM)
            snr = (S_nu * G_nu * S_SNR_CONST).sum(axis=1) / np.sqrt(n_comp)
            hits = np.where(ok & (snr > p.SNR_threshold))[0]

            need = p.TARGET_DETECTIONS - len(detected)
            if len(hits) >= need:
                used = hits[:need]
                n_generated += int(used[-1]) + 1
            else:
                used = hits
                n_generated += n
            for i in used:
                fluence = (lum[i] / (dl ** 2 * z_factor)) * (wi_z * 1000.0)
                detected.append((lum[i], fluence, gain_ratio[i], snr[i]))

        if len(detected) == p.TARGET_DETECTIONS:
            detection_rate = (len(detected) / n_generated) * f_area
            psi = (0.015 * ((1 + z) ** 2.7) / (1 + ((1 + z) / 2.9) ** 5.6)) if p.USE_SFR else 0.015
            nfrb_sfr = psi * zg.dvol(z_idx, area_tel) * zg.dt_yr[z_idx] * p.fscale
            n_pop = detection_rate * n_ratio_for(L_min_z) * nfrb_sfr
            out.results.append({"z": z, "DM": DM, "n_total_population": n_pop})
            w = n_pop / p.TARGET_DETECTIONS
            for lum_i, fl, gr, s in detected:
                out.events.append({"DM": DM, "z": z, "fluence": fl, "weight": w, "lum_true": lum_i,
                                   "gain_ratio": gr, "fluence_obs": fl * gr, "lum_inferred": lum_i * gr,
                                   "snr": s})
            if verbose and z_idx % 500 == 0:
                print(f"  z={z:.2f}: pop {n_pop:.1f}")

    out.elapsed_s = time.time() - t0
    if not out.results:
        raise RuntimeError(f"[{tel.key}] no redshift bin reached TARGET_DETECTIONS; check the beam "
                           f"geometry / csv inputs printed above.")
    if verbose:
        print(f"  done: {len(out.results)} z-bins, n_frb (weighted) = {out.n_frb:.1f}, {out.elapsed_s:.1f} s")
    return out
