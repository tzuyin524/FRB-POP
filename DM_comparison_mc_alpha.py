"""
FRB Population Monte-Carlo Pipeline
====================================
Refactored from `DM_comparison_mc_alpha.ipynb` into a single, config/CLI
driven script. The underlying physics (radiometer equation, Gaussian beam,
Schechter/standard-candle luminosity sampling, DM smearing, SFR-weighted
volume rates, cosmological vs. Euclidean distances) is unchanged from the
notebook.

For every (dish size, intrinsic alpha) combination it simulates the FRB
population, then writes ONE file per combination containing the paired
DM-distribution array and observed (fitted) spectral-index array:

    DM_alpha/DM_alpha_{dish_size}_{alpha_intrinsic}.nparray

(the file is numpy .npz content under a custom extension -- read it back
with `np.load(path)`, see `load_result()` below).

Usage
-----
Run with all defaults (dish sizes 12/60/300 m, alpha=-1.5):
    python frb_pipeline.py

Tune from the command line:
    python frb_pipeline.py --dish-sizes 12 60 300 --alphas -1.5 -2.0 \
        --lum-kind schechter --L-star 1e15 --alpha-sch -1.5 \
        --L-min-sch 1e10 --L-max-sch 1e16 \
        --z-max 5.0 --dz 0.01 --target-detections 10 \
        --out-dir DM_alpha

Or point to a JSON config file (CLI flags override anything in it):
    python frb_pipeline.py --config my_config.json

Example my_config.json:
{
  "dish_sizes": [12, 60, 300],
  "alphas_intrinsic": [-1.5, -2.0],
  "lum_kind": "schechter",
  "L_star": 1e15,
  "alpha_sch": -1.5,
  "L_min_sch": 1e10,
  "L_max_sch": 1e16,
  "L_std": 2.4e14,
  "use_cosmology": true,
  "use_sfr": true,
  "use_smearing": true,
  "use_spectral_index": true,
  "use_beam": true,
  "z_min": 0.01,
  "z_max": 5.0,
  "dz": 0.01,
  "snr_threshold": 10.0,
  "t_obs": 0.001,
  "target_detections": 10,
  "fscale": 2e-8,
  "seed": 42,
  "out_dir": "DM_alpha"
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from astropy.cosmology import FlatLambdaCDM

try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:  # older scipy
    from scipy.integrate import cumtrapz as _cumtrapz
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
K_BOLTZMANN = 1.38e-23
C_LIGHT = 3.0e8


# ===========================================================================
# CONFIG OBJECTS -- every tunable knob lives here
# ===========================================================================

@dataclass
class TelescopeConfig:
    """One telescope / instrument, defined entirely by dish diameter `D`
    plus a shared set of receiver/backend parameters (fixed across the dish
    sweep, matching the notebook's "differs only by dish size" setup)."""

    D: float                    # dish diameter [m]  (the tunable knob)
    eta: float = 0.6            # aperture efficiency
    Trec: float = 22.0          # receiver temperature [K]
    Tsky: float = 3.0           # sky temperature [K]
    BW: float = 300e6           # bandwidth [Hz]
    nu_cen: float = 1.4e9       # centre frequency [Hz]
    n_comp: int = 16            # channels used for spectral-index fitting
    n_chan: int = 1024          # channels used for DM smearing
    tsamp: float = 1.0e-3       # sampling time [s]

    def gain(self) -> float:
        return np.pi * self.D ** 2 / 4 * self.eta / 2.0 / K_BOLTZMANN * 1e-26

    def nu_min(self) -> float:
        return self.nu_cen - self.BW / 2

    def nu_array(self) -> np.ndarray:
        return np.linspace(self.nu_min(), self.nu_cen + self.BW / 2, self.n_comp + 1)

    def bw_comp(self) -> float:
        return self.BW / self.n_comp

    def bw_chan(self) -> float:
        return self.BW / self.n_chan

    def theta_max(self) -> float:
        return 2 * 1.22 * C_LIGHT / self.D / self.nu_min()

    def sky_area_deg2(self) -> float:
        fwhm_rad = 1.22 * (C_LIGHT / self.nu_min()) / self.D
        r_deg = 2.0 * (fwhm_rad * 180 / np.pi)
        return np.pi * r_deg ** 2


@dataclass
class LuminosityConfig:
    """kind="schechter" uses a truncated Schechter function; kind="std" uses
    a standard candle at L_std."""

    kind: str = "schechter"
    L_star: float = 1e15
    alpha_sch: float = -1.5
    L_min_sch: float = 1e10
    L_max_sch: float = 1e16
    L_std: float = 2.4e14


@dataclass
class PhysicsFlags:
    use_cosmology: bool = True
    use_sfr: bool = True
    use_smearing: bool = True
    use_spectral_index: bool = True
    use_beam: bool = True


@dataclass
class SimulationConfig:
    z_min: float = 0.01
    z_max: float = 5.0
    dz: float = 0.01
    H0: float = 70.0
    Om0: float = 0.3
    SNR_threshold: float = 10.0
    t_obs: float = 0.001
    target_detections: int = 10
    fscale: float = 0.2e-7
    max_iterations: int = 1_000_000
    seed: int = 42

    def z_array(self) -> np.ndarray:
        return np.arange(self.z_min, self.z_max, self.dz)

    def cosmology(self) -> FlatLambdaCDM:
        return FlatLambdaCDM(H0=self.H0, Om0=self.Om0)


# ===========================================================================
# PHYSICS FUNCTIONS
# ===========================================================================

def gaussian_beam_gain(nu, G0, theta, D, c=C_LIGHT):
    coeff = (theta ** 2 * D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * c ** 2)
    return G0 * np.exp(-coeff * nu ** 2)


def schechter_function(L, L_star, alpha, phi_star=1.0):
    x = L / L_star
    return phi_star * x ** alpha * np.exp(-x)


def create_truncated_schechter_sampler(L_min, L_max, L_star, alpha):
    L_grid = np.logspace(np.log10(L_min), np.log10(L_max), 2000)
    phi_grid = schechter_function(L_grid, L_star, alpha)
    cdf = _cumtrapz(phi_grid, L_grid, initial=0)
    cdf = cdf / cdf[-1]
    return interp1d(cdf, L_grid, kind="linear", bounds_error=False,
                     fill_value=(L_min, L_max))


def std_sampler(L_candle: float) -> Callable[[float], float]:
    def sampler(u):
        return L_candle
    return sampler


# ===========================================================================
# RESULT
# ===========================================================================

@dataclass
class ZBinResult:
    z: float
    DM: float
    n_total_population: float
    alpha_inferred: np.ndarray
    alpha_error: np.ndarray
    snr_coherent: np.ndarray


@dataclass
class TelescopeResult:
    D: float
    alpha_intrinsic: float
    bins: List[ZBinResult] = field(default_factory=list)
    alpha_inferred: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_error: np.ndarray = field(default_factory=lambda: np.array([]))
    snr_coherent: np.ndarray = field(default_factory=lambda: np.array([]))

    def dm_per_detection(self) -> np.ndarray:
        """DM repeated once per individual VALID detected FRB -- paired 1:1
        with `alpha_observed()`."""
        dms = []
        for b in self.bins:
            n_valid = int(np.sum(~np.isnan(b.alpha_inferred)))
            dms.extend([b.DM] * n_valid)
        return np.array(dms)

    def alpha_observed(self) -> np.ndarray:
        """Observed (fitted) spectral index per detected FRB, NaNs dropped --
        paired 1:1 with `dm_per_detection()`."""
        return self.alpha_inferred[~np.isnan(self.alpha_inferred)]

    def dm_bins(self) -> np.ndarray:
        return np.array([b.DM for b in self.bins])

    def population_weights(self) -> np.ndarray:
        return np.array([b.n_total_population for b in self.bins])


# ===========================================================================
# CORE SIMULATION  (physics unchanged from the notebook)
# ===========================================================================

def run_single_telescope(
    telescope: TelescopeConfig,
    alpha_intrinsic: float,
    lf: LuminosityConfig,
    flags: PhysicsFlags,
    sim: SimulationConfig,
    verbose: bool = True,
) -> TelescopeResult:
    D = telescope.D
    Trec, Tsky, BW, Np = telescope.Trec, telescope.Tsky, telescope.BW, 2
    n_comp = telescope.n_comp
    nu_cen = telescope.nu_cen
    nu_min = telescope.nu_min()
    nu_array = telescope.nu_array()
    BW_comp = telescope.bw_comp()
    BW_chan = telescope.bw_chan()

    G0 = telescope.gain()
    theta_max = telescope.theta_max()
    area_tel = telescope.sky_area_deg2()

    z_array = sim.z_array()
    dz = sim.dz
    cosmo = sim.cosmology()
    H0_km_s_Mpc = cosmo.H0.value
    DH_Mpc = 299792.458 / H0_km_s_Mpc

    freq_powers_scatter = np.array(
        [(nu_min * BW_comp * (i + 1)) ** (-4) for i in range(n_comp)]
    )

    if verbose:
        print(f"{'='*60}\nDISH SIZE: {D} m   alpha_intrinsic: {alpha_intrinsic}")
        print(f"  G0={G0:.4e}, theta_max={theta_max:.6f} rad, sky_area={area_tel:.4f} deg^2")

    np.random.seed(sim.seed)

    L_min_by_z: Dict[float, float] = {}
    nu0 = nu_cen
    for z in z_array:
        dl = (cosmo.luminosity_distance(z).value * 1000.0 if flags.use_cosmology
              else (DH_Mpc * z) * 1000.0)
        G_max = gaussian_beam_gain(nu0, G0, 0, D, C_LIGHT)
        S_min = sim.SNR_threshold * (Trec + Tsky) / (G_max * np.sqrt(BW * sim.t_obs * Np))
        z_factor = (1 + z) ** (-1 - alpha_intrinsic) if flags.use_spectral_index else 1.0
        L_min = S_min * (dl ** 2) * z_factor
        L_min_by_z[z] = float(np.clip(L_min, lf.L_min_sch, lf.L_max_sch))

    sampler_cache: Dict[float, Callable[[float], float]] = {}

    bins: List[ZBinResult] = []
    alpha_inferred_all: List[float] = []
    alpha_error_all: List[float] = []
    snr_all: List[float] = []

    _INNER_COEFF_CONST = (D ** 2 * 2.355 ** 2) / (2 * 1.22 ** 2 * C_LIGHT ** 2)
    S_SNR_CONST = np.sqrt(BW_comp * sim.t_obs * Np) / (Trec + Tsky)
    nu_scaling = (nu_array / nu0) ** alpha_intrinsic if flags.use_spectral_index else np.ones_like(nu_array)

    for z_idx, z in enumerate(z_array):
        zmin, zmax = z - 0.5 * dz, z + 0.5 * dz
        L_min_z = L_min_by_z[z]

        if lf.kind.lower() == "std":
            if L_min_z > lf.L_std:
                break
            truncated_sampler = std_sampler(lf.L_std)
        else:
            if L_min_z > 8.952e15:
                break
            if L_min_z not in sampler_cache:
                sampler_cache[L_min_z] = create_truncated_schechter_sampler(
                    L_min_z, lf.L_max_sch, lf.L_star, lf.alpha_sch
                )
            truncated_sampler = sampler_cache[L_min_z]

        DM = 1000 * z
        wi_z = 0.001 * (1 + z)
        BW_chan_MHz = BW_chan / 1e6
        nu_cen_GHz = nu_cen / 1e9

        if flags.use_smearing:
            w_DM = 8.3e-6 * DM * BW_chan_MHz / (nu_cen_GHz ** 3)
            w_scatter_comp = (
                1.9e-7 * (DM ** 1.5) * (1 + (DM ** 3) * 3.55e-5) * 3 * freq_powers_scatter * 1e-3
            )
            w_scatter = np.sqrt(np.sum(w_scatter_comp ** 2) / len(w_scatter_comp))
            w_sampling = telescope.tsamp * 1e-3
        else:
            w_DM = w_scatter = w_sampling = 0.0

        w_obs_factor = np.sqrt(wi_z ** 2 + w_DM ** 2 + w_scatter ** 2 + w_sampling ** 2)

        dl = (cosmo.luminosity_distance(z).value * 1000.0 if flags.use_cosmology
              else (DH_Mpc * z) * 1000.0)
        z_factor = (1 + z) ** (-1 - alpha_intrinsic) if flags.use_spectral_index else 1.0
        theta_hp = np.sqrt(np.log(2) / (_INNER_COEFF_CONST * nu0 ** 2))

        shell_alpha_inferred: List[float] = []
        shell_alpha_error: List[float] = []
        shell_snr_coherent: List[float] = []

        n_generated = 0
        n_detected = 0

        while n_detected < sim.target_detections and n_generated < sim.max_iterations:
            n_generated += 1
            theta_frb = theta_max * np.sqrt(np.random.random())
            if theta_frb > 4.0 * theta_hp:
                continue

            lum = float(truncated_sampler(np.random.uniform(0, 1)))
            S0_frb = (lum / (dl ** 2 * z_factor)) * (wi_z / w_obs_factor)
            S_nu = S0_frb * nu_scaling

            if flags.use_beam:
                G_nu = G0 * np.exp(-(_INNER_COEFF_CONST * theta_frb ** 2) * nu_array ** 2)
            else:
                G_nu = G0 * np.exp(-(_INNER_COEFF_CONST * 0.0 ** 2) * nu_array ** 2)

            snr_spectrum = S_nu * G_nu * S_SNR_CONST
            snr_coherent = np.sum(snr_spectrum) / np.sqrt(n_comp)

            if snr_coherent > sim.SNR_threshold:
                n_detected += 1
                shell_snr_coherent.append(snr_coherent)
                try:
                    log_nu = np.log10(nu_array / nu0)
                    log_snr = np.log10(snr_spectrum)
                    sigma_log_snr = 1.0 / (snr_spectrum * np.log(10))
                    weights = 1.0 / sigma_log_snr ** 2
                    coeffs = np.polyfit(log_nu, log_snr, 1, w=weights, cov=True)
                    shell_alpha_inferred.append(coeffs[0][0])
                    shell_alpha_error.append(np.sqrt(coeffs[1][0, 0]))
                except Exception:
                    shell_alpha_inferred.append(np.nan)
                    shell_alpha_error.append(np.nan)

        if n_detected == sim.target_detections:
            detection_rate = n_detected / n_generated

            if lf.kind.lower() == "std":
                N_ratio = 1.0 if lf.L_std >= L_min_z else 0.0
            else:
                L_grid_full = np.logspace(10, 16, 5000)
                phi_grid_full = schechter_function(L_grid_full, lf.L_star, lf.alpha_sch)
                cdf_full = _cumtrapz(phi_grid_full, L_grid_full, initial=0)
                cdf_full /= cdf_full[-1]
                cdf_interp = interp1d(L_grid_full, cdf_full, kind="linear")
                N_ratio = 1.0 - cdf_interp(L_min_z)

            if flags.use_cosmology:
                dt = (cosmo.age(zmin).value - cosmo.age(zmax).value) * 1e9
                dvol = (cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value) * (
                    area_tel / (4.0 * np.pi * ((180.0 / np.pi) ** 2))
                )
            else:
                r_Mpc = DH_Mpc * z
                dr_Mpc = DH_Mpc * dz
                area_sr = area_tel * (np.pi / 180.0) ** 2
                dvol = (r_Mpc ** 2) * dr_Mpc * area_sr
                dt = (dr_Mpc * 3.0857e22) / C_LIGHT / (3600.0 * 24.0 * 365.0)

            psi = (0.015 * ((1 + z) ** 2.7) / (1 + ((1 + z) / 2.9) ** 5.6)) if flags.use_sfr else 0.015
            n_total_population = float(detection_rate * N_ratio * psi * dvol * dt * sim.fscale)

            bins.append(ZBinResult(
                z=z, DM=DM, n_total_population=n_total_population,
                alpha_inferred=np.array(shell_alpha_inferred),
                alpha_error=np.array(shell_alpha_error),
                snr_coherent=np.array(shell_snr_coherent),
            ))
            alpha_inferred_all.extend(shell_alpha_inferred)
            alpha_error_all.extend(shell_alpha_error)
            snr_all.extend(shell_snr_coherent)

            if verbose and z_idx % 50 == 0:
                print(f"  z={z:.2f}: Pop {n_total_population:.1f}, "
                      f"alpha (mean): {np.nanmean(shell_alpha_inferred):.3f}"
                      f"+/-{np.nanstd(shell_alpha_inferred):.3f}")

    if verbose:
        print(f"  -> {len(bins)} redshift bins with {sim.target_detections} detections each\n")

    return TelescopeResult(
        D=D, alpha_intrinsic=alpha_intrinsic, bins=bins,
        alpha_inferred=np.array(alpha_inferred_all),
        alpha_error=np.array(alpha_error_all),
        snr_coherent=np.array(snr_all),
    )


# ===========================================================================
# OUTPUT
# ===========================================================================

def _fmt(x: float) -> str:
    """Filename-safe numeric formatting, e.g. 12.0 -> '12', -1.5 -> '-1.5'."""
    return f"{x:g}"


def save_result(out_dir: Path, res: TelescopeResult) -> Path:
    """Write DM_alpha/DM_alpha_{dish_size}_{alpha_intrinsic}.nparray
    containing the paired DM distribution + observed spectral index arrays
    (readable back with `np.load(path)`, see `load_result`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"DM_alpha_{_fmt(res.D)}_{_fmt(res.alpha_intrinsic)}.nparray"
    with open(path, "wb") as f:
        np.savez(
            f,
            DM=res.dm_per_detection(),
            alpha_observed=res.alpha_observed(),
            dish_size=res.D,
            alpha_intrinsic=res.alpha_intrinsic,
        )
    return path


def load_result(path) -> dict:
    """Read back a file written by `save_result`."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ===========================================================================
# CONFIG LOADING (JSON file + CLI overrides)
# ===========================================================================

DEFAULTS = dict(
    dish_sizes=[12, 60, 300],
    alphas_intrinsic=[-1.5],
    eta=0.6, Trec=22.0, Tsky=3.0, BW=300e6, nu_cen=1.4e9,
    n_comp=16, n_chan=1024, tsamp=1.0e-3,
    lum_kind="schechter",
    L_star=1e15, alpha_sch=-1.5, L_min_sch=1e10, L_max_sch=1e16, L_std=2.4e14,
    use_cosmology=True, use_sfr=True, use_smearing=True,
    use_spectral_index=True, use_beam=True,
    z_min=0.01, z_max=5.0, dz=0.01,
    snr_threshold=10.0, t_obs=0.001, target_detections=10,
    fscale=0.2e-7, seed=42, max_iterations=1_000_000,
    out_dir="DM_alpha",
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FRB population Monte-Carlo pipeline: tune dish size(s) "
                     "and intrinsic alpha, get DM + observed-spectral-index arrays."
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file. CLI flags below override its values.")

    p.add_argument("--dish-sizes", type=float, nargs="+", default=None,
                   help="One or more dish diameters [m], run simultaneously.")
    p.add_argument("--alphas", "--alphas-intrinsic", dest="alphas_intrinsic",
                   type=float, nargs="+", default=None,
                   help="One or more intrinsic spectral indices to sweep.")

    tel = p.add_argument_group("telescope backend params (shared across dish sizes)")
    tel.add_argument("--eta", type=float, default=None)
    tel.add_argument("--Trec", type=float, default=None)
    tel.add_argument("--Tsky", type=float, default=None)
    tel.add_argument("--BW", type=float, default=None)
    tel.add_argument("--nu-cen", type=float, default=None)
    tel.add_argument("--n-comp", type=int, default=None)
    tel.add_argument("--n-chan", type=int, default=None)
    tel.add_argument("--tsamp", type=float, default=None)

    lf = p.add_argument_group("luminosity function")
    lf.add_argument("--lum-kind", choices=["schechter", "std"], default=None)
    lf.add_argument("--L-star", type=float, default=None)
    lf.add_argument("--alpha-sch", type=float, default=None)
    lf.add_argument("--L-min-sch", type=float, default=None)
    lf.add_argument("--L-max-sch", type=float, default=None)
    lf.add_argument("--L-std", type=float, default=None)

    fl = p.add_argument_group("physics toggles")
    fl.add_argument("--use-cosmology", action=argparse.BooleanOptionalAction, default=None)
    fl.add_argument("--use-sfr", action=argparse.BooleanOptionalAction, default=None)
    fl.add_argument("--use-smearing", action=argparse.BooleanOptionalAction, default=None)
    fl.add_argument("--use-spectral-index", action=argparse.BooleanOptionalAction, default=None)
    fl.add_argument("--use-beam", action=argparse.BooleanOptionalAction, default=None)

    sm = p.add_argument_group("simulation / redshift grid")
    sm.add_argument("--z-min", type=float, default=None)
    sm.add_argument("--z-max", type=float, default=None)
    sm.add_argument("--dz", type=float, default=None)
    sm.add_argument("--snr-threshold", type=float, default=None)
    sm.add_argument("--t-obs", type=float, default=None)
    sm.add_argument("--target-detections", type=int, default=None)
    sm.add_argument("--fscale", type=float, default=None)
    sm.add_argument("--seed", type=int, default=None)
    sm.add_argument("--max-iterations", type=int, default=None)

    p.add_argument("--out-dir", type=str, default=None,
                   help="Output folder (default: DM_alpha).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-z-bin progress prints.")
    return p


def resolve_config(argv: Optional[List[str]] = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    cfg = dict(DEFAULTS)

    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    # any CLI flag the user actually passed (non-None) overrides config/defaults
    cli_overrides = {k: v for k, v in vars(args).items()
                      if k not in ("config", "quiet") and v is not None}
    cfg.update(cli_overrides)
    cfg["quiet"] = args.quiet
    return cfg


# ===========================================================================
# MAIN
# ===========================================================================

def main(argv: Optional[List[str]] = None):
    cfg = resolve_config(argv)
    verbose = not cfg["quiet"]

    lf = LuminosityConfig(
        kind=cfg["lum_kind"], L_star=cfg["L_star"], alpha_sch=cfg["alpha_sch"],
        L_min_sch=cfg["L_min_sch"], L_max_sch=cfg["L_max_sch"], L_std=cfg["L_std"],
    )
    flags = PhysicsFlags(
        use_cosmology=cfg["use_cosmology"], use_sfr=cfg["use_sfr"],
        use_smearing=cfg["use_smearing"], use_spectral_index=cfg["use_spectral_index"],
        use_beam=cfg["use_beam"],
    )
    sim = SimulationConfig(
        z_min=cfg["z_min"], z_max=cfg["z_max"], dz=cfg["dz"],
        SNR_threshold=cfg["snr_threshold"], t_obs=cfg["t_obs"],
        target_detections=cfg["target_detections"], fscale=cfg["fscale"],
        seed=cfg["seed"], max_iterations=cfg["max_iterations"],
    )

    out_dir = Path(cfg["out_dir"])
    written = []

    for D in cfg["dish_sizes"]:
        telescope = TelescopeConfig(
            D=D, eta=cfg["eta"], Trec=cfg["Trec"], Tsky=cfg["Tsky"], BW=cfg["BW"],
            nu_cen=cfg["nu_cen"], n_comp=cfg["n_comp"], n_chan=cfg["n_chan"],
            tsamp=cfg["tsamp"],
        )
        for alpha_intrinsic in cfg["alphas_intrinsic"]:
            res = run_single_telescope(telescope, alpha_intrinsic, lf, flags, sim, verbose=verbose)
            path = save_result(out_dir, res)
            written.append(path)
            print(f"Saved {path}  (DM: {res.dm_per_detection().shape}, "
                  f"alpha_observed: {res.alpha_observed().shape})")

    print(f"\nWrote {len(written)} file(s) to {out_dir}/")
    return written


if __name__ == "__main__":
    main()
