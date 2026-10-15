"""Shared, tunable simulation parameters.

Edit the defaults here (or build a SimParams(...) in the driver notebook) and every
telescope picks the change up. Per-telescope exceptions go in Telescope.overrides
(see telescopes.py), never in the physics code.
"""
from dataclasses import dataclass, replace, fields


@dataclass(frozen=True)
class SimParams:
    # ---- intrinsic spectrum ------------------------------------------------
    alpha_intrinsic: float = 0.0          # fixed intrinsic spectral index

    # ---- effect switches ---------------------------------------------------
    USE_COSMOLOGY: bool = True            # cosmological vs Euclidean distances / volumes / time
    USE_SFR: bool = True                  # SFR-weighted rate vs constant
    USE_SMEARING: bool = True             # DM smearing + scattering + sampling widths
    USE_SPECTRAL_INDEX: bool = True       # (1+z)^(-1-alpha) and (nu/nu0)^alpha terms
    USE_BEAM: bool = True                 # beam gain pattern vs on-axis gain everywhere

    # ---- luminosity function ----------------------------------------------
    LUMINOSITY_FUNCTION: str = "schechter"   # "schechter" or "std"
    L_STAR: float = 3e13                  # Jy kpc^2
    ALPHA_SCH: float = -1.2               # faint-end slope
    L_MIN_SCH: float = 1e6
    L_MAX_SCH: float = 1e16
    L_BREAK_SCH: float = 8.952e15         # stop the z loop once L_min(z) exceeds this (schechter)
    L_std: float = 1e12                   # Jy kpc^2 (standard candle)

    # ---- rate / detection --------------------------------------------------
    fscale: float = 100 * 1e-7            # FRB rate scaling factor
    SNR_threshold: float = 8.0
    t_obs: float = 0.001                  # s
    Np: int = 2                           # polarisations

    # ---- redshift grid and Monte-Carlo controls ----------------------------
    dz: float = 0.001
    z_min: float = 0.01
    z_max: float = 5.0
    TARGET_DETECTIONS: int = 10           # detections required per z-bin
    BATCH_SIZE: int = 4096
    MAX_DRAWS_PER_BIN: int = 10000
    H0: float = 70.0
    Om0: float = 0.3

    def with_overrides(self, **kw) -> "SimParams":
        valid = {f.name for f in fields(self)}
        bad = set(kw) - valid
        if bad:
            raise KeyError(f"Unknown SimParams field(s): {sorted(bad)}")
        return replace(self, **kw) if kw else self
