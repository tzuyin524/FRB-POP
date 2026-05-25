"""
N(DM) for three single-dish telescopes (ASKAP=12m, PKS=60m, FAST=300m).

Calibration: ASKAP just detects a *median*-luminosity FRB (L = L*) at z=1
(i.e. DM = 1000 under the linear Macquart simplification).

Scaling vs. dish diameter D:
  S_min  ∝ 1/D^2   (radiometer eq, fixed T_sys, BW, η, n_p, w)
  Ω_FoV  ∝ 1/D^2   (diffraction-limited primary beam)
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfc
import matplotlib.pyplot as plt

# --- Cosmology ---------------------------------------------------------------
H0, Om = 67.4, 0.315
Ol = 1.0 - Om
c_kms = 299792.458
DH = c_kms / H0
Mpc_to_m = 3.0857e22

def E(z):
    return np.sqrt(Om * (1 + z)**3 + Ol)

z_grid = np.linspace(1e-4, 6.0, 6001)
DC = np.concatenate([[0.0],
                     cumulative_trapezoid(1.0 / E(z_grid), z_grid)]) * DH   # Mpc
dVc_dz = 4 * np.pi * DH * DC**2 / E(z_grid)                                  # Mpc^3
DM_grid = 1000.0 * z_grid                                                    # pc/cm^3

# --- Calibration: L_* set by ASKAP horizon at z=1 ----------------------------
# Choose ASKAP S_min in Jy (arbitrary; only S_min^ASKAP * D_ASKAP^-2 ratio matters,
# but we need an absolute number to set L_*).  Use 1 Jy.
Smin_ASKAP_Jy = 1.0
Smin_ASKAP    = Smin_ASKAP_Jy * 1e-26                       # W/m^2/Hz

z_horizon_med = 1.0
DC_h_m = np.interp(z_horizon_med, z_grid, DC) * Mpc_to_m
L_star = 4 * np.pi * DC_h_m**2 * (1 + z_horizon_med) * Smin_ASKAP    # W/Hz
print(f"Calibrated L_* = {L_star:.3e} W/Hz")

sigma = 1.0   # natural-log width of the LF

# --- Telescopes --------------------------------------------------------------
telescopes = [
    ("ASKAP",  12.0,  "tab:blue"),
    ("PKS",    60.0,  "tab:orange"),
    ("FAST",  300.0,  "tab:green"),
]

D_ASKAP = 12.0

def Smin_for(D):
    return Smin_ASKAP * (D_ASKAP / D)**2

def Omega_for(D):
    # FoV in arbitrary units, normalised so ASKAP = 1
    return (D_ASKAP / D)**2

# --- L_min and p_det ---------------------------------------------------------
def L_min(z, Smin):
    DC_m = np.interp(z, z_grid, DC) * Mpc_to_m
    return 4 * np.pi * DC_m**2 * (1 + z) * Smin

def p_det(z, Smin, L_star, sigma):
    Lm = L_min(z, Smin)
    arg = (np.log(Lm) - np.log(L_star)) / (sigma * np.sqrt(2))
    return 0.5 * erfc(arg)

# --- dN/dDM (per unit observer time, in arbitrary units of Phi0/4pi * Mpc^3) -
def dN_dDM(D):
    Smin = Smin_for(D)
    Omega = Omega_for(D)
    pd = p_det(z_grid, Smin, L_star, sigma)
    return Omega * (1.0/1000.0) * (1.0/(1 + z_grid)) * dVc_dz * pd, pd

# --- Plot --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Absolute dN/dDM per primary beam-time, ASKAP-units
ax = axes[0]
for name, D, col in telescopes:
    curve, _ = dN_dDM(D)
    ax.plot(DM_grid, curve, label=fr"{name}  ($D={D:g}$ m)", color=col)
ax.set_xlim(0, 5000)
ax.set_yscale("log")
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$dN/d\mathrm{DM}$ per unit time  (ASKAP-FoV units)")
ax.set_title(r"Absolute rate per primary-beam pointing  ($\Omega \propto D^{-2}$)")
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False)

# (b) Shape — peak normalised, to compare DM distributions
ax = axes[1]
for name, D, col in telescopes:
    curve, _ = dN_dDM(D)
    ax.plot(DM_grid, curve / curve.max(),
            label=fr"{name}  ($D={D:g}$ m)", color=col)
ax.set_xlim(0, 6000)
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$dN/d\mathrm{DM}$  (peak-normalised)")
ax.set_title(r"Shape of detected DM distribution")
ax.grid(alpha=0.3)
ax.legend(frameon=False)

fig.suptitle(
    fr"FRB N(DM) for D = 12, 60, 300 m  —  calibrated so ASKAP horizon "
    fr"$L_*$ at DM=1000 (z=1);  log-normal LF $\sigma_{{\ln L}} = {sigma}$",
    y=1.02)
fig.tight_layout()
out = "/Users/mbailes/frb_dndm_telescopes.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")

# --- Summary numbers ---------------------------------------------------------
print("\nPeak DM and integrated rate (ASKAP-units) for each telescope:")
for name, D, _ in telescopes:
    curve, pd = dN_dDM(D)
    DM_peak = DM_grid[np.argmax(curve)]
    R_tot = np.trapezoid(curve, DM_grid)            # arbitrary units
    print(f"  {name:6s} D={D:5.1f} m   "
          f"S_min={Smin_for(D)/1e-26:7.4f} Jy   "
          f"Omega/Omega_ASKAP={Omega_for(D):.4f}   "
          f"DM_peak={DM_peak:5.0f}   R_total={R_tot:.3e}")
