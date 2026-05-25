"""
dN/dDM for a log-normal FRB luminosity function under DM = 1000 z,
constant comoving rate density, flat LCDM.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfc
import matplotlib.pyplot as plt

# --- Cosmology (Planck-ish flat LCDM) ----------------------------------------
H0 = 67.4                  # km/s/Mpc
Om = 0.315
Ol = 1.0 - Om
c_kms = 299792.458         # km/s
DH = c_kms / H0            # Hubble distance, Mpc

def E(z):
    return np.sqrt(Om * (1 + z)**3 + Ol)

# Comoving distance D_C(z) by cumulative trapezoid integration
z_grid = np.linspace(1e-4, 5.0, 5001)
inv_E = 1.0 / E(z_grid)
DC = np.concatenate([[0.0], cumulative_trapezoid(inv_E, z_grid)]) * DH      # Mpc
DL = (1 + z_grid) * DC                                                       # Mpc
dVc_dz = 4 * np.pi * DH * DC**2 / E(z_grid)                                  # Mpc^3

# --- Detection threshold and luminosity function -----------------------------
# Convert Mpc -> m for L_min
Mpc_to_m = 3.0857e22

# Fiducial detection threshold: 1 Jy peak for a 1-ms pulse (Parkes-ish).
Smin_Jy = 1.0

# Log-normal LF: scan median L_* (W/Hz) chosen so the rollover lies at
# observable DMs.  Roughly: z_horizon ~ where L_* equals the limiting L.
L_star_cases = [(1e26, "faint LF"),
                (1e27, "medium LF"),
                (1e28, "bright LF")]
sigma_cases = [0.5, 1.0, 2.0]   # natural-log widths

def L_min(z, Smin_Wm2Hz):
    # L_min = 4 pi D_C^2 (1+z) S_min, with D_C in metres
    DC_m = np.interp(z, z_grid, DC) * Mpc_to_m
    return 4 * np.pi * DC_m**2 * (1 + z) * Smin_Wm2Hz

def p_det(z, Smin_Wm2Hz, L_star, sigma):
    Lm = L_min(z, Smin_Wm2Hz)
    arg = (np.log(Lm) - np.log(L_star)) / (sigma * np.sqrt(2))
    return 0.5 * erfc(arg)

# --- dN/dDM (proportional; drop constant prefactor Phi0 Omega/(4 pi)) --------
# DM = 1000 z  =>  z = DM/1000;   dN/dDM = (1/1000) * dN/dz
# dN/dz \propto (1/(1+z)) * dV/dz * p_det(z)
DM_grid = 1000.0 * z_grid

def dN_dDM_shape(Smin_Jy, sigma, L_star):
    Smin = Smin_Jy * 1e-26
    pd = p_det(z_grid, Smin, L_star, sigma)
    return (1.0/1000.0) * (1.0/(1 + z_grid)) * dVc_dz * pd, pd

# --- Plot --------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, (Lstar, label) in zip(axes, L_star_cases):
    for sig in sigma_cases:
        curve, _ = dN_dDM_shape(Smin_Jy, sig, Lstar)
        ax.plot(DM_grid, curve / curve.max(),
                label=fr"$\sigma_{{\ln L}}={sig}$")
    # also overlay the unweighted volume kernel for reference (no LF cut)
    vol = (1.0/1000.0) * (1.0/(1 + z_grid)) * dVc_dz
    ax.plot(DM_grid, vol / vol.max(), 'k--', lw=1, alpha=0.5,
            label=r"$p_\mathrm{det}=1$ (volume only)")
    ax.set_xlim(0, 4000)
    ax.set_xlabel(r"DM (pc cm$^{-3}$)")
    ax.set_title(fr"{label}: $L_*=10^{{{int(np.log10(Lstar))}}}$ W Hz$^{{-1}}$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
axes[0].set_ylabel(r"$dN/d\mathrm{DM}$  (peak-normalised)")
fig.suptitle(
    fr"FRB detections per unit DM:  log-normal LF,  $S_\min={Smin_Jy}$ Jy,  "
    r"DM$=1000\,z$,  constant $\Phi_0$,  flat $\Lambda$CDM",
    y=1.02)
fig.tight_layout()
out = "/Users/mbailes/frb_dndm.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")

# --- Print where each curve peaks --------------------------------------------
print(f"\nPeak DM (pc/cm^3) for S_min = {Smin_Jy} Jy:")
for Lstar, label in L_star_cases:
    for sig in sigma_cases:
        curve, _ = dN_dDM_shape(Smin_Jy, sig, Lstar)
        DM_peak = DM_grid[np.argmax(curve)]
        print(f"  L*={Lstar:.0e} ({label:10s})  sigma={sig:.1f}  "
              f"->  DM_peak ~ {DM_peak:6.0f}")
