"""
Add spectral index alpha to the K-correction:
    L_min(z) = 4 pi D_C^2 (1+z)^(1-alpha) S_min(DM)
where S_nu ∝ nu^alpha (alpha < 0 => fainter at higher frequencies).

Keeps:  Madau-Dickinson SFR weighting,
        log-normal LF (sigma = 1/sqrt(2)),
        L_* = ASKAP-DM=1000 calibration / 4,
        DM smearing  S_min(DM) = S_min^(0) sqrt(1 + (DM/1000)^2),
        DM = 1000 z.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import erfc
import matplotlib.pyplot as plt

# Cosmology
H0, Om = 67.4, 0.315
Ol = 1.0 - Om
DH = 299792.458 / H0
Mpc_to_m = 3.0857e22
def E(z): return np.sqrt(Om*(1+z)**3 + Ol)

z_grid = np.linspace(1e-4, 8.0, 8001)
DC = np.concatenate([[0.0],
                     cumulative_trapezoid(1.0/E(z_grid), z_grid)]) * DH
dVc_dz = 4*np.pi*DH*DC**2 / E(z_grid)
DM_grid = 1000.0 * z_grid

# SFR (Madau-Dickinson)
def psi(z): return 0.015*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
w_SFR = psi(z_grid) / psi(0.0)

# DM smearing factor
smear = np.sqrt(1.0 + (DM_grid/1000.0)**2)

# LF and calibration
Smin_ASKAP = 1.0 * 1e-26
DC_h = np.interp(1.0, z_grid, DC) * Mpc_to_m
L_star = (4*np.pi * DC_h**2 * 2 * Smin_ASKAP) / 4.0
sigma = 1.0 / np.sqrt(2)

D_ASKAP = 12.0
telescopes = [("ASKAP", 12.0,  "tab:blue"),
              ("PKS",   60.0,  "tab:orange"),
              ("FAST", 300.0,  "tab:green")]
def Smin0(D): return Smin_ASKAP * (D_ASKAP/D)**2

def p_det(z, D, alpha):
    DC_m = np.interp(z, z_grid, DC) * Mpc_to_m
    s = np.interp(z, z_grid, smear)
    Lm = 4*np.pi * DC_m**2 * (1+z)**(1.0-alpha) * Smin0(D) * s
    return 0.5 * erfc((np.log(Lm) - np.log(L_star)) / (sigma*np.sqrt(2)))

def pdf_DM(D, alpha):
    pd = p_det(z_grid, D, alpha)
    f = w_SFR * (1.0/1000.0) * (1.0/(1+z_grid)) * dVc_dz * pd
    return f / trapezoid(f, DM_grid)

# --- Plot: one panel per alpha ----------------------------------------------
alphas = [0.0, -2.0]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, a in zip(axes, alphas):
    for name, D, col in telescopes:
        p = pdf_DM(D, a)
        cdf = cumulative_trapezoid(p, DM_grid, initial=0.0)
        med = np.interp(0.5, cdf, DM_grid)
        ax.plot(DM_grid, p, color=col, lw=2,
                label=fr"{name}  (med {med:.0f})")
    ax.set_xlim(0, 6000)
    ax.set_xlabel(r"DM (pc cm$^{-3}$)")
    ax.set_title(fr"$\alpha = {a:+.0f}$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
axes[0].set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$  (pc$^{-1}$ cm$^3$)")
fig.suptitle(
    fr"FRB DM distribution vs spectral index $\alpha$  ($S_\nu \propto \nu^\alpha$)"
    fr"  —  SFR-weighted, $\sigma_{{\ln L}}={sigma:.3f}$, "
    fr"$L_*$=ASKAP-DM=1000/4, +DM smearing",
    y=1.02)
fig.tight_layout()
out = "/home/thsu/FRB_population/frb_dm_distribution/figures/frb_dndm_alpha.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
plt.show()

# --- Quartile table ---------------------------------------------------------
print("\nDM quartiles (pc/cm^3):")
print(f"{'telescope':8s}  {'alpha':>5s}   {'Q25':>5s} {'Q50':>5s} {'Q75':>5s} {'Q90':>5s}")
for name, D, _ in telescopes:
    for a in alphas:
        p = pdf_DM(D, a)
        cdf = cumulative_trapezoid(p, DM_grid, initial=0.0)
        q = [np.interp(qq, cdf, DM_grid) for qq in (.25,.5,.75,.9)]
        print(f"{name:8s}  {a:+5.1f}   " + "  ".join(f"{x:5.0f}" for x in q))
