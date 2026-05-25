"""
Add intra-channel DM smearing to the sensitivity:
    S_min(DM) = S_min^(0) * sqrt(1 + (DM/1000)^2)
applied to all telescopes equally.  Keeps SFR(z) weighting (Madau-Dickinson),
log-normal LF, DM=1000z.  Compares with vs without smearing.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
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

# SFR weight (Madau-Dickinson 2014)
def psi(z): return 0.015*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
w_SFR = psi(z_grid) / psi(0.0)

# Calibration / LF
Smin_ASKAP = 1.0 * 1e-26                                     # 1 Jy
DC_h = np.interp(1.0, z_grid, DC) * Mpc_to_m
L_star = (4*np.pi * DC_h**2 * 2 * Smin_ASKAP) / 4.0          # 4x fainter
sigma = 1.0 / np.sqrt(2)

D_ASKAP = 12.0
telescopes = [("ASKAP", 12.0,  "tab:blue"),
              ("PKS",   60.0,  "tab:orange"),
              ("FAST", 300.0,  "tab:green")]

def Smin0(D): return Smin_ASKAP * (D_ASKAP/D)**2

# DM smearing factor on S_min, evaluated on z_grid (DM = 1000 z)
smear = np.sqrt(1.0 + (DM_grid/1000.0)**2)

def p_det(z, D, with_smear):
    DC_m = np.interp(z, z_grid, DC) * Mpc_to_m
    s = np.interp(z, z_grid, smear) if with_smear else 1.0
    Lm = 4*np.pi * DC_m**2 * (1+z) * Smin0(D) * s
    return 0.5 * erfc((np.log(Lm) - np.log(L_star)) / (sigma*np.sqrt(2)))

def pdf_DM(D, with_smear):
    pd = p_det(z_grid, D, with_smear)
    f = w_SFR * (1.0/1000.0) * (1.0/(1+z_grid)) * dVc_dz * pd
    return f / np.trapezoid(f, DM_grid)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, ws in zip(axes, [False, True]):
    for name, D, col in telescopes:
        p = pdf_DM(D, ws)
        cdf = cumulative_trapezoid(p, DM_grid, initial=0.0)
        med = np.interp(0.5, cdf, DM_grid)
        ax.plot(DM_grid, p, color=col, lw=2,
                label=fr"{name}  (median DM $\approx${med:.0f})")
    ax.set_xlim(0, 6000)
    ax.set_xlabel(r"DM (pc cm$^{-3}$)")
    ax.set_title("With DM smearing  $S_\\min\\,\\sqrt{1+(DM/1000)^2}$"
                 if ws else "No smearing")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
axes[0].set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$  (pc$^{-1}$ cm$^3$)")
fig.suptitle(
    fr"SFR-weighted (Madau-Dickinson),  log-normal LF "
    fr"$\sigma_{{\ln L}}={sigma:.3f}$,  $L_*$ = ASKAP-DM=1000 / 4",
    y=1.02)
fig.tight_layout()
out = "/Users/mbailes/frb_dndm_smear.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")

print("\nDM quartiles (pc/cm^3):")
print(f"{'telescope':10s}  {'mode':16s}  {'Q25':>6s} {'Q50':>6s} {'Q75':>6s} {'Q90':>6s}")
for name, D, _ in telescopes:
    for ws, lab in [(False,"no smear"), (True,"with smear")]:
        p = pdf_DM(D, ws)
        cdf = cumulative_trapezoid(p, DM_grid, initial=0.0)
        q = [np.interp(qq, cdf, DM_grid) for qq in (.25,.5,.75,.9)]
        print(f"{name:10s}  {lab:16s}  " + "  ".join(f"{x:5.0f}" for x in q))
