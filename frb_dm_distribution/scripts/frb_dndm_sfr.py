"""
Add SFR(z) weighting (Madau-Dickinson 2014) to the FRB rate density.
Compare constant Phi_0 vs SFR-weighted detected DM distributions for
ASKAP/PKS/FAST.
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

# --- Madau-Dickinson 2014 SFR ------------------------------------------------
def psi_MD14(z):
    return 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)

w_SFR = psi_MD14(z_grid) / psi_MD14(0.0)   # normalised so w(z=0)=1

# --- LF and detection --------------------------------------------------------
Smin_ASKAP = 1.0 * 1e-26
DC_h = np.interp(1.0, z_grid, DC) * Mpc_to_m
L_star = (4*np.pi * DC_h**2 * 2 * Smin_ASKAP) / 4.0   # 4x fainter
sigma = 1.0 / np.sqrt(2)

D_ASKAP = 12.0
telescopes = [("ASKAP", 12.0,  "tab:blue"),
              ("PKS",   60.0,  "tab:orange"),
              ("FAST", 300.0,  "tab:green")]

def Smin_for(D): return Smin_ASKAP * (D_ASKAP/D)**2

def p_det(z, Smin):
    DC_m = np.interp(z, z_grid, DC) * Mpc_to_m
    Lm = 4*np.pi * DC_m**2 * (1+z) * Smin
    return 0.5 * erfc((np.log(Lm) - np.log(L_star)) / (sigma*np.sqrt(2)))

def pdf_DM(D, sfr_weight=False):
    pd = p_det(z_grid, Smin_for(D))
    f = (1.0/1000.0) * (1.0/(1+z_grid)) * dVc_dz * pd
    if sfr_weight:
        f = f * w_SFR
    return f / np.trapezoid(f, DM_grid)

# --- Plot --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, sfr in zip(axes, [False, True]):
    for name, D, col in telescopes:
        p = pdf_DM(D, sfr_weight=sfr)
        cdf = cumulative_trapezoid(p, DM_grid, initial=0.0)
        med = np.interp(0.5, cdf, DM_grid)
        ax.plot(DM_grid, p, color=col, lw=2,
                label=fr"{name}  (median DM $\approx${med:.0f})")
    ax.set_xlim(0, 6000)
    ax.set_xlabel(r"DM (pc cm$^{-3}$)")
    ax.set_title("SFR-weighted (Madau–Dickinson)" if sfr else r"Constant $\Phi_0$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
axes[0].set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$  (pc$^{-1}$ cm$^3$)")
fig.suptitle(
    fr"FRBs tracking SFR vs constant rate  —  log-normal LF "
    fr"$\sigma_{{\ln L}}={sigma:.3f}$,  $L_*$ = ASKAP-DM=1000 / 4",
    y=1.02)
fig.tight_layout()
out = "/Users/mbailes/frb_dndm_sfr.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")

# --- Quartile table ----------------------------------------------------------
print("\nDM quartiles (pc/cm^3):")
print(f"{'telescope':10s}  {'mode':12s}  {'Q25':>6s} {'Q50':>6s} {'Q75':>6s} {'Q90':>6s}")
for name, D, _ in telescopes:
    for sfr, lab in [(False,"const Phi0"), (True,"SFR(z)")]:
        p = pdf_DM(D, sfr_weight=sfr)
        cdf = cumulative_trapezoid(p, DM_grid, initial=0.0)
        q = [np.interp(qq, cdf, DM_grid) for qq in (.25,.5,.75,.9)]
        print(f"{name:10s}  {lab:12s}  "
              + "  ".join(f"{x:5.0f}" for x in q))

# --- Show the SFR weight itself for reference -------------------------------
print("\nSFR(z)/SFR(0) sample:")
for zz in [0, 0.5, 1, 1.9, 3, 5]:
    print(f"  z={zz:>4.1f}  ->  w_SFR = {psi_MD14(zz)/psi_MD14(0):.2f}")
