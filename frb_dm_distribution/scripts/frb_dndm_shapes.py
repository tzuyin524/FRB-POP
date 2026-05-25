"""
Shapes only: PDF of detected DM for ASKAP/PKS/FAST.
Drop FoV weighting (constant in DM); normalise each curve to unit area
so they are proper p(DM | detected) distributions.
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

# Calibrate L_*: ASKAP horizon for L=L* at z=1 (DM=1000), then make 4x fainter
Smin_ASKAP = 1.0 * 1e-26                         # 1 Jy
DC_h = np.interp(1.0, z_grid, DC) * Mpc_to_m
L_star_orig = 4*np.pi * DC_h**2 * 2 * Smin_ASKAP
L_star = L_star_orig / 4.0
print(f"L_* (orig) = {L_star_orig:.3e},  L_* (4x fainter) = {L_star:.3e}  W/Hz")

sigma = 1.0 / np.sqrt(2)   # tightened by sqrt(2) from previous run
D_ASKAP = 12.0
telescopes = [("ASKAP", 12.0,  "tab:blue"),
              ("PKS",   60.0,  "tab:orange"),
              ("FAST", 300.0,  "tab:green")]

def Smin_for(D): return Smin_ASKAP * (D_ASKAP/D)**2

def p_det(z, Smin):
    DC_m = np.interp(z, z_grid, DC) * Mpc_to_m
    Lm = 4*np.pi * DC_m**2 * (1+z) * Smin
    return 0.5 * erfc((np.log(Lm) - np.log(L_star)) / (sigma*np.sqrt(2)))

def pdf_DM(D):
    pd = p_det(z_grid, Smin_for(D))
    f = (1.0/1000.0) * (1.0/(1+z_grid)) * dVc_dz * pd
    return f / np.trapezoid(f, DM_grid)

# --- Plot --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
for name, D, col in telescopes:
    p = pdf_DM(D)
    DM_med = np.interp(0.5, np.cumsum(p)*np.diff(np.r_[0, DM_grid]), DM_grid)
    ax.plot(DM_grid, p, color=col, lw=2,
            label=fr"{name}  ($D={D:g}$ m,  median DM $\approx${DM_med:.0f})")
ax.set_xlim(0, 6000)
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$  (pc$^{-1}$ cm$^{3}$)")
ax.set_title(
    fr"Detected-DM distribution  —  log-normal LF $\sigma_{{\ln L}}={sigma}$, "
    r"DM$=1000\,z$, constant $\Phi_0$"
    + "\n$L_*$ is 4$\\times$ fainter than the ASKAP-DM=1000 calibration")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
out = "/Users/mbailes/frb_dndm_shapes_4x_fainter_sig0p71.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")

# Quartiles for each telescope
print("\nDM quartiles (pc/cm^3):")
for name, D, _ in telescopes:
    p = pdf_DM(D)
    cdf = cumulative_trapezoid(p, DM_grid, initial=0.0)
    q25, q50, q75, q90 = (np.interp(q, cdf, DM_grid) for q in (.25,.5,.75,.9))
    print(f"  {name:6s}  Q25={q25:5.0f}  Q50={q50:5.0f}  "
          f"Q75={q75:5.0f}  Q90={q90:5.0f}")
