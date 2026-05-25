"""
Generate the six figures used in the pedagogical document.
Each figure isolates one new physical effect, building cumulatively:
  fig1: Euclidean standard candle (hard cutoff)
  fig2: Euclidean + log-normal LF
  fig3: + cosmology (DM=1000z, flat LCDM)
  fig4: + SFR(z) weighting (Madau-Dickinson)
  fig5: + intra-channel DM smearing  S_min ~ sqrt(1+(DM/1000)^2)
  fig6: + spectral index alpha (K-correction)
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfc
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
})

D_ASKAP = 12.0
telescopes = [("ASKAP", 12.0,  "tab:blue"),
              ("PKS",   60.0,  "tab:orange"),
              ("FAST", 300.0,  "tab:green")]

# ---------------------------------------------------------------------------
# Cosmology (used from fig3 onwards)
# ---------------------------------------------------------------------------
H0, Om = 67.4, 0.315
Ol = 1.0 - Om
DH = 299792.458 / H0
Mpc_to_m = 3.0857e22
def E(z): return np.sqrt(Om*(1+z)**3 + Ol)

z_grid = np.linspace(1e-4, 8.0, 8001)
DC = np.concatenate([[0.0],
                     cumulative_trapezoid(1.0/E(z_grid), z_grid)]) * DH
dVc_dz = 4*np.pi*DH*DC**2 / E(z_grid)
DM_grid_cosmo = 1000.0 * z_grid

# Calibration: ASKAP just detects median FRB (L=L*) at z=1, then 4x fainter
Smin_ASKAP_Jy = 1.0
Smin_ASKAP    = Smin_ASKAP_Jy * 1e-26
DC_h = np.interp(1.0, z_grid, DC) * Mpc_to_m
L_star = (4*np.pi * DC_h**2 * 2 * Smin_ASKAP) / 4.0
sigma  = 1.0 / np.sqrt(2)

def Smin0(D): return Smin_ASKAP * (D_ASKAP/D)**2

# SFR weighting (Madau-Dickinson 2014)
def psi(z): return 0.015*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
w_SFR = psi(z_grid) / psi(0.0)

# ---------------------------------------------------------------------------
# Figure 1: Euclidean standard candle  (hard horizon, no LF spread)
# ---------------------------------------------------------------------------
# In Euclidean limit, calibrate so ASKAP horizon DM_max(ASKAP) = 1000.
# Standard-candle d_max(D) ∝ D (sensitivity gain), and DM ∝ d.
# Therefore DM_max(D) = 1000 (D/D_ASKAP).
DM_grid_eucl = np.linspace(0, 30000, 6001)

fig, ax = plt.subplots(figsize=(7.3, 4.4))
for name, D, col in telescopes:
    DM_max = 1000.0 * (D / D_ASKAP)
    pdf = np.where(DM_grid_eucl <= DM_max, DM_grid_eucl**2, 0.0)
    norm = np.trapezoid(pdf, DM_grid_eucl)
    ax.plot(DM_grid_eucl, pdf/norm, color=col, lw=2,
            label=fr"{name}  ($D={D:g}$ m, $\mathrm{{DM}}_{{\max}}={DM_max:.0f}$)")
ax.set_xlim(0, 28000)
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$")
ax.set_title(r"Effect 1 — Euclidean standard candle: $dN/d\mathrm{DM}\propto \mathrm{DM}^2$ up to a hard horizon")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("/home/thsu/FRB_population/frb_dm_distribution/figures/fig1_euclidean.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Euclidean + log-normal LF (smooth tails, no cosmology)
# ---------------------------------------------------------------------------
# Pick L* and S_min so ASKAP horizon for L=L* is at DM_max=1000.
# In Euclidean: L_min(d) = 4 pi d^2 S_min, with d ∝ DM.
# Fix d_*(ASKAP) = some reference distance; use 1 Gpc -> DM=1000.
d_ref_m = 1e3 * Mpc_to_m                   # 1 Gpc in metres = ASKAP horizon
L_star_eucl = 4*np.pi * d_ref_m**2 * Smin_ASKAP   # so L*(ASKAP) hits S_min at d_ref
# We are using a SEPARATE L* for the Euclidean panel, chosen self-consistently.
def Lmin_eucl(DM, D):
    d = (DM/1000.0) * d_ref_m
    return 4*np.pi * d**2 * Smin0(D)
def pdet_eucl(DM, D):
    Lm = Lmin_eucl(DM, D)
    return 0.5 * erfc((np.log(np.maximum(Lm, 1e-300)) - np.log(L_star_eucl))
                      / (sigma*np.sqrt(2)))

fig, ax = plt.subplots(figsize=(7.3, 4.4))
DM_e = np.linspace(0.5, 12000, 4001)
for name, D, col in telescopes:
    pdf = DM_e**2 * pdet_eucl(DM_e, D)
    pdf /= np.trapezoid(pdf, DM_e)
    ax.plot(DM_e, pdf, color=col, lw=2, label=fr"{name}  ($D={D:g}$ m)")
ax.set_xlim(0, 12000)
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$")
ax.set_title(r"Effect 2 — Add a log-normal luminosity function: tails fill in beyond the standard-candle horizon")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("/home/thsu/FRB_population/frb_dm_distribution/figures/fig2_lf.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figures 3-5: cosmological model (re-use the pdf_DM machinery)
# ---------------------------------------------------------------------------
# DM smearing factor on S_min, evaluated on z_grid (DM = 1000 z)
smear_arr = np.sqrt(1.0 + (DM_grid_cosmo/1000.0)**2)

def Lmin_cosmo(z, D, alpha=0.0, smear=False):
    DC_m = np.interp(z, z_grid, DC) * Mpc_to_m
    s = np.interp(z, z_grid, smear_arr) if smear else 1.0
    return 4*np.pi * DC_m**2 * (1+z)**(1.0-alpha) * Smin0(D) * s

def pdet_cosmo(z, D, alpha=0.0, smear=False):
    Lm = Lmin_cosmo(z, D, alpha, smear)
    return 0.5 * erfc((np.log(np.maximum(Lm,1e-300)) - np.log(L_star))
                      / (sigma*np.sqrt(2)))

def pdf_DM(D, sfr=False, alpha=0.0, smear=False):
    pd = pdet_cosmo(z_grid, D, alpha, smear)
    f  = (1.0/1000.0)*(1.0/(1+z_grid))*dVc_dz*pd
    if sfr: f = f * w_SFR
    return f / np.trapezoid(f, DM_grid_cosmo)

# --- Figure 3: cosmology (constant Phi_0, alpha=0) ---
fig, ax = plt.subplots(figsize=(7.3, 4.4))
for name, D, col in telescopes:
    p = pdf_DM(D, sfr=False, alpha=0.0)
    cdf = cumulative_trapezoid(p, DM_grid_cosmo, initial=0)
    med = np.interp(0.5, cdf, DM_grid_cosmo)
    ax.plot(DM_grid_cosmo, p, color=col, lw=2,
            label=fr"{name}  ($D={D:g}$ m, median DM $\approx${med:.0f})")
ax.set_xlim(0, 6000)
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$  (pc$^{-1}$ cm$^3$)")
ax.set_title(r"Effect 3 — Cosmology: $(1+z)^{-1}\,dV_c/dz$ peaks at $z\!\sim\!2.4$, plus the $(1+z)$ K-tax")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("/home/thsu/FRB_population/frb_dm_distribution/figures/fig3_cosmology.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Figure 4: + SFR weighting ---
fig, ax = plt.subplots(figsize=(7.3, 4.4))
for name, D, col in telescopes:
    p_no  = pdf_DM(D, sfr=False, alpha=0.0)
    p_sfr = pdf_DM(D, sfr=True,  alpha=0.0)
    ax.plot(DM_grid_cosmo, p_no,  color=col, lw=1.0, ls="--", alpha=0.6)
    ax.plot(DM_grid_cosmo, p_sfr, color=col, lw=2.0,
            label=fr"{name}  with SFR$(z)$")
ax.plot([], [], 'k--', lw=1, alpha=0.6, label="(constant $\\Phi_0$ for reference)")
ax.set_xlim(0, 6000)
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$  (pc$^{-1}$ cm$^3$)")
ax.set_title(r"Effect 4 — Madau–Dickinson SFR weighting truncates the high-DM tail beyond cosmic noon")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("/home/thsu/FRB_population/frb_dm_distribution/figures/fig4_sfr.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Figure 5: + DM smearing (on top of SFR) -----------------------------
fig, ax = plt.subplots(figsize=(7.3, 4.4))
for name, D, col in telescopes:
    p_no = pdf_DM(D, sfr=True, alpha=0.0, smear=False)
    p_sm = pdf_DM(D, sfr=True, alpha=0.0, smear=True)
    ax.plot(DM_grid_cosmo, p_no, color=col, lw=1.0, ls="--", alpha=0.6)
    ax.plot(DM_grid_cosmo, p_sm, color=col, lw=2.0,
            label=fr"{name}  with smearing")
ax.plot([], [], 'k--', lw=1, alpha=0.6, label="(no smearing for reference)")
ax.set_xlim(0, 6000)
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$  (pc$^{-1}$ cm$^3$)")
ax.set_title(r"Effect 5 — DM smearing  $S_\min\!\to\! S_\min\sqrt{1+(\mathrm{DM}/1000)^2}$  hits PKS hardest")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("/home/thsu/FRB_population/frb_dm_distribution/figures/fig5_smear.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Figure 6: + spectral index alpha (on top of SFR + smearing) ---------
fig, ax = plt.subplots(figsize=(7.3, 4.4))
for name, D, col in telescopes:
    p0 = pdf_DM(D, sfr=True, alpha=0.0,  smear=True)
    pa = pdf_DM(D, sfr=True, alpha=-2.0, smear=True)
    ax.plot(DM_grid_cosmo, p0, color=col, lw=1.0, ls="--", alpha=0.6)
    ax.plot(DM_grid_cosmo, pa, color=col, lw=2.0,
            label=fr"{name}  $\alpha=-2$")
ax.plot([], [], 'k--', lw=1, alpha=0.6, label="($\\alpha=0$ for reference)")
ax.set_xlim(0, 6000)
ax.set_xlabel(r"DM (pc cm$^{-3}$)")
ax.set_ylabel(r"$p(\mathrm{DM}\,|\,\mathrm{detected})$  (pc$^{-1}$ cm$^3$)")
ax.set_title(r"Effect 6 — Spectral index $\alpha=-2$ adds a $(1+z)^{3}$ K-correction on top of smearing")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("/home/thsu/FRB_population/frb_dm_distribution/figures/fig6_alpha.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Print quartile summary table for the doc
print("\nQuartiles for each successive effect (cosmological model):")
header = f"{'tel':6s}  {'config':18s} {'Q25':>5s} {'Q50':>5s} {'Q75':>5s} {'Q90':>5s}"
print(header)
def q(p):
    cdf = cumulative_trapezoid(p, DM_grid_cosmo, initial=0)
    return [np.interp(qq, cdf, DM_grid_cosmo) for qq in (.25,.5,.75,.9)]
for name, D, _ in telescopes:
    rows = [
        ("cosmo only",          pdf_DM(D, sfr=False, alpha=0.0,  smear=False)),
        ("+ SFR(z)",            pdf_DM(D, sfr=True,  alpha=0.0,  smear=False)),
        ("+ smearing",          pdf_DM(D, sfr=True,  alpha=0.0,  smear=True)),
        ("+ alpha=-2",          pdf_DM(D, sfr=True,  alpha=-2.0, smear=True)),
    ]
    for tag, p in rows:
        qq = q(p)
        print(f"{name:6s}  {tag:18s} " + " ".join(f"{x:5.0f}" for x in qq))
print("done")
