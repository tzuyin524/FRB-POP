
# ============================================================================
# ASKAP FLYEYE - REDSHIFT CHUNK ANALYSIS
# z_range: 1.26 to 1.31
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps, cumtrapz
from scipy.optimize import curve_fit
from scipy.special import erf
import sympy as sp
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
import time

print(f"Starting redshift chunk: z = 1.26 to 1.31")
start_time = time.time()

# ============================================================================
# TELESCOPE PARAMETERS - ASKAP FLYEYE
# ============================================================================
D = 12
G0 = 0.035
nu0 = 1.4e9
Np = 2
Trec = 70
Tsky = 3
BW = 336e6
nu_cen = 1.32e9
nu_min = nu_cen - BW / 2
nu_max = nu_cen + BW / 2
n_comp = 16
BW_comp = BW / n_comp
n_chan = 2048
BW_chan = BW / n_chan
t = 0.001
c = 3.0e8
area = 160  # Field of view in deg^2

# FRB Parameters
L_STAR = 1e15
ALPHA_SCH = -1.5
L_MIN_SCH = 1e10
L_MAX_SCH = 1e16

# Schechter function
def schechter_function(L, L_star, alpha, phi_star=1.0):
    x = L / L_star
    return phi_star * x**alpha * np.exp(-x)

# Prepare inverse CDF for Schechter function
L_grid = np.logspace(np.log10(L_MIN_SCH), np.log10(L_MAX_SCH), 10000)
phi_grid = schechter_function(L_grid, L_STAR, ALPHA_SCH)
cdf_grid = cumtrapz(phi_grid, L_grid, initial=0)
cdf_grid = cdf_grid / cdf_grid[-1]
schechter_inverse_cdf = interp1d(cdf_grid, L_grid, kind='linear', 
                                 bounds_error=False, fill_value=(L_MIN_SCH, L_MAX_SCH))

def schechter_luminosity(L_star, alpha, L_min, L_max, size=1):
    u = np.random.uniform(0, 1, size=size)
    L_samples = schechter_inverse_cdf(u)
    return L_samples[0] if size == 1 else L_samples

# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================
def gaussian_beam_gain(nu, G0, theta, D, c):
    coeff = (theta**2 * D**2 * 2.355**2) / (2 * 1.22**2 * c**2)
    return G0 * np.exp(-coeff * nu**2)

def SNR(S, t, G, BW, Np, Trec, Tsky):
    return S * G * np.sqrt(BW * t * Np) / (Trec + Tsky)

# ============================================================================
# SETUP COSMOLOGY
# ============================================================================
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
dz = 0.01
z_array = np.arange(1.26, 1.31, dz)

print(f"Processing {len(z_array)} redshift bins...")

# ============================================================================
# MAIN LOOP - GENERATE FRBs AND DETECT
# ============================================================================
all_dm = []
all_theta = []
all_L = []
all_snr = []
all_alpha_inferred = []

np.random.seed(42)
fscale = 0.2 * 1e-7
theta_max = 2 * 1.22 * c / 13.5 / nu_min
wi = 0.001
alpha_intrinsic = 0

for z in z_array:
    zmin = z - 0.5 * dz
    zmax = z + 0.5 * dz
    
    # Calculate SFR and volumes
    psi = 0.015 * ((1 + z)**2.7) / (1 + ((1 + z) / 2.9)**5.6)
    dvol = (cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value) * (area / (4. * np.pi * ((180. / np.pi)**2)))
    
    t1 = cosmo.age(zmax).value
    t2 = cosmo.age(zmin).value
    dt = (t2 - t1) * 1.e+9
    
    nfrb_shell = int(np.round(psi * dvol * dt * fscale))
    
    for i in range(nfrb_shell):
        # Generate FRB luminosity
        lum = schechter_luminosity(L_STAR, ALPHA_SCH, L_MIN_SCH, L_MAX_SCH)
        all_L.append(lum)
        
        # DM from redshift
        DM = 1000 * z
        all_dm.append(DM)
        
        # Redshifted width
        wi_z = wi * (1 + z)
        
        # DM smearing
        nu_cen_GHz = nu_cen / 1e9
        BW_chan_MHz = BW_chan / 1e6
        w_DM = 8.3e-6 * DM * BW_chan_MHz / (nu_cen_GHz**3)
        
        # Scattering
        freq_powers_scatter = np.array([(nu_min * BW_comp * (j + 1))**(-4) for j in range(n_comp)])
        w_scatter_comp = 1.9e-7 * (DM**1.5) * (1 + (DM**3) * 3.55e-5) * 3 * freq_powers_scatter * 1e-3
        w_scatter = np.sqrt(np.sum(w_scatter_comp**2) / len(w_scatter_comp))
        w_sampling = 1.265e-3
        
        # Total observed width
        w_obs = np.sqrt(wi_z**2 + w_DM**2 + w_scatter**2 + w_sampling**2)
        
        # Flux density
        dl = cosmo.luminosity_distance(z).value * 1000  # kpc
        S0_frb = lum / ((dl**2) * ((1 + z)**(-1 - alpha_intrinsic))) * (wi_z / w_obs)
        
        # Angular offset
        theta_frb = theta_max * np.sqrt(np.random.random())
        all_theta.append(theta_frb)
        
        # SNR check
        S_nu_low = S0_frb * (nu_min / nu0)**alpha_intrinsic
        G_max = gaussian_beam_gain(nu0, G0, 0, D, c)
        snr_max = SNR(S_nu_low, t, G_max, BW, Np, Trec, Tsky)
        
        if snr_max > 10:
            all_snr.append(snr_max)
            all_dm_detected = all_dm[-1]  # Last DM for detected FRB
        
# ============================================================================
# SAVE RESULTS
# ============================================================================
all_dm = np.array(all_dm)
all_theta = np.array(all_theta)
all_L = np.array(all_L)
all_snr = np.array(all_snr)

# Save DM array
filename = f'dm_detected_askap_z1.26_1.31.npy'
np.save(f'/home/thsu/FRB_population/ASKAP_flyeye_DM/{filename}', all_dm)
print(f"\nSaved: {filename}")
print(f"  DM points: {len(all_dm)}")
print(f"  Detected (SNR>10): {len(all_snr)}")

elapsed_time = time.time() - start_time
print(f"\nChunk completed in {elapsed_time:.2f} seconds")
print(f"Time per FRB: {elapsed_time/max(len(all_dm),1)*1000:.4f} ms")
