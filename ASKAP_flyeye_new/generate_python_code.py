import os
from pathlib import Path
import numpy as np

# Create output directory
output_dir = Path("/home/thsu/FRB_population/ASKAP_flyeye_new")
output_dir.mkdir(exist_ok=True)

# Define redshift bins: (0.01, 0.02), (0.02, 0.03), ..., (1.69, 1.70)
dz = 0.01
z_start = 0.01
z_end = 1.70

redshift_ranges = []
z_min = z_start
while z_min < z_end:
    z_max = min(z_min + dz, z_end)
    if z_max <= z_end:
        redshift_ranges.append((z_min, z_max))
    z_min += dz

print(f"Generated {len(redshift_ranges)} redshift bins:")
print(f"First bin: z = {redshift_ranges[0][0]:.2f} to {redshift_ranges[0][1]:.2f}")
print(f"Last bin: z = {redshift_ranges[-1][0]:.2f} to {redshift_ranges[-1][1]:.2f}")

# Template code for each chunk
template_code = '''# ============================================================================
# ASKAP FLYEYE - REDSHIFT CHUNK ANALYSIS
# z_range: {z_min:.2f} to {z_max:.2f}
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

print(f"Starting redshift chunk: z = {z_min:.2f} to {z_max:.2f}")
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
dz = {dz:.4f}
z_array = np.arange({z_min:.4f}, {z_max:.4f}, dz)

print(f"Processing {{len(z_array)}} redshift bins...")

# ============================================================================
# MAIN LOOP - GENERATE FRBs AND DETECT
# ============================================================================
all_dm = []
all_theta = []
all_L = []
all_snr = []

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

# ============================================================================
# SAVE RESULTS
# ============================================================================
all_dm = np.array(all_dm)
all_theta = np.array(all_theta)
all_L = np.array(all_L)
all_snr = np.array(all_snr)

# Save DM array
filename = f'dm_detected_askap_z{z_min:.2f}_{z_max:.2f}.npy'
np.save(f'./{{filename}}', all_dm)      
print(f"\\nSaved: {{filename}}")
print(f"  Total FRB points: {{len(all_dm)}}")
print(f"  Detected (SNR>10): {{len(all_snr)}}")

elapsed_time = time.time() - start_time
print(f"\\nChunk completed in {{elapsed_time:.2f}} seconds")
if len(all_dm) > 0:
    print(f"Time per FRB: {{elapsed_time/len(all_dm)*1000:.4f}} ms")
'''

# Generate code files for each chunk
for i, (z_min, z_max) in enumerate(redshift_ranges):
    filename = output_dir / f"run_askap_z{z_min:.2f}_{z_max:.2f}.py"
    
    code = template_code.format(z_min=z_min, z_max=z_max, dz=dz)
    
    with open(filename, 'w') as f:
        f.write(code)
    
    if (i + 1) % 10 == 0:
        print(f"Created {i+1} files...")

print(f"\nSuccessfully created {len(redshift_ranges)} script files in {output_dir}")

# Create combiner script
combiner_code = '''# ============================================================================
# SCRIPT TO COMBINE ALL REDSHIFT CHUNKS
# ============================================================================

import numpy as np
from pathlib import Path
import glob

print("\\nCombining all DM arrays from redshift chunks...")

# Find all DM files
dm_files = sorted(glob.glob('dm_detected_askap_z*.npy'))

if not dm_files:
    print("ERROR: No DM files found!")
    exit(1)

print(f"Found {len(dm_files)} files")

# Load and combine
all_dm_combined = []
for filename in dm_files:
    dm_array = np.load(filename)
    all_dm_combined.extend(dm_array)
    print(f"  {filename}: {len(dm_array)} points")

all_dm_combined = np.array(all_dm_combined)

# Save combined array
output_file = 'dm_detected_askap_combined_full.npy'
np.save(output_file, all_dm_combined)

print(f"\\nSuccessfully combined {len(dm_files)} chunks")
print(f"Total DM points: {len(all_dm_combined)}")
print(f"Saved to: {output_file}")

# Print statistics
print(f"\\nDM Statistics:")
print(f"  Min: {all_dm_combined.min():.1f}")
print(f"  Max: {all_dm_combined.max():.1f}")
print(f"  Mean: {all_dm_combined.mean():.1f}")
print(f"  Median: {np.median(all_dm_combined):.1f}")
'''

combiner_file = output_dir / "combine_all_chunks.py"
with open(combiner_file, 'w') as f:
    f.write(combiner_code)

print(f"Created combiner script: {combiner_file}")

print(f"\n{'='*80}")
print("READY TO RUN:")
print(f"{'='*80}")
print(f"cd /home/thsu/FRB_population/ASKAP_flyeye_new")
print(f"for f in run_askap_z*.py; do python $f & done && wait")
print(f"python combine_all_chunks.py")
print(f"{'='*80}")