# ============================================================================
# RUN ALL COMBINATIONS (Powers of 2 D0 multipliers: 1,2,4,8,...,2048; α = 0, -2, -4; lumi_func = std, schechter)
# ============================================================================

from pathlib import Path
import time
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import gaussian_kde
from astropy import units as u
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

# ============================================================================
# SIMULATION FUNCTION
# ============================================================================

def run_single_simulation(alpha_val, lumi_func, diameter_multiplier=1):
    """Run FRB simulation for a given alpha, luminosity function, and diameter"""
    
    # ============ SETUP ============
    # Generate telescope params dynamically based on diameter multiplier
    # D0 = 12 meters, so D = diameter_multiplier * 12
    D_base = 12  # meters
    D = diameter_multiplier * D_base
    TELESCOPE_NAME = f"{diameter_multiplier}d0"
    
    # Fixed telescope specs
    eta = 0.6
    Trec = 22
    Tsky = 3
    BW = 300e6
    nu_cen = 1.4e9
    n_comp = 16
    n_chan = 2048
    tsamp = 1.e-3
    
    k = 1.38e-23
    c = 3.0e8
    
    G0 = np.pi * D*D/4 * eta/2.0/k*1e-26
    Np = 2
    BW_comp = BW / n_comp
    BW_chan = BW / n_chan
    nu_min = nu_cen - BW / 2
    nu_max = nu_cen + BW / 2
    nu_array = np.linspace(nu_min, nu_max, n_comp+1)
    
    # Redshift parameters
    dz = 0.01
    z_min = 0.01
    z_max = 5.0
    z_array_opt = np.arange(z_min, z_max, dz)
    
    L_STAR = 1e15
    ALPHA_SCH = -1.5
    L_MIN_SCH = 1e10
    L_MAX_SCH = 1e16
    L_std = 1e14
    
    theta_max = 2*1.22*c/D/nu_min
    fscale = 0.2 * 1e-7
    
    SNR_threshold = 10
    t_obs = 0.001
    TARGET_DETECTIONS = 10
    
    cosmo_obj = FlatLambdaCDM(H0=70, Om0=0.3)
    
    fwhm_rad = 1.22 * (c / nu_min) / D
    fwhm_deg = fwhm_rad * (180 / np.pi)
    r_deg = 2.0 * fwhm_deg
    area = np.pi * (r_deg)**2
    
    freq_powers_scatter = np.array([(nu_min * BW_comp * (i+1))**(-4) for i in range(n_comp)])
    
    np.random.seed(42)
    
    # ============ DEFINE FUNCTIONS ============
    def gaussian_beam_gain(nu, G0, theta, D, c):
        coeff = (theta**2 * D**2 * 2.355**2) / (2 * 1.22**2 * c**2)
        return G0 * np.exp(-coeff * nu**2)
    
    def schechter_function(L, L_star, alpha, phi_star=1.0):
        x = L / L_star
        return phi_star * x**alpha * np.exp(-x)
    
    def create_truncated_schechter_sampler(L_min, L_max, L_star, alpha):
        L_grid = np.logspace(np.log10(L_min), np.log10(L_max), 2000)
        phi_grid = schechter_function(L_grid, L_star, alpha)
        cdf = cumtrapz(phi_grid, L_grid, initial=0)
        cdf = cdf / cdf[-1]
        inv_cdf = interp1d(cdf, L_grid, kind='linear', 
                           bounds_error=False, fill_value=(L_min, L_max))
        return inv_cdf
    
    def std_sampler(L_candle):
        def sampler(u):
            return L_candle
        return sampler
    
    # ============ PRECOMPUTE L_min ============
    L_min_by_z = {}
    nu0 = nu_cen
    for z in z_array_opt:
        dl = cosmo_obj.luminosity_distance(z).value * 1000
        G_max = gaussian_beam_gain(nu0, G0, 0, D, 3.0e8)
        S_min = SNR_threshold * (Trec + Tsky) / (G_max * np.sqrt(BW * t_obs * Np))
        z_factor = (1 + z)**(-1 - alpha_val)
        L_min = S_min * (dl**2) * z_factor
        L_min_by_z[z] = np.clip(L_min, L_MIN_SCH, L_MAX_SCH)
    
    sampler_cache = {}
    
    # ============ MAIN SIMULATION ============
    results_optimized = []
    start_time = time.time()
    
    _COEFF_CONST = (2.355**2) / (2 * 1.22**2 * (3.0e8**2))
    S_SNR_CONST = np.sqrt(BW_comp * t_obs * Np) / (Trec + Tsky)
    _INNER_COEFF_CONST = (D**2 * 2.355**2) / (2 * 1.22**2 * c**2)
    nu_scaling = (nu_array / nu0)**alpha_val
    
    for z_idx, z in enumerate(z_array_opt):
        zmin = z - 0.5 * dz
        zmax = z + 0.5 * dz
        L_min_z = L_min_by_z[z]
        
        if lumi_func.lower() == "std":
            if L_min_z > L_std:
                break
            truncated_sampler = std_sampler(L_std)
        else:
            if L_min_z > 8.952e15:
                break
            if L_min_z not in sampler_cache:
                sampler_cache[L_min_z] = create_truncated_schechter_sampler(
                    L_min_z, L_MAX_SCH, L_STAR, ALPHA_SCH
                )
            truncated_sampler = sampler_cache[L_min_z]
        
        DM = 1000 * z
        wi_z = 0.001 * (1 + z)
        BW_chan_MHz = BW_chan / 1e6
        nu_cen_GHz = nu_cen / 1e9
        
        w_DM = 8.3e-6 * DM * BW_chan_MHz / (nu_cen_GHz**3)
        w_scatter_comp = 1.9e-7 * (DM**1.5) * (1 + (DM**3) * 3.55e-5) * 3 * freq_powers_scatter * 1e-3
        w_scatter = np.sqrt(np.sum(w_scatter_comp**2) / len(w_scatter_comp))
        w_sampling = tsamp*1e-3
        
        w_obs_factor = np.sqrt(wi_z**2 + w_DM**2 + w_scatter**2 + w_sampling**2)
        dl = cosmo_obj.luminosity_distance(z).value * 1000
        z_factor = (1 + z)**(-1 - alpha_val)
        
        theta_hp = np.sqrt(np.log(2) / (_INNER_COEFF_CONST * nu0**2))
        
        n_generated = 0
        n_detected = 0
        detected_luminosities = []
        detected_snrs = []
        max_iterations = 1000000
        
        while n_detected < TARGET_DETECTIONS and n_generated < max_iterations:
            n_generated += 1
            
            theta_frb = theta_max * np.sqrt(np.random.random())
            
            if theta_frb > 4.0 * theta_hp:
                continue
            
            lum = float(truncated_sampler(np.random.uniform(0, 1)))
            
            S0_frb = (lum / (dl**2 * z_factor)) * (wi_z / w_obs_factor)
            S_nu = S0_frb * nu_scaling
            G_nu = G0 * np.exp(-(_INNER_COEFF_CONST * theta_frb**2) * nu_array**2)
            
            snr_spectrum = S_nu * G_nu * S_SNR_CONST
            snr_coherent = np.sum(snr_spectrum) / np.sqrt(n_comp)
            
            if snr_coherent > SNR_threshold:
                n_detected += 1
                detected_luminosities.append(lum)
                detected_snrs.append(snr_coherent)
        
        if n_detected == TARGET_DETECTIONS:
            detection_rate = n_detected / n_generated
            
            if lumi_func.lower() == "std":
                N_ratio = 1.0 if L_std >= L_min_z else 0.0
            else:
                L_grid_full = np.logspace(10, 16, 5000)
                phi_grid_full = schechter_function(L_grid_full, L_STAR, ALPHA_SCH)
                cdf_full = cumtrapz(phi_grid_full, L_grid_full, initial=0)
                cdf_full /= cdf_full[-1]
                cdf_interp = interp1d(L_grid_full, cdf_full, kind='linear')
                N_ratio = 1.0 - cdf_interp(L_min_z)
            
            psi = 0.015 * ((1+z)**2.7) / (1 + ((1+z)/2.9)**5.6)
            dt = (cosmo_obj.age(zmin).value - cosmo_obj.age(zmax).value) * 1e9
            dvol = (cosmo_obj.comoving_volume(zmax).value - cosmo_obj.comoving_volume(zmin).value) * (area /(4.*np.pi*((180./np.pi)**2)))
            
            nfrb_sfr = psi * dvol * dt * fscale
            n_total_population = detection_rate * N_ratio * nfrb_sfr
            
            results_optimized.append({
                'luminosity_function': lumi_func,
                'z': z, 'L_min': L_min_z, 'n_generated': n_generated, 'n_detected': n_detected,
                'detection_rate': detection_rate, 'mean_SNR': np.mean(detected_snrs),
                'mean_detected_L': np.mean(detected_luminosities), 'N_ratio': N_ratio,
                'psi': psi, 'dvol': dvol, 'dt': dt, 'nfrb_sfr_base': nfrb_sfr,
                'n_total_population': n_total_population, 'alpha_intrinsic': alpha_val
            })
    
    end_time = time.time()
    
    # ============ SAVE RESULTS ============
    if results_optimized:
        z_results = np.array([r['z'] for r in results_optimized])
        alpha_intrinsic_results = np.array([r['alpha_intrinsic'] for r in results_optimized])
        L_min_results = np.array([r['L_min'] for r in results_optimized])
        n_generated_results = np.array([r['n_generated'] for r in results_optimized])
        n_detected_results = np.array([r['n_detected'] for r in results_optimized])
        detection_rate_results = np.array([r['detection_rate'] for r in results_optimized])
        mean_snrs = np.array([r['mean_SNR'] for r in results_optimized])
        N_ratio_results = np.array([r['N_ratio'] for r in results_optimized])
        n_population_results = np.array([r['n_total_population'] for r in results_optimized])
        
        df_summary = pd.DataFrame({
            'z': z_results,
            'luminosity_function': [lumi_func] * len(z_results),
            'alpha_intrinsic': alpha_intrinsic_results,
            'L_min_Jy_kpc2': L_min_results,
            'L_min_over_L_star': L_min_results / L_STAR,
            'n_generated': n_generated_results,
            'n_detected': n_detected_results,
            'detection_rate_10_per_n': detection_rate_results,
            'N_ratio_detectable_L': N_ratio_results,
            'mean_detected_L': np.array([r['mean_detected_L'] for r in results_optimized]),
            'mean_SNR': np.array([r['mean_SNR'] for r in results_optimized]),
            'SFR_psi': np.array([r['psi'] for r in results_optimized]),
            'comoving_volume_Mpc3': np.array([r['dvol'] for r in results_optimized]),
            'time_Gyr': np.array([r['dt'] for r in results_optimized]),
            'nfrb_sfr_base': np.array([r['nfrb_sfr_base'] for r in results_optimized]),
            'n_total_population_scaled': n_population_results
        })
        
        Path("DM_distribution_new").mkdir(exist_ok=True)
        csv_file = f"DM_distribution_new/DM_{TELESCOPE_NAME}_alpha{alpha_val}_{lumi_func.lower()}_scaled.csv"
        df_summary.to_csv(csv_file, index=False)
        
        return {
            'diameter': diameter_multiplier,
            'diameter_m': D,
            'alpha': alpha_val,
            'lumi_func': lumi_func,
            'file': csv_file,
            'n_results': len(results_optimized),
            'time_sec': end_time - start_time,
            'total_pop': n_population_results.sum()
        }
    return None


# ============================================================================
# EXECUTE ALL RUNS (Powers of 2 D0 multipliers: 1, 2, 4, 8, ..., 2048)
# ============================================================================

if __name__ == "__main__":
    D_base = 12  # meters
    
    # D0 multipliers following 2^x pattern: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    DIAMETER_RANGE = [2**i for i in range(8)]  # 2^0 through 2^6
    
    # Display diameter mapping
    print("\n" + "="*100)
    print("RUNNING ALL PARAMETER COMBINATIONS - POWERS OF 2 D0 MULTIPLIERS")
    print("="*100)
    print(f"\nD0 Multipliers (2^x) and Actual Sizes:")
    print(f"{'2^x':<6} {'D0 Multiplier':<18} {'Diameter (m)':<18} {'Diameter (km)':<15}")
    print("─" * 60)
    
    for i, d_mult in enumerate(DIAMETER_RANGE):
        d_m = d_mult * D_base
        print(f"2^{i:<4} {d_mult:<18} {d_m:<18.1f} {d_m/1000:<15.3f}")
    
    print(f"\nTotal Diameters: {len(DIAMETER_RANGE)}")
    print(f"D0 Range: {DIAMETER_RANGE[0]} to {DIAMETER_RANGE[-1]}")
    print(f"Diameter Range: {DIAMETER_RANGE[0]*D_base:.1f}m to {DIAMETER_RANGE[-1]*D_base:.1f}m")
    print("="*100 + "\n")
    
    alpha_values = [0, -2, -4]
    lumi_funcs = ['std', 'schechter']
    
    total_runs = len(DIAMETER_RANGE) * len(alpha_values) * len(lumi_funcs)
    run_count = 0
    all_run_results = []
    
    start_total_time = time.time()
    
    for diameter_mult in DIAMETER_RANGE:
        for alpha_val in alpha_values:
            for lumi_func in lumi_funcs:
                run_count += 1
                diameter_m = diameter_mult * D_base
                print(f"\n{'─'*100}")
                print(f"RUN {run_count}/{total_runs}: D={diameter_mult}D0 ({diameter_m}m), α={alpha_val:2d}, {lumi_func.upper()}")
                print(f"{'─'*100}")
                
                result = run_single_simulation(alpha_val, lumi_func, diameter_mult)
                
                if result:
                    all_run_results.append(result)
                    print(f"✓ Completed in {result['time_sec']:.1f}s")
                    print(f"  Saved: {result['file']}")
                    print(f"  Results: {result['n_results']} z-bins, Total Population: {result['total_pop']:.2f}")
                else:
                    print(f"✗ No results generated for this run")
    
    end_total_time = time.time()
    total_time_hours = (end_total_time - start_total_time) / 3600
    total_time_minutes = ((end_total_time - start_total_time) % 3600) / 60
    
    print(f"\n{'='*100}")
    print(f"ALL {run_count} RUNS COMPLETED!")
    print(f"Total time: {total_time_hours:.1f}h {total_time_minutes:.1f}m")
    if run_count > 0:
        print(f"Average time per run: {(end_total_time - start_total_time) / run_count:.1f}s")
    print(f"{'='*100}\n")
    
    # Summary table
    if all_run_results:
        print("\nSUMMARY OF ALL RUNS:")
        print(f"{'D0':<8} {'D(m)':<12} {'α':<5} {'Lumi':<12} {'Time (s)':<10} {'Population':<15} {'z-bins':<8}")
        print("─" * 80)
        for r in all_run_results:
            print(f"{r['diameter']:<8} {r['diameter_m']:<12.0f} {r['alpha']:<5d} {r['lumi_func']:<12} {r['time_sec']:<10.1f} {r['total_pop']:<15.2f} {r['n_results']:<8}")
        
        print(f"\n{'='*100}")
        total_pop = sum(r['total_pop'] for r in all_run_results)
        print(f"Total populations across all runs: {total_pop:.2f}")
        print(f"{'='*100}\n")