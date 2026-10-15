import os, numpy as np, pandas as pd
def make(d):
    os.makedirs(os.path.join(d, "MeerKat_beams"), exist_ok=True)
    rng = np.random.default_rng(1)
    def freqs(nu0, bw, n): return np.linspace(nu0 - bw / 2, nu0 + bw / 2, n) / 1e6
    # long-format Tsky csvs
    for name, nu0, bw, n in [("ASKAP_Tsky.csv", 1.32e9, 336e6, 17), ("Parkes_Tsky.csv", 1.382e9, 340e6, 17)]:
        f = freqs(nu0, bw, n); pd.DataFrame({"freq_MHz": f, "Tsky": 3 + 2 * np.exp(-(f - f.min()) / 200)}).to_csv(os.path.join(d, name), index=False)
    # FAST: wide gain csv + wide Tsky csv
    fm = [1050, 1150, 1250, 1350, 1450]
    g = {f"Gain_{f}": 15 + 3 * rng.random(19) for f in fm}; pd.DataFrame(g).to_csv(os.path.join(d, "FAST_beam_gain.csv"), index=False)
    pd.DataFrame([{f"Tsky_{f}": 4 + (1450 - f) / 400 for f in fm}]).to_csv(os.path.join(d, "FAST_Tsky.csv"), index=False)
    # DSA110: arcsec beams, 9 rows
    f = freqs(1.405e9, 250e6, 9)
    pd.DataFrame({"freq_MHz": f, "sigma_avg": 60 + (1500 - f) / 20, "fwhm_avg": 140 + (1500 - f) / 8, "beam_sep": 200.0, "Tsky": 3.5 + (1500 - f) / 300}).to_csv(os.path.join(d, "DSA110_beam_parameters.csv"), index=False)
    # Apertif: arcsec, 11 rows
    f = freqs(1.37e9, 300e6, 11)
    pd.DataFrame({"freq_MHz": f, "sigma_avg": 60 + (1500 - f) / 20, "fwhm_avg": 140 + (1500 - f) / 8, "beam_sep": 250.0, "Tsky": 3.2 + (1500 - f) / 300}).to_csv(os.path.join(d, "Apertif_beam_parameters.csv"), index=False)
    # MeerKAT: degrees, 17 rows
    f = freqs(1.284e9, 856e6, 17)
    pd.DataFrame({"freq_MHz": f, "beam_sep": 0.03, "sigma_x": 0.004 + (1700 - f) * 2e-6, "sigma_y": 0.006 + (1700 - f) * 2e-6, "Tsky": 3 + (1700 - f) / 400}).to_csv(os.path.join(d, "MeerKat_beams", "beam_parameters.csv"), index=False)
