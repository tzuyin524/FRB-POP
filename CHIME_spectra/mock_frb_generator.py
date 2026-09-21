#!/usr/bin/env python3
"""
mock_frb_generator.py

Generates synthetic CHIME/FRB-baseband-style HDF5 files matching the on-disk
layout that run_baseband_spectral_index.py expects, with a *known, injected*
spectral index, DM, scattering timescale, and dead-channel mask -- so you can
run the real pipeline on a mock file and check whether it recovers what was
put in.

Injected signal model
----------------------
- 2D Gaussian radiometer noise, std = Tsys/G, independent per (channel, pol,
  time sample).
- amplitude_norm (the pipeline's calibration gain) is written as all-ones, so
  raw power *is* flux density in Jy -- this keeps calibration out of the test
  and means the injected/recovered numbers are directly comparable in Jy.
- FRB = time_profile(t) x freq_envelope(f):
    * time_profile: Gaussian in time, centered at an arrival time that is
      delayed per-channel by the standard cold-plasma dispersion law for the
      given DM (same formula/convention as the pipeline's incoherent_dedisp:
      delay(f) = DM/2.41e-4 * (f^-2 - f_ref^-2), f_ref = top of the recorded
      band) -- i.e. the mock file contains a *dispersed* burst, exactly like
      real coherently-dedispersed-but-not-yet-incoherently-corrected data.
    * freq_envelope: either a power law S(f) = S0*(f/f_ref)^alpha (alpha
      drawn uniformly from [-4, -1] by default, matching typical FRB spectral
      indices) or a Gaussian bump in frequency.
    * each channel's pulse is then convolved with a one-sided exponential
      scattering kernel, tau(f) = tau_ref * (f/f_ref)^scatter_index
      (scatter_index = -4 by default, the standard thin-screen scaling).
- A configurable fraction of the 1024-channel grid is marked dead/missing
  (excluded from index_map/freq/id entirely, just like un-recorded channels
  in real data), either scattered randomly or in a few contiguous blocks
  (more realistic RFI-flagged sub-bands).

Usage:
    python mock_frb_generator.py --out mock_frb.h5 --dm 300 --alpha -2.0
    python mock_frb_generator.py --out mock_frb.h5 --freq-shape gaussian --peak-freq 600 --peak-width 80
    python mock_frb_generator.py --batch 20 --outdir mock_frbs/   # random alpha in [-4,-1] each, logs truth to CSV
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

# Matches incoherent_dedisp's convention in the pipeline:
#   dm_delay = dm / DM_CONST * (fr**-2 - freq_ref**-2)   [seconds, DM in pc/cm^3, freq in MHz]
DM_CONST = 2.41e-4


def make_freq_grid(n_chan=1024, freq_top=800.19921875, freq_bot=400.390625):
    """1024-channel CHIME-like grid: index 0 = top of band, descending in freq.
    freq must vary linearly with channel index -- the pipeline recovers
    freq_full via a linear fit to (freq_id, freq_centre), so a non-linear
    grid here would silently corrupt every frequency downstream."""
    freq_id_all = np.arange(n_chan)
    freq_centre_all = np.linspace(freq_top, freq_bot, n_chan)
    return freq_id_all, freq_centre_all


def pick_dead_channels(n_chan, dead_frac=0.05, mode="random", rng=None, n_blocks=3):
    """Boolean mask, True = dead/missing (excluded from index_map/freq/id)."""
    rng = rng or np.random.default_rng()
    dead = np.zeros(n_chan, dtype=bool)
    n_dead = int(round(dead_frac * n_chan))
    if n_dead == 0:
        return dead
    if mode == "block":
        remaining = n_dead
        for i in range(n_blocks):
            if remaining <= 0:
                break
            blocks_left = n_blocks - i
            this_len = remaining if blocks_left == 1 else int(rng.integers(1, max(2, remaining // blocks_left + 1)))
            this_len = min(this_len, remaining, n_chan)
            start = int(rng.integers(0, max(1, n_chan - this_len)))
            dead[start:start + this_len] = True
            remaining -= this_len
    else:
        idx = rng.choice(n_chan, size=n_dead, replace=False)
        dead[idx] = True
    return dead


def min_duration_for_dm(dm, freq_top, freq_bot, margin_s=0.1):
    """Minimum recording length (s) needed for the full band to contain the
    dispersion sweep for `dm`, plus a fixed margin for off-pulse buffer
    (needed for noise estimation) and pulse/scattering width.

    IMPORTANT: tiedbeam_power in real CHIME/FRB baseband files is loaded with
    a comment saying it's "already coherently dedispersed" -- but the
    pipeline's own incoherent_dedisp() step (using the same DM_coherent
    stored in the file) still does real, non-trivial work: it shifts each
    channel by dm/2.41e-4*(f^-2 - f_ref^-2), i.e. up to several *seconds* for
    a typical extragalactic DM across the 400-800 MHz band (e.g. ~7.8 s for
    DM=400). That only makes sense if the as-recorded data still contains
    that full sweep and incoherent_dedisp is what removes it -- so this mock
    generator reproduces that sweep on injection, and the recording MUST be
    at least this long or the burst simply never appears in the low-frequency
    channels (it "arrives" after the recording ends). This is also why real
    baseband files are large (seconds of data at ~us sampling, x1024
    channels) -- don't be surprised by GB-scale output for realistic DMs.
    """
    sweep = dm / DM_CONST * (freq_bot ** -2 - freq_top ** -2)
    return sweep + margin_s


def dm_delay(freq_mhz, dm, freq_ref):
    """Seconds. Positive for freq_mhz < freq_ref (lower freq arrives later),
    matching the pipeline's incoherent_dedisp convention exactly."""
    return dm / DM_CONST * (freq_mhz ** -2 - freq_ref ** -2)


def scattering_kernel(tau_s, dt, n_kernel):
    """One-sided exponential scattering tail, normalized to unit area
    (preserves fluence rather than peak amplitude, as real scattering does)."""
    t = np.arange(n_kernel) * dt
    k = np.exp(-t / tau_s)
    k /= k.sum()
    return k


def build_mock_arrays(
    n_chan=1024,
    freq_top=800.19921875,
    freq_bot=400.390625,
    dt=2.56e-6,
    duration_s=None,
    npol=2,
    dm=300.0,
    alpha=None,
    freq_shape="powerlaw",
    peak_freq=600.0,
    peak_width_mhz=100.0,
    freq_ref_signal=600.0,
    pulse_width_s=1.5e-3,
    scatter_tau_ref_s=0.0,
    scatter_freq_ref=600.0,
    scatter_index=-4.0,
    tsys_over_g=1.0,
    snr_target=25.0,
    dead_frac=0.05,
    dead_mode="random",
    burst_frac_of_recording=None,
    seed=None,
    allow_short_duration=False,
):
    """Pure-numpy core: builds the (power, freq_id, freq, ...) arrays with no
    h5py dependency, so the injection math can be sanity-checked (or fed to a
    from-scratch reimplementation of the pipeline's steps) without needing
    h5py installed at all. gen_mock_frb() below just writes these to disk.
    Returns (power, freq_id, freq, amplitude_norm, ctime, dt, truth_dict).

    duration_s: if None (default), auto-computed from `dm` via
    min_duration_for_dm() so the full dispersion sweep fits -- this is almost
    always what you want. Only pass an explicit value if you know it covers
    the sweep (see min_duration_for_dm's docstring for why this matters);
    otherwise low-frequency channels will silently contain no burst at all.
    Passing a too-short duration_s raises unless allow_short_duration=True.
    """
    rng = np.random.default_rng(seed)
    if alpha is None:
        alpha = float(rng.uniform(-4.0, -1.0))

    freq_id_all, freq_centre_all = make_freq_grid(n_chan, freq_top, freq_bot)
    dead = pick_dead_channels(n_chan, dead_frac, dead_mode, rng)
    recorded = ~dead
    freq_id = freq_id_all[recorded]
    freq = freq_centre_all[recorded]
    n_rec = freq_id.size

    freq_ref_disp = float(freq.max())  # == freq.max() used by the pipeline itself
    min_dur = min_duration_for_dm(dm, float(freq.max()), float(freq.min()), margin_s=4 * pulse_width_s + 0.05)
    if duration_s is None:
        duration_s = min_dur
    elif duration_s < min_dur and not allow_short_duration:
        raise ValueError(
            f"duration_s={duration_s:.4f}s is shorter than the ~{min_dur:.4f}s needed to contain "
            f"the DM={dm} dispersion sweep across the recorded band ({freq.min():.1f}-{freq.max():.1f} MHz) "
            f"plus a pulse/off-pulse margin. Low-frequency channels would silently get no injected "
            f"signal at all. Pass duration_s=None to auto-size it, or allow_short_duration=True to "
            f"override (e.g. you've deliberately restricted DM/band for a quick test)."
        )

    n_time = int(round(duration_s / dt))
    # Default: place the burst so its earliest-arriving (highest-freq) edge
    # has some off-pulse buffer before it, and the latest-arriving
    # (lowest-freq) edge still has buffer after it -- i.e. right after the
    # margin built into min_duration_for_dm, not at the recording midpoint
    # (the midpoint is wrong once duration is dominated by the DM sweep
    # itself, since the sweep is very asymmetric: the burst is "early" at
    # high freq and "late" at low freq).
    if burst_frac_of_recording is None:
        t_burst = 2 * pulse_width_s + 0.025
    else:
        t_burst = burst_frac_of_recording * duration_s

    # ---- noise: independent Gaussian, std = tsys_over_g, per chan/pol/time ----
    power = rng.normal(0.0, tsys_over_g, size=(n_rec, npol, n_time)).astype(np.float32)

    # ---- frequency envelope, normalized to a peak of 1 ----
    if freq_shape == "powerlaw":
        s_freq = (freq / freq_ref_signal) ** alpha
    elif freq_shape == "gaussian":
        s_freq = np.exp(-0.5 * ((freq - peak_freq) / peak_width_mhz) ** 2)
    else:
        raise ValueError(f"unknown freq_shape {freq_shape!r}")
    s_freq = s_freq / np.max(s_freq)

    amplitude0 = snr_target * tsys_over_g  # peak per-sample signal amplitude
    t_axis = np.arange(n_time) * dt

    for i, f in enumerate(freq):
        arrival = t_burst + dm_delay(f, dm, freq_ref_disp)
        pulse = np.exp(-0.5 * ((t_axis - arrival) / pulse_width_s) ** 2)
        if scatter_tau_ref_s > 0:
            tau = scatter_tau_ref_s * (f / scatter_freq_ref) ** scatter_index
            n_k = max(1, int(round(10 * tau / dt)))
            if n_k > 1:
                kernel = scattering_kernel(tau, dt, n_k)
                pulse = fftconvolve(pulse, kernel, mode="full")[:n_time]
        power[i, :, :] += amplitude0 * s_freq[i] * pulse[np.newaxis, :]

    amplitude_norm = np.ones((n_rec, npol), dtype=np.float64)
    ctime = np.zeros(n_rec, dtype=np.float64)
    truth = {
        "alpha_injected": alpha,
        "dm": dm,
        "freq_shape": freq_shape,
        "peak_freq_injected": peak_freq if freq_shape == "gaussian" else None,
        "n_dead": int(dead.sum()),
        "n_recorded": n_rec,
        "scatter_tau_ref_s": scatter_tau_ref_s,
        "pulse_width_s": pulse_width_s,
        "snr_target": snr_target,
        "t_burst": t_burst,
    }
    return power, freq_id, freq, amplitude_norm, ctime, dt, truth


def gen_mock_frb(out_path, **kwargs):
    """Build mock arrays and write them to an HDF5 file in the exact layout
    run_baseband_spectral_index.py expects. Returns the truth dict."""
    power, freq_id, freq, amplitude_norm, ctime, dt, truth = build_mock_arrays(**kwargs)
    npol = power.shape[1]

    import h5py
    with h5py.File(out_path, "w") as f_out:
        dset = f_out.create_dataset("tiedbeam_power", data=power)
        dset.attrs["DM_coherent"] = float(truth["dm"])
        f_out.attrs["delta_time"] = float(dt)

        idx = f_out.create_group("index_map")
        freq_grp = idx.create_group("freq")
        freq_grp.create_dataset("id", data=freq_id.astype(np.int64))
        freq_grp.create_dataset("centre", data=freq.astype(np.float64))
        idx.create_dataset("amplitude_norm", data=amplitude_norm)

        t0_grp = f_out.create_group("time0")
        # All channels share the same absolute start time -- real data has
        # small per-channel offsets from buffering, but those are irrelevant
        # to testing the DM/alpha/scattering recovery, so 0 for all keeps the
        # injected dm_delay() the *only* thing incoherent_dedisp has to undo.
        t0_grp.create_dataset("ctime", data=ctime)

    truth["file"] = str(out_path)
    return truth


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="mock_frb.h5", help="Output HDF5 path (single-file mode)")
    p.add_argument("--batch", type=int, default=0, help="If >0, generate this many mock files instead of one")
    p.add_argument("--outdir", default="mock_frbs", help="Output directory for --batch mode")
    p.add_argument("--dm", type=float, default=300.0)
    p.add_argument("--alpha", type=float, default=None, help="Spectral index; random in [-4,-1] if omitted")
    p.add_argument("--freq-shape", choices=["powerlaw", "gaussian"], default="powerlaw")
    p.add_argument("--peak-freq", type=float, default=600.0, help="Gaussian freq_shape: peak frequency (MHz)")
    p.add_argument("--peak-width", type=float, default=100.0, help="Gaussian freq_shape: width (MHz)")
    p.add_argument("--pulse-width-ms", type=float, default=1.5, help="Intrinsic Gaussian pulse width (ms)")
    p.add_argument("--scatter-tau-ms", type=float, default=0.0, help="Scattering time at scatter-freq-ref (ms); 0 = no scattering")
    p.add_argument("--scatter-freq-ref", type=float, default=600.0)
    p.add_argument("--scatter-index", type=float, default=-4.0)
    p.add_argument("--tsys-over-g", type=float, default=1.0)
    p.add_argument("--snr-target", type=float, default=25.0)
    p.add_argument("--dead-frac", type=float, default=0.05)
    p.add_argument("--dead-mode", choices=["random", "block"], default="random")
    p.add_argument("--duration-s", type=float, default=None,
                    help="Recording length (s). Default: auto-computed from --dm so the full "
                         "dispersion sweep fits (see min_duration_for_dm) -- almost always what you want.")
    p.add_argument("--allow-short-duration", action="store_true",
                    help="Override the duration safety check (only if you know what you're doing)")
    p.add_argument("--dt", type=float, default=2.56e-6)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    common = dict(
        dm=args.dm, freq_shape=args.freq_shape, peak_freq=args.peak_freq,
        peak_width_mhz=args.peak_width, pulse_width_s=args.pulse_width_ms * 1e-3,
        scatter_tau_ref_s=args.scatter_tau_ms * 1e-3, scatter_freq_ref=args.scatter_freq_ref,
        scatter_index=args.scatter_index, tsys_over_g=args.tsys_over_g, snr_target=args.snr_target,
        dead_frac=args.dead_frac, dead_mode=args.dead_mode, duration_s=args.duration_s, dt=args.dt,
        allow_short_duration=args.allow_short_duration,
    )

    if args.batch > 0:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(args.batch):
            out_path = outdir / f"mock_frb_{i:03d}.h5"
            truth = gen_mock_frb(out_path, alpha=args.alpha, seed=(args.seed + i if args.seed is not None else None), **common)
            rows.append(truth)
            print(f"[{i+1}/{args.batch}] wrote {out_path} (alpha={truth['alpha_injected']:.3f}, "
                  f"dm={truth['dm']:.1f}, n_dead={truth['n_dead']})")
        csv_path = outdir / "truth.csv"
        with open(csv_path, "w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} mock files + truth table: {csv_path}")
    else:
        truth = gen_mock_frb(args.out, alpha=args.alpha, **common)
        print(f"Wrote {args.out}")
        print(truth)


if __name__ == "__main__":
    main()
