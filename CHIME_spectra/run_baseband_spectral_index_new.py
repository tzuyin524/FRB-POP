#!/usr/bin/env python3
"""
run_baseband_spectral_index.py

Batch version of the "CHIME/FRB Baseband Catalog 1 — waterfall + spectral index"
notebook. Runs the full single-FRB pipeline (load power -> Stokes I -> fill missing
channels -> S/N -> downsample -> RFI excision -> incoherent dedispersion -> final cut
-> boxcar search -> fluence/flux calibration -> average-error correction -> ~20 MHz
frequency binning -> spectral index fit -> Gaussian peak-frequency fit) for every
beamformed HDF5 file found under DEFAULT_INPUT_DIR, saves the "final cut with
boxcar-derived spectral index" figure for each FRB (per-channel spectrum, binned
spectrum, spectral-index fit with its 1-sigma band, and the Gaussian-fit peak
frequency with its 1-sigma band, all drawn on the spectrum panel), and writes a
summary CSV with frb_name, alpha, alpha_err, peak_freq_gauss, peak_freq_gauss_err.

Usage:
    python run_baseband_spectral_index.py
    python run_baseband_spectral_index.py --input-dir /path/to/h5 --output-dir /path/to/out
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("Agg")  # headless / batch-safe backend
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

DEFAULT_INPUT_DIR = "/fred/oz002/thsu/beamformed_files/beamformed_files"
DEFAULT_OUTPUT_DIR = "/home/thsu/FRB_population/CHIME_spectra/baseband_spectra"
DEFAULT_PATTERN = "*.h5"

DOWN_FACTOR = 64
SN_THRESHOLD = 5
WINDOW_MS = 80  # ms, window around the burst for the "final cut"
SMOOTH_WINDOW = 5  # channels, for the smoothed peak-frequency estimate
FREQ_BIN_MHZ = 20.0  # MHz, target width of the frequency bins used for the spectral/Gaussian fits

# Opt-in fallback for channels whose amplitude_norm calibration gain is
# missing/non-positive (see _fill_bad_norm_from_neighbors). Off by default --
# leaving a bad-calibration channel as NaN is the conservative/honest choice;
# only turn this on once you've confirmed (via the [DIAG] empty-bin printout)
# that the gaps are short and neighbor-interpolation is a defensible fill,
# not a real multi-channel calibration-solution gap.
INTERP_BAD_AMPLITUDE_NORM = False
NORM_INTERP_MAX_GAP_CHAN = 3  # max consecutive bad channels a fill is allowed to bridge


def _mad_scale(x):
    """Robust (outlier-resistant) estimate of standard deviation via the median
    absolute deviation, scaled to match a Gaussian's std (factor 1.4826).

    Using this instead of np.nanstd for a cross-channel *reference* threshold
    matters for the iterative RFI-excision loop below: with a plain std, each
    round of flagging shrinks and "cleans" the surviving population, which
    tightens the reference scale and makes the next borderline channel look
    like a bigger outlier than it really is -- a feedback loop that can
    progressively eat an entire contiguous sub-band that was never actually
    RFI. MAD is far less sensitive to the very outliers being clipped, so the
    reference scale stays stable across iterations instead of ratcheting
    tighter each pass.
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad


def _safe_div(numerator, denominator):
    """Elementwise divide, but treat a zero or non-finite denominator as NaN
    (instead of producing +/-inf).

    This matters more than it looks: np.nanmean/np.nanstd/np.nanmedian only
    skip NaN -- they do NOT skip inf. So a single dead or zero-variance
    channel that divides-by-zero into inf, if it ever ends up inside an
    array that a *later* nanmean/nanstd/nanmedian is computed over (e.g. a
    per-channel z-score computed across the whole 1024-channel band), can
    silently blow up that statistic to inf/NaN for the *entire* band, not
    just the one bad channel. Routing zero/degenerate divisions through NaN
    instead keeps that contamination local to the channel that was actually
    bad.
    """
    denominator = np.where(
        (np.asarray(denominator) == 0) | ~np.isfinite(denominator), np.nan, denominator
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / denominator


def _fill_bad_norm_from_neighbors(normalization_factor, freq_id, max_gap_chan=NORM_INTERP_MAX_GAP_CHAN):
    """Opt-in fallback (see INTERP_BAD_AMPLITUDE_NORM) that linearly
    interpolates a replacement calibration gain, per polarization, for
    channels whose amplitude_norm is NaN/non-positive -- but only when a good
    value exists within `max_gap_chan` channels on at least one side.

    Deliberately conservative in two ways:
      1. Never touches anything outside the recorded band
         [freq_id.min(), freq_id.max()] -- padding rows have no data at all,
         so interpolating there would be pure invention, not a fill.
      2. A bad channel is only filled if it's within max_gap_chan of a real
         (finite, positive) neighbor. A short 1-3 channel dropout surrounded
         by good calibration is a reasonable thing to bridge; a wide
         contiguous block of bad channels (e.g. a genuine calibration-solution
         gap spanning tens of channels) is left as NaN rather than
         extrapolated across, since there's no nearby good value to anchor to.

    Returns a new array (input is not modified in place). Prints how many
    channel-pol entries were actually filled.
    """
    filled = normalization_factor.copy()
    row_lo, row_hi = int(freq_id.min()), int(freq_id.max())
    n_filled_total = 0

    for pol in range(filled.shape[1]):
        col = filled[row_lo : row_hi + 1, pol]
        good_idx = np.where(np.isfinite(col))[0]
        if good_idx.size < 2:
            continue
        bad_idx = np.where(~np.isfinite(col))[0]
        if bad_idx.size == 0:
            continue

        interp_vals = np.interp(bad_idx, good_idx, col[good_idx])
        for bi, val in zip(bad_idx, interp_vals):
            left = good_idx[good_idx < bi]
            right = good_idx[good_idx > bi]
            left_gap = (bi - left.max()) if left.size else np.inf
            right_gap = (right.min() - bi) if right.size else np.inf
            if min(left_gap, right_gap) <= max_gap_chan:
                col[bi] = val
                n_filled_total += 1

        filled[row_lo : row_hi + 1, pol] = col

    if n_filled_total > 0:
        print(
            f"    [INFO] Interpolated amplitude_norm for {n_filled_total} "
            f"channel-pol entries from neighbors (max_gap_chan={max_gap_chan}).",
            flush=True,
        )
    return filled


def incoherent_dedisp(array_in, dm, time0, freq, freq_id, dt=2.56e-6, freq_ref=400):
    """Shift each channel in time to align the burst across the band (relative to
    freq_ref), using the DM and each channel's absolute recording start time."""
    array_out = np.zeros_like(array_in) + np.nan
    for t, fr, f_id in zip(time0, freq, freq_id):
        dm_delay = dm / 2.41e-4 * (fr ** -2 - freq_ref ** -2)
        bins_shift = np.round((t - dm_delay) / dt).astype(int)
        array_out[f_id] = np.roll(array_in[f_id], bins_shift, axis=-1)
    return array_out


def boxcar_search(profile, max_width_frac=0.25):
    """Slide dyadic boxcar widths over `profile`, return the (width, start_bin, S/N)
    of the boxcar that maximizes S/N -- i.e. an automatic on-burst duration/position."""
    n = len(profile)
    max_width = max(1, int(n * max_width_frac))
    widths = [2 ** i for i in range(0, int(np.log2(max_width)) + 1)]
    best_snr, best_width, best_start = -np.inf, 1, int(np.nanargmax(profile))
    for w in widths:
        kernel = np.ones(w) / np.sqrt(w)
        conv = np.convolve(np.nan_to_num(profile), kernel, mode="valid")
        idx = int(np.nanargmax(conv))
        if conv[idx] > best_snr:
            best_snr, best_width, best_start = conv[idx], w, idx
    return best_width, best_start, best_snr


def nan_aware_smooth(spectrum, window):
    """Moving-average smoothing across a 1D array that may contain NaNs. NaN channels
    (missing/RFI-masked) don't count toward neighboring averages. Returns (smoothed,
    coverage), where coverage[i] is the number of real (non-NaN) channels that
    contributed to smoothed[i] -- a window with only 1-2 real channels in it (e.g. near
    the edge of a large masked/RFI gap) isn't meaningfully smoothed, so callers should
    treat coverage as a per-point reliability weight."""
    if window <= 1:
        coverage = (~np.isnan(spectrum)).astype(float)
        return spectrum.copy(), coverage
    kernel = np.ones(window)
    filled = np.where(np.isnan(spectrum), 0.0, spectrum)
    valid = (~np.isnan(spectrum)).astype(float)
    numerator = np.convolve(filled, kernel, mode="same")
    coverage = np.convolve(valid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = numerator / coverage
    smoothed[coverage == 0] = np.nan
    return smoothed, coverage


def sample_offburst_noise_spectrum(spectrum_full, normalization_factor, exclude_range):
    """Estimate a per-channel noise level (RMS about zero, in calibrated
    flux units), so the on-burst spectrum can be normalized into
    signal-to-noise units before fitting (rather than left as an additive
    baseline correction -- S/N normalization is a multiplicative reshaping of
    the spectrum, not just an offset removal; see the discussion at the call
    site for why that distinction matters).

    Computed PER CHANNEL from ALL of that channel's own off-burst time
    samples (everything outside `exclude_range`, the on-burst cut window
    given as (start_bin, end_bin) on the same full-resolution time axis as
    `spectrum_full`) -- not from one shared window applied to every channel.

    This matters because incoherent_dedisp shifts each channel by a
    different amount (the DM delay is frequency-dependent, largest for
    low-frequency channels) via np.roll, so a single randomly-chosen time
    window can contain real data for some channels while landing entirely on
    invalid/wrapped samples for others. That was disproportionately failing
    low-frequency channels (see the 400-500 MHz [INFO]/[DIAG] counts).
    Estimating noise independently per channel from its own full off-burst
    coverage sidesteps that: any channel with at least a couple of finite
    off-burst samples gets a usable estimate, and it no longer depends on
    where one shared random window happened to land. It also uses more
    samples than a single short window did, for a more stable std estimate.

    Averages over polarization and calibrates by normalization_factor (same
    calibration as flux_channel_box_raw) before taking the per-channel RMS.
    Dead/missing channels are NaN in `spectrum_full` and therefore
    contribute nothing via np.nanmean/np.nanstd.

    Returns None (the caller should then skip normalization and warn) if
    `exclude_range` covers the entire time axis, leaving no off-burst region
    at all.
    """
    n_total = spectrum_full.shape[-1]
    excl_start = max(0, int(exclude_range[0]))
    excl_end = min(n_total, int(exclude_range[1]))

    if excl_start <= 0 and excl_end >= n_total:
        return None

    off_burst = np.concatenate(
        [spectrum_full[..., :excl_start], spectrum_full[..., excl_end:]], axis=-1
    )  # (freq, npol, n_offburst)
    with np.errstate(invalid="ignore"):
        pol_avg = np.nanmean(off_burst, axis=1)  # (freq, n_offburst)
        calibrated = _safe_div(pol_avg, np.nanmean(normalization_factor, axis=1)[:, None])
        # True RMS about zero (sqrt(mean(x^2))), not std about the sample's
        # own mean -- the off-pulse region is already baseline-subtracted
        # (see spect_off_pulse, now a per-channel median) so it should sit
        # near zero, and RMS-about-zero is the noise-power estimator that
        # matches "rms of the off-pulse region".
        noise_channel = np.sqrt(np.nanmean(calibrated ** 2, axis=1))  # (freq,)
    return noise_channel


def bin_spectrum(freq_full, flux, mask, chan_err=None, bin_width_mhz=FREQ_BIN_MHZ):
    """Block-average `flux` into ~`bin_width_mhz`-wide frequency bins.

    How the binning works:
      1. The channel spacing is measured from `freq_full` (the median spacing
         between consecutive channels, in MHz/channel).
      2. `n_chan_per_bin` = round(bin_width_mhz / channel_spacing) -- the
         number of consecutive channels grouped into one bin so that each
         bin spans close to `bin_width_mhz`.
      3. Channels are grouped in order into bins of that size (the last bin
         may be partial if the channel count doesn't divide evenly).
      4. Within each bin, dead/masked channels (mask == False, including
         NaN flux) are simply skipped -- they contribute zero weight to that
         bin's average. A bin's flux is the plain/uniform mean of whatever
         real channels it contains (channels are NOT inverse-noise weighted
         here); a bin with zero real channels is left as NaN (bin_mask ==
         False) rather than being filled with any placeholder.
      5. Each bin's frequency (bin_freq) is the plain mean of all channel
         frequencies in that bin (frequency is defined for every channel
         regardless of whether its flux is masked).
      6. Each bin's flux error (bin_flux_err):
           - If `chan_err` is given (per-channel noise, same shape/units as
             `flux` -- e.g. from sample_offburst_noise_spectrum), the error is
             propagated directly from those channel noise levels rather than
             measured from their scatter:
                 bin_flux_err = sqrt(sum(chan_err_i^2)) / n_real_channels
             (standard error propagation for a plain/unweighted mean of n
             independent measurements, each with its own known noise
             chan_err_i). Only the real channels in the bin (mask == True)
             contribute to the sum, matching how bin_flux itself is
             averaged. A bin with zero real channels is NaN.
           - If `chan_err` is not given, falls back to the old behavior: the
             standard error of the mean estimated from the in-bin scatter of
             the channels actually present (per-channel scatter /
             sqrt(count)). Bins with fewer than 2 real channels have no
             meaningful scatter to estimate from, so bin_flux_err is NaN for
             those.
         Either way, bin_flux_err is what gets drawn as the error bar on the
         binned spectrum point (matplotlib treats a NaN errorbar as "no error
         bar").

    Returns (bin_freq, bin_flux, bin_flux_err, bin_mask, n_chan_per_bin).
    """
    chan_width = float(np.abs(np.median(np.diff(freq_full))))
    n_chan_per_bin = max(1, int(round(bin_width_mhz / chan_width)))

    n_chan = len(freq_full)
    n_bins = int(np.ceil(n_chan / n_chan_per_bin))
    pad = n_bins * n_chan_per_bin - n_chan

    freq_pad = np.pad(freq_full.astype(float), (0, pad), constant_values=np.nan)
    flux_pad = np.pad(np.where(mask, flux, np.nan), (0, pad), constant_values=np.nan)
    weight_pad = np.pad(mask.astype(float), (0, pad), constant_values=0.0)

    freq_grid = freq_pad.reshape(n_bins, n_chan_per_bin)
    flux_grid = flux_pad.reshape(n_bins, n_chan_per_bin)
    weight_grid = weight_pad.reshape(n_bins, n_chan_per_bin)

    bin_freq = np.nanmean(freq_grid, axis=1)

    count = weight_grid.sum(axis=1)
    flux_filled = np.where(weight_grid > 0, flux_grid, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_flux = flux_filled.sum(axis=1) / count
    bin_flux[count == 0] = np.nan

    if chan_err is not None:
        err_pad = np.pad(np.where(mask, chan_err, np.nan), (0, pad), constant_values=np.nan)
        err_grid = err_pad.reshape(n_bins, n_chan_per_bin)
        err_sq_filled = np.where(weight_grid > 0, err_grid ** 2, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bin_flux_err = np.sqrt(err_sq_filled.sum(axis=1)) / count
        bin_flux_err[count == 0] = np.nan
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            bin_flux_std = np.nanstd(flux_grid, axis=1)
            bin_flux_err = bin_flux_std / np.sqrt(count)
        bin_flux_err[count < 2] = np.nan

    bin_mask = count > 0

    return bin_freq, bin_flux, bin_flux_err, bin_mask, n_chan_per_bin


def gaussian_model(freq, amplitude, mu, sigma, offset):
    """Gaussian bandpass model: baseline offset plus a Gaussian bump centered at mu."""
    return offset + amplitude * np.exp(-0.5 * ((freq - mu) / sigma) ** 2)


def fit_gaussian_peak_freq(
    freq_full, flux_channel_box, mask_box, primary_guess, fallback_peak_freq,
    smoothed_spectrum=None, valid_smoothed=None,
    manual_mu_guess=None, manual_mu_bounds=None, flux_err=None,
):
    """Fit a Gaussian to the on-burst spectrum, zero-weighting dead/masked channels.

    A channel weight of 0 contributes nothing to the least-squares cost, which is
    mathematically identical to leaving that channel out of the fit entirely -- so
    that's how this is implemented (rather than passing weight=0 with an infinite
    sigma into curve_fit, which would blow up on NaN flux values at dead channels).
    Only real, unmasked channels (mask_box) ever influence the fit.

    If `flux_err` is given (same shape as flux_channel_box, e.g. bin_flux_err),
    it's passed to curve_fit as `sigma` with `absolute_sigma=True` so
    well-measured points constrain the fit more than noisy ones, and pcov comes
    back in real physical units rather than rescaled to force a reduced
    chi-square of 1. A point with no usable error estimate (NaN or <=0 -- e.g.
    a bin with only 1 real channel, see bin_spectrum) falls back to the median
    sigma among points that do have one, rather than being dropped or given an
    inconsistent (unweighted) treatment relative to its neighbors. If none of
    the valid points have a usable error at all, the fit is unweighted, same
    as before.

    Many FRB spectra don't show a clean interior turnover within the CHIME band --
    they can look more like a monotonic power-law rise toward one edge, and a poorly
    seeded fit can converge to a degenerate local minimum with mu pinned at the band
    edge. To find a genuine in-band bump automatically (without per-source tuning),
    this tries several candidate starting points for mu and keeps whichever converged
    fit has the lowest residual:
      1. manual_mu_guess, if supplied (highest priority -- for manual override tools)
      2. local maxima detected by scipy.signal.find_peaks on the smoothed spectrum
         (this is what catches a genuine bump like the one at ~500 MHz)
      3. primary_guess (typically peak_guess_binned)
      4. a dense grid spread across the allowed mu range, as a fallback

    The best converged fit is returned as-is, including cases where mu lands at or
    near the allowed mu range's edge.

    manual_mu_bounds, if supplied as (lo, hi), restricts where the fit is allowed to
    place mu -- useful for manually telling the fit "the real bump is somewhere in
    this sub-band" -- without discarding any data (amplitude/offset are still
    informed by the full valid spectrum).

    Returns (mu, popt, pcov) -- popt/pcov are None if no fit converged.
    """
    freq_valid = freq_full[mask_box]
    flux_valid = flux_channel_box[mask_box]

    if freq_valid.size < 5:
        print(
            f"    [WARN] Only {freq_valid.size} valid channel(s) -- too few to fit a "
            f"Gaussian; falling back to the binned peak-flux frequency for peak_freq_gauss.",
            flush=True,
        )
        return fallback_peak_freq, None, None

    sigma_valid = None
    if flux_err is not None:
        sigma_valid = np.asarray(flux_err)[mask_box].astype(float).copy()
        good_sigma = np.isfinite(sigma_valid) & (sigma_valid > 0)
        if good_sigma.any():
            sigma_valid[~good_sigma] = np.nanmedian(sigma_valid[good_sigma])
        else:
            sigma_valid = None  # nothing usable at all -- fall back to unweighted


    freq_min, freq_max = freq_full.min(), freq_full.max()
    band_width = freq_max - freq_min

    if manual_mu_bounds is not None:
        mu_lo, mu_hi = manual_mu_bounds
    else:
        mu_lo, mu_hi = freq_min, freq_max

    sigma0 = max((freq_valid.max() - freq_valid.min()) / 6, 1.0)
    amplitude0 = max(np.nanmax(flux_valid) - np.nanmedian(flux_valid), 1e-6)
    offset0 = np.nanmedian(flux_valid)
    bounds = (
        [0.0, mu_lo, 1e-3, -np.inf],
        [np.inf, mu_hi, band_width, np.inf],
    )

    # ---- Build candidate starting points for mu ----
    candidates = []
    if manual_mu_guess is not None:
        candidates.append(manual_mu_guess)

    if smoothed_spectrum is not None and valid_smoothed is not None:
        ok = valid_smoothed & np.isfinite(smoothed_spectrum)
        if np.any(ok):
            spec_for_peaks = np.where(ok, smoothed_spectrum, -np.inf)
            local_scale = np.nanmax(smoothed_spectrum[ok]) - np.nanmedian(smoothed_spectrum[ok])
            prominence = max(local_scale * 0.1, 1e-9)
            peak_idx, _props = find_peaks(spec_for_peaks, prominence=prominence)
            # restrict to peaks within the allowed mu range
            for idx in peak_idx:
                pf = freq_full[idx]
                if mu_lo <= pf <= mu_hi:
                    candidates.append(pf)

    candidates.append(primary_guess)
    candidates.extend(np.linspace(mu_lo, mu_hi, 15))

    # ---- Multi-start fit: keep whichever converged attempt has the lowest residual ----
    best_popt, best_pcov, best_cost = None, None, np.inf
    for mu0 in candidates:
        p0 = [amplitude0, mu0, sigma0, offset0]
        try:
            popt, pcov = curve_fit(
                gaussian_model, freq_valid, flux_valid, p0=p0, bounds=bounds,
                sigma=sigma_valid, absolute_sigma=True, maxfev=10000
            )
        except (RuntimeError, ValueError):
            continue
        cost = np.sum((flux_valid - gaussian_model(freq_valid, *popt)) ** 2)
        if cost < best_cost:
            best_cost, best_popt, best_pcov = cost, popt, pcov

    if best_popt is None:
        print(
            "    [WARN] Gaussian fit did not converge from any starting point -- "
            "falling back to the binned peak-flux frequency for peak_freq_gauss.",
            flush=True,
        )
        return fallback_peak_freq, None, None

    mu = float(best_popt[1])
    return mu, best_popt, best_pcov


def process_frb(filepath, output_dir, manual_gauss_mu_guess=None, manual_gauss_mu_bounds=None):
    """Run the full single-FRB pipeline on one beamformed h5 file. Saves the final-cut
    boxcar spectral-index figure to output_dir and returns a result dict with
    frb_name, alpha, alpha_err, peak_freq_gauss, peak_freq_gauss_err.
    Raises on failure (caller handles/logs).

    manual_gauss_mu_guess / manual_gauss_mu_bounds: optional manual overrides for the
    Gaussian peak-frequency fit (see fit_gaussian_peak_freq) -- intended for the
    single-source refit tool, not the batch run.
    """

    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        # ---- Load power (stored, coherently dedispersed) ----
        power = f["tiedbeam_power"][:]

        # ---- Total intensity (average over the two linear polarizations) ----
        total_intensity = np.mean(power, axis=1)

        # ---- Fill missing channels with NaN so channel index <-> physical freq lines up ----
        freq_id = f["index_map"]["freq"]["id"][:]
        waterfall = np.zeros([1024, total_intensity.shape[1]]) + np.nan
        waterfall[freq_id] = total_intensity

        # ---- Convert to S/N (per-channel mean/std normalization) ----
        sn = waterfall - np.nanmean(waterfall, axis=1)[:, np.newaxis]
        sn = _safe_div(sn, np.nanstd(sn, axis=1)[:, np.newaxis])

        # ---- Downsample in time ----
        sn = sn[..., : sn.shape[-1] // DOWN_FACTOR * DOWN_FACTOR]
        sn_down = (
            np.nanmean(
                sn.reshape(sn.shape[0], sn.shape[1] // DOWN_FACTOR, DOWN_FACTOR),
                axis=-1,
            )
            * np.sqrt(DOWN_FACTOR)
        )

        # ---- RFI excision (iterative outlier flagging on channel std/mean) ----
        idx_rfi_out = np.zeros(sn_down.shape[0], dtype=bool)
        for _ in range(10):
            spect = np.nanstd(sn_down, axis=1)
            spect -= np.nanmedian(spect)
            spect = _safe_div(spect, _mad_scale(spect))
            idx_rfi = spect > SN_THRESHOLD
            sn_down[idx_rfi] = np.nan
            idx_rfi_out = idx_rfi_out | idx_rfi

            spect = np.nanmean(sn_down, axis=1)
            spect -= np.nanmedian(spect)
            spect = _safe_div(spect, _mad_scale(spect))
            idx_rfi = spect > SN_THRESHOLD
            sn_down[idx_rfi] = np.nan
            idx_rfi_out = idx_rfi_out | idx_rfi

        # ---- Incoherent dedispersion ----
        t0 = f["time0"]["ctime"][:] - f["time0"]["ctime"][:].min()  # seconds
        freq = f["index_map"]["freq"]["centre"][:]
        freq_ref = freq.max()
        dm = f["tiedbeam_power"].attrs["DM_coherent"]
        dt = f.attrs["delta_time"] * DOWN_FACTOR  # seconds, after downsampling

        sn_dedisp = incoherent_dedisp(sn_down, dm, t0, freq, freq_id, dt=dt, freq_ref=freq_ref)

        # ---- Final cut around the burst ----
        window_bins = int(WINDOW_MS / dt / 1000)
        profile = np.nansum(sn_dedisp, axis=0) / np.sqrt(
            np.sum(~np.isnan(np.nansum(sn_dedisp, axis=1)))
        )
        bin_start = np.nanargmax(profile) - window_bins // 2
        bin_end = np.nanargmax(profile) + window_bins // 2
        # Clip to valid bounds -- if the burst sits close to the start/end of the
        # recording, an out-of-range bin_start/bin_end (e.g. negative bin_start) would
        # otherwise slice inconsistently at the downsampled vs full-res scale later on
        # (Python's negative-index wraparound applies differently once scaled by
        # DOWN_FACTOR), causing a boolean-index length mismatch further down.
        bin_start = int(max(0, bin_start))
        bin_end = int(min(bin_end, len(profile)))
        if bin_end <= bin_start:
            raise ValueError(
                f"Burst window collapsed after clipping to array bounds "
                f"(bin_start={bin_start}, bin_end={bin_end}, len(profile)={len(profile)}) "
                f"-- burst peak likely too close to the edge of the recording."
            )

        _slope, _intercept = np.polyfit(freq_id, freq, 1)
        freq_full = _slope * np.arange(1024) + _intercept  # MHz, every channel row

        # freq_full is a linear extrapolation across all 1024 rows, but only the
        # rows in freq_id were ever actually recorded -- rows outside
        # [freq_id.min(), freq_id.max()] are pure extrapolation with no data
        # (permanently NaN), not RFI. Restrict the *plotted* row range to the
        # recorded channels so that padding doesn't show up as a blank band;
        # real RFI/missing-channel gaps *within* that range are left alone.
        row_lo, row_hi = int(freq_id.min()), int(freq_id.max())
        freq_full_plot = freq_full[row_lo : row_hi + 1]
        freq_lim = (freq_full_plot.min(), freq_full_plot.max())

        wfall_cut = sn_dedisp[row_lo : row_hi + 1, bin_start:bin_end]
        profile_cut = profile[bin_start:bin_end]
        time_ms = np.arange(wfall_cut.shape[1]) * dt * 1000

        box_width, box_start, box_snr = boxcar_search(profile_cut)
        box_end = min(box_start + box_width, len(time_ms) - 1)
        box_time_start = time_ms[box_start]
        box_time_end = time_ms[box_end]
        box_width_us = box_width * dt * 1e6

        frb_name = filepath.stem.split("_")[0]

        # ---- Fluence / flux calibration setup ----
        power_clean_full = np.zeros([1024, 2, total_intensity.shape[1]]) + np.nan
        power_clean_full[freq_id] = power
        power_clean_full[np.isnan(np.nanmean(sn_down, axis=1))] = np.nan
        power_clean_full = incoherent_dedisp(
            power_clean_full, dm, t0, freq, freq_id, dt=f.attrs["delta_time"], freq_ref=freq_ref
        )
        power_clean = power_clean_full[..., bin_start * DOWN_FACTOR : bin_end * DOWN_FACTOR]

        off_pulse = profile.copy() - np.nanmedian(profile)
        while np.nanmax(np.abs(off_pulse)) > 3:
            off_pulse[np.abs(off_pulse) > 3] = np.nan
            off_pulse -= np.nanmean(off_pulse)
            off_pulse = _safe_div(off_pulse, np.nanstd(off_pulse))
        off_pulse = off_pulse[bin_start:bin_end]
        idx_off = np.repeat(~np.isnan(off_pulse), DOWN_FACTOR)
        bg = power_clean[..., idx_off]

        # Median (not mean) of the off-pulse region, per channel/pol -- more
        # robust than the mean against a stray off-pulse outlier sample (e.g.
        # a brief RFI blip that survived the earlier excision) pulling the
        # baseline estimate away from the bulk of the noise.
        spect_off_pulse = np.nanmedian(bg, axis=-1)
        spectrum = power_clean - spect_off_pulse[..., np.newaxis]
        # Same background subtraction applied to the *full* (unsliced) recording, so
        # a random off-burst window can be drawn from anywhere outside the on-burst
        # cut for the average-error estimate below.
        spectrum_full = power_clean_full - spect_off_pulse[..., np.newaxis]

        normalization_factor = np.zeros([1024, 2]) + np.nan
        if "amplitude_norm" not in f["index_map"]:
            raise KeyError(
                f"{filepath.name}: index_map/amplitude_norm is missing from this file "
                "entirely (not just a few bad channels) -- there is no calibration "
                "gain available at all, so flux/fluence can't be computed for this "
                "recording. Skipping (see the per-file failure summary at the end)."
            )
        normalization_factor[freq_id] = f["index_map"]["amplitude_norm"][:]
        normalization_factor[np.isnan(np.nanmean(sn_down, axis=1))] = np.nan
        # A zero/negative amplitude_norm is not a usable calibration gain --
        # left in, it would silently flip the sign of (or blow up) that
        # channel's flux via the divide below instead of dropping cleanly to
        # NaN like a missing gain does. Treat it as bad up front so it's
        # counted consistently everywhere downstream (this matches the
        # norm_bad definition the [DIAG] empty-bin printout already uses).
        normalization_factor[normalization_factor <= 0] = np.nan

        if INTERP_BAD_AMPLITUDE_NORM:
            normalization_factor = _fill_bad_norm_from_neighbors(normalization_factor, freq_id)

        # ---- Boxcar-derived spectral index ----
        box_start_fullres = box_start * DOWN_FACTOR
        box_end_fullres = min(box_end * DOWN_FACTOR, spectrum.shape[-1])

        # Pol- and on-burst-time-averaged power, per channel -- computed once and
        # reused below. `numerator_full` (pre-calibration) is kept as its own name
        # purely so the [DIAG] empty-bin breakdown further down can tell "on-burst
        # window itself was NaN" apart from "normalization was NaN" apart from "the
        # off-burst error subtraction below is what killed it"; it used to be
        # recomputed from scratch with a second identical nanmean/nanmean call.
        numerator_full = np.nanmean(
            np.nanmean(spectrum[..., box_start_fullres:box_end_fullres], axis=1), axis=1
        )
        # _safe_div (rather than a bare "/") so a channel with an exactly-zero
        # or non-finite mean normalization_factor gives NaN instead of +/-inf --
        # an inf here would otherwise be able to contaminate any later
        # nanmean/nanstd/nanmedian computed across the whole band (see
        # _safe_div's docstring for why that matters).
        flux_channel_box_raw = _safe_div(numerator_full, np.nanmean(normalization_factor, axis=1))

        # ---- Per-channel noise estimate: sample a random 10 ms off-burst
        # window (anywhere outside the on-burst cut) and estimate each
        # channel's noise std from it. The on-burst flux is then DIVIDED by
        # this per-channel noise (see below), so everything downstream --
        # binning, the alpha fit, the Gaussian peak fit, the plotted
        # spectrum -- operates on a signal-to-noise spectrum, not a physical
        # flux-density spectrum. The reported alpha is therefore the
        # power-law index of S/N(freq), not of flux density S(freq).
        noise_channel = sample_offburst_noise_spectrum(
            spectrum_full,
            normalization_factor,
            exclude_range=(bin_start * DOWN_FACTOR, bin_end * DOWN_FACTOR),
        )
        if noise_channel is not None:
            # sample_offburst_noise_spectrum's noise_channel is the std of
            # INDIVIDUAL (pol-averaged) off-burst time samples -- not yet
            # averaged down. flux_channel_box_raw, the quantity this error is
            # for, is itself a MEAN over n_box_samples on-burst time samples
            # (see the nanmean over box_start_fullres:box_end_fullres above),
            # so its noise is smaller than the single-sample noise by the
            # standard error-of-the-mean factor 1/sqrt(n_box_samples).
            # Skipping this scaling silently inflates every propagated bin
            # error (and therefore every fit weight) by sqrt(n_box_samples),
            # which for a typical box width of a few hundred full-res samples
            # is roughly a 10-20x overestimate.
            n_box_samples = max(1, box_end_fullres - box_start_fullres)
            noise_channel = noise_channel / np.sqrt(n_box_samples)
            # A channel can only get a meaningful error bar if it has a real
            # (finite, positive) noise estimate from its own off-burst
            # coverage. A channel with no usable noise estimate (e.g.
            # genuinely no off-burst samples at all for that channel) is
            # excluded from the fit entirely (via mask_box below) rather than
            # kept with a fabricated/inconsistent error.
            noise_bad = ~np.isfinite(noise_channel) | (noise_channel <= 0)
            if np.any(noise_bad):
                print(
                    f"    [INFO] {int(noise_bad.sum())} channel(s) had no usable "
                    f"off-burst noise estimate -- excluding those channels from "
                    f"the binned fit (no error bar available).",
                    flush=True,
                )

            # ---- Normalize flux by its per-channel noise (S/N units) ----
            # flux_channel_box_raw is in physical (Jy) units; noise_channel is
            # that same channel's noise, already scaled (above) to match the
            # on-burst MEAN's noise rather than a single sample's. Dividing
            # converts the spectrum to S/N units and is what's carried
            # through binning and both fits from here on.
            #
            # This also fixes the per-channel error that goes into
            # bin_spectrum: by construction, Var(raw/noise) =
            # Var(raw)/noise^2 = noise^2/noise^2 = 1 (to first order, treating
            # noise_channel itself as a fixed/known calibration rather than a
            # noisy estimate). So every valid channel now carries the SAME
            # per-channel error of 1 into the bin average, instead of the
            # channel-dependent noise_channel used previously -- a bin's
            # error becomes the textbook sqrt(N)/N = 1/sqrt(N) standard error
            # of a mean of N unit-variance points (see bin_spectrum).
            flux_channel_box = _safe_div(flux_channel_box_raw, noise_channel)
            flux_channel_box[noise_bad] = np.nan
            chan_err_for_binning = np.where(noise_bad, np.nan, 1.0)
        else:
            print(
                f"    [WARN] No off-burst region available for {frb_name} (on-burst "
                f"cut spans the whole recording) -- cannot estimate per-channel "
                f"noise, so the spectrum CANNOT be normalized to S/N units for "
                f"this file. Falling back to the raw physical-flux spectrum with "
                f"scatter-based bin errors -- treat this file's alpha/peak_freq "
                f"as not directly comparable to the S/N-normalized files.",
                flush=True,
            )
            flux_channel_box = flux_channel_box_raw
            chan_err_for_binning = None

        # Only exclude channels that are truly dead/RFI-flagged (non-finite) or
        # have no valid frequency -- NOT channels that simply have noise-negative
        # flux. A per-channel flux>0 cut here would throw away real, live
        # channels before they ever reach the bin average: for a marginal-S/N
        # burst, individual channels routinely dip below zero from noise even
        # when the underlying signal is real, and averaging ~n_chan_per_bin of
        # them together (see bin_spectrum) is exactly what recovers that signal
        # without bias. A bin can still come out net-negative if there's truly
        # no signal in it -- that's handled downstream (fit_mask, log-scale
        # plot) by requiring bin-level flux > 0, not per-channel.
        mask_box = np.isfinite(flux_channel_box) & np.isfinite(freq_full)

        # ---- Bin to ~FREQ_BIN_MHZ-wide channels (see bin_spectrum for how the
        # binning and dead-channel skipping works -- NaN/dead channels are
        # skipped, e.g. [1,1,1,1,nan,1] bins as a mean of 5, not 6), then fit
        # the spectral index and the Gaussian peak on the binned spectrum.
        bin_freq, bin_flux, bin_flux_err, bin_mask, n_chan_per_bin = bin_spectrum(
            freq_full, flux_channel_box, mask_box,
            chan_err=chan_err_for_binning, bin_width_mhz=FREQ_BIN_MHZ,
        )

        # Diagnostic: for any empty bin that falls inside the actually-recorded
        # band (not the extrapolated padding), report how many channels landed
        # in it and how many of those were RFI-excised (idx_rfi_out) vs simply
        # never recorded (missing from freq_id) -- so an empty bin can be
        # attributed to a specific cause instead of just "no data".
        recorded_freq_lo = min(freq_full[row_lo], freq_full[row_hi])
        recorded_freq_hi = max(freq_full[row_lo], freq_full[row_hi])
        empty_in_band = ~bin_mask & (bin_freq >= recorded_freq_lo) & (bin_freq <= recorded_freq_hi)
        if np.any(empty_in_band):
            recorded = np.zeros(1024, dtype=bool)
            recorded[freq_id] = True
            n_chan = len(freq_full)
            n_pad = int(np.ceil(n_chan / n_chan_per_bin)) * n_chan_per_bin - n_chan
            recorded_pad = np.pad(recorded, (0, n_pad), constant_values=False)
            rfi_pad = np.pad(idx_rfi_out, (0, n_pad), constant_values=True)
            recorded_grid = recorded_pad.reshape(-1, n_chan_per_bin)
            rfi_grid = rfi_pad.reshape(-1, n_chan_per_bin)

            norm_mean = np.nanmean(normalization_factor, axis=1)
            norm_bad = ~np.isfinite(norm_mean) | (norm_mean <= 0)
            norm_bad_pad = np.pad(norm_bad, (0, n_pad), constant_values=True)
            norm_bad_grid = norm_bad_pad.reshape(-1, n_chan_per_bin)

            # Two more possible culprits, checked in the order they happen in
            # the pipeline: (1) the on-burst window itself (numerator_full,
            # before normalization/S-N-normalization) was already NaN -- e.g.
            # this channel's raw recorded data doesn't actually cover the
            # on-burst time window; (2) numerator+normalization were both
            # fine, but this channel had no usable off-burst noise estimate
            # (see sample_offburst_noise_spectrum), making noise_channel
            # bad/NaN and excluding an otherwise-good flux value.
            numerator_bad = ~np.isfinite(numerator_full)
            numerator_bad_pad = np.pad(numerator_bad, (0, n_pad), constant_values=True)
            numerator_bad_grid = numerator_bad_pad.reshape(-1, n_chan_per_bin)

            if noise_channel is not None:
                error_bad = ~np.isfinite(noise_channel) | (noise_channel <= 0)
            else:
                error_bad = np.zeros(1024, dtype=bool)
            error_bad_pad = np.pad(error_bad, (0, n_pad), constant_values=False)
            error_bad_grid = error_bad_pad.reshape(-1, n_chan_per_bin)

            for i in np.where(empty_in_band)[0]:
                not_rfi = recorded_grid[i] & ~rfi_grid[i]
                n_recorded = int(recorded_grid[i].sum())
                n_rfi = int((recorded_grid[i] & rfi_grid[i]).sum())
                n_norm_bad = int((not_rfi & norm_bad_grid[i]).sum())
                remaining = not_rfi & ~norm_bad_grid[i]
                n_numerator_bad = int((remaining & numerator_bad_grid[i]).sum())
                remaining = remaining & ~numerator_bad_grid[i]
                n_error_bad = int((remaining & error_bad_grid[i]).sum())
                n_unexplained = int((remaining & ~error_bad_grid[i]).sum())
                print(
                    f"    [DIAG] empty bin at {bin_freq[i]:.1f} MHz: "
                    f"{n_recorded}/{n_chan_per_bin} recorded, {n_rfi} RFI-excised, "
                    f"{n_norm_bad} bad amplitude_norm, "
                    f"{n_numerator_bad} on-burst window itself NaN (no real data "
                    f"there), {n_error_bad} have no usable noise estimate for S/N "
                    f"normalization, {n_unexplained} still "
                    f"unexplained.",
                    flush=True,
                )

        fit_mask = bin_mask & np.isfinite(bin_flux) & (bin_flux > 0) & np.isfinite(bin_freq)
        log_f_box = np.log10(bin_freq[fit_mask])
        log_s_box = np.log10(bin_flux[fit_mask])

        # Weight each bin by its inverse log-flux error (np.polyfit's `w` is
        # 1/sigma, not 1/sigma^2), converting the linear-space bin_flux_err via
        # standard error propagation: sigma[log10(x)] = sigma[x] / (x * ln 10).
        # A bin with no usable error (NaN/<=0, e.g. only 1 real channel behind
        # it -- see bin_spectrum) falls back to the median sigma among bins
        # that do have one, so it still contributes but isn't given an
        # inconsistent implicit weight relative to its neighbors. If fewer
        # than 2 bins have a usable error at all, weighting isn't meaningful
        # and the fit falls back to unweighted (matches prior behavior).
        sigma_log_box = bin_flux_err[fit_mask] / (bin_flux[fit_mask] * np.log(10))
        good_sigma_box = np.isfinite(sigma_log_box) & (sigma_log_box > 0)
        if good_sigma_box.sum() >= 2:
            sigma_log_box = sigma_log_box.copy()
            sigma_log_box[~good_sigma_box] = np.nanmedian(sigma_log_box[good_sigma_box])
            coeffs_box, cov_box = np.polyfit(
                log_f_box, log_s_box, 1, w=1.0 / sigma_log_box, cov="unscaled"
            )
        else:
            print(
                "    [INFO] Fewer than 2 bins have a usable flux error -- fitting "
                "the spectral index unweighted.",
                flush=True,
            )
            coeffs_box, cov_box = np.polyfit(log_f_box, log_s_box, 1, cov=True)
        alpha_box, const_box = coeffs_box
        alpha_box_err = np.sqrt(cov_box[0, 0])
        log_fit_box = alpha_box * log_f_box + const_box
        # Propagate the (alpha, const) covariance to a 1-sigma band on the fit line:
        # Var[log10(flux_pred)] = log_f^2 * Var[alpha] + 2*log_f * Cov[alpha, const] + Var[const]
        log_fit_box_err = np.sqrt(
            log_f_box ** 2 * cov_box[0, 0] + 2 * log_f_box * cov_box[0, 1] + cov_box[1, 1]
        )
        fit_line_box = 10 ** log_fit_box
        fit_line_box_lo = 10 ** (log_fit_box - log_fit_box_err)
        fit_line_box_hi = 10 ** (log_fit_box + log_fit_box_err)

        # Internal seed/fallback for the Gaussian fit only -- not reported as an
        # output peak frequency in its own right.
        peak_guess_binned = float(
            bin_freq[np.nanargmax(np.where(bin_mask, bin_flux, -np.inf))]
        )

        # Frequency of peak flux density -- Gaussian fit to the ~FREQ_BIN_MHZ binned
        # spectrum, skipping bins with no real channels behind them (bin_mask).
        # Automatically seeded from detected local bumps in the binned spectrum
        # (falls back to peak_guess_binned if none are found), or from
        # manual_gauss_mu_guess/manual_gauss_mu_bounds if supplied.
        peak_freq_gauss, gauss_popt, gauss_pcov = fit_gaussian_peak_freq(
            bin_freq, bin_flux, bin_mask,
            primary_guess=peak_guess_binned, fallback_peak_freq=peak_guess_binned,
            smoothed_spectrum=bin_flux, valid_smoothed=bin_mask,
            manual_mu_guess=manual_gauss_mu_guess, manual_mu_bounds=manual_gauss_mu_bounds,
            flux_err=bin_flux_err,
        )
        peak_freq_gauss_err = (
            float(np.sqrt(gauss_pcov[1, 1])) if gauss_pcov is not None else None
        )

        # ---- Figure: final cut, boxcar-derived spectral index ----
        fig = plt.figure(figsize=(7, 6), dpi=150)
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 3), wspace=0.0, hspace=0.0)
        ax_time = fig.add_subplot(gs[0, 0])
        ax_wfall = fig.add_subplot(gs[1, 0], sharex=ax_time)
        ax_spec = fig.add_subplot(gs[1, 1], sharey=ax_wfall)

        ax_wfall.imshow(
            wfall_cut,
            interpolation="none",
            aspect="auto",
            origin="upper",
            extent=[time_ms[0], time_ms[-1], freq_lim[0], freq_lim[1]],
        )
        ax_wfall.set_xlabel("Time (ms)")
        ax_wfall.set_ylabel("Frequency (MHz)")

        ax_time.plot(time_ms, profile_cut, lw=1, color="tab:blue")
        ax_time.axvspan(box_time_start, box_time_end, color="tab:blue", alpha=0.25, lw=0)
        ax_time.set_xlim(time_ms[0], time_ms[-1])
        plt.setp(ax_time.get_xticklabels(), visible=False)
        ax_time.tick_params(axis="x", direction="in")
        ax_time.set_yticks([])
        ax_time.text(
            0.97,
            0.92,
            f"{frb_name}\n{dm:.1f} pc/cc\n{box_width_us:.1f} us\n"
            f"$\\alpha$={alpha_box:.2f}$\\pm${alpha_box_err:.2f}",
            transform=ax_time.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

        ax_spec.errorbar(
            bin_flux[bin_mask], bin_freq[bin_mask],
            xerr=bin_flux_err[bin_mask],
            fmt="o", ms=3, color="tab:blue", ecolor="tab:blue",
            elinewidth=1, capsize=2, zorder=2,
        )
        # ax_spec.fill_betweenx(
        #     bin_freq[fit_mask], fit_line_box_lo, fit_line_box_hi,
        #     color="black", alpha=0.2, zorder=2, lw=0,
        # )
        ax_spec.plot(fit_line_box, bin_freq[fit_mask], "-", lw=1, zorder=3, color="black", label="Spectral index fit", alpha = 0.65)
        if gauss_popt is not None:
            freq_grid = np.linspace(freq_lim[0], freq_lim[1], 300)
            flux_gauss_curve = gaussian_model(freq_grid, *gauss_popt)
            ax_spec.plot(flux_gauss_curve, freq_grid, color="tab:blue", ls="-.", lw=1, label="Gaussian fit")
        if peak_freq_gauss_err is not None and np.isfinite(peak_freq_gauss_err):
            # A poorly-constrained Gaussian fit can return a huge mu uncertainty
            # (e.g. a nearly-flat spectrum with very few binned points). Clip the
            # shaded span to just outside the physical band rather than letting
            # it (and therefore the shared y-axis autoscale) run off to +/-1e5
            # MHz.
            band_width = freq_lim[1] - freq_lim[0]
            span_lo = max(peak_freq_gauss - peak_freq_gauss_err, freq_lim[0] - 0.1 * band_width)
            span_hi = min(peak_freq_gauss + peak_freq_gauss_err, freq_lim[1] + 0.1 * band_width)
            ax_spec.axhspan(span_lo, span_hi, color="tab:blue", alpha=0.15, zorder=1, lw=0)
        ax_spec.axhline(peak_freq_gauss, color="tab:blue", ls="-", lw=1.2, label="Peak Frequency")
        ax_spec.set_xscale("log")
        ax_spec.xaxis.set_major_locator(LogLocator(base=10, numticks=3))
        ax_spec.xaxis.set_minor_formatter(NullFormatter())
        plt.setp(ax_spec.get_xticklabels(), fontsize=7, rotation=45)
        plt.setp(ax_spec.get_yticklabels(), visible=False)
        ax_spec.tick_params(axis="y", direction="in")
        ax_spec.set_xlabel("S/N", fontsize=8)
        ax_spec.legend(fontsize=5, loc="upper right", framealpha=0.8)

        # Backstop: imshow's extent already implies this range, but lock it
        # explicitly so nothing plotted on the shared ax_spec axis (an
        # errorbar, axhspan, etc.) can silently drag the whole y-axis outside
        # the physical band.
        ax_wfall.set_ylim(freq_lim[0], freq_lim[1])

        out_png = Path(output_dir) / f"{filepath.stem}_boxcar_spectral_index.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    return {
        "frb_name": frb_name,
        "dm": dm,
        "freq_top": freq_lim[1],
        "freq_low": freq_lim[0],
        "alpha": alpha_box,
        "alpha_err": alpha_box_err,
        "peak_freq_gauss": peak_freq_gauss,
        "peak_freq_gauss_err": peak_freq_gauss_err,
        "png_path": str(out_png),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing beamformed h5 files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save PNGs and the summary CSV")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern for input files (default: *.h5)")
    parser.add_argument("--recursive", action="store_true", help="Search input_dir recursively for matching files")
    parser.add_argument(
        "--csv-name", default="baseband_spectral_index_summary.csv", help="Filename for the output summary CSV"
    )
    parser.add_argument(
        "--interp-bad-norm",
        action="store_true",
        help=(
            "Interpolate amplitude_norm for channels with missing/non-positive "
            "calibration gain from nearby good channels (see "
            "_fill_bad_norm_from_neighbors / NORM_INTERP_MAX_GAP_CHAN). Off by "
            "default -- only enable after checking via [DIAG] output that the "
            "gaps are short, not a wide calibration-solution gap."
        ),
    )
    args = parser.parse_args()

    if args.interp_bad_norm:
        global INTERP_BAD_AMPLITUDE_NORM
        INTERP_BAD_AMPLITUDE_NORM = True

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        files = sorted(input_dir.rglob(args.pattern))
    else:
        files = sorted(input_dir.glob(args.pattern))

    if not files:
        print(f"No files matching '{args.pattern}' found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s) in {input_dir}")

    results = []
    failures = []

    for i, filepath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing {filepath.name} ...", flush=True)
        try:
            result = process_frb(filepath, output_dir)
            results.append(result)
            print(
                f"    -> alpha = {result['alpha']:.3f} +/- {result['alpha_err']:.3f}, "
                f"peak_freq_gauss = {result['peak_freq_gauss']:.1f} "
                f"+/- {result['peak_freq_gauss_err'] if result['peak_freq_gauss_err'] is not None else float('nan'):.1f} MHz"
            )
        except Exception as exc:
            print(f"    !! FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            failures.append((filepath.name, str(exc)))

    csv_path = output_dir / args.csv_name
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "frb_name",
                "dm",
                "freq_top",
                "freq_low",
                "alpha",
                "alpha_err",
                "peak_freq_gauss",
                "peak_freq_gauss_err",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "frb_name": r["frb_name"],
                    "dm": r["dm"],
                    "freq_top": r["freq_top"],
                    "freq_low": r["freq_low"],
                    "alpha": r["alpha"],
                    "alpha_err": r["alpha_err"],
                    "peak_freq_gauss": r["peak_freq_gauss"],
                    "peak_freq_gauss_err": r["peak_freq_gauss_err"],
                }
            )

    print()
    print(f"Done. {len(results)} succeeded, {len(failures)} failed out of {len(files)} total.")
    print(f"Summary CSV: {csv_path}")
    if failures:
        print("Failed files:")
        for name, err in failures:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
