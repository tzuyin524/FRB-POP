"""
Batch-run the CHIME/FRB Catalog-1 waterfall / boxcar spectral-index pipeline
over every *_waterfall.h5 file in INPUT_DIR.

Mirrors chimecat1_spectral_index.ipynb (boxcar-selected spectrum version):
  - loads wfall/spec/ts/model_* from the h5 file
  - RFI-masks the waterfall
  - finds the burst peak/width/S-N via a matched-filter boxcar search
  - per channel: subtracts the median of the off-pulse (non-burst) region
    as a baseline, averages the baseline-subtracted flux over just the
    boxcar's on-pulse time bins, then divides by that channel's off-pulse
    rms noise to get one S/N-normalized spectral value per channel. Since
    the channel's own noise is divided out, each (live) channel's error
    on that normalized value is just 1/sqrt(n_on_pulse) -- the standard
    error on the mean, in units of the channel's noise.
  - bins the per-channel spectrum and its error to ~5-MHz-wide frequency
    bins (factor chosen per-file), drops dead (always-zero/NaN) channels,
    and fits the spectral index (F_nu ~ nu^alpha) on the binned spectrum
    with an error-weighted least-squares fit in log-log space
  - re-plots waterfall + time series (boxcar window shaded, model overlaid)
    + boxcar spectrum (with power-law fit AND Gaussian fit both overlaid),
    saved as <frb_name>_wfall_spectral_index.png
  - appends frb_name, dm, freq_top, freq_low (top/bottom of the observing
    band, in MHz, from the file's frequency axis), alpha, alpha_err (from
    the boxcar power-law fit), peak_freq (simple argmax of the boxcar
    spectrum, no fit), and peak_freq_gauss / peak_freq_gauss_err (centroid
    +/- 1-sigma uncertainty of a Gaussian fit to that same boxcar
    spectrum, with dead/always-zero channels zero-weighted, in MHz) to a
    summary CSV

Adjust the CONFIG block below, then run with:
    python plot_waterfalls.py
"""

import csv
import glob
import os
import traceback

import h5py
import matplotlib

matplotlib.use("Agg")  # headless -- no display needed on a cluster
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.stats
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
INPUT_DIR = "/fred/oz002/thsu/waterfalls"
OUTPUT_DIR = "/home/thsu/FRB_population/CHIME_spectra/chimecat1_spectra"
CSV_PATH = os.path.join(OUTPUT_DIR, "chimecat1_alpha.csv")

FBIN = 64        # frequency binning factor for the *waterfall display* (fit uses full res)
TBIN = 1         # time binning factor for the *waterfall display* (fit uses full res)
FIT_FMIN = None  # MHz, restrict spectral-index fit band (or None)
FIT_FMAX = None  # MHz
TARGET_FREQ_BIN_MHZ = 5  # target width (MHz) of each freq bin for the
                          # boxcar spectrum panel/fit; actual factor is
                          # chosen per-file (nearest divisor of nchan)

# ----------------------------------------------------------------------
# 2. Helper functions (unchanged from chimecat1_spectral_index.ipynb)
# ----------------------------------------------------------------------


def boxcar_kernel(width):
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)


def find_burst(ts, min_width=1, max_width=128):
    """Matched-filter search over boxcar widths to find the peak time bin,
    best boxcar width, and S/N of the burst."""
    min_width = int(min_width)
    max_width = int(max_width)
    widths = list(range(min_width, min(max_width + 1, len(ts) - 2)))
    snrs = np.empty_like(widths, dtype=float)
    peaks = np.empty_like(widths, dtype=int)
    for i, w in enumerate(widths):
        convolved = scipy.signal.convolve(ts, boxcar_kernel(w), mode="same")
        peaks[i] = np.nanargmax(convolved)
        snrs[i] = convolved[peaks[i]]
    best_idx = np.nanargmax(snrs)
    return peaks[best_idx], widths[best_idx], snrs[best_idx]


def bin_freq_channels(data, fbin_factor=4):
    """Average adjacent freq channels (display only). Falls back to the
    largest factor <= fbin_factor that evenly divides nchan, since not
    every FRB's waterfall in a batch necessarily has the same channel
    count (the notebook version raises instead, since it only ever
    handles one file at a time)."""
    num_chan = data.shape[0]
    factor = fbin_factor
    while factor > 1 and num_chan % factor != 0:
        factor -= 1
    if factor != fbin_factor:
        print(f"    [warn] FBIN={fbin_factor} does not divide nchan={num_chan}; "
              f"using FBIN={factor} for display binning instead")
    return np.nanmean(
        data.reshape((num_chan // factor, factor) + data.shape[1:]), axis=1
    )


def bin_time_samples(data, tbin_factor=4):
    """Average adjacent time samples together (display only). Trims any
    remainder samples that don't fit evenly into tbin_factor."""
    ntime = data.shape[1]
    tbin_factor = max(1, min(tbin_factor, ntime))
    ntime_trim = (ntime // tbin_factor) * tbin_factor
    data = data[:, :ntime_trim]
    new_shape = data.shape[:1] + (ntime_trim // tbin_factor, tbin_factor)
    return np.nanmean(data.reshape(new_shape), axis=2)


def bin_1d(arr, factor):
    n = arr.shape[0]
    n_trim = (n // factor) * factor
    return np.nanmean(arr[:n_trim].reshape(-1, factor), axis=1)


def bin_1d_sum(arr, factor):
    n = arr.shape[0]
    n_trim = (n // factor) * factor
    return np.nansum(arr[:n_trim].reshape(-1, factor), axis=1)


def freq_bin_factor_for_width(plot_freq, nchan, target_mhz=20):
    """Choose an integer number of native frequency channels to average
    together so each output bin is as close as possible to `target_mhz`
    wide, restricted to values that evenly divide nchan (channel count/
    width can differ between files in a batch, so this is computed fresh
    per file rather than hardcoded)."""
    chan_width = abs(np.nanmedian(np.diff(plot_freq)))
    if not np.isfinite(chan_width) or chan_width <= 0:
        return 1
    ideal = max(1, int(round(target_mhz / chan_width)))
    divisors = [d for d in range(1, nchan + 1) if nchan % d == 0]
    factor = min(divisors, key=lambda d: abs(d - ideal))
    return factor


def argmax_peak_frequency(freq, spec):
    """
    Simple, fit-free peak frequency: the frequency bin at which `spec` is
    maximal. No weighting/dead-channel handling beyond whatever filtering
    the caller already applied to `freq`/`spec` -- this is meant to be a
    quick, assumption-free cross-check against the Gaussian-fit centroid
    from `fit_peak_frequency`, not a replacement for it.
    """
    freq = np.asarray(freq, dtype=float)
    spec = np.asarray(spec, dtype=float)
    finite = np.isfinite(freq) & np.isfinite(spec)
    if finite.sum() == 0:
        raise ValueError("No finite frequency/spectrum bins -- cannot find peak_freq.")
    return float(freq[finite][np.nanargmax(spec[finite])])


def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def fit_peak_frequency(freq, spec, dead_frac, dead_thresh=0.5):
    """
    Fit a Gaussian to `spec` vs `freq` and return its centroid (mu) as the
    peak frequency, in MHz, along with the 1-sigma uncertainty on that
    centroid (from the fit covariance matrix) and the full best-fit
    parameter vector (amp, mu, sigma, offset) so the Gaussian can be
    overplotted later.

    `dead_frac` is, per frequency bin, the fraction of underlying full-res
    channels that were flagged dead (always-zero/non-finite) before binning.
    Bins that are more than `dead_thresh` dead are given zero weight in the
    fit (via an effectively-infinite sigma in curve_fit) rather than being
    dropped from the arrays outright -- so a fully-dead bin cannot pull the
    fit toward zero, but it also doesn't shrink the array/change indexing.

    Falls back to a simple weighted argmax (still zero-weighting dead bins)
    if the Gaussian fit fails to converge; in that case the returned error
    is NaN and the returned popt is None (nothing to overplot).

    Returns
    -------
    mu_fit : float
        Peak (centroid) frequency in MHz.
    mu_err : float
        1-sigma uncertainty on mu_fit, in MHz (NaN if the fit failed/fell
        back to argmax).
    popt : ndarray or None
        Best-fit Gaussian parameters (amp, mu, sigma, offset), or None if
        the fit failed.
    """
    freq = np.asarray(freq, dtype=float)
    spec = np.asarray(spec, dtype=float)
    dead_frac = np.asarray(dead_frac, dtype=float)

    finite = np.isfinite(freq) & np.isfinite(spec)
    weight = np.where(dead_frac > dead_thresh, 0.0, 1.0)
    weight[~finite] = 0.0

    if weight.sum() < 4:
        raise ValueError("Fewer than 4 live (non-dead) frequency bins -- "
                          "cannot fit a peak frequency.")

    # sigma = 1/weight for curve_fit's weighted least squares; zero-weight
    # bins get a very large (not infinite, to stay numerically safe) sigma
    # so they contribute ~nothing to the fit.
    sigma_arr = np.where(weight > 0, 1.0, 1.0e8)
    spec_filled = np.where(finite, spec, 0.0)  # value is irrelevant where weight=0

    amp0 = np.nanmax(spec_filled * weight) - np.nanmin(spec_filled[weight > 0])
    mu0 = freq[weight > 0][np.nanargmax(spec_filled[weight > 0])]
    sigma0 = (freq[weight > 0].max() - freq[weight > 0].min()) / 6.0
    offset0 = np.nanmedian(spec_filled[weight > 0])
    p0 = [amp0 if amp0 > 0 else 1.0, mu0, max(sigma0, 1.0), offset0]

    try:
        popt, pcov = curve_fit(
            gaussian, freq, spec_filled, p0=p0, sigma=sigma_arr,
            absolute_sigma=False, maxfev=10000,
        )
        mu_fit = popt[1]
        if not (freq[weight > 0].min() <= mu_fit <= freq[weight > 0].max()):
            raise RuntimeError("Fitted centroid fell outside the live frequency range")
        if np.all(np.isfinite(pcov)):
            mu_err = float(np.sqrt(pcov[1, 1]))
        else:
            mu_err = float("nan")
        return float(mu_fit), mu_err, popt
    except (RuntimeError, ValueError) as exc:
        print(f"    [warn] Gaussian peak-frequency fit failed ({exc}); "
              "falling back to zero-weighted argmax")
        live = weight > 0
        mu_fallback = float(freq[live][np.nanargmax(spec_filled[live])])
        return mu_fallback, float("nan"), None


def mask_rfi(wfall, spec, model_wfall=None, var_factor=3):
    """Flag channels with anomalously high variance or spectrum outliers,
    exactly as in the CHIME/FRB tutorial."""
    q1 = np.nanquantile(spec, 0.25)
    q3 = np.nanquantile(spec, 0.75)
    iqr = q3 - q1

    channel_variance = np.nanvar(wfall, axis=1)
    mean_channel_variance = np.nanmean(channel_variance)

    with np.errstate(invalid="ignore"):
        rfi_mask = (
            (channel_variance > var_factor * mean_channel_variance)
            | (spec[::-1] < q1 - 1.5 * iqr)
            | (spec[::-1] > q3 + 1.5 * iqr)
        )
    wfall = wfall.copy()
    wfall[rfi_mask, ...] = np.nan
    if model_wfall is not None:
        model_wfall = model_wfall.copy()
        model_wfall[rfi_mask, ...] = np.nan
    spec = spec.copy()
    spec[rfi_mask[::-1]] = np.nan
    return wfall, spec, model_wfall


def _log_linear(x, alpha, intercept):
    return alpha * x + intercept


def fit_spectral_index(freq_mhz, spec, spec_err=None, freq_range=None):
    """
    Fit F_nu ~ nu^alpha to the on-pulse spectrum in log-log space:
    log(F) = alpha * log(nu) + const.

    If `spec_err` is given (same shape as `spec`, e.g. from off-burst
    noise), the fit is a weighted least squares: `spec_err` is propagated
    into log space (sigma_log10(F) = spec_err / (F * ln 10)) and used as
    `sigma` in scipy.optimize.curve_fit with absolute_sigma=True, so
    noisier/less-certain bins pull the fit less. Without `spec_err`, falls
    back to an ordinary (unweighted) least-squares fit.

    Returns a dict with alpha, alpha_err, intercept, r_value, n_channels_used.
    """
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spec = np.asarray(spec, dtype=float)
    if spec_err is not None:
        spec_err = np.asarray(spec_err, dtype=float)

    good = np.isfinite(freq_mhz) & np.isfinite(spec)
    if freq_range is not None:
        good &= (freq_mhz >= freq_range[0]) & (freq_mhz <= freq_range[1])
    # spectrum can be background-subtracted and go negative/zero; log() needs > 0
    good &= spec > 0
    if spec_err is not None:
        good &= np.isfinite(spec_err) & (spec_err > 0)

    if good.sum() < 5:
        raise ValueError(
            f"Only {good.sum()} usable channels (positive flux, non-NaN) for the fit; "
            "try relaxing freq_range or check RFI masking."
        )

    log_f = np.log10(freq_mhz[good])
    log_s = np.log10(spec[good])

    # Unweighted correlation coefficient, reported regardless of fit method
    r_value = float(scipy.stats.pearsonr(log_f, log_s)[0])

    if spec_err is not None:
        log_s_err = spec_err[good] / (spec[good] * np.log(10))
        popt, pcov = curve_fit(
            _log_linear, log_f, log_s, sigma=log_s_err, absolute_sigma=True,
        )
        alpha, intercept = float(popt[0]), float(popt[1])
        alpha_err = float(np.sqrt(pcov[0, 0]))
    else:
        result = scipy.stats.linregress(log_f, log_s)
        alpha, intercept, alpha_err = result.slope, result.intercept, result.stderr
        r_value = result.rvalue

    return {
        "alpha": alpha,
        "alpha_err": alpha_err,
        "intercept": intercept,
        "r_value": r_value,
        "n_channels_used": int(good.sum()),
    }


# ----------------------------------------------------------------------
# 3. Per-FRB processing
# ----------------------------------------------------------------------


def process_one(h5path, freq_range):
    """Load one waterfall file, find the boxcar, fit alpha on the boxcar
    spectrum, make the figure, save the PNG.
    Returns (frb_name, alpha, alpha_err)."""
    with h5py.File(h5path, "r") as f:
        data = f["frb"]
        eventname = data.attrs["tns_name"]
        if isinstance(eventname, bytes):
            eventname = eventname.decode()
        dm = float(data.attrs["dm"][()])
        scatterfit = bool(data.attrs["scatterfit"][()])

        wfall = data["wfall"][:]
        model_wfall = data["model_wfall"][:]
        plot_time = data["plot_time"][:]
        plot_freq = data["plot_freq"][:]
        ts = data["ts"][:]
        model_ts = data["model_ts"][:]
        spec = data["spec"][:]
        extent = list(data["extent"][:])

    # Top/bottom of the observing band (e.g. 800 / 400 MHz for CHIME),
    # straight from the file's frequency axis.
    freq_top = float(np.nanmax(plot_freq))
    freq_low = float(np.nanmin(plot_freq))

    wfall, spec, model_wfall = mask_rfi(wfall, spec, model_wfall)
    ts = np.nansum(wfall, axis=0)
    model_ts = np.nansum(model_wfall, axis=0)
    dt = np.median(np.diff(plot_time))

    # ---- Identify the boxcar window from the matched-filter search ----
    peak, width, snr = find_burst(ts)
    box_lo = int(peak - width // 2)
    box_hi = box_lo + width - 1  # inclusive
    box_lo = max(box_lo, 0)
    box_hi = min(box_hi, len(ts) - 1)
    print(f"    boxcar: peak bin={peak}, width={width} bins ({width * dt:.3f} ms), "
          f"S/N={snr:.2f}, using bins {box_lo}-{box_hi}")

    # ---- Step 1: per-channel baseline (median off-pulse) subtraction ----
    time_sel = slice(box_lo, box_hi + 1)
    n_on_pulse = box_hi - box_lo + 1

    wfall_for_spec = wfall.copy()
    dead_chan = (np.all(wfall_for_spec == 0, axis=1)
                 | np.all(~np.isfinite(wfall_for_spec), axis=1))
    wfall_for_spec[dead_chan, :] = np.nan
    wfall_ascending = wfall_for_spec[::-1, :]      # match plot_freq's ascending order
    dead_chan_ascending = dead_chan[::-1]

    off_pulse_mask = np.ones(wfall.shape[1], dtype=bool)
    off_pulse_mask[time_sel] = False
    if off_pulse_mask.sum() < 10:
        raise ValueError(
            f"Only {off_pulse_mask.sum()} off-burst time samples available "
            "-- too few to estimate noise."
        )

    # Per-channel baseline = median of the off-pulse (noise-only) samples.
    # Subtracting it removes each channel's DC offset so the noise floor
    # sits at ~0 before the on-pulse region is combined.
    with np.errstate(invalid="ignore"):
        baseline_per_chan = np.nanmedian(wfall_ascending[:, off_pulse_mask], axis=1)
    wfall_baseline_sub = wfall_ascending - baseline_per_chan[:, None]

    # ---- Step 2: per-channel RMS noise -- std of the (baseline-subtracted)
    # off-pulse region. Computed before the on-pulse spectrum below since
    # it's now used to normalize it (S/N per channel), not just to build
    # the error bars.
    with np.errstate(invalid="ignore"):
        noise_per_chan = np.nanstd(wfall_baseline_sub[:, off_pulse_mask], axis=1)

    # ---- Step 3: on-pulse spectrum -- mean (not sum) over the boxcar's
    # time bins, per channel, of the baseline-subtracted data, divided by
    # that channel's RMS noise -- i.e. a per-channel S/N spectrum rather
    # than a flux spectrum. ----
    with np.errstate(invalid="ignore"):
        spec_sel_full = (
            np.nanmean(wfall_baseline_sub[:, time_sel], axis=1) / noise_per_chan
        )
    # Explicitly re-mask dead channels so they never masquerade as real
    # zero-flux measurements in the binning step below.
    spec_sel_full[dead_chan_ascending] = np.nan

    # Error on spec_sel_full: it's (mean of n_on_pulse samples with per-
    # sample sigma = noise_per_chan) / noise_per_chan, so the noise_per_chan
    # cancels and the error on the *normalized* value is just 1/sqrt(n_on_pulse)
    # -- the same for every (live) channel, since normalizing by a channel's
    # own noise removes that channel's noise scale from the uncertainty too.
    chan_err_full = np.full_like(noise_per_chan, 1.0 / np.sqrt(n_on_pulse))
    chan_err_full[dead_chan_ascending] = np.nan  # NaN for dead channels

    # Diagnostic: if noise_per_chan is nearly constant across all live
    # channels, the input wfall was likely already noise-normalized
    # upstream (common for pre-processed FRB data products) -- in that
    # case flat error bars downstream are real, not a bug in this script.
    _live_noise = noise_per_chan[~dead_chan_ascending]
    if _live_noise.size:
        print(f"    off-burst noise per channel: min={np.nanmin(_live_noise):.4g}, "
              f"max={np.nanmax(_live_noise):.4g}, "
              f"std/mean={np.nanstd(_live_noise) / np.nanmean(_live_noise):.3g}")

    # ---- Step 4: bin the data (average channels into freq bins), with
    # error propagation for each bin ----
    nchan = wfall.shape[0]
    freq_bin_sel = freq_bin_factor_for_width(plot_freq, nchan, TARGET_FREQ_BIN_MHZ)
    actual_bin_mhz = freq_bin_sel * abs(np.nanmedian(np.diff(plot_freq)))
    print(f"    freq binning: {freq_bin_sel} channels/bin "
          f"(~{actual_bin_mhz:.1f} MHz) for spectral fit")

    spec_sel_binned = bin_1d(spec_sel_full, freq_bin_sel)   # mean over live channels
    freq_sel_binned = bin_1d(plot_freq, freq_bin_sel)
    # fraction of each bin's underlying full-res channels that were dead
    dead_frac_binned = bin_1d(dead_chan_ascending.astype(float), freq_bin_sel)

    # Error on a bin's mean of n_live independent channels, each with its
    # own noise chan_err_full[i]: var(mean) = (1/n_live^2) * sum(sigma_i^2).
    # bin_1d_sum uses nansum, so NaN (dead) channels contribute 0 to both
    # sums, leaving only live channels in the propagation.
    live_chan_ascending = ~dead_chan_ascending
    n_live_per_bin = bin_1d_sum(live_chan_ascending.astype(float), freq_bin_sel)
    sumsq_err_per_bin = bin_1d_sum(chan_err_full ** 2, freq_bin_sel)
    with np.errstate(invalid="ignore", divide="ignore"):
        spec_err_binned = np.sqrt(sumsq_err_per_bin) / n_live_per_bin

    # Peak frequency: Gaussian fit to the (unfiltered) binned spectrum,
    # zero-weighting bins that are mostly/entirely dead channels rather
    # than dropping them outright
    peak_freq_gauss, peak_freq_gauss_err, gauss_popt = fit_peak_frequency(
        freq_sel_binned, spec_sel_binned, dead_frac_binned
    )
    print(f"    peak frequency (Gaussian fit) = {peak_freq_gauss:.1f} "
          f"+/- {peak_freq_gauss_err:.1f} MHz")

    # For the alpha fit and the spectrum panel: drop bins that are more
    # than half dead channels (matches the intent of the old isfinite()
    # check, but isfinite() alone can't catch this -- see note above)
    good_bins = dead_frac_binned <= 0.5
    spec_sel_binned = spec_sel_binned[good_bins]
    freq_sel_binned = freq_sel_binned[good_bins]
    spec_err_binned = spec_err_binned[good_bins]
    print(f"    {good_bins.sum()}/{len(good_bins)} freq bins retained "
          f"({(~good_bins).sum()} dead bins removed)")

    # Simple, fit-free peak frequency: just the frequency bin with the
    # highest S/N in the (dead-bin-filtered) boxcar spectrum -- a quick
    # cross-check against the Gaussian-fit centroid above.
    peak_freq = argmax_peak_frequency(freq_sel_binned, spec_sel_binned)
    print(f"    peak frequency (argmax) = {peak_freq:.1f} MHz")

    fit_sel = fit_spectral_index(
        freq_sel_binned, spec_sel_binned, spec_err=spec_err_binned, freq_range=freq_range
    )
    print(
        f"    alpha = {fit_sel['alpha']:.2f} +/- {fit_sel['alpha_err']:.2f} "
        f"(R = {fit_sel['r_value']:.3f}, {fit_sel['n_channels_used']} bins used)"
    )

    # ---- Replot: waterfall + time series (boxcar shaded) + boxcar spectrum ----
    wfall_binned = bin_freq_channels(wfall, FBIN)
    wfall_binned = bin_time_samples(wfall_binned, TBIN)

    fig = plt.figure(figsize=(7, 6), dpi=150)
    gs = gridspec.GridSpec(
        ncols=2, nrows=2, figure=fig, width_ratios=[3, 1], height_ratios=[1, 3],
        hspace=0.0, wspace=0.0,
    )
    ax_im = plt.subplot(gs[2])
    ax_ts = plt.subplot(gs[0], sharex=ax_im)
    ax_spec = plt.subplot(gs[3], sharey=ax_im)

    peak_idx = np.argmax(ts)
    t = plot_time - plot_time[peak_idx]
    t = t - dt / 2.0
    t = np.append(t, t[-1] + dt)  # ntime_full + 1 bin edges

    ntime_full = wfall.shape[1]
    ntime_trim = (ntime_full // TBIN) * TBIN
    extent[0] = t[0]
    extent[1] = t[ntime_trim]

    plot_wfall = np.ma.masked_invalid(wfall_binned)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    vmin = np.nanpercentile(wfall_binned, 1)
    vmax = np.nanpercentile(wfall_binned, 99)

    ax_im.imshow(plot_wfall, aspect="auto", interpolation="none", extent=extent,
                 vmin=vmin, vmax=vmax, cmap=cmap)
    ax_ts.plot(t, np.append(ts, ts[-1]), color="tab:gray", drawstyle="steps-post",
               label="data")

    burst_cmap = plt.cm.viridis
    mcolor = burst_cmap(0.25) if scatterfit else burst_cmap(0.5)
    ax_ts.plot(t, np.append(model_ts, model_ts[-1]), color=mcolor, drawstyle="steps-post",
               label="model")

    # Shade the boxcar window on the time-series panel
    box_x_lo = t[box_lo]
    box_x_hi = t[box_hi + 1]
    ax_ts.axvspan(box_x_lo, box_x_hi, color="tab:blue", alpha=0.25, zorder=0)
    ax_ts.legend(fontsize=6, loc="upper left", framealpha=0.8)

    # boxcar spectrum + fit
    ax_spec.errorbar(spec_sel_binned, freq_sel_binned, xerr=spec_err_binned,
                      color="tab:blue", marker=".", ls="-", lw=1, elinewidth=0.8,
                      capsize=1.5, label="data")

    freq_fit = np.linspace(np.nanmin(freq_sel_binned), np.nanmax(freq_sel_binned), 200)
    spec_fit = 10 ** fit_sel["intercept"] * freq_fit ** fit_sel["alpha"]
    ax_spec.plot(spec_fit, freq_fit, color="black", lw=1.5, ls="-",
                 label=fr"Spectral index fit", alpha=0.65)

    # Overplot the Gaussian fit used for the peak frequency, if it converged
    if gauss_popt is not None:
        spec_gauss_fit = gaussian(freq_fit, *gauss_popt)
        ax_spec.plot(
            spec_gauss_fit, freq_fit, color="tab:blue", lw=1.2, ls="-.",
            label=fr"Gaussian fit"
        )
    ax_spec.legend(fontsize=5, loc="upper right")

    plt.setp(ax_ts.get_xticklabels(), visible=False)
    ax_ts.set_yticks([])
    ax_ts.set_xlim(extent[0], extent[1])
    plt.setp(ax_spec.get_yticklabels(), visible=False)
    ax_spec.set_xticks([])
    ax_spec.set_ylim(extent[2], extent[3])

    ax_im.set_xlabel("Time [ms]")
    ax_im.set_ylabel("Frequency [MHz]")
    ax_ts.text(
        0.98, 0.85,
        f"{eventname}\nDM: {dm:.1f} pc/cc\nSNR: {snr:.2f}\n"
        fr"$\alpha$ = {fit_sel['alpha']:.2f} $\pm$ {fit_sel['alpha_err']:.2f}"
        f"\nboxcar: {width} bins ({width * dt:.2f} ms)",
        transform=ax_ts.transAxes, ha="right", va="top", fontsize=8,
    )

    # fig.suptitle(f"{eventname}", y=0.98)

    savepath = os.path.join(OUTPUT_DIR, f"{eventname}_wfall_spectral_index.png")
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure saved to {savepath}")

    return (
        eventname, dm, freq_top, freq_low,
        fit_sel["alpha"], fit_sel["alpha_err"],
        peak_freq, peak_freq_gauss, peak_freq_gauss_err,
    )


# ----------------------------------------------------------------------
# 4. Run over every waterfall file in INPUT_DIR
# ----------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    freq_range = None
    if FIT_FMIN is not None or FIT_FMAX is not None:
        freq_range = (FIT_FMIN if FIT_FMIN is not None else -np.inf,
                      FIT_FMAX if FIT_FMAX is not None else np.inf)

    h5paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*_waterfall.h5")))
    print(f"Found {len(h5paths)} waterfall files in {INPUT_DIR}")

    results = []
    failed = []

    for i, h5path in enumerate(h5paths, 1):
        basename = os.path.basename(h5path)
        print(f"[{i}/{len(h5paths)}] {basename}")
        try:
            result = process_one(h5path, freq_range)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 -- keep the batch going
            print(f"    [FAILED] {exc}")
            traceback.print_exc()
            failed.append((basename, str(exc)))
        finally:
            plt.close("all")

    # ---- Write summary CSV ----
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frb_name", "dm", "freq_top", "freq_low",
            "alpha", "alpha_err",
            "peak_freq", "peak_freq_gauss", "peak_freq_gauss_err",
        ])
        for row in results:
            writer.writerow(row)
    print(f"\nWrote {len(results)} rows to {CSV_PATH}")

    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for basename, err in failed:
            print(f"  {basename}: {err}")


if __name__ == "__main__":
    main()
