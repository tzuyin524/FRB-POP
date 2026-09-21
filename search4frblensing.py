##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     20/02/2025                 #
# Date (updated):     12/05/2025                 #
##################################################
#                                                #
#                                                #
# Search for lensing in FRB voltage data         #
##################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.fft import next_fast_len
import argparse
from utils import average
from math import ceil
import sys, warnings, os
from make_dynspec import make_ds, baseline_correction, flag_chan
import multiprocessing as mp 
import time
from lensingutils import *
import io
from contextlib import redirect_stdout

ARGTYPES = {'c': str, 'x':str, 'y': str, 'delDM': float, 'cfreq': float, 'bw': float, 'delDM_fast': bool, 
            'DMmin': float, 'DMmax': float, 'DMstep': float, 'cpus': int, 'rms_w': float, 'rms_g': float, 
            'stDev': int, 'Wsigma': float, 'wtype': str, 'idt': float, 'idf': float, 'tN': int, 'chanflag': str,
            'save_data': bool, 'showplots': bool, 'dsI': str, 'resetdsI': bool, 'o': str, 'n': str, 'noclean': bool,
            'cfs': bool, 't': float, 'nFFT': int, 'save_config': bool, 'more_plots': bool, 'acf_crop': str, 'dsI_crop': str,'edge_samp': int}
            

class _empty:
    pass

def get_args():

    desc = """
    Search for Lensing in FRB voltage data. Requires complex voltage data X and Y.
    """

    acf_crop_desc = """ Crop used to search for peak in frequency-averaged ACF i.e. acf_crop=5:10,-10:-5 will search for 
    peak within the ranges -10->-5 milliseconds and 5 to 10 milliseconds, excluding everything else. Must be in accending order if given ranges. can also
    give single sample values i.e. acf_crop = 5:10, 12 to also include the sample at 12 milliseconds. By default this script will search the entire avaliable 
    ACF window.
    """

    parser = argparse.ArgumentParser(description = desc)
    parser.add_argument("-c", help = "Config file", type = str, default = None)
    parser.add_argument("--save_config", help = "filepath to config file to save params to", type = str, default = None)

    parser.add_argument("-x", help = "X voltage data", required = False, type = str)
    parser.add_argument("-y", help = "Y voltage data", required = False, type = str)

    parser.add_argument("--nFFT", help = "channelisation - Number of channels", type = int, default = 336)
    parser.add_argument("-t", help = "Window length of dynspec in time [ms] centered at maximum of burst", default = 100.0, type = float)
    parser.add_argument("--delDM", help = "Apply a delta DM to the copy [pc cm^-3]", default = 0.0, type = float)
    parser.add_argument("--cfreq", help = "Central frequency [MHz]", type = float, default = 919.5)
    parser.add_argument("--bw", help = "Bandwidth [MHz]", type = float, default = 336)
    parser.add_argument("--delDM_fast", action = "store_true")
    parser.add_argument("--dsI_crop", help = "Overide automatic on pulse searching algorithm and specify manual crop [ms] from starting sample (example: 10:20)", type = str, default = "")

    # in case searching over a range of DM values
    parser.add_argument("--DMmin", help = "Minimum delta DM to search [pc cm^-3]", default = None, type = float)
    parser.add_argument("--DMmax", help = "Maximum delta DM to search [pc cm^-3]", default = None, type = float)
    parser.add_argument("--DMstep", help = "Step size for DM range to search [pc cm^-3]", default = 0.1, type = float)
    parser.add_argument("--cpus", help = "Number of processes (CPUS) to run in parallel", default = 1, type = int)

    # baseline arguments (for correction)
    parser.add_argument("--rms_w", help = "Width of rms region for S/N estimation [ms]", type = float, default = 20.0)
    parser.add_argument("--rms_g", help = "Width of window seperating on pulse and off pulse windows (guard) [ms]", type = float, default = 10.0)
    parser.add_argument("--stDev", help = "stDev of gaussian smoothing filter in samples [if set to zero, will not perform weighting]", type = int, default = 3)
    parser.add_argument("--Wsigma", help = "S/N threshold for zeroing baseline in weight function [if set to zero, will not zero any weights]", type = float, default = 3.0)
    parser.add_argument("--wtype", help = "Weighting type, by default [mf - matched filter with gaussian smoothing], options include [mf, boxcar]", type = str, default = "mf")

    # optional variables
    parser.add_argument('--idt', help = "Time resolution [ms] at nyquist sampling", default = 0.00000297619, type = float)
    parser.add_argument('--idf', help = "Freq resolution [MHz] at a channelization of 336", default = 1.0, type = float)
    parser.add_argument('--tN', help = "Averaging factor to help with finding peak", default = 100, type = int)
    parser.add_argument("--chanflag", help = "Channels to flag e.g. '919.5, 1020:1040'", default = None, type = str)
    parser.add_argument('--edge_samp', help = "Number of samples to flag at each edge of the ACF lag axis", type = int, default = 0)
    
    # cleaning options
    parser.add_argument('--pixel_threshold', help = "Threshold S/N for individual pixels before cleaning", type = float, default = 50.0)
    parser.add_argument('--acf_crop', help = acf_crop_desc, type = str, default = None)

    # plotting paramters, does not affect acf
    parser.add_argument("--save_data", help = "Save data such as acf, X and Y crops etc.", action = "store_true")
    parser.add_argument("--showplots", help = "Show interactive plots", action = "store_true")
    parser.add_argument("--more_plots", help = "Show more plots, only if showplots is True", action = "store_true")
    parser.add_argument("--dsI", help = "Raw dynspec file for initial cropping. If no file is found, will make dynspec", type = str, default = None)
    parser.add_argument("--resetdsI", help = "reset dsI file", action = "store_true")
    parser.add_argument("-o", help = "Output dir", type = str, default = os.getcwd())
    parser.add_argument("-n", help = "file prefix name", type = str, default = "")
    parser.add_argument("--noclean", help = "Don't do ACF cleaning", action = "store_true")
    parser.add_argument("--cfs", help = "Coherently add together frequency channels in acf (np.sum(acf, axis = 0)) instead of incoherently (np.sum(np.abs(acf), axis = 0))", action = "store_true")

    args = parser.parse_args()

    if args.c is not None:
        if os.path.exists(args.c):
            load_config(args)   
            
            # overide config with user inputted values
            inputted_args = sys.argv[1:]

            for arg in ARGTYPES.keys():
                if len(arg) == 1:
                    input_arg = "-" + arg
                elif len(arg) > 1:
                    input_arg = "--" + arg
                if input_arg in inputted_args:
                    ind = inputted_args.index(input_arg)
                    if ARGTYPES[arg] == bool:
                        # only works if action == "store_true"
                        setattr(args, arg, True)
                    elif ARGTYPES[arg] == float:
                        setattr(args, arg, float(inputted_args[ind + 1]))
                    elif ARGTYPES[arg] == int:
                        setattr(args, arg, int(inputted_args[ind + 1]))
                    elif ARGTYPES[arg] == str:
                        setattr(args, arg, inputted_args[ind + 1])
                    else:
                        ValueError(f"Argument type [{ARGTYPES[arg]}] not supported...")
            
                    
        else:
            print("Config file path does not exist!")

    



    # format args.nFFT in case of list given
    args.singledm = True
    args.dmrange = np.array([args.delDM])
    if (args.DMmin is not None) and (args.DMmax is not None):
        args.singledm = False
        args.dmrange = np.arange(args.DMmin, args.DMmax, args.DMstep)

    return args



def load_config(args):
    """
    Load config file
    """

    with open(args.c, 'r') as file:
        lines = file.readlines()
    
    for line in lines:

        line_parts = line.split(':', 1)

        par = line_parts[0].strip()
        if par in ARGTYPES:
            val = line_parts[1].strip()
            if ARGTYPES[par] == bool:
                if (val == "") or (val == "False"):
                    setattr(args, par, False)
                elif val == "True":
                    setattr(args, par, True)
            
            elif ARGTYPES[par] == float:
                if val == "":
                    setattr(args, par, None)
                else:
                    setattr(args, par, float(val))
            elif ARGTYPES[par] == int:
                if val == "":
                    setattr(args, par, None)
                else:
                    setattr(args, par, int(val))
            elif ARGTYPES[par] == str:
                if val == "":
                    setattr(args, par, None)
                else:
                    setattr(args, par, val)
            else:
                ValueError(f"input type [{ARGTYPES[par]}] not currently supported!")
    print(args.__dict__)
    return






def save_config(args):
    """
    Save current args to .txt (config file)
    """

    with open(args.save_config, 'w') as file:
        for par in ARGTYPES:
            val = getattr(args, par)
            if ARGTYPES[par] == bool:
                if val:
                    file.write(f"{par}: {'True'}\n")
                else:
                    file.write(f"{par}: {'False'}\n")
            
            else:
                if val is None:
                    file.write(f"{par}: \n")
                else:
                    file.write(f"{par}: {val}\n")
            
    return




def load_data(args):
    """
    
    Load data and perform cropping

    Parameters
    ----------
    args : input arguments
    """

    data = {}
    data['X'] = None        # On pulse window of complex voltage X
    data['Y'] = None        # On pulse window of complex voltage Y
    data['Xerr'] = None     # Off pulse window of comples voltage X
    data['Yerr'] = None     # Off pulse window of comples voltage Y
    data['dsI'] = None      # On pulse window of Stokes I dynspec
    data['dsX'] = None      # On pulse complex X dynspec
    data['dsY'] = None      # On pulse complex Y dynspec
    data['dsXerr'] = None
    data['dsIerr'] = None   # Off pulse window of Stokes I dynspec
    data['tI'] = None       # On pulse window of Stokes I time series
    data['tIfull'] = None   # Full Stokes I time series
    data['tIerr'] = None    # RMS (Variance) in time samples
    data['tImf'] = None     # tI matched filter - Smoothed Gaussian Power times series
    data['acf'] = None      # Channelised auto-correlation voltage dynspec

    data['XdeDM'] = None
    data['YdeDM'] = None
    data['dsXdeDM'] = None
    data['dsYdeDM'] = None

    # load in data
    get_crop(args, data, nFFT = 336)

    return data






def get_crop(args, data, nFFT: int = 336):
    """
    Get crop by making dynamic spectrum

    Parameters
    ----------
    X : complex X data
    Y : complex Y data
    """

    datapath = os.path.join(args.o, args.dsI)

    temp_args = _empty()
    temp_args.do_chanflag = True

    make_new_dsI = False
    if args.dsI is None:
        make_new_dsI = True
    if datapath is not None:
        if not os.path.isfile(datapath):
            make_new_dsI = True
    if args.resetdsI:
        make_new_dsI = True

    if make_new_dsI:
        # load in data
        X = np.load(args.x, "r")
        Y = np.load(args.y, "r")

        print(f"Loaded [X] data from: {args.x}")
        print(f"Loaded [Y] data from: {args.y}")

        # make dynspec synamic spectrum
        I = make_ds(X, Y, nFFT = nFFT)
        I[0] *= 1e-12

        if args.dsI is not None:
            with open(datapath, 'wb') as file:
                print(f"Saving full Stokes I dynspec as [{datapath}] for quick loading later...")
                np.save(file, I)

    else:
        print(f"Quick loading full Stokes I dynspec from [{datapath}]...")
        I = np.load(datapath, 'r')

    I_raw = I.copy()
    I_raw -= np.mean(I_raw, axis = 1)[:, None]

    chanflag,_ = flag_chan(I_raw, 10, 1000, temp_args, None)

    # get bounds
    I_raw[chanflag] = np.nan
    
    data['tIfull'] = average(np.nanmean(I_raw, axis = 0), N = args.tN)

    # parameters for on pulse and off pulse crop
    peak_samp = int(np.argmax(data['tIfull']) * args.tN * 336)
    width_samp = int(args.t / args.idt / 2)

    if args.dsI_crop is not None:
        if args.dsI_crop != "":
            bounds = args.dsI_crop.split(':')
            manual_crop_min = float(bounds[0].strip()) / args.idt
            manual_crop_max = float(bounds[1].strip()) / args.idt
            width = manual_crop_max - manual_crop_min
            width_samp = int(width / 2)
            peak_samp = int(manual_crop_min) + width_samp

            args.t = float(bounds[1].strip()) - float(bounds[0].strip())

    rms_g_samp = int(args.rms_g / args.idt)
    rms_w_samp = int(args.rms_w / args.idt / 2)

    onpulse_window = slice(peak_samp - width_samp,peak_samp + width_samp)
    offpulse_window = slice(peak_samp - width_samp - rms_g_samp - rms_w_samp,
                            peak_samp - width_samp - rms_g_samp)
   
    # crop
    X = np.load(args.x, mmap_mode = "r")
    Y = np.load(args.y, mmap_mode = "r")

    data['X'], data['Y'] = X[onpulse_window], Y[onpulse_window]
    data['Xerr'], data['Yerr'] = X[offpulse_window], Y[offpulse_window]

    args.peak_samp = peak_samp
    args.onpulse_window = onpulse_window
    args.offpulse_window = offpulse_window

    print(f"On-pulse window: {onpulse_window}")
    print(f"Off-pulse window: {offpulse_window}")
    print(f"Peak sample: {peak_samp}")


    return






def get_weights(I, Ierr, args):
    """
    Get weights
    """

    W_min = 0.0

    
    gaussian_W = apply_gaussian_smooth(I, args.stDev)
    W = np.interp(np.linspace(0, 1.0, args.nsamp),
                np.linspace(0, 1.0, gaussian_W.size), gaussian_W)
    
    if args.Wsigma > 0.0:
        W[W < args.Wsigma * Ierr] = W.max() * W_min
    else:
        print(f"No threshold flagging in weight function will be done as Wsigma = {args.Wsigma}")
    
    if args.wtype == "boxcar":
        W[W > 0.0] = W.max()

    else:
        ValueError(f"Weight type [{args.wtype}] invalid, use either 'mf' or 'boxcar'!")
    
    return W




def get_channelflagging(args, data):

    freqs = get_freqs(args.cfreq, args.bw)

    # find prior flagged channels in data
    prior_zapstr = get_zapstr(data['dsI'], freqs)

    # get user input zapstr and combine with priors
    zapstr = combine_zapchan(prior_zapstr, args.chanflag)

    return zapstr, freqs




def setup_acf(args, data):
    """
    Setup for auto correlations. 
    This function does the following:

    1. Make baseline corrected Stokes I dynspec
    2. make matched filter based on Stokes I power
    3. auto channel flagging
    4. Make Complex Polarisation dynspec for autocorrelation

    Parameters
    ----------
    args : input arguments
    data : Dictionary of data 
    """
    print("\n#" + "-"*30 + "#")
    print(f"Channelising to {args.nFFT} Channels")
    print("Doing Setup for correlation")
    print("#" + "-"*30 + "#\n")

    nsamp = int(data['X'].size // args.nFFT)
    args.nsamp = nsamp

    # make stokes I dynspec
    print("Stokes I dynamic spectrum")
    # Stokes I (total intensity) dynspec of the on-pulse crop, from make_ds(data['X'], data['Y'], nFFT=336)
    data['dsI'] = make_ds(data['X'], data['Y'], nFFT = 336)

    print("Stokes Ierr dynamic spectrum")
    # off-pulse (noise) crop — used for baseline/rms estimation
    data['dsIerr'] = make_ds(data['Xerr'], data['Yerr'], nFFT = 336)

    data['baseline_correction'] = np.mean(data['dsIerr'], axis = 1)[:, None]
    data['rms_norm'] = np.std(data['dsIerr'], axis = 1)[:, None]

    # baseline correction
    data['dsI'] -= data['baseline_correction']
    data['dsIerr'] -= data['baseline_correction']

    # normalise by rms
    # data['dsI'] /= data['rms_norm']
    # data['dsIerr'] /= data['rms_norm']

    # Channel flagging
    freqs = get_freqs(args.cfreq, args.bw, data['dsI'].shape[0])

    # find prior flagged channels in data
    prior_zapstr = get_zapstr(data['dsI'], freqs)

    # get user input zapstr and combine with priors
    data['zapstr'] = combine_zapchan(prior_zapstr, args.chanflag)

    flaggedchans = np.array(zap_chan(freqs, data['zapstr']))
    print(flaggedchans)

    # average dynspec
    data['dsI'] = average(data['dsI'], axis = 1, N = args.tN)
    data['dsIerr'] = average(data['dsIerr'], axis = 1, N = args.tN)

    # combine flagging
    # flaggedchans = np.unique(np.append(auto_flaggedchans, flaggedchans))

    data['dsI'][flaggedchans.tolist()] = np.nan
    data['dsIerr'][flaggedchans.tolist()] = np.nan

    data['flaggedchans'] = flaggedchans.copy()

    # now get flaggedchans for acf, for use later
    freqs_acf = get_freqs(args.cfreq, args.bw, args.nFFT)
    data['flaggedchans_acf'] = np.array(zap_chan(freqs_acf, data['zapstr']))

    # get weight function
    data['tI'] = np.nanmean(data['dsI'], axis = 0)
    data['tIerr'] = np.std(np.nanmean(data['dsIerr'], axis = 0))

    data['tImf'] = get_weights(data['tI'], data['tIerr'], args)

    # make complex pol dynspec
    # complex dynspecs of each polarization (X, Y)
    data['dsX'] = make_complex_ds(data['X'], args.nFFT)
    data['dsY'] = make_complex_ds(data['Y'], args.nFFT)

    # make off-pulse complex pol dynspec
    data['dsXerr'] = make_complex_ds(data['Xerr'], args.nFFT)
    data['dsYerr'] = make_complex_ds(data['Yerr'], args.nFFT)
    data['dsXerr_baseline'] = np.mean(data['dsXerr'])

    # baseline correction
    # data['dsX'] -= 

    data['dsX_unweighted'] = data['dsX'].copy()
    data['dsY_unweighted'] = data['dsY'].copy()

    # weight 
    if args.stDev > 0.0:
        data['dsX'] *= data['tImf'][None, :] / data['tImf'].max()
        data['dsY'] *= data['tImf'][None, :] / data['tImf'].max()


    return








def corr_delta_dm(args, X, Y, dsX, dsY, baseline, tIerr, flaggedchans, dm, return_data = False):
    """
    Do correlation for delta DM [dm]

    Parameters
    ----------
    args : input pars 
    X : Complex X polarisation time series
    Y : Complex Y polarisation time series
    dsX : Complex X dynspec (weighted if applicable)
    dsY : Complex Y dynspec (weighted if applicable)
    dm : Delta DM to apply 
    """
    nsamp = int(X.size // args.nFFT)

    # Make X and Y dedispersed products
    print("Dedispersing X and Y")
    XdeDM = coherent_desperse(X, args.cfreq, args.bw, args.cfreq - args.bw/2, dm)
    YdeDM = coherent_desperse(Y, args.cfreq, args.bw, args.cfreq - args.bw/2, dm)

    # make X and Y dedispersed dynspec products
    print("Making dedispersed complex products")
    dsXdeDM = make_complex_ds(XdeDM, args.nFFT)
    dsYdeDM = make_complex_ds(YdeDM, args.nFFT)

    # get weights
    print("Applying weights")
    dsIdeDM = make_ds(XdeDM, YdeDM, nFFT = 336)
    dsIdeDM = average(dsIdeDM, axis = 1, N = args.tN)
    dsIdeDM -= baseline
    dsIdeDM[data['flaggedchans'].tolist()] = np.nan
    tIdeDM = np.nanmean(dsIdeDM, axis = 0)

    weights = get_weights(tIdeDM, tIerr, args)
    
    # apply weights
    if args.stDev > 0.0:
        # complex dynspecs of the dedispersed copy,
        dsXdeDM *= weights[None, :] / weights.max()
        dsYdeDM *= weights[None, :] / weights.max()

    # construct ACF
    # It's a cross-correlation between the original complex-voltage dynspec and a dedispersed copy of itself
    print("Calculating channelised ACF")
    # (from lag -(nsamp-1) to +(nsamp-1))
    acf = np.zeros((args.nFFT, 2*nsamp - 1), dtype = X.dtype)
    for i in range(args.nFFT):
        acf[i, :] = (correlate(dsX[i,:], dsXdeDM[i,:])/np.sqrt(correlate(np.abs(dsX[i,:])**2, np.abs(dsXdeDM[i,:])**2))
                    + correlate(dsY[i,:], dsYdeDM[i,:])/np.sqrt(correlate(np.abs(dsY[i,:])**2, np.abs(dsYdeDM[i,:])**2)))
    
    if not args.cfs:
        acf = np.abs(acf)

    # flag channels
    # get flagging for acf 
    freqs = get_freqs(args.cfreq, args.bw, args.nFFT)
    acf[flaggedchans.tolist()] = np.nan

    # clean acf
    if not args.noclean:
        print("Cleaning ACF")
        acf = clean_acf(args, acf, dm, flaggedchans.size)

    # get acf peak
    peak_val = np.nanmax(np.abs(np.nanmean(acf, axis = 0)))

    if return_data:
        return np.abs(acf), np.nanmean(acf, axis = 0), dsIdeDM, tIdeDM, weights
    else:
        return peak_val









def clean_acf(args, acf, dm, nflagchans):
    """
    Clean acf

    Parameters
    ----------
    acf :np.ndarray or array-like
        2D acf dynspec
    """

    cleaned_acf = acf.copy()

    # Remove Samples with bad channels
    mask = np.ones(cleaned_acf.shape[0], dtype = bool)
    mask[data['flaggedchans_acf'].tolist()] = False
    rawI_acf = np.mean(cleaned_acf[mask], axis = 0)
    cleaned_acf[:, np.isnan(rawI_acf)] = np.nan

    # clean zero lag (include dm shifting)
    zero_peak_idx = cleaned_acf.shape[1]//2

    if dm != 0.0:
        acf_shape = cleaned_acf.shape
        # cleaned_acf = cleaned_acf.flatten()

        # get dm shifts
        f = np.linspace(args.cfreq + args.bw/2 - args.bw / args.nFFT / 2,
                        args.cfreq - args.bw/2 + args.bw / args.nFFT / 2, args.nFFT)

        dmshifts_idx = np.asarray(np.round(4.14938e3 * dm * (1/f**2 - 1/(args.cfreq - args.bw/2)**2) / (args.idt * args.nFFT * 1e-3)), dtype = int)
        dmshifts_w = np.max(dmshifts_idx) - np.min(dmshifts_idx)
        print(dmshifts_w)

        if dmshifts_w < int(0.010 / (args.idt * args.nFFT)):
            dmshifts_w = int(0.01 / (args.idt * args.nFFT))

        if dmshifts_w < 10: 
            dmshifts_w *= 2.0
        elif dmshifts_w < 50:
            dmshifts_w *= 1.4
        else:
            dmshifts_w *= 1.2

        # cleaned_acf[:, int(zero_peak_idx - dmshifts_w) : int(zero_peak_idx + dmshifts_w)] = np.nan
        cleaned_acf[:, int(zero_peak_idx - 2*dmshifts_w) : int(zero_peak_idx + 2*dmshifts_w)] = np.nan

        if args.edge_samp > 0:
            cleaned_acf[:, :args.edge_samp] = np.nan
            cleaned_acf[:, -args.edge_samp:] = np.nan  
        # for i in range(-10, 11):
        #     cleaned_acf[dmshifts_idx + i] = np.nan
            # cleaned_acf[dmshifts_idx - 1] = np.nan
            # cleaned_acf[dmshifts_idx + 1] = np.nan

        # left and right sample remove as well
        # reshape
        # cleaned_acf = cleaned_acf.reshape(acf_shape)
    else:
        # width = int(0.01 / (args.idt * args.nFFT))
        # cleaned_acf[:, zero_peak_idx - width : zero_peak_idx + width] = np.nan
        cleaned_acf[:, zero_peak_idx] = np.nan

    # get rid of inf values
    I_acf = np.nanmean(cleaned_acf, axis = 0)
    cleaned_acf[:, np.abs(I_acf) == np.inf] = np.nan

    # remove very bright pixels
    rms_mean = np.nanstd(cleaned_acf)
    cleaned_acf[cleaned_acf > args.pixel_threshold * rms_mean] = rms_mean

    # Apply the acf_crop (if given) by setting all outside samples to NaN.
    if (args.acf_crop is not None):
        if args.acf_crop != "":
            acf_t = np.linspace(-args.t, args.t, I_acf.size)
            crop_samples = zap_chan(acf_t, args.acf_crop)   # repurpose zap_chan code (cause i'm LAZY)
            x_mask = np.ones(acf_t.size, dtype = bool)
            x_mask[crop_samples] = False
            cleaned_acf[:, x_mask] = np.nan




    return cleaned_acf


def dm_zero_lag_halfwidth_ms(args, dm, n_lag_samples):
    """
    Reproduce clean_acf()'s zero-lag mask half-width for a given trial DM,
    converted to ms, so the same physical exclusion width can be reused for
    power_acf()'s zero-lag masking instead of guessing a separate constant.

    This mirrors clean_acf()'s dmshifts_w calculation exactly (same
    formula, same heuristic scaling), then converts the resulting
    half-width from samples to ms using the lag axis spacing implied by
    n_lag_samples (the size of the ACF trace, spanning [-args.t, args.t]).

    Parameters
    ----------
    args : input arguments
    dm : trial delta DM [pc cm^-3] used for the coherent ACF
    n_lag_samples : size of the lag axis (e.g. data['tIacf'].size), used
        to get the sample spacing consistent with clean_acf's own axis

    Returns
    -------
    half_width_ms : the mask half-width [ms] - i.e. what to pass as
        zero_guard_ms to power_acf() for a matching zero-lag exclusion
    """
    dt_samp = 2 * args.t / (n_lag_samples - 1)

    if dm == 0.0:
        # clean_acf only masks a single central sample when dm == 0
        return dt_samp / 2.0

    f = np.linspace(args.cfreq + args.bw/2 - args.bw / args.nFFT / 2,
                    args.cfreq - args.bw/2 + args.bw / args.nFFT / 2, args.nFFT)

    dmshifts_idx = np.asarray(np.round(4.14938e3 * dm * (1/f**2 - 1/(args.cfreq - args.bw/2)**2)
                    / (args.idt * args.nFFT * 1e-3)), dtype = int)
    dmshifts_w = np.max(dmshifts_idx) - np.min(dmshifts_idx)

    if dmshifts_w < int(0.010 / (args.idt * args.nFFT)):
        dmshifts_w = int(0.01 / (args.idt * args.nFFT))

    if dmshifts_w < 10:
        dmshifts_w *= 2.0
    elif dmshifts_w < 50:
        dmshifts_w *= 1.4
    else:
        dmshifts_w *= 1.2

    # clean_acf masks [zero_peak_idx - 2*dmshifts_w : zero_peak_idx + 2*dmshifts_w],
    # i.e. a half-width of 2*dmshifts_w samples
    half_width_ms = 2 * dmshifts_w * dt_samp

    return half_width_ms


def power_acf(args, data, zero_guard_ms):
    """
    Compute the power (Stokes I) auto-correlation function as an
    independent cross-check on the coherent/complex ACF peak lag.

    Correlates the frequency-averaged Stokes I power time series
    (data['tI']) against itself. No dedispersion, no phase information,
    no per-channel normalization - just a plain intensity ACF.

    Parameters
    ----------
    args : input arguments
    data : data dictionary (expects data['tI'] to already be populated,
           i.e. setup_acf() must have been run first)
    zero_guard_ms : half-width [ms] of the zero-lag region to mask out.
        Pick this based on the known/visible burst width - e.g. from the
        coherent ACF's own zero-lag gap, or the burst duration seen in
        the _crop.png diagnostic plot. There is no DM-dependent formula
        here (unlike clean_acf's dmshifts_w) because power_acf never
        dedisperses or channelises anything - the zero-lag width here is
        set entirely by the burst's own temporal width, not by DM smearing.

    Returns
    -------
    lags_ms : 1D array, time-lag axis [ms]
    pacf_masked : 1D array, power ACF (NaN near zero lag, and at edges
        if args.edge_samp is set)
    peak_samp_pacf : int, index of the peak (excluding zero lag/edges)
    peak_lag_ms : float, time-lag [ms] of the peak
    """

    tI = data['tI']

    # mean-subtract so the correlation isn't dominated by the baseline level
    tI_sub = tI - np.nanmean(tI)
    pacf = correlate(tI_sub, tI_sub)

    # tI spans [-args.t/2, args.t/2], so its self-correlation spans
    # [-args.t, args.t], same convention as the coherent ACF's lag axis
    lags_ms = np.linspace(-args.t, args.t, pacf.size)
    dt_samp = 2 * args.t / (pacf.size - 1)
    zero_idx = pacf.size // 2

    zero_guard_samp = int(round(zero_guard_ms / dt_samp))

    pacf_masked = pacf.astype(float).copy()
    pacf_masked[zero_idx - zero_guard_samp : zero_idx + zero_guard_samp + 1] = np.nan

    if getattr(args, 'edge_samp', 0) > 0:
        pacf_masked[:args.edge_samp] = np.nan
        pacf_masked[-args.edge_samp:] = np.nan

    peak_samp_pacf = np.nanargmax(np.abs(pacf_masked))
    peak_lag_ms = lags_ms[peak_samp_pacf]

    return lags_ms, pacf_masked, peak_samp_pacf, peak_lag_ms


def plot_diagnostics(args, data):
    """
    Diagnostics plotting
    """

    print(f"[nFFT = {args.nFFT}] Making plots\n")

    x = np.linspace(-args.t / 2, args.t / 2, data['tI'].size)


    ################################################################
    # Plot Gaussian smoothing function
    ################################################################

    fig2, ax2 = plt.subplots(3, 1, figsize = (12,8))
    ax2 = ax2.flatten()

    ax2[0].plot(data['tIfull'], 'k')
    ylim = ax2[0].get_ylim()
    ax2[0].plot([args.peak_samp / args.tN / 336]*2, ylim, '--r', label = "Burst peak")
    ax2[0].plot([args.onpulse_window.start / args.tN / 336]*2, ylim, '--b', label = "On pulse window")
    ax2[0].plot([args.onpulse_window.stop / args.tN / 336]*2, ylim, '--b')
    ax2[0].plot([args.offpulse_window.start / args.tN / 336]*2, ylim, '--m', label = "Off pulse window")
    ax2[0].plot([args.offpulse_window.stop / args.tN / 336]*2, ylim, '--m')
    ax2[0].set_ylim(ylim)
    ax2[0].get_xaxis().set_visible(False)
    ax2[0].set_ylabel("Flux Density (arb.)", fontsize = 16)
    ax2[0].legend(loc = "upper right", fancybox = True)



    def plot_weight_function(ax, tI, weights, do_labels = True):
        ax.plot(x, tI, 'k')

        if args.stDev > 0.0:
            ax.plot(np.linspace(-args.t / 2, args.t / 2, weights.size), weights, '--r')
        ax.set_xlabel("Time [ms]", fontsize = 16)
        ax.set_ylabel("Flux Density (arb.)", fontsize = 16)
        xlim = ax.get_xlim()

        for i, sig in enumerate([1, 2, 3, 5, 10, 20]):
            label = f"{sig}"
            if not do_labels:
                label = None
            ax.plot(xlim, [sig * data['tIerr']]*2, '--', label = label)
        ax.set_xlim(xlim)
    
    
    plot_weight_function(ax2[1], data['tI'], data['tImf'])
    ax2[1].legend(title = "S/N threshold", fancybox = True, loc = "upper right")

    # also include plot of best delDM weight function
    plot_weight_function(ax2[2], data['tIdeDM'], data['tImfdeDM'], False)
    ax2[2].plot([],[], 'k', linewidth = 1.5, label = f"Weights from de-dispersed data [DM = {data['peak_dm']:.6f} $\\mathrm{{pc cm^{{-3}}}}$]")
    ax2[2].legend(fancybox = True, loc = "upper right")


    fig2.tight_layout()
    fig2.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(os.path.join(args.o, f"{args.n}_crop.png"))    
    
    
    
    
    ################################################################
    # Plot dedispersed dynamic spectrum
    ################################################################
    fig4 = plt.figure(figsize = (10, 10))
    plt.imshow(data['dsIdeDM'], aspect = 'auto', extent = [-args.t / 2, args.t / 2, args.cfreq - args.bw/2, args.cfreq + args.bw/2])
    plt.xlabel("Time [ms]", fontsize = 16)
    plt.ylabel("Freq [MHz]", fontsize = 16)
    plt.title("De-dispersed Stokes I dynspec", fontsize = 16)    
    plt.savefig(os.path.join(args.o, f"{args.n}_dsI_despersed.png"))




    ################################################################
    # Make figure comparing Stokes I dynspec and ACF power dynspec
    ################################################################

    # 3 rows now: new row 0 holds the power-ACF panel (positive lag only)
    # above the coherent ACF trace panel (row 1, same column) - everything
    # else keeps the same relative layout, just shifted down one row.
    fig, ax = plt.subplots(3, 3, figsize = (16, 11),
                            gridspec_kw = {'height_ratios':[1, 1, 3], 'width_ratios':[3,3,1]})
    ax = ax.flatten()

    # row 0: power ACF panel (col1), col0/col2 unused this row
    ax[0].set_axis_off()
    ax[2].set_axis_off()

    # row 1 (same roles as the old row 0)
    # Power time series
    ax[3].plot(x, data['tI'], 'k')
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    ax[3].set_title("Intensity")

    ax[5].set_axis_off()

    # row 2 (same roles as the old row 1)
    # Stokes I dynspec
    ax[6].imshow(data['dsI'], aspect = 'auto', extent = [-args.t / 2, args.t / 2, args.cfreq - args.bw/2, args.cfreq + args.bw/2])
    ax[6].set_xlabel("Time [ms]", fontsize = 16)
    ax[6].set_ylabel("Freq [MHz]", fontsize = 16)
    ax[6].sharex(ax[3])

    # coherent ACF trace + peak
    x_acf = np.linspace(-args.t, args.t, data['tIacf'].size)
    peak_samp = np.nanargmax(np.abs(data['tIacf']))
    peak_lag_ms_coherent = x_acf[peak_samp]

    # power ACF trace + peak (independent cross-check). Zero-lag mask width
    # is matched to what clean_acf() already uses for the coherent ACF
    # (2*dmshifts_w, converted to ms), so both panels exclude the same
    # physical time-lag region around zero rather than two different,
    # inconsistently-chosen widths.
    zero_guard_ms = dm_zero_lag_halfwidth_ms(args, data['peak_dm'], data['tIacf'].size)
    print(f"Zero-lag mask half-width used (both ACFs): {zero_guard_ms:.6f} ms")
    lags_ms, pacf, peak_samp_pacf, peak_lag_ms_pacf = power_acf(args, data, zero_guard_ms=zero_guard_ms)

    # the power ACF is symmetric about zero lag (self-correlation of
    # burst + echo always produces peaks at +lag and -lag), while the
    # coherent ACF (a cross-correlation, not self-correlation) is not -
    # so compare magnitudes, not signed lags
    print(f"Coherent ACF peak lag:      {peak_lag_ms_coherent:.6f} ms")
    print(f"Power ACF peak lag:         {peak_lag_ms_pacf:.6f} ms")
    print(f"Coherent ACF |peak lag|:    {abs(peak_lag_ms_coherent):.6f} ms")
    print(f"Power ACF |peak lag|:       {abs(peak_lag_ms_pacf):.6f} ms")
    print(f"|Difference| (abs lags):    {abs(abs(peak_lag_ms_coherent) - abs(peak_lag_ms_pacf)):.6f} ms")

    # power ACF panel - show the full trace (both positive and negative
    # lag), sharing the x-axis with the coherent ACF panel below it so the
    # two are aligned on the same time-lag coordinate. Only the positive
    # side of the (symmetric) peak is marked, since that's the physically
    # meaningful one to read off.
    ax[1].plot(lags_ms, np.abs(pacf), 'b')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title("Power ACF")
    ax[1].sharex(ax[4])

    peak_lag_ms_pacf_pos = abs(peak_lag_ms_pacf)
    peak_idx_pos = np.nanargmin(np.abs(lags_ms - peak_lag_ms_pacf_pos))
    ax[1].plot(lags_ms[peak_idx_pos], np.abs(pacf)[peak_idx_pos], 'r', marker = "*",
                markersize = 12, zorder = 3, alpha = 0.7, label = f"Peak: {peak_lag_ms_pacf_pos:.4f} ms")
    ax[1].legend(fontsize = 8, loc = "upper right")

    # ACF time series
    ax[4].plot(x_acf, np.abs(data['tIacf']), 'k')
    ax[4].get_xaxis().set_visible(False)
    ax[4].get_yaxis().set_visible(False)
    ax[4].plot(x_acf[peak_samp], np.abs(data['tIacf'])[peak_samp], 'r', marker = "*", markersize = 12, 
                zorder = 0, alpha = 0.5)

    # ACF dynspec
    ax[7].imshow(data['acf'], aspect = 'auto', extent = [-args.t, args.t, args.cfreq - args.bw/2, args.cfreq + args.bw/2])
    ax[7].set_xlabel("Time-lag [ms]", fontsize = 16)
    ax[7].get_yaxis().set_visible(False)
    ax[7].sharex(ax[4])
    ax[7].sharey(ax[6])
    

    # Frequency components of ACF peak
    freqs = np.linspace(args.cfreq + args.bw//2 - args.bw/args.nFFT, 
                        args.cfreq - args.bw//2 + args.bw/args.nFFT, args.nFFT)
    ax[8].plot(data['acf'][:, peak_samp], freqs, 'k')
    ax[8].plot([np.abs(data['tIacf'])[peak_samp]]*2, [freqs[-1], freqs[0]], 'r--')
    ax[8].set_ylim([args.cfreq - args.bw/2, args.cfreq + args.bw/2])
    ax[8].get_yaxis().set_visible(False)
    ax[8].xaxis.tick_top()
    ax[8].set_xlabel("ACF power", fontsize = 16)
    ax[8].xaxis.set_label_position('top')
    # ax[8].sharey(ax[6])

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)

    plt.savefig(os.path.join(args.o, f"{args.n}_acf.png"))

    # stash power ACF products on data in case they're wanted for saving
    # or further inspection downstream
    data['pacf'] = pacf
    data['pacf_lags_ms'] = lags_ms
    data['peak_lag_ms_pacf'] = peak_lag_ms_pacf
    data['peak_lag_ms_coherent'] = peak_lag_ms_coherent







    ################################################################
    # Plot acf power VS DM
    ################################################################
    if not args.singledm:
        fig3 = plt.figure(figsize = (12,8))
        plt.plot(args.dmrange, data['corr_dm_list'], 'k', linewidth = 1.5)
        plt.xlabel("DM [$\mathrm{pc cm^{-3}}$]", fontsize = 16)
        plt.ylabel("ACF power (arb.)", fontsize = 16)
        fig3.tight_layout()
        plt.savefig(os.path.join(args.o, f"{args.n}_acf_dm.png"))



    #################################################################
    # Plot DM smearing as a function of channelisation
    #################################################################
    if args.more_plots:
        fig5, ax = plt.subplots(4, 1, figsize = (10,14))
        ax = ax.flatten()
        
        # calc dm smearing
        ftop = args.cfreq + args.bw/2
        fbottom = args.cfreq - args.bw/2

        nFFTarr = np.arange(1, 6720)

        delDMtop = args.idt * nFFTarr * 1e-3 / (4.14938e3 * (1/(ftop - 336/nFFTarr)**2 - 1/ftop**2))
        delDMbot = args.idt * nFFTarr * 1e-3 / (4.14938e3 * (1/fbottom**2 - 1/(fbottom + 336/nFFTarr)**2))

        ranges = [slice(0, 32), slice(32, 168), slice(168, 1024), slice(1024, 6720)]
        for i in range(4):
            ax[i].plot(nFFTarr[ranges[i]], delDMtop[ranges[i]], color = "#FD314C", linewidth = 2, label = "Maximum delta DM")
            ax[i].plot(nFFTarr[ranges[i]], delDMbot[ranges[i]], color = "#267DFF", linewidth = 2, label = "Minimum delta DM")
            if i == 3:
                ax[i].set_xlabel("# channels", fontsize = 14)
            ax[i].set_ylabel("DM [pc $cm^{-3}$]", fontsize = 14)    
            ylim = ax[i].get_ylim()
            if args.DMstep is not None:
                ax[i].plot([nFFTarr[ranges[i]][0], nFFTarr[ranges[i]][-1]], [args.DMstep]*2, 'k--', label = "DM step")
                ax[i].set_ylim(ylim)
            ax[i].grid()
        ax[0].legend()
        ax[0].set_title("Min DM to cause coherence smearing")

        fig5.subplots_adjust(wspace = 0)
        plt.savefig(os.path.join(args.o, f"{args.n}_DMsmear.png"))


    return




def print_info_and_statistics(args, data):
    """
    Print info and statistics about the data
    """

    if not args.singledm:
        print(f"Optimal DM: {data['peak_dm']}")

    # calculate Noise in ACF and the acf SNR, This calculation assumes that the noise level in both the
    # X and Y components are the same
    # peak_samp = np.nanargmax(data['tIacf'])
    # nsamps = int(data['acf_sample_count'][peak_samp])
    # acfcrop = data['tIacf'][peak_samp - nsamps:peak_samp + nsamps].copy()

    # peak_val = data['tIacf'][peak_samp]
    # acfcrop[acfcrop.size // 2] = 0.0
    # rms = np.std(acfcrop[int(0.2*nsamps):acfcrop.size - int(0.2*nsamps)])
    # print(np.abs(peak_val), rms)
    # print(f"\nACF S/N: {np.abs(peak_val) / rms}")
    # print(f"# samples at ACF peak: {nsamps}")


    return 





def save_data(args, data):
    """
    SAVE autocorrelation results
    """

    print(f"Saving ACF products as .npy files...")

    def _save(filepath, data):
        with open(os.path.join(args.o,filepath), 'wb') as file:
            np.save(file, data)
    
    if (args.save_config is not None) and (args.save_config != ""):
        print("Saving args to config file...")
        save_config(args)

    # acf products
    if args.save_data:
        _save(f"{args.n}_acf_dsI.npy", data['acf'])
        _save(f"{args.n}_acf_tI.npy", data['tIacf'])
        _save(f"{args.n}_acf_dsX.npy", data['dsX_unweighted'])
        _save(f"{args.n}_acf_dsY.npy", data['dsY_unweighted'])
        _save(f"{args.n}_crop_X.npy", data['X'])
        _save(f"{args.n}_crop_Y.npy", data['Y'])
        _save(f"{args.n}_dsX.npy", data['dsX'])
        _save(f"{args.n}_dsY.npy", data['dsY'])

        # save weights
        _save(f"{args.n}_W1.npy", data['tImf'])
        _save(f"{args.n}_W2.npy", data['tImfdeDM'])

        # save power acf products too
        if 'pacf' in data and data['pacf'] is not None:
            _save(f"{args.n}_power_acf.npy", data['pacf'])

    # save dms and list of corr vals if applicable
    if not args.singledm:
        print("Saving DM trial data...")
        _save(f"{args.n}_trial_DMs.npy", data['dm_list'])
        _save(f"{args.n}_corr_vals.npy", data['corr_dm_list'])

        # print info to .txt file
        with open(f"{os.path.join(args.o, args.n)}_info.txt", 'w') as file:
            file.write(f"Best delta DM: {data['peak_dm']}\n") 
            if 'peak_lag_ms_coherent' in data:
                file.write(f"Coherent ACF peak lag [ms]: {data['peak_lag_ms_coherent']}\n")
            if 'peak_lag_ms_pacf' in data:
                file.write(f"Power ACF peak lag [ms]: {data['peak_lag_ms_pacf']}\n")


    
    return

    



def _run(args, shared_list_corr_vals, shared_list_dms, list_dms):
    """
    Wrapper function to process deltaDM function in processing pool

    Parameters
    ----------
    shared_list_corr_vals : list of corr vals shared across processes
    shared_list_dms : list of dms processed shared across processes
    list_dms : dms to process in process instance
    """
    global X, Y, dsX, dsY, baseline, tIerr, flaggedchans
    trap = io.StringIO()
    for _, dm in enumerate(list_dms):

        # process correlation for delta dm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with redirect_stdout(trap):
                peak_val = corr_delta_dm(args, X, Y, dsX, dsY, baseline, tIerr, flaggedchans, dm)

        # add result to shared list 
        if dm not in shared_list_dms:
            shared_list_corr_vals.append(peak_val)
            shared_list_dms.append(dm)
        
            # print progress
            progressBar(args.ntask, len(shared_list_dms), "Corr Delta DMs:", f"Complete: {time.time() - args.start_time:.2f}s Elapsed")




if __name__ == "__main__":

    print("#"*32)
    print("Searching for lensing in FRB...")
    print("#"*32)

    args = get_args()

    # load data and crop
    data = load_data(args)


    #########################################################################
    # Make matched filter using Power spectra. Also make complex pol dynspec
    #
    #########################################################################
    setup_acf(args, data)

    if args.singledm:
        print("\n#" + "-"*45 + "#")
        print(f"Dedispersing to a single delta DM: {args.dmrange[0]:.4f} pc/cm^3")
        print("#" + "-"*45 + "#\n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data['acf'], data['tIacf'], data['dsIdeDM'], data['tIdeDM'], data['tImfdeDM'] = corr_delta_dm(
                args, data['X'], data['Y'], data['dsX'], data['dsY'], data['baseline_correction'], data['tIerr'], 
                data['flaggedchans_acf'], args.dmrange[0], True)

        data['peak_dm'] = args.dmrange[0]

    else:
        print("#" + "-"*40 + "#")
        print(f"Dedispersing to a range of delta DMs: min = {args.DMmin:.4f}, max = {args.DMmax:.4f}, step = {args.DMstep:.6f} pc/cm^3")
        print("#" + "-"*40 + "#\n")

        # set weighted X and Y to global variables for processing Pool
        global X, Y, dsX, dsY, baseline, tIerr, flaggedchans
        X = data['X'].copy()
        Y = data['Y'].copy()
        dsX = data['dsX'].copy()
        dsY = data['dsY'].copy()
        baseline = data['baseline_correction'].copy()
        tIerr = data['tIerr'].copy()
        flaggedchans = data['flaggedchans_acf'].copy()


        ############################################
        #   Setup processes to perform 
        #   autocorrelation for a number
        #   Of delta DMs. 
        ############################################
        task_arr = split_tasks_into_processes(args.dmrange, args.cpus)

        # setup processing pool
        pool = mp.Pool(args.cpus)
        manager = mp.Manager()
        # create shared lists of completed DM values
        shared_list_corr_vals = manager.list()
        shared_list_dms = manager.list()

        args.start_time = time.time()   # initial start time, for time-keeping during processing
        args.ntask = args.dmrange.size # set number of tasks, for logging
        # create processes
        for task_batch in task_arr:
            pool.apply_async(_run, (args, shared_list_corr_vals, shared_list_dms, task_batch,),)

        # start process
        pool.close()
        pool.join()


        # Summarise pool processing and get best delta DM
        data['peak_dm'], data['dm_list'], data['corr_dm_list'] = get_best_deltaDM(data, shared_list_corr_vals, shared_list_dms)

        # run corr for optimal DM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data['acf'], data['tIacf'], data['dsIdeDM'], data['tIdeDM'], data['tImfdeDM'] = corr_delta_dm(
                                            args, X, Y, dsX, dsY, baseline, tIerr, flaggedchans, data['peak_dm'], True)
    #################################
    # diagnostic plotting and info
    #
    #################################
    plot_diagnostics(args, data)

    # print stats
    print_info_and_statistics(args, data)

    # save data
    save_data(args, data)

    print("search4frblensing.py Completed!\n")

    if args.showplots:
        print("Plotting data...")
        plt.show()


    # END OF SCRIPT
