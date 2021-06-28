#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:33:53 2021

@author: chess-lin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ppxf.ppxf_util as util
from spectrum import Spectrum
from astropy.io import fits

def convolve_test(flux, lsf):
    
    fwhm_dif = np.sqrt((lsf**2 - fwhm_model**2).clip(0))
    
    sigma = fwhm_dif/2.355/header['CDELT1'] # Sigma difference in pixels
    
    convolved_signal = util.gaussian_filter1d(flux, sigma)  # perform convolution with variable sigma
    
    return convolved_signal

model_file = '../data/models/tmpWzZ2t1/Mku1.30Zm0.40T00.1000_iPp0.00_baseFe_linear_FWHM_2.51.fits'

muse_lsf_file = '../data/misc_data/muse_manual_resolution.csv'

with fits.open(model_file) as hdu:
    header = hdu[0].header
    flux = hdu[0].data
    
muse_lsf = pd.read_csv(muse_lsf_file)
fwhm_model = 2.51

model = Spectrum(flux = flux,
                 wave = [header['CRVAL1'], header['CDELT1']],
                 medium = 'air', sampling_type = 'linear')
    
# Mask the mode to MUSE
trim = np.ma.masked_outside(model.wave, 
                     muse_lsf['lambda'].min(), muse_lsf['lambda'].max()).mask
model.trim_w_mask(np.invert(trim))
model.normalize_median()

# General
lsf_interp = np.interp(model.wave, muse_lsf['lambda'], muse_lsf['fwhm_A'])
model_conv_interp = convolve_test(model.flux, lsf_interp)

# ELODIE style
lsf_piecewise = np.zeros_like(model.flux)
interval = int(np.ceil(lsf_piecewise.size / 4))

inter_1 = np.repeat(np.median(lsf_interp[0*interval:1*interval]), interval)
inter_2 = np.repeat(np.median(lsf_interp[1*interval:2*interval]), interval)
inter_3 = np.repeat(np.median(lsf_interp[2*interval:3*interval]), interval)
inter_4 = np.repeat(np.median(lsf_interp[3*interval:4*interval]), interval)

lsf_piecewise = np.concatenate((inter_1, inter_2, inter_3, inter_4))[:model.flux.size]

model_conv_piecewise = convolve_test(model.flux, lsf_piecewise)

# A single LSF
lsf_median = np.full_like(model.flux, np.median(lsf_interp))
model_conv_median = convolve_test(model.flux, lsf_median)

# Plot LSF
fig, ax = plt.subplots()
ax.step(model.wave, lsf_median, where = 'mid', label = 'Single LSF')
ax.step(model.wave, lsf_piecewise, where = 'mid', label = 'Piecewise LSF')
ax.step(model.wave, lsf_interp, where = 'mid', label = 'General')
ax.legend()

ax.set_xlabel(r'$\textbf{Wavelength} \mathbf{(\AA)}$')
ax.set_ylabel(r'$\textbf{FWHM in } \AA$')
ax.set_title(r'LSF for convolution')

plt.savefig('../plots/lsf_comparison.pdf')

# Plot flux
fig, ax = plt.subplots()
ax.step(model.wave, model_conv_piecewise, where = 'mid', label = 'Model',
        color = 'Grey')
ax.step(model.wave, model_conv_median, where = 'mid', label = 'Single')
ax.step(model.wave, model_conv_piecewise, where = 'mid', label = 'Piecewise')
ax.step(model.wave, model_conv_interp, where = 'mid', label = 'General')
ax.legend()

ax.set_xlabel(r'$\textbf{Wavelength} \mathbf{(\AA)}$')
ax.set_ylabel(r'$\textbf{Flux density in arbitrary units}$')
ax.set_title(r'Convolution comparison')

plt.savefig('../plots/convolution_comparison.pdf')


