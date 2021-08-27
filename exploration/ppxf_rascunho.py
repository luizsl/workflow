#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 17:31:49 2021

@author: chess-lin
"""

import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from spectrum import Spectrum
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from scipy.constants import physical_constants
from os import path
import ppxf as ppxf_package
import matplotlib.pyplot as plt

#######################################################              test
ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
# model_file = ('../models/tmpWzZ2t1/Mku1.30Zm0.40T00.1000_iPp0.00_baseFe'
#               '_linear_FWHM_2.51.fits')
# vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30Z*.fits')
vazdekis = glob.glob('../data/models/tmpWzZ2t1/Mku1.30Z*.fits')
obs_file = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

#########################################################################

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

# create a single model to be used as template
with fits.open(vazdekis[0]) as hdu:
    flux = hdu['PRIMARY'].data
    first_wave = hdu['PRIMARY'].header['CRVAL1']
    step_wave =  hdu['PRIMARY'].header['CDELT1']
    model = Spectrum(flux, wave = [first_wave, step_wave],
                     medium = 'air', sampling_type = 'linear')
    model.normalize_median()

with fits.open(obs_file) as hdu:
    data = hdu['DATA'].data
    flux = data[:, 105:106, 105:106].sum(axis=1).sum(axis=1)
    error_d = hdu['STAT'].data
    error = error_d[:, 105:106, 105:106].sum(axis=1).sum(axis=1)
    first_wave = hdu['DATA'].header['CRVAL3']
    step_wave = hdu['DATA'].header['CD3_3']
    obs = Spectrum(flux, wave = [first_wave, step_wave],
                   medium = 'air', sampling_type = 'linear', flux_unc = error)
    # obs.flux = np.nan_to_num(obs.flux)
    mask = (obs.wave > np.min(model.wave)) & (obs.wave < np.max(model.wave))
    obs.trim_w_mask(mask)
    obs.normalize_median()
    
    wave_ln = np.e**(np.arange(np.log(obs.wave[0]), 
                               np.log(obs.wave[-1]), 
                               1*np.log(obs.wave[1]/obs.wave[0])))
    
    obs.resampling(wave_ln, new_sampling_type = 'ln')
    # obs.flux = np.nan_to_num(obs.flux)
    obs.plot()

########## Observation Properties ###########

z = 0.005  # redshift estimate  TO DO: get the estimation automatically from some place
velscale = np.log(obs.wave[1]/obs.wave[0])*C   #(eq.8 of Cappellari 2017)

############# Model properties ##############

fwhm_model = 2.51 #Vazdekis+10

########### muse properties #################

# interpolate lsf for each pixel of observations
fwhm_muse = pd.read_csv('../data/misc_data/muse_manual_resolution.csv',
                        delimiter = ',', index_col = False)
fwhm_obs = np.interp(model.wave, fwhm_muse['lambda'], fwhm_muse['fwhm_A'])

############# Fit configuration #############

fwhm_dif = np.sqrt((fwhm_obs**2 - fwhm_model**2).clip(0))
sigma = fwhm_dif / (2.355*np.e**(model.wave[1]/model.wave[0])) # Sigma difference in pixels

templates = np.empty((wave_ln.size, len(vazdekis)))
for j, file in enumerate(vazdekis):
    with fits.open(vazdekis[j]) as hdu:
        flux = hdu['PRIMARY'].data
        first_wave = hdu['PRIMARY'].header['CRVAL1']
        step_wave =  hdu['PRIMARY'].header['CDELT1']
        model = Spectrum(flux, wave = [first_wave, step_wave],
                         medium = 'air', sampling_type = 'linear')
        model.convolve(sigma)
        model.resampling(wave_ln, new_sampling_type = 'ln')
        model.flux = np.nan_to_num(model.flux)
        model.normalize_median()
        templates[:, j] = model.flux

noise = np.sqrt(obs.flux_unc)/np.nanmedian(np.sqrt(obs.flux_unc))
noise[0] = np.nanmean(noise)
wave_range_model = [np.min(model.wave), np.max(model.wave)]
dv = C*np.log(model.wave[0]/obs.wave[0])    # eq.(8) of Cappellari (2017)
goodpixels = util.determine_goodpixels(np.log(obs.wave), wave_range_model, z)
vel = C*np.log(1 + z)   # eq.(8) of Cappellari (2017)
start = [vel, 200.]  # (km/s), starting guess for [V, sigma]

pp = ppxf(templates[:, :], obs.flux, noise, velscale, start,
          goodpixels = goodpixels, plot = True, moments=4,
          degree = 10, vsyst = dv, clean = True, lam=obs.wave)
