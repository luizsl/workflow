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

#######################################################              test
ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
# model_file = ('../models/tmpWzZ2t1/Mku1.30Zm0.40T00.1000_iPp0.00_baseFe'
#               '_linear_FWHM_2.51.fits')
vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30Z*.fits')
obs_file = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

#########################################################################

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

# create a single model to be used as template
# model_file = ''
with fits.open(vazdekis[0]) as hdu:
    flux = hdu['PRIMARY'].data
    first_wave = hdu['PRIMARY'].header['CRVAL1']
    step_wave =  hdu['PRIMARY'].header['CDELT1']
    model = Spectrum(flux, wave = [first_wave, step_wave],
                     medium = 'air', sampling_type = 'linear')
    model.normalize_median()
            
# read observation
# obs_file = ''
with fits.open(obs_file) as hdu:
    hdu.info()
    flux = hdu['DATA'].data[:, 100, 100]
    first_wave = hdu['DATA'].header['CRVAL3']
    step_wave = hdu['DATA'].header['CD3_3']
    obs = Spectrum(flux, wave = [first_wave, step_wave],
                   medium = 'air', sampling_type = 'linear')
    wave_ln = np.e**np.array(np.log(obs.wave[0]) + np.arange(3000) * np.log(obs.wave[1]/obs.wave[0]))
    obs.rebinning(wave_ln, new_sampling_type = 'ln')
    # mask = (obs.wave > np.min(model.wave)) & (obs.wave < np.max(model.wave))
    # obs.trim_w_mask(mask)
    # obs.normalize_median()

########## Observation Properties ###########

z = 0.001  # redshift estimate  TO DO: get the estimation automatically from some place

#assert it's sampled in ln
obs.flux, obs.wave, velscale = util.log_rebin([obs.wave[0], obs.wave[-1]], obs.flux)

#velscale = np.log(obs.wave[1]/obs.wave[0])*C   #(eq.8 of Cappellari 2017)

############# Model properties ##############

fwhm_model = 2.51 #Vazdekis+10

model.flux, model.wave, velscale_temp = util.log_rebin([np.min(model.wave), np.max(model.wave)],
                         model.flux, velscale=velscale)

########### muse properties #################

# interpolate lsf for each pixel of observations
fwhm_muse = pd.read_csv('../muse_description/muse_manual_resolution.csv',
                        delimiter = ',', index_col = False)
fwhm_obs = np.interp(np.e**model.wave, fwhm_muse['lambda'], fwhm_muse['fwhm_A'])

############# Fit configuration #############

fwhm_dif = np.sqrt((fwhm_obs**2 - fwhm_model**2).clip(0))
sigma = fwhm_dif / (2.355*np.e**(model.wave[1]/model.wave[0])) # Sigma difference in pixels

a = util.log_rebin([np.min(model.wave), np.max(model.wave)], model.flux, velscale=velscale_temp)[0]
  
noise = np.full_like(obs.flux, 0.0166)       # Assume constant noise per pixel here
wave_range_model = [np.min(model.wave), np.max(model.wave)]
dv = C*np.log(model.wave[0]/obs.wave[0])    # eq.(8) of Cappellari (2017)
goodpixels = util.determine_goodpixels(np.log(obs.wave), wave_range_model, z)
vel = C*np.log(1 + z)   # eq.(8) of Cappellari (2017)
start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    
pp = ppxf(model.flux, obs.flux, noise, velscale, start,
          goodpixels = goodpixels, plot = True, moments=4,
          degree = 12, vsyst = dv, clean = False, lam=obs.wave)



# a = np.array([[[ 2,  8,  9],
#         [ 4,  5, 10],
#         [ 3,  6,  7]],
#        [[ 2,  8,  9],
#         [ 4,  5, 10],
#         [ 3,  6,  7]]])

# b = np.array([[0, 7, 0],
#        [3, 0, 9],
#        [0, 5, 0]])