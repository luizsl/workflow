#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:38:11 2021

@author: Luiz


            **************** Temporary ******************
"""

import numpy as np
from astropy.io import fits

flux_path = '../run_ppxf/flux_obs.dat'
flux_unc_path = '../run_ppxf/flux_obs_unc.dat' 
original_cube = '../../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'
shape_obs = (2586, 104293)


new_hdul = fits.HDUList()
new_hdul.append(fits.ImageHDU())
new_hdul.append(fits.ImageHDU())
new_hdul.append(fits.ImageHDU())

with fits.open(original_cube) as hdu:
    header0 = hdu[0].header
    header1 = hdu[1].header
    header2 = hdu[2].header
    
wave = data.wave_obs
wave_log = np.log10(wave)

obs_flux = np.memmap(filename = flux_path, dtype = 'float32', mode='r',
                     shape = shape_obs)
obs_flux = obs_flux.reshape((-1,) + (317, 329))

obs_unc_flux = np.memmap(filename = flux_unc_path, dtype = 'float32', mode='r',
                         shape = shape_obs)
obs_unc_flux = obs_unc_flux.reshape((-1,) + (317, 329))


new_hdul[1].header = header1[:-2]
new_hdul[1].name = 'data'
new_hdul[1].data = obs_flux
new_hdul[1].header['CTYPE3'] = 'AWAV-LOG'
new_hdul[1].header['CRVAL3'] = wave_log[0]
new_hdul[1].header['CD3_3'] = np.median(np.diff(wave_log))

new_hdul[2].header = header2[:-2]
new_hdul[2].name = 'stat'
new_hdul[2].data = obs_unc_flux
new_hdul[2].header['CTYPE3'] = 'AWAV-LOG'
new_hdul[2].header['CRVAL3'] = wave_log[0]
new_hdul[2].header['CD3_3'] = np.median(np.diff(wave_log))


new_hdul.writeto('input_cube_ppxf.fits', overwrite = True)

hdu_res = fits.open('input_cube_ppxf.fits')
