#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:16:51 2021

@author: Luiz
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import spectcube as sc


def build_input(input_cube_ppxf_path, stellar_flux_path):
    
    # input_cube_ppxf_path = '../run_ppxf/input_cube_ppxf.fits'
    # stellar_flux_path = '../run_ppxf/NGC613_1/bestfit.fits'

    with fits.open(input_cube_ppxf_path) as hdu:
        header0 = hdu[0].header
        header1 = hdu[1].header
        header2 = hdu[2].header
    
        flux_obs = np.array(hdu[1].data, dtype = 'float32')
        flux_obs = flux_obs[:, 100:110, 100:110]
        
        flux_unc_obs = np.array(hdu[2].data, dtype = 'float32')
        flux_unc_obs = flux_unc_obs[:, 100:110, 100:110]
        
        old_log_wave = \
            sc.util.build_wave_array(wave = [header1['CRVAL3'], header1['CD3_3']],
                                     sampling_type = 'log',
                                     size = header1['NAXIS3'])
        new_lin_wave = \
            sc.util.fit_wave_interval(wave = old_log_wave, 
                                      old_sampling = 'log',
                                      new_sampling = 'linear')
            
        flux_obs_res, _, flux_unc_obs_res = \
            sc.resampling(flux = flux_obs, 
                          old_wave = old_log_wave, old_sampling_type = 'log',
                          new_wave = new_lin_wave, new_sampling_type = 'linear',
                          flux_err = flux_unc_obs)
    
        del flux_obs, flux_unc_obs, hdu
        flux_obs_res = np.array(flux_obs_res, dtype = np.single)
        flux_unc_obs_res = np.array(flux_unc_obs_res, dtype = np.single)
            
    new_hdul = fits.HDUList()
    new_hdul.append(fits.ImageHDU())
    new_hdul.append(fits.ImageHDU())
    new_hdul.append(fits.ImageHDU())
    new_hdul.append(fits.ImageHDU())
    new_hdul.append(fits.ImageHDU())

    new_hdul[0].header = header0
    
    new_hdul[1].header = header1[:-2]
    new_hdul[1].name = 'sci'
    new_hdul[1].data = flux_obs_res
    new_hdul[1].header['CTYPE3'] = 'AWAV'
    new_hdul[1].header['CRVAL3'] = new_lin_wave[0]
    new_hdul[1].header['CD3_3'] = new_lin_wave[1] - new_lin_wave[0]
    
    new_hdul[2].header = header2[:-2]
    new_hdul[2].name = 'error'
    new_hdul[2].data = flux_unc_obs_res
    new_hdul[2].header['CTYPE3'] = 'AWAV'
    new_hdul[2].header['CRVAL3'] = new_lin_wave[0]
    new_hdul[2].header['CD3_3'] = new_lin_wave[1] - new_lin_wave[0]
    
    # new_hdul[2].header = header1[:-2]
    new_hdul[3].name = 'mask'
    mask_nan = np.any(np.isnan(flux_obs_res[:, ...]), axis = 0)
    new_hdul[3].data = np.array(mask_nan, dtype = int)
    # new_hdul[2].header['CTYPE3'] = 'AWAV'
    # new_hdul[2].header['CRVAL3'] = new_lin_wave[0]
    # new_hdul[2].header['CD3_3'] = new_lin_wave[1] - new_lin_wave[0]
    
    del flux_obs_res, flux_unc_obs_res
    
    with fits.open(stellar_flux_path) as hdu:
        stellar_flux = np.array(hdu[0].data, dtype = 'float32')
        stellar_flux = stellar_flux[:, 100:110, 100:110]
        
        stellar_flux_res, _, _ = \
            sc.resampling(flux = stellar_flux, 
                          old_wave = old_log_wave, old_sampling_type = 'log',
                          new_wave = new_lin_wave, new_sampling_type = 'linear')
        stellar_flux_res = np.array(stellar_flux_res, dtype = np.single)     
            
    new_hdul[4].header = header2[:-2]
    new_hdul[4].name = 'stellar'
    new_hdul[4].data = stellar_flux_res
    new_hdul[4].header['CTYPE3'] = 'AWAV'
    new_hdul[4].header['CRVAL3'] = new_lin_wave[0]
    new_hdul[4].header['CD3_3'] = new_lin_wave[1] - new_lin_wave[0]
    
    new_hdul.info()
    
    new_hdul.writeto('input_cube_ifscube.fits', overwrite = True)

def run_ifscube(input_ifscube_path, conf_ifscube_path):
    
    os.system(f'cubefit -oc {conf_ifscube_path} {input_ifscube_path}')

    
if __name__ == '__main__':
    
    build_input(input_cube_ppxf_path = '../run_ppxf/input_cube_ppxf.fits',
                stellar_flux_path = '../run_ppxf/NGC613_1/bestfit.fits')
    
    run_ifscube(input_ifscube_path = 'input_cube_ifscube.fits',
                conf_ifscube_path = 'halpha_cube_muse.cfg')


#%%
# x = 9

# hdu_res = fits.open('input_cube_ifscube_linefit.fits')

# # print(hdu_res['status'].data)

# plt.plot(hdu_res['restwave'].data,
#          hdu_res['fitspec'].data[:,x,0] - hdu_res['stellar'].data[:,x,0], label = 'spectrum')
# # plt.plot(hdu_res['restwave'].data, hdu_res['var'].data[:,x,0])
# # plt.plot(hdu_res['restwave'].data, hdu_res['model'].data[:,x,0], label = 'model')
# plt.plot(hdu_res['restwave'].data, hdu_res['fitcont'].data[:,x,0], label = 'continuum')
# plt.plot(hdu_res['restwave'].data,
#          hdu_res['model'].data[:,x,0] - hdu_res['stellar'].data[:,x,0], label = 'fit')
# plt.legend()

# # hdu_res['solution'].data[:,2,0]

