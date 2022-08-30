#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:40:35 2022

@author: Luiz
"""
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter


class SnrMap:
    def __init__(self, path, z=None):
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul, \
            tempfile.TemporaryFile() as flux_file, \
            tempfile.TemporaryFile() as flux_unc_file:
            
            self.wcs = WCS(hdul[1].header)
            
            where_nan = ~np.isfinite(hdul['DATA'].data[-1, ...])

            self.flux = np.memmap(
                flux_file, dtype='float32', mode='w+',
                shape=hdul['DATA'].data.shape)
            
            hdul['DATA'].data[-1, ...][where_nan] = \
                hdul['DATA'].data[-2, ...][where_nan]
            self.flux[:] = hdul['DATA'].data[:]
            
            self.flux_unc = np.memmap(
                flux_unc_file, dtype='float32', mode='w+',
                shape=hdul['STAT'].data.shape)
            hdul['STAT'].data[-1, ...][where_nan] = \
                hdul['STAT'].data[-2, ...][where_nan]
            self.flux_unc[:] = hdul['STAT'].data[:]
            
            self.first = hdul['data'].header['CRVAL3']
            self.step = hdul['data'].header['CD3_3']
            self.n_pix = hdul['data'].header['NAXIS3']
            
            self.wave = self.first + np.arange(self.n_pix)*self.step
            
            if z is not None:
                self.wave = self.wave / (1+z)
                
    def full_axis(self):
        # noise is the variance
        signal = np.mean(self.flux, axis=0)
        noise = np.mean(self.flux_unc, axis=0)
        snr = signal / np.sqrt(noise)
        return snr
        
    def range_axis(self, rng:list):
        # noise is the variance
        assert len(rng) == 2
        w = (self.wave > rng[0]) & (self.wave < rng[1])
        signal = np.mean(self.flux[w], axis=0)
        noise = np.mean(self.flux_unc[w], axis=0)
        snr = signal / np.sqrt(noise)
        return snr
    
if __name__ == '__main__':
    # Read the file with the MUSE datacube. The redshift can be included when 
    # instantiating the object. That is useful when computing the 
    # signal-to-noise (S/N) within a interval of wavelength

    # path = '../data/fov_sample_1_3.fits'
    path = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits.gz'
    t = SnrMap(path, 0.004951)
    
    # Build coorcinates centred on the brightest spaxel
    flux = np.nansum(t.flux, axis=0)
    ym, xm = np.unravel_index(np.nanargmax(flux), flux.shape)
    row = np.arange(flux.shape[0])
    col = np.arange(flux.shape[1])
    pixsize = 0.2
    x = (col - xm)*pixsize
    y = (row - ym)*pixsize

    #%%
    kinematics_path = '../data_products/NGC0613_DATACUBE_FINAL_clean/XSLAgeMh_2/ppxf/sol.fits'
    
    with fits.open(kinematics_path) as hdul:
        data = hdul[0].data
        vel = data[0,:,:]
        sig = data[1,:,:]
        h_3 = data[2,:,:]
        h_4 = data[3,:,:]
        
    # Plot kinematics 
    plt.style.use('../src/fig_conf.mplstyle')
    
    # fig, ax = plt.subplots(1, 2, figsize=(12,4))
    ax = np.full((2,2), fill_value=None, dtype=object)
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(1, 1),
                          left=0.05, right=0.94, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.25)
    ax[0, 0] = fig.add_subplot(gs[0, 0])
    ax[0, 1] = fig.add_subplot(gs[0, 1])
    
    ax[1, 0] = fig.add_subplot(gs[1, 0])
    ax[1, 1] = fig.add_subplot(gs[1, 1])
    
    extent = [x[0]-pixsize, x[-1]-pixsize, y[0]-pixsize, y[-1]-pixsize]
    
    vel_plot = ax[0,0].imshow(
        vel, origin='lower',
        vmin=-120, vmax= 120,
        extent=extent)
    vel_cbar = plt.colorbar(vel_plot, ax=ax[0,0])
    ax[0,0].set_title(r'V$_{\star}$')
    vel_cbar.set_label(r'km\,s$^{-1}$', fontsize=12,
                   rotation = 270, labelpad=15)
    
    sig_plot = ax[0,1].imshow(
        sig, origin='lower',
        vmin=50, vmax= 125,
        extent=extent)
    sig_cbar = plt.colorbar(sig_plot, ax=ax[0,1])
    ax[0,1].set_title(r'$\sigma_{\star}$')
    sig_cbar.set_label(r'km\,s$^{-1}$', fontsize=12,
                   rotation = 270, labelpad=15)
    
    h_3_plot = ax[1,0].imshow(
        h_3, origin='lower',
        vmin=-0.15, vmax= 0.15,
        extent=extent)
    h_3_cbar = plt.colorbar(h_3_plot, ax=ax[1,0])
    ax[1,0].set_title(r'h$_{3}$')
    

    h_4_plot = ax[1,1].imshow(
        h_4, origin='lower',
        vmin=-0.05, vmax= 0.15,
        extent=extent)
    h_4_cbar = plt.colorbar(h_4_plot, ax=ax[1,1])
    ax[1,1].set_title(r'h$_{4}$')

    ax[0,0].set_xlim(-34, 33)
    ax[0,1].set_ylim(-34, 31)
    ax[1,0].set_xlabel('arcsec')
    ax[0,0].set_ylabel('arcsec')
    ax[1,0].set_xlim(-34, 33)
    ax[1,1].set_ylim(-34, 31)
    ax[1,1].set_xlabel('arcsec')
    ax[1,0].set_ylabel('arcsec')
    
    plt.savefig('../plots/kinematics_map_NGC613.pdf')
    plt.show()
    
    #%%
    stellar_age_path = '../data_products/NGC0613_DATACUBE_FINAL_clean/XSLAgeMh_2/ppxf/mean_log10_age_light.fits'
    stellar_mh_path = '../data_products/NGC0613_DATACUBE_FINAL_clean/XSLAgeMh_2/ppxf/mean_mh_light.fits'
    
    with fits.open(stellar_age_path) as hdul:
        stellar_age = hdul[0].data

    with fits.open(stellar_mh_path) as hdul:
        stellar_mh = hdul[0].data
        
        
    # Plot stellar parameters 
    plt.style.use('../src/fig_conf.mplstyle')
    
    # fig, ax = plt.subplots(1, 2, figsize=(12,4))
    ax = np.full((1,2), fill_value=None, dtype=object)
    fig = plt.figure(figsize=(8, 3.2))
    gs = fig.add_gridspec(1, 2,  width_ratios=(1, 1),
                          left=0.05, right=0.94, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.25)
    ax[0, 0] = fig.add_subplot(gs[0, 0])
    ax[0, 1] = fig.add_subplot(gs[0, 1])
    
    extent = [x[0]-pixsize, x[-1]-pixsize, y[0]-pixsize, y[-1]-pixsize]
    
    age_plot = ax[0,0].imshow(
        10**stellar_age/1e9, origin='lower',
        vmin=3, vmax= 11,
        extent=extent)
    age_cbar = plt.colorbar(age_plot, ax=ax[0,0])
    ax[0,0].set_title(r'Mean stellar age')
    age_cbar.set_label(r'[Gyr]', fontsize=12,
                   rotation = 270, labelpad=15)
    
    mh_plot = ax[0,1].imshow(
        stellar_mh, origin='lower',
        vmin=-0.15, vmax= 0.15,
        extent=extent)
    mh_cbar = plt.colorbar(mh_plot, ax=ax[0,1])
    ax[0,1].set_title(r'Mean metallicity')
    mh_cbar.set_label(r'[M/H]', fontsize=12,
                   rotation = 270, labelpad=15)
    
    ax[0,0].set_xlim(-34, 33)
    ax[0,0].set_ylim(-34, 31)
    ax[0,0].set_xlabel('arcsec')
    ax[0,0].set_ylabel('arcsec')
    ax[0,1].set_xlim(-34, 33)
    ax[0,1].set_ylim(-34, 31)
    ax[0,1].set_xlabel('arcsec')
    # ax[0,1].set_ylabel('arcsec')
    
    plt.savefig('../plots/age_mh_map_NGC613.pdf')
    plt.show()
    
    #%%
    stellar_reddening = '../data_products/NGC0613_DATACUBE_FINAL_clean/XSLAgeMh_2/ppxf/reddening.fits'
    
    with fits.open(stellar_reddening) as hdul:
        stellar_reddening = hdul[0].data
        
    # Plot stellar parameters 
    plt.style.use('../src/fig_conf.mplstyle')
    
    # fig, ax = plt.subplots(1, 2, figsize=(12,4))
    ax = np.full((1,1), fill_value=None, dtype=object)
    fig = plt.figure(figsize=(6, 3.2))
    gs = fig.add_gridspec(1, 1)
    ax[0, 0] = fig.add_subplot(gs[0, 0])
    
    extent = [x[0]-pixsize, x[-1]-pixsize, y[0]-pixsize, y[-1]-pixsize]
    
    red_plot = ax[0,0].imshow(
        stellar_reddening, origin='lower',
        vmin=0.01, vmax= 0.35,
        extent=extent)
    red_cbar = plt.colorbar(red_plot, ax=ax[0,0])
    ax[0,0].set_title(r'Stellar Reddening')
    red_cbar.set_label(r'E(B-V)', fontsize=12,
                   rotation = 270, labelpad=15)
    
    ax[0,0].set_xlim(-34, 33)
    ax[0,0].set_ylim(-34, 31)
    ax[0,0].set_xlabel('arcsec')
    ax[0,0].set_ylabel('arcsec')
    # ax[0,1].set_xlim(-34, 33)
    # ax[0,1].set_ylim(-34, 31)
    # ax[0,1].set_xlabel('arcsec')
    # # ax[0,1].set_ylabel('arcsec')
    
    plt.savefig('../plots/stellar_reddening_map_NGC613.pdf')
    plt.show()