#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:47:10 2022

@author: Luiz
"""
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS


class SnrMap:
    def __init__(self, path, z=None):
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul, \
            tempfile.TemporaryFile() as flux_file, \
            tempfile.TemporaryFile() as flux_unc_file:
            
            self.wcs = WCS(hdul[1].header)
                
            self.flux = np.memmap(
                flux_file, dtype='float32', mode='w+',
                shape=hdul['DATA'].data.shape)
            self.flux[:] = hdul['DATA'].data[:]
            
            self.flux_unc = np.memmap(
                flux_unc_file, dtype='float32', mode='w+',
                shape=hdul['STAT'].data.shape)
            self.flux_unc[:] = hdul['STAT'].data[:]
            
            self.first = hdul['data'].header['CRVAL3']
            self.step = hdul['data'].header['CD3_3']
            self.n_pix = hdul['data'].header['NAXIS3']
            
            self.wave = self.first + np.arange(self.n_pix)*self.step
            
            if z is not None:
                self.wave = self.wave / (1+z)
                
    def full_axis(self):
        signal = np.nansum(self.flux, axis=0)
        noise = np.nansum(self.flux_unc, axis=0)
        snr = signal / (np.sqrt(noise * self.n_pix))
        return snr
        
    def range_axis(self, rng:list):
        assert len(rng) == 2
        w = (self.wave > rng[0]) & (self.wave < rng[1])
        signal = np.nansum(self.flux[w], axis=0)
        noise = np.nansum(self.flux_unc[w], axis=0)
        snr = signal / (np.sqrt(noise * w.sum()))
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
    
    # First, we compute the S/N with the full spectral axis and with the slice
    # of the spectral axis between 5450 and 5550 A.
    plt.style.use('../src/fig_conf.mplstyle')
    
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    extent = [x[0]-pixsize, x[-1]-pixsize, y[0]-pixsize, y[-1]-pixsize]
    
    im0 = ax[0].imshow(t.full_axis(), origin='lower',
                       vmin=3, vmax= 80, extent=extent)
    
    ker_0 = gaussian_filter(t.full_axis(), 4)
    cont_0 = ax[0].contour(ker_0, levels=[10, 15, 20, 30, 40, 60],
                           colors='k', linewidths=0.8, linestyles='dashdot',
                           extent=extent)
    ax[0].clabel(cont_0, fontsize=10, inline=True)
    ax[0].set_title('full spectral axis')

    
    im1 = ax[1].imshow(t.range_axis([5450, 5550]), origin='lower',
                       vmin=3, vmax= 80, extent=extent)
    ker_1 = gaussian_filter(t.range_axis([5450, 5550]), 4)
    cont_1 = ax[1].contour(ker_1, levels=[10, 15, 20, 30, 40, 60],
                           colors='k', linewidths=0.8, linestyles='dashdot',
                           extent=extent)
    ax[1].clabel(cont_1, fontsize=10, inline=True)
    ax[1].set_title('5450 - 5550 \AA')

    
    cbar = plt.colorbar(im1, ax=ax[:])
    cbar.set_label('average signal-to-noise', fontsize=12,
                   rotation = 270, labelpad=20)
    
    ax[0].set_xlim(-34, 33)
    ax[0].set_ylim(-34, 31)
    ax[0].set_xlabel('arcsec')
    ax[0].set_ylabel('arcsec')
    ax[1].set_xlim(-34, 33)
    ax[1].set_ylim(-34, 31)
    ax[1].set_xlabel('arcsec')
    ax[1].set_ylabel('arcsec')
    
    plt.savefig('../plots/SNR_map_NGC613_without_minimal.pdf')
    plt.show()

    # Also remove the spaxels with S/N <= 3 and redo the plot

    plt.style.use('../src/fig_conf.mplstyle')

    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    extent = [x[0]-pixsize, x[-1]-pixsize, y[0]-pixsize, y[-1]-pixsize]

    gp_0 = np.where(t.full_axis() > 3, t.full_axis(), np.nan)
    im0 = ax[0].imshow(gp_0, origin='lower',
                       vmin=3, vmax= 80, extent=extent)
    ker_0 = gaussian_filter(gp_0, 4)
    cont_0 = ax[0].contour(ker_0, levels=[10, 15, 20, 30, 40, 60],
                           colors='k', linewidths=0.8, linestyles='dashdot',
                           extent=extent)
    ax[0].clabel(cont_0, fontsize=10, inline=True)
    ax[0].set_title('full spectral axis')
    
    gp_1 = np.where(t.range_axis([5450, 5550]) > 3, t.range_axis([5450, 5550]),
                    np.nan)
    im1 = ax[1].imshow(gp_1, origin='lower',
                       vmin=3, vmax= 80, extent=extent)
    ker_1 = gaussian_filter(gp_1, 4)
    cont_1 = ax[1].contour(ker_1, levels=[10, 15, 20, 30, 40, 60],
                           colors='k', linewidths=0.8, linestyles='dashdot',
                           extent=extent)
    ax[1].clabel(cont_1, fontsize=10, inline=True)
    ax[1].set_title('5450 - 5550 \AA')
    
    cbar = plt.colorbar(im1, ax=ax[:])
    cbar.set_label('average signal-to-noise', fontsize=12,
                   rotation = 270, labelpad=20)
    
    ax[0].set_xlim(-34, 33)
    ax[0].set_ylim(-34, 31)
    ax[0].set_xlabel('arcsec')
    ax[0].set_ylabel('arcsec')
    ax[1].set_xlim(-34, 33)
    ax[1].set_ylim(-34, 31)
    ax[1].set_xlabel('arcsec')
    ax[1].set_ylabel('arcsec')
    plt.savefig('../plots/SNR_map_NGC613_with_minimal_3.pdf')
    plt.show()
    
    # Compute the percentage of spaxels removed clipping the S/N to 3
    #
    # Applying the clip in the average S/N map compute along the whole spectral
    # axis 
    frac_full = np.isfinite(t.full_axis()).sum() / np.isfinite(gp_0).sum()
    frac_full = (frac_full - 1) * 100
    print(f'Only {frac_full: .2f}% of spaxels are removed in the full spectral axis')
    
    # Compute the fraction with the S/N computed between 5450 and 5550 angstrons 
    frac_slice = np.isfinite(t.range_axis([5450, 5550])).sum() / np.isfinite(gp_1).sum()
    frac_slice = (frac_slice - 1) * 100
    print(f'Only {frac_slice: .2f}% of spaxels are removed with S/N computed between 5450 and 5550 A')    
