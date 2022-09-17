#!/usr/bin/env python


from os import path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning


# file_dir = '../../data/toy_trick.fits'

galaxy_dir = '../../data_products/fov_sample_1_3/XSLAgeMh/ppxf/galaxy.fits'
noise_dir = '../../data_products/fov_sample_1_3/XSLAgeMh/ppxf/noise.fits'
bestfit_dir = '../../data_products/fov_sample_1_3/XSLAgeMh/ppxf/bestfit.fits'

# with fits.open(file_dir) as hdul:
#     flux = hdul['data'].data
#     flux_noise =  hdul['stat'].data

with fits.open(galaxy_dir) as hdul:
    flux = hdul[0].data

with fits.open(noise_dir) as hdul:
    flux_noise = hdul[0].data

with fits.open(bestfit_dir) as hdul:
    stellar = hdul[0].data

wave = 4750 + 1.25*np.arange(3682)

signal_full = np.nanmean(flux, axis=0)
noise_full = np.nanmean(flux_noise**2, axis=0)
snr_full = signal_full/ np.sqrt(noise_full)

signal = np.nanmean(flux[100:115], axis=0)
noise = np.nanmean(flux_noise[100:115]**2, axis=0)
snr = signal/np.sqrt(noise)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(snr_full, origin='lower')
ax[1].imshow(snr, origin='lower', vmin=3, vmax=90)

fig, ax = plt.subplots(1, 1)
ax.imshow(snr/snr_full, origin='lower')

#%%
# flux = np.nansum(signal, axis=0)
jm = np.nanargmax(signal)
sig = signal.ravel()
noi= np.sqrt(noise.ravel())

valid = np.isfinite(sig)

row, col = map(np.ravel, np.indices(signal.shape))

pixsize = 0.2
x = (col - col[jm])*pixsize
y = (row - row[jm])*pixsize

target_sn = 10

x = x[valid]
y = y[valid]
sig = sig[valid]
noi = noi[valid]

binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
    x, y, sig, noi, target_sn, pixelsize=0.2, plot=1, quiet=0)


# if __name__ == '__main__':

#     voronoi_binning_example()
#     plt.tight_layout()