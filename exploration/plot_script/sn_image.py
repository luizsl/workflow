#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:21:01 2021

@author: chess-lin
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm

cube_file = '../../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

# read data
with fits.open(cube_file) as hdu:
    wcs = WCS(hdu[1].header)
    n = np.double(hdu[1].header['NAXIS3'])
    flux_cube = np.nansum(hdu[1].data, axis = 0, dtype = np.double)
    err_cube = np.nansum(hdu[2].data, axis = 0, dtype = np.double)

# Create map
sn_per_pixel = (flux_cube/np.sqrt(err_cube))/(np.sqrt(n))

# Plot
plt.style.use('fig_conf.mplstyle')

norm = simple_norm(sn_per_pixel.clip(1,80), stretch = 'asinh')
ax = plt.subplot(projection=wcs[1])

im = ax.imshow(sn_per_pixel, origin = 'lower', cmap = cm.CMRmap, norm = norm)

cbar = plt.colorbar(im)
cbar.set_ticks([1] + np.linspace(10,80,8).tolist())
cbar.set_ticklabels([r'$<$ 1'] 
                    + np.linspace(10,70,7, dtype = int).tolist() 
                    + [r'$>$ 80'])

ax.grid(color = 'white', ls = 'dotted')
ax.set_xlabel(r'$\textbf{Right Ascension (J2000)}$')
ax.set_ylabel(r'$\textbf{Declination (J2000)}$')
ax.set_title(r'SNR')

plt.savefig('snr_ngc613.pdf')
