#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:21:01 2021

@author: Luiz
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm

def build_map(cube_file, name_out):
    '''
    Produces SNR map from MUSE data.

    Parameters
    ----------
    cube_file : Muse fits
        MUSE cube with the data
    name_out : string
        name of the pdf output

    Returns
    -------
    Produce a .pdf file with the resulting figure
    '''
    with fits.open(cube_file) as hdu:
        wcs = WCS(hdu[1].header)
        n_pixel = np.double(hdu[1].header['NAXIS3'])
        flux_cube = np.nansum(hdu[1].data, axis = 0, dtype = np.double)
        err_cube = np.nansum(hdu[2].data, axis = 0, dtype = np.double)

    # Create map
    # There's a division by nan whose result is also nan.
    # This is not a problem, then I'm disabling the warnings.
    np.seterr(divide = 'ignore', invalid = 'ignore')
    sn_per_pixel = (flux_cube/np.sqrt(err_cube)) * (1./np.sqrt(n_pixel))

    # Plot

    norm = simple_norm(sn_per_pixel.clip(1,80), stretch = 'asinh')
    ax = plt.subplot(projection=wcs[1])

    im = ax.imshow(sn_per_pixel, origin = 'lower', cmap = cm.CMRmap, norm = norm)

    cbar = plt.colorbar(im)
    cbar.set_ticks([1] + np.linspace(10,80,8).tolist())
    cbar.set_ticklabels([r'$<$ 1']
                        + np.linspace(10,70,7, dtype = int).tolist()
                        + [r'$>$ 80'])
    cbar.minorticks_off()

    ax.grid(color = 'white', ls = 'dotted')
    ax.set_xlabel(r'$\textbf{Right Ascension (J2000)}$')
    ax.set_ylabel(r'$\textbf{Declination (J2000)}$')
    ax.set_title(r'SNR')

    plt.savefig(name_out)
