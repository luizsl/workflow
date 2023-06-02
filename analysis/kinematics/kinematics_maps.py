#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:28:44 2023

@author: Luiz
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

kin_path = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
            'XSLAgeMh_3/ppxf_emission_line_binned100_2components/'
            'sol.fits'
)

flux_path = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
            'XSLAgeMh_3/ppxf_emission_line_binned100_2components/'
            'corrected_flux.fits'
)

rms_path = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
            'XSLAgeMh_3/ppxf_emission_line_binned100_2components/'
            'amplitude_rms.fits'
)
 
with fits.open(kin_path) as hdul:
    kin_map = hdul[0].data

with fits.open(flux_path) as hdul:
    flux_map = hdul[0].data
    
with fits.open(rms_path) as hdul:
    rms_map = hdul[0].data
    
extent = [x_full.min() - 0.1, x_full.max() - 0.1,
          y_full.min() - 0.1, y_full.max() - 0.1]

# cmap = 'Spectral_r'
# cmap = 'coolwarm'
# cmap = 'bwr'
cmap = 'seismic'
vlim = 200

# mask = flux_map[0] > 1e3
# kin_map[9][~mask] = np.nan

with plt.style.context(['science', 'nature']):
    fig, ax = plt.subplots()
    im = ax.imshow(
        # kin_map[4], origin='lower', extent=extent,
        # flux_map[0], origin='lower', extent=extent,
        rms_map[10], origin='lower', extent=extent,
        # cmap=cmap,
        # vmin=-140, vmax=100,
        # vmin=-vlim, vmax=vlim
        )
    plt.colorbar(im)
    