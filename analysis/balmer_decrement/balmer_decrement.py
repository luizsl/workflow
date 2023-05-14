#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:30:36 2022

@author: Luiz

Compute extinction based on Balmer decrement"""

import json
# import re

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.io import fits

file_metadata = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
                  'XSLAgeMh_3/ppxf_emission_line_binned100_1components/'
                  'metadata.json'
)

flux_path = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
             'XSLAgeMh_3/ppxf_emission_line_binned100_1components/'
             'corrected_flux.fits'
)

# file_metadata = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
#                  'XSLAgeMh_3/ppxf_emission_line_1components/'
#                  'metadata.json'
#                  )

# flux_path = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
#                'XSLAgeMh_3/ppxf_emission_line_1components/'
#                'corrected_flux.fits'
#                )
       
with fits.open(flux_path) as hdul:
    corrected_flux = hdul[0].data

with open(file_metadata) as fp:
    out_metadata = json.load(fp)

# gas_names = np.asarray(out_ppxf['results']['gas_names'])

xbin = np.asarray(out_metadata['obs']['xbin'])
ybin = np.asarray(out_metadata['obs']['ybin'])
x_full = np.asarray(out_metadata['obs']['x_full'])
y_full = np.asarray(out_metadata['obs']['y_full'])
grid = np.column_stack((x_full, y_full))

ha_1 = corrected_flux[9]
hb_1 = corrected_flux[8]
ha_hb_1 = ha_1/hb_1

# ha_2 = corrected_flux[19]
# hb_2 = corrected_flux[18]
# ha_hb_2 = ha_2/hb_2
with plt.style.context(['science', 'nature']):
    extent = [x_full.min() - 0.1, x_full.max() - 0.1,
              y_full.min() - 0.1, y_full.max() - 0.1]
    
    # Scatter plot
    fig, ax = plt.subplots(1,2, tight_layout=True, figsize=(10, 4))
    # ax[0].scatter(xbin, ybin, c=ha_hb_1, alpha=0.8, s=1)
    im0 = ax[0].imshow(ha_1/hb_1, origin='lower', extent=extent)
    ax[0].title.set_text(r'Balmer decrement (H$\alpha$ / H$\beta$)')
    plt.colorbar(im0)
    # ax[1].scatter(xbin, ybin, c=ha_hb_2, alpha=0.8, s=1)
    
    #  Plot of interpolation
    
    # inter = interpolate.LinearNDInterpolator
    # inter = interpolate.RBFInterpolator
    
    
    # _interp_ha_hb_1 = inter(np.column_stack([ybin, xbin]), ha_hb_1)
    # inter_ha_hb_1 = _interp_ha_hb_1(grid)
    
    # _interp_ha_hb_2 = inter(np.column_stack([ybin, xbin]), ha_hb_2)
    # inter_ha_hb_2 = _interp_ha_hb_2(grid)
    
    # Scatter plot
    # fig, ax = plt.subplots()
    ebv = 1.97 * np.log10( (ha_1/hb_1) / 2.87 )
    av = ebv * 3.1
    im1 = ax[1].imshow(av, origin='lower', extent=extent)
    # im1 = ax[1].imshow(ebv, origin='lower', extent=extent, vmax=0.6)
    ax[1].title.set_text(r'A$_V$ (R$_V$ = 3.1)')
    plt.colorbar(im1)

    # TODO: Remove plot and save in proper directory
    plt.savefig('balmer_decrement_attenuation.pdf')
    
#%%
# import matplotlib.colors as colors

# shape = (len(set(y_full.round(5))),
#          len(set(x_full.round(5))))

# extent = [x_full.min() - 0.1, x_full.max() - 0.1,
#           y_full.min() - 0.1, y_full.max() - 0.1]
# vmin = 2.5
# vmax = 10
# alpha = 0.7
# levels = np.arange(vmin, vmax, .5)
# norm = colors.AsinhNorm(vmin=vmin, vmax=vmax, clip=True)

# fig, ax = plt.subplots(1, 2)

# ax[0].imshow(inter_ha_hb_1.reshape(shape), origin='lower', extent=extent,
#              norm=norm)
# ax[0].contour(
#     inter_ha_hb_1.reshape(shape), origin='lower', extent=extent,
#     norm=norm, alpha=alpha, levels=levels)
# # ax[0].scatter(xbin, ybin, c='k', alpha=0.8, s=1)

# ax[1].imshow(inter_ha_hb_2.reshape(shape), origin='lower', extent=extent,
#              norm=norm)
# ax[1].contour(
#     inter_ha_hb_2.reshape(shape), origin='lower', extent=extent,
#     norm=norm, alpha=alpha, levels=levels)
# # ax[1].scatter(xbin, ybin, c='k', alpha=0.8, s=1)
#%%
# finite = (ha_hb_1 < 10) & (ha_hb_1 > 2)
# plt.contour(xbin[finite], ybin[finite], ha_hb_1[finite], levels=100)

# #%%
# # Interpolation with selection
# fig, ax = plt.subplots(1, 2)
# ax[0].scatter(xbin, ybin, c=ha_hb_1)
# ax[1].scatter(xbin, ybin, c=ha_hb_2)