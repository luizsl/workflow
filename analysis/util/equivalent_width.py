#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:04:13 2023

@author: Luiz
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


obs_path = (
    '../../data/NGC613/Muse/'
    'NGC0613_DATACUBE_FINAL_clean.fits'
    )

stellar_path = (
    '../../data_products/NGC0613_DATACUBE_FINAL_clean/'
    'XSLAgeMh_3/ppxf/'
    'bestfit.fits'
    )

emission_path = (
    '../../data_products/NGC0613_DATACUBE_FINAL_clean/'
    'XSLAgeMh_3/ppxf_emission_line_1components/'
    'corrected_flux.fits'
    )

# with open(stellar_meta_path) as f:
#     stellar_meta = json.load(f)
#     wave_stellar = np.asarray(stellar_meta['obs']['wave_obs'])
#     stellar_norm =np.asarray(stellar_meta['obs']['obs_norm_factor'])
                             
with fits.open(stellar_path, memmap=True) as hdul:
    stellar_bestfit = hdul[0].data * 1e-20
    
with fits.open(obs_path, memmap=True) as hdul:
    obs = hdul[1].data * 1e-20
    bunit = hdul[1].header['BUNIT']
    step = hdul[1].header['CD3_3']
    first = hdul[1].header['CRVAL3']
    npix = hdul[1].header['NAXIS3']
    wave = first + np.arange(npix)*step
    wave_obs = wave / (1 + 0.004951)
    
with fits.open(emission_path) as hdul:
    flux = hdul[0].data

ha = flux[9]
cont_spectrum = obs

#%%

win_red = (6600, 6650)
win_blue = (6480, 6500)
lamb_c = 6559

band_red = (wave > min(win_red)) & (wave < max(win_red))
band_blue = (wave > min(win_blue)) & (wave < max(win_blue))

cont_blue = cont_spectrum[band_blue].mean(axis=0)
cont_red = cont_spectrum[band_red].mean(axis=0)

lamb_blue = wave[band_blue].mean()
lamb_red = wave[band_red].mean()

cont = (cont_red-cont_blue) * (lamb_c-lamb_blue)/(lamb_red-lamb_blue) + cont_blue

fig, ax = plt.subplots(1,3, tight_layout=True, figsize=(9.5, 3.6))

im0 = ax[0].imshow(cont, origin='lower')
ax[0].title.set_text(
    'Pseudo-continuum'
    # fr'$\lambda\lambda$ {min(win_red)}, {max(win_blue)} $\AA$'
)
fig.colorbar(im0, ax=ax[0], shrink=0.6)

im1 = ax[1].imshow(ha * 1e-20, origin='lower')
ax[1].title.set_text(r'H$\alpha$ flux')
fig.colorbar(im1, ax=ax[1], shrink=0.6)

im2 = ax[2].imshow(ha*1e-20 / cont, origin='lower', vmax=70)
ax[2].title.set_text(r'H$\alpha$ EW')
fig.colorbar(im2, ax=ax[2], shrink=0.6)

# TODO: Remove plot and save in proper directory
plt.savefig('cont_lum_eqw_ha.pdf')
