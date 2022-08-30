#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:41:34 2022

@author: Luiz
"""
import os
import json

import numpy as np
import spectcube as sc
import matplotlib.pyplot as plt
from astropy.io import fits

def check_inside(wave, fixed_mask:list):
    for bound in fixed_mask:
        if (wave > bound[0]) & (wave < bound[1]):
            return True
    return False

N_PIXEL_MUSE = 3680

# root_dir ='../data_products/NGC0613_DATACUBE_FINAL_clean/XSLAgeMh_2/ppxf'
root_dir ='../data_products/fov_sample_1_3/XSLAgeMh_3/ppxf'
metadata_file = 'metadata.json'
goodpixels_file = 'goodpixels.fits'
bestfit_file = 'bestfit.fits'
galaxy_file = 'galaxy.fits'

metadata_path = os.path.join(root_dir, metadata_file)
with open(metadata_path) as f:
    metadata = json.load(f)

goodpixels_path = os.path.join(root_dir, goodpixels_file)
with fits.open(goodpixels_path, memmap=True) as hdul:
    goodpixels = np.single(hdul[0].data)

bestfit_path = os.path.join(root_dir, bestfit_file)
with fits.open(bestfit_path, memmap=True) as hdul:
    bestfit = np.half(hdul[0].data)

galaxy_path = os.path.join(root_dir, galaxy_file)
with fits.open(galaxy_path, memmap=True) as hdul:
    galaxy = np.half(hdul[0].data)
        
wave = np.asarray(metadata['obs']['wave_obs'])
bound = sc.util._build_edges(wave, 'ln')

fixed_mask = metadata['conf']['observation']['fixed_spectral_mask']

n_fixed_mask = 0

for j, value in enumerate(wave):
    if check_inside(value, fixed_mask):
        n_fixed_mask += 1
        
n_goodpixels = np.nansum(np.isfinite(goodpixels), axis=0, dtype=float)
n_goodpixels[n_goodpixels == 0] = np.nan

n_full_mask = N_PIXEL_MUSE - n_goodpixels
n_dyna_mask = n_full_mask - n_fixed_mask

fraction_full_mask = n_full_mask / N_PIXEL_MUSE
fraction_dyna_mask = n_dyna_mask / N_PIXEL_MUSE
fraction_fixed_mask = n_fixed_mask / N_PIXEL_MUSE


#%% Plot fraction dynamic mask

plt.style.use('../src/fig_conf.mplstyle')

fig, axs = plt.subplots(1,2, figsize=(10,4), constrained_layout=True,
                        tight_layout=False)

im0 = axs[0].imshow(fraction_full_mask, origin='lower',
                    )
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(fraction_dyna_mask, origin='lower',
                   )
plt.colorbar(im1, ax=axs[1])

axs[0].set_title('fraction masked (fixed + dynamic)')
axs[1].set_title('fraction masked (dynamic)')
fig.suptitle(r'$\sigma$=2.5, width emission mask = 750 km/s')
# plt.savefig('../plots/mask/example_mask_fov_sigma2d5_750kms_xsl.pdf')

#%%
fig, axs = plt.subplots(1,1, figsize=(10,4))

plt.plot(wave, galaxy[:, 52,51])
plt.plot(wave, bestfit[:, 52,51])


# plt.plot(wave, galaxy[:, 154,154])
# plt.plot(wave, bestfit[:,154,154])

#
# fig, axs = plt.subplots(1,1)

# for x,y in np.ndindex(10,10):
#     print(x, y)
    
# shade regions
for j in range(wave.size):
    if check_inside(wave[j], fixed_mask):
        lw = bound[j]
        up = bound[j+1]
        axs.axvspan(lw, up, color='silver', alpha=1, lw=0)
    elif j not in goodpixels[:, 52,51]:
        lw = bound[j]
        up = bound[j+1]
        axs.axvspan(lw, up, color='darkseagreen', alpha=1, lw=0)
axs.set_yscale('log')
axs.set_xlabel(r'wavelength')
axs.set_ylabel(r'flux density [a. u.]')
fig.suptitle(r'$\sigma$=2.5, width emission mask = 750 km/s')
# plt.savefig('../plots/mask/example_mask_spectrum_sigma2d5_750kms_xsl.pdf')
