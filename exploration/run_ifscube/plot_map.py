#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 18:21:48 2021

@author: Luiz
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes, make_rgb_axes

#%%
# hdu = fits.open('../../data_products/NGC613/miles/ifscube_1/input_cube_linefit.fits', memmap = True)

# print(hdu_res['status'].data)
# x = 5
# # plt.plot(hdu_res['restwave'].data, hdu_res['fitspec'].data[:,x,0], label = 'spectrum')
# # plt.plot(hdu_res['restwave'].data, hdu_res['stellar'].data[:,x,0], label = 'stellar')
# plt.plot(hdu_res['restwave'].data,hdu_res['fitspec'].data[:,x,0] - hdu_res['stellar'].data[:,x,0], label = 'emission')
# # plt.plot(hdu_res['restwave'].data, hdu_res['var'].data[:,x,0])
# # plt.plot(hdu_res['restwave'].data, hdu_res['model'].data[:,x,0], label = 'emission')
# plt.plot(hdu_res['restwave'].data, hdu_res['fitcont'].data[:,x,0], label = 'continuum')
# plt.plot(hdu_res['restwave'].data,
#          hdu_res['model'].data[:,x,0] - hdu_res['stellar'].data[:,x,0], label = 'emission fit')
# plt.legend()

# hdu_res['solution'].data[:,2,0]

#%%  plot the used fov

# path = '../../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

# hdu_im = fits.open(path)
# image = np.nansum(np.nansum(hdu_im[1].data, axis = 1), axis = 1)
# #%%

# hdu = fits.open('input_cube_ifscube_linefit.fits')
# plt.style.use('../fig_conf.mplstyle')

#%%

# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## hb_narrow
# flux_narrow = hdu['flux_m'].data[0]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title(r'H$_{\beta}$ (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## hb_broad
# flux_broad = hdu['flux_m'].data[9]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title(r'H$_{\beta}$ (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('hb_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ## Oiii narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title(r'H$_{\beta}$ (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('hb_sum.pdf')

#%%

# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## hb_narrow
# flux_narrow = hdu['flux_m'].data[5]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title(r'H$_{\alpha}$ (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## hb_broad
# flux_broad = hdu['flux_m'].data[14]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title(r'H$_{\alpha}$ (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('ha_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ## ha narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title(r'H$_{\alpha}$ (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('ha_sum.pdf')

#%%

# flux_ha = hdu['flux_m'].data[5] + hdu['flux_m'].data[14]
# flux_hb = hdu['flux_m'].data[0] + hdu['flux_m'].data[9]
# ratio = flux_ha/flux_hb

# ## ha/hb
# fig, ax = plt.subplots()
# im0 = ax.imshow(ratio, cmap = 'afmhot', origin = 'lower', vmin = 0, vmax = 100)
# cbar0 = plt.colorbar(im0, ax = ax)
# # cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title(r'H$_{\alpha}$ / H$_{\beta}$')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('ha_hb_sum.pdf')
#%%

#Oiii5007

# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## Oiii_narrow
# flux_narrow = hdu['flux_m'].data[2]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title('[OIII]$\lambda$5007 (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## Oiii_broad
# flux_broad = hdu['flux_m'].data[11]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title('[OIII]$\lambda$5007 (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('oiii_5007_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ## Oiii narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title('[OIII]$\lambda$5007 (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('oiii_5007_sum.pdf')

# #%%

# #Oiii4958

# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## Oiii_narrow
# flux_narrow = hdu['flux_m'].data[1]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title('[OIII]$\lambda$4958 (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## Oiii_broad
# flux_broad = hdu['flux_m'].data[10]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title('[OIII]$\lambda$4958 (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('oiii_4958_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ## Oiii narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title('[OIII]$\lambda$4958 (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('oiii_4958_sum.pdf')

# #%%

# ## Nii 6548

# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## Oi6300_narrow
# flux_narrow = hdu['flux_m'].data[4]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title('[SiI]$\lambda$6548 (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## Oi6300_broad
# flux_broad = hdu['flux_m'].data[13]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title('[NII]$\lambda$6548 (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('nii_6548_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ## Oi6300 narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title('[NII]$\lambda$6584 (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('nii_6548_sum.pdf')

# #%%

# ## Nii 6584

# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## Oi6300_narrow
# flux_narrow = hdu['flux_m'].data[6]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title('[NII]$\lambda$6584 (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## Oi6300_broad
# flux_broad = hdu['flux_m'].data[15]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title('[NII]$\lambda$6584 (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('nii_6584_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ## Oi6300 narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title('[NII]$\lambda$6584 (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('nii_6584_sum.pdf')

# #%%

# ## Sii 6717

# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## narrow
# flux_narrow = hdu['flux_m'].data[7]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title('[SII]$\lambda$6717 (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## broad
# flux_broad = hdu['flux_m'].data[16]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title('[SII]$\lambda$6717 (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('sii_6717_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ##  narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title('[SII]$\lambda$6717 (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('sii_6717_sum.pdf')

# #%%

# ## Oi 6300


# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## narrow
# flux_narrow = hdu['flux_m'].data[3]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title('[OI]$\lambda$6300 (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## Oi6300_broad
# flux_broad = hdu['flux_m'].data[12]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title('[OI]$\lambda$6300 (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('oi_6300_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ##  narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title('[OI]$\lambda$6300 (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('oi_6300_sum.pdf')

# #%%

# ## Sii 6730


# fig, ax = plt.subplots(ncols=2, figsize = (10, 4.0))

# ## narrow
# flux_narrow = hdu['flux_m'].data[8]
# im0 = ax[0].imshow(flux_narrow/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax[0])
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[0].set_title('[SII]$\lambda$6730 (Narrow)')
# ax[0].set_xlabel('Pixel')
# ax[0].set_ylabel('Pixel')

# ## Oi6300_broad
# flux_broad = hdu['flux_m'].data[17]
# im1 = ax[1].imshow(flux_broad/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar1 = plt.colorbar(im1, ax = ax[1])
# cbar1.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax[1].set_title('[SII]$\lambda$6730 (Broad)')
# ax[1].set_xlabel('Pixel')
# ax[1].set_ylabel('Pixel')

# plt.savefig('sii_6730_broad_narrow.pdf')


# fig, ax = plt.subplots()

# ##  narrow + broad
# im0 = ax.imshow((flux_narrow + flux_broad)/1e-9, cmap = 'afmhot', origin = 'lower')
# cbar0 = plt.colorbar(im0, ax = ax)
# cbar0.set_label(r'$ \text{Flux} \, ( 10^{-29} \erg / \s / \cm^{2})$')

# ax.set_title('[SII]$\lambda$6730 (Narrow + Broad)')
# ax.set_xlabel('Pixel')
# ax.set_ylabel('Pixel')

# plt.savefig('sii_6730_sum.pdf')

# #%%

# # plt.imshow(hdu['solution'].data[2], cmap = 'afmhot', origin = 'lower')
# hdu = fits.open('../../data_products/NGC613/miles/ifscube_3/input_cube_linefit.fits', memmap = True)



# plt.plot(hdu['restwave'].data, hdu['fitspec'].data[:, 155, 130] - 
#          hdu['stellar'].data[:, 155, 130])

# plt.plot(hdu['restwave'].data, hdu['model'].data[:, 155, 130] - 
#          hdu['stellar'].data[:, 155, 130])

# #%% Hb model map

# flux_hb_model = np.nansum(hdu['model'].data[123:133, ...], axis = 0)
# plt.imshow(np.log10(flux_hb_model), origin = 'lower', vmin = 2.6, vmax = 4.6)

# #%% Hb velocity map

# hb_velocity = hdu['solution'].data[1]
# plt.imshow(hb_velocity, origin = 'lower', vmin = 1300, vmax = 1600)

# #%% Hb velocity map

# hb_sigma = hdu['solution'].data[1]
# plt.imshow(hb_sigma, origin = 'lower')

# #%% Hb map

# flux_hb = np.nansum(hdu['fitspec'].data[123:133, ...], axis = 0)
# plt.imshow(np.log10(flux_hb), origin = 'lower', vmin = 2.6, vmax = 4.6)

# #%% Ha map

# flux_ha = np.nansum(hdu['model'].data[1784:1800, ...], axis = 0)
# plt.imshow(np.log10(flux_ha), origin = 'lower', vmin = 2.8, vmax = 5.5)

#%%



hdul = fits.open('../data_products/NGC613/miles/ifscube_5/input_cube_5_linefit.fits')

#%%
chi = hdul['red_chi'].data
plt.imshow(chi,
           vmin = 0.3,
           vmax = 2.5,
           origin = 'lower')

#%%
param = hdul['solution'].data[38]
fig, ax = plt.subplots(1,1)
ax.imshow(param,
          vmin = 250,
          vmax = 400,
          cmap = 'Spectral',
          origin = 'lower')

param = hdul['solution'].data[37]
fig, ax = plt.subplots(1,1)
ax.imshow(param,
          vmin = 1480-200,
          vmax = 1480+200,
          cmap = 'Spectral',
          origin = 'lower')

param = hdul['solution'].data[36]
fig, ax = plt.subplots(1,1)
ax.imshow(param,
          vmin = 10,
          vmax = 2e2,
          cmap = 'Reds',
          origin = 'lower')

param = hdul['solution'].data[2]
fig, ax = plt.subplots(1,1)
ax.imshow(param,
          vmax = 160,
          cmap = 'Spectral',
          origin = 'lower')

param = hdul['solution'].data[1]
fig, ax = plt.subplots(1,1)
ax.imshow(param - 1480,
          vmin = -200,
          vmax = +200,
          cmap = 'Spectral_r',
          origin = 'lower')

#%%
param = hdul['solution'].data[0]
fig, ax = plt.subplots(1,1)

plt.imshow(np.log10(param.clip(1e-3)),
          vmin = 1.2,
          vmax = 4.5,
          origin = 'lower')

#%%
param = hdul['solution'].data[36]
fig, ax = plt.subplots(1,1)

plt.imshow(np.log10(param.clip(1e-3)),
          vmin = 0.8,
          vmax = 3.3,
          origin = 'lower')

#%%
grad_param = np.gradient(param)
plt.imshow(grad_param[0],
           vmax = 50,
           vmin = -50,
           cmap = 'Spectral',
           origin = 'lower')
#%%
# flux_ha_narrow = flux_ha_narrow / np.nanmax(flux_ha_narrow)
# flux_ha_narrow = np.uint8(255*flux_ha_narrow)

# flux_ha_broad_blue = flux_ha_broad_blue / np.nanmax(flux_ha_broad_blue)
# flux_ha_broad_blue = np.uint8(255*flux_ha_broad_blue)

fig, ax = plt.subplots(1,1)
ax.imshow(flux_ha_broad_blue, origin = 'lower')


# a = np.dstack((flux_ha_narrow,
#                np.zeros_like(flux_ha_broad_blue),
#                flux_ha_broad_blue))
# plt.imshow(a, origin = 'lower')

# plt.plot(hdu['restwave'].data, hdu['fitcont'].data[:, 0, 0])
# plt.plot(hdu['restwave'].data, hdu['model'].data[:, 0, 0])
# plt.plot(hdu['restwave'].data, hdu['fitspec'].data[:, 0, 0])
# plt.plot(hdu['restwave'].data, hdu['stellar'].data[:, 0, 0])

#%%


import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

hdul = fits.open('../../data_products/NGC613/miles/ifscube_6/input_cube_6_linefit.fits')

# data = hdul['solution'].data

# plt.imshow(data[1], origin = 'lower')


param = hdul['solution'].data[38]
fig, ax = plt.subplots(1,1)
ax.imshow(param,
          vmin = 250,
          vmax = 400,
          cmap = 'Spectral',
          origin = 'lower')

param = hdul['solution'].data[37]
fig, ax = plt.subplots(1,1)
ax.imshow(param,
          vmin = 1480-200,
          vmax = 1480+200,
          cmap = 'Spectral_r',
          origin = 'lower')

param = hdul['solution'].data[36]
fig, ax = plt.subplots(1,1)
ax.imshow(param,
          vmin = 1,
          vmax = 1.5e3,
          # cmap = 'Reds',
          origin = 'lower')

param = hdul['solution'].data[2]
fig, ax = plt.subplots(1,1)
ax.imshow(param,
          vmax = 160,
          cmap = 'Spectral',
          origin = 'lower')

param = hdul['solution'].data[1]
fig, ax = plt.subplots(1,1)
ax.imshow(param - 1480,
          vmin = -200,
          vmax = +200,
          cmap = 'Spectral_r',
          origin = 'lower')

#%%
param = hdul['solution'].data[42]
fig, ax = plt.subplots(1,1)
plt.imshow(np.log10(param.clip(1e-3)),
          vmin = 1.2,
          vmax = 4.5,
          origin = 'lower')

#%%

i = 0
hdul['parnames'].data[3*i]

j = 10
hdul['parnames'].data[3*j]


k = 20
hdul['parnames'].data[3*k]


param = hdul['solution'].data[3*i]
fig, ax = plt.subplots(1,1)
plt.imshow(np.log10(param),
            vmin= 1,
            vmax = 4.6,
          origin = 'lower')

param = hdul['solution'].data[3*i+1]
fig, ax = plt.subplots(1,1)
plt.imshow(param,
            # vmin= 0,
            # vmax = 4.6,
            origin = 'lower')

param1 = hdul['solution'].data[3*i+2]
fig, ax1 = plt.subplots(1,1)
ax1.imshow(param1,
           vmin = 60,
           vmax = 130,
           origin = 'lower')



param2 = hdul['solution'].data[3*j+0]
fig, ax2 = plt.subplots(1,1)
ax2.imshow(np.log10(param2),
            vmin = 1,
            vmax = 3,
            origin = 'lower')

param3 = hdul['solution'].data[3*k+0]
fig, ax2 = plt.subplots(1,1)
ax2.imshow(np.arcsinh(param3),
            vmin = 3,
            vmax = 7 ,
            origin = 'lower')



fig, ax = plt.subplots(1,1)
ax.hist(param1[50:250, 50:250].flatten(), bins = 500)
ax.hist(param2[50:250, 50:250].flatten(), bins = 500)

#%%
fig, ax = plt.subplots(1,1)
plt.plot(hdul['restwave'].data,
         hdul['model'].data[:, 183, 168] - hdul['stellar'].data[:, 183, 168])
plt.plot(hdul['restwave'].data,
         hdul['fitspec'].data[:, 183, 168] - hdul['stellar'].data[:, 183, 168])

from scipy import ndimage

img = ndimage.sobel(param)
#%%
plt.imshow(img,
           vmin = 30,
           origin = 'lower')
