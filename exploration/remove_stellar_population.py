#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:29:36 2021

@author: Luiz
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

obs_path = 'run_ppxf/flux_obs.dat'
obs_unc_path = 'run_ppxf/flux_obs_unc.dat'

bestfit_path = 'run_ppxf/NGC613_1/bestfit.fits'

shape_obs = (2586, 104293)


obs_flux = np.memmap(filename = obs_path, dtype = 'float32', mode='r',
                shape = shape_obs)
obs_flux = obs_flux.reshape((-1,) + (317, 329))

obs_flux_unc = np.memmap(filename = obs_unc_path, dtype = 'float32', mode='r',
                shape = shape_obs)
obs_flux_unc = obs_flux.reshape((-1,) + (317, 329))

with fits.open(bestfit_path) as hdu:
    data_stellar = hdu[0].data

data_out = obs_flux - data_stellar

del obs_flux, data_stellar

wave_obs = wave
# wave_obs = data.wave_obs
wave_fix = wave_obs * ((1480/299792.458)+1)

stacked = np.nansum(data_out[:, :, :], axis = 1)
stacked = np.nansum(stacked, axis = 1)

plt.plot(wave_fix, obs_flux[:, 165, 140], label = 'single')
plt.plot(wave_fix, data_out[:, 165, 140], label = 'single')

plt.plot(wave_fix, stacked, label = 'stacked')
plt.legend()


#%%


# plt.plot(wave_fix, stacked, label = 'stacked')
# plt.plot(stacked, label = 'stacked')

# plt.legend()

#%%
image_oiii = np.nansum(data_out[281:285, ...], axis = 0)

im = plt.imshow(np.log10(image_oiii.clip(0.001)), vmin = -0.5, vmax = 0.4,
                cmap = 'Blues')
plt.colorbar(im)

#%%

plt.plot(wave_fix, np.nansum(np.nansum(obs_flux[:, :, :],axis = 1), axis = 1), label = 'single')

plt.plot(wave_fix, obs_flux[:, 170, 140], label = 'single')
plt.plot(wave_fix, obs_flux[:, 175, 140], label = 'single')
plt.plot(wave_fix, obs_flux[:, 180, 140], label = 'single')
# plt.plot(wave, obs_flux_unc[:, 165, 140], label = 'single')

