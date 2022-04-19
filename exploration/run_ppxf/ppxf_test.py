#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:18:35 2022

@author: chess-lin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:54:40 2021

@author: Luiz
"""
from datetime import datetime
from time import perf_counter as clock

import _pickle as pickle
import numpy as np
import ppxf.ppxf_util as util
from ppxf.ppxf import ppxf
from scipy.constants import physical_constants


def mask_wavelength(wave, intervals =[]):
    mask = np.full_like(wave, False, dtype = bool)
    
    for lower, upper in intervals:
        temp = np.ma.masked_inside(wave, lower, upper)
        mask = np.logical_or(mask, temp.mask)
    
    # Invert mask to include in the fit
    mask = ~mask
    return mask

flux = prep.obs.mmap_flux_obs[:,0:15]

flux = flux.sum(axis=1)
factor = np.nanmedian(flux)
galaxy = flux
#/factor

flux_unc = prep.obs.mmap_flux_obs_unc[:,0:15]**2
flux_unc = flux_unc.sum(axis=1)
noise= np.sqrt(flux_unc)
#/factor

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s


# frac = prep.meta.wave_obs[1]/prep.meta.wave_obs[0]    # Constant lambda fraction per pixel
frac = prep.obs.meta['wave_obs'][1]/prep.obs.meta['wave_obs'][0]
velscale = np.log(frac)*C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
# z = prep.meta.z # redshift estimate

# dv = C*np.log(prep.model.meta['wave_model'][0]/prep.obs.meta['wave_obs'][0])    # eq.(8) of Cappellari (2017)
# goodpixels = util.determine_goodpixels(np.log(prep.obs.meta['wave_obs']), prep.meta.limit_model, z =0, width=2500)

vel = C*np.log(1 + 0)   # q.(8) of Cappellari (2017)
start = [vel, 3*velscale] # (km/s), starting guess for [V, sigma]

# flux = prep.obs.mmap_flux_obs[:].sum(axis=1)
galaxy[-1] =galaxy[-2]
noise[-1] =noise[-2]

# lam_range_gal = [np.min(prep.obs.meta['wave_obs']), np.max(prep.obs.meta['wave_obs'])]
# gas_template, gas_names,gas_names = util.emission_lines(
#     ln_lam_temp = np.log(prep.model.meta['wave_model']), lam_range_gal=lam_range_gal,
#     FWHM_gal = 2.6,tie_balmer=True)

# shape = prep.model.mmap_flux_model.shape[0]
# star_template = prep.model.mmap_flux_model.reshape(shape, -1)

shape = miles.templates.shape[0]
star_template = miles.templates.reshape(shape, -1)

# template = np.column_stack([star_template, gas_template])
template = star_template
# n_stars = star_template.shape[1]
# n_gas = len(gas_names)
# component = np.array([0]*n_stars + [1]*n_gas)
# gas_component = np.array(component) > 0  # gas_component=True for gas templates

mask_list = [
    [4000, 4770],
    [4850, 4880],
    [4950, 4970], 
    [4990, 5025],
    [5190, 5210],
    [6250, 6380],
    [6530, 6600],
    [6700, 6750],
    [7560, 7610],
    [8710, 8725],
    [9000, 9500]]

mask = mask_wavelength(prep.obs.meta['wave_obs'], mask_list)
# start = [start, [0, 70]]
        

import matplotlib.pyplot as plt

pp = ppxf(
    template, 
    galaxy,
    noise,
    velscale, start, moments=4, degree=-1, mdegree=-1,
    clean=True,
    reg_dim=(24,6), regul=1/0.4,
    reddening=0, 
    # goodpixels=goodpixels,
    velscale_ratio=2,
    mask = mask,
    # component = component, gas_component=gas_component, gas_names=gas_names,gas_reddening=0,
    lam=prep.obs.meta['wave_obs'], 
    # lam_temp=prep.model.meta['wave_model'],
    lam_temp=miles.lam_temp,
    plot = False, quiet=False)

# print("Formal errors:")
# print("     dV    dsigma   dh3      dh4")
# print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

# print('Elapsed time in PPXF: %.2f s' % (clock() - t))


fig, ax = plt.subplots()
ax = pp.plot()
# plt.plot(prep.obs.meta['wave_obs']*0.1, noise)

fig, ax = plt.subplots()
light_weights = pp.weights
#[~gas_component]
light_weights = light_weights.reshape(24,6)
light_weights /= light_weights.sum()
# plt.imshow(light_weights)

miles.plot(light_weights)
