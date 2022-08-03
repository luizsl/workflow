#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:18:35 2022

@author: Luiz
"""

from datetime import datetime
from time import perf_counter as clock

import extinction
import _pickle as pickle
import numpy as np
from ppxf.ppxf import ppxf, robust_sigma
from scipy.constants import physical_constants

prep = data

def clip_outliers(galaxy, bestfit, goodpixels, fixed_goodpixels=None, sigma=3):
    """
    Adapted from Michele Cappellari's example

    Repeat the fit after clipping bins deviants more than 3*sigma
    in relative error until the bad bins don't change any more.
    """
    if fixed_goodpixels is None:
        fixed_goodpixels = np.arange(galaxy.size)
    while True:
        goodpixels = np.intersect1d(goodpixels, fixed_goodpixels)
        scale = galaxy[goodpixels] @ bestfit[goodpixels]/np.sum(bestfit[goodpixels]**2)
        resid = scale*bestfit[goodpixels] - galaxy[goodpixels]
        err = robust_sigma(resid, zero=1)
        ok_old = goodpixels.copy()
        goodpixels = np.flatnonzero(np.abs(bestfit - galaxy) < sigma*err)
        goodpixels = np.intersect1d(goodpixels, fixed_goodpixels)
        if np.array_equal(goodpixels, ok_old):
            break
    return goodpixels

def mask_wavelength(wave, intervals =[]):
    mask = np.full_like(wave, False, dtype = bool)

    for lower, upper in intervals:
        temp = np.ma.masked_inside(wave, lower, upper)
        mask = np.logical_or(mask, temp.mask)

    # Invert mask to include in the fit
    mask = ~mask

    goodpixels = np.arange(mask.size)[mask]
    return goodpixels

def dered(spectrum, wave=None, law='calzetti00', r_v=4.05,
          ebv=None):
    assert ebv is not None
    assert wave is not None
    a_v = ebv * r_v

    if law == 'fm07':
        ext_mag = extinction.__getattribute__(law)(wave=wave, a_v=a_v)
    else:
        ext_mag = extinction.__getattribute__(law)(wave=wave, a_v=a_v, r_v=r_v)

    dered_spectrum = extinction.remove(ext_mag, spectrum)

    return dered_spectrum

galaxy = prep.obs.flux_grid[:, 1]
noise = prep.obs.flux_grid_unc[:, 1]

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s


# frac = prep.meta.wave_obs[1]/prep.meta.wave_obs[0]    # Constant lambda fraction per pixel
frac = prep.obs.meta['wave_obs'][1]/prep.obs.meta['wave_obs'][0]
velscale = np.log(frac)*C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
# z = prep.meta.z # redshift estimate


vel = C*np.log(1 + 0)   # q.(8) of Cappellari (2017)
start = [vel, 3*velscale] # (km/s), starting guess for [V, sigma]



def fixed_mask_wavelength(wave, intervals =[]):
    mask = np.full_like(wave, False, dtype = bool)

    for lower, upper in intervals:
        temp = np.ma.masked_inside(wave, lower, upper)
        mask = np.logical_or(mask, temp.mask)

    # Invert mask to include in the fit
    mask = ~mask

    goodpixels = np.arange(mask.size)[mask]
    return goodpixels

# mask_list = [
#     [4000, 4770],
#     [4850, 4880],
#     [4950, 4970],
#     [4990, 5025],
#     [5524, 5574],
#     [5190, 5210],
#     [5831, 5975],
#     [6250, 6380],
#     [6530, 6600],
#     [6700, 6750],
#     [7560, 7610],
#     [8710, 8725],
#     [9000, 9500]]

# fixed_mask_list = [
#     [4000, 4770],
#     [9200, 9500]]

# goodpixels = mask_wavelength(prep.obs.meta['wave_obs'], mask_list)
# fixed_goodpixels = fixed_mask_wavelength(prep.obs.meta['wave_obs'], fixed_mask_list)

goodpixels = prep.obs.meta['guess_goodpixels']
fixed_goodpixels = prep.obs.meta['fixed_goodpixels']

#     TEMPLATES MINE

# reg_dim = prep.model.flux_grid.shape[1:]

shape = prep.model.flux_grid.shape[0]
star_template = prep.model.flux_grid.reshape(shape, -1)

# template = np.column_stack([star_template, gas_template])
template = star_template
# n_stars = star_template.shape[1]
# n_gas = len(gas_names)
# component = np.array([0]*n_stars + [1]*n_gas)
# gas_component = np.array(component) > 0  # gas_component=True for gas templates


lam_temp=prep.model.meta['wave_model']


#

degree = 10

pp = ppxf(
    template,
    galaxy,
    noise,
    velscale, start, moments=4, degree=-1, mdegree=degree,
    clean=False,
    # reg_dim=reg_dim, regul=1/0.01,
    # reddening=0,
    goodpixels=np.intersect1d(fixed_goodpixels, goodpixels),
    # goodpixels=goodpixels,
    velscale_ratio=2,
    # mask = mask,
    # component = component, gas_component=gas_component, gas_names=gas_names,gas_reddening=0,
    lam=prep.obs.meta['wave_obs'],
    lam_temp=lam_temp,
    # lam_temp=miles.lam_temp,
    plot = False, quiet=False)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax = pp.plot()

good = clip_outliers(pp.galaxy, pp.bestfit, pp.goodpixels, fixed_goodpixels, sigma=2.5)

pp = ppxf(
    template,
    galaxy,
    noise,
    velscale, start, moments=4, degree=-1, mdegree=degree,
    clean=False,
    # reg_dim=reg_dim, regul=1/0.01,
    # reddening=0,
    # goodpixels=np.intersect1d(fixed_goodpixels, good),
    goodpixels=good,
    velscale_ratio=2,
    # mask = mask,
    # component = component, gas_component=gas_component, gas_names=gas_names,gas_reddening=0,
    lam=prep.obs.meta['wave_obs'],
    lam_temp=lam_temp,
    # lam_temp=miles.lam_temp,
    plot = False, quiet=False)
# print("Formal errors:")
# print("     dV    dsigma   dh3      dh4")
# print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

# print('Elapsed time in PPXF: %.2f s' % (clock() - t))


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax = pp.plot()


#%%
fig, ax = plt.subplots()
light_weights = pp.weights
#[~gas_component]
light_weights = light_weights.reshape(reg_dim)
light_weights /= light_weights.sum()
plt.imshow(light_weights)

    # prep.model.plot(light_weights)

    # plt.plot(prep.obs.flux_grid[:, 0])
    # plt.plot(prep.obs.flux_grid_unc[:, 0])

#%%

wave = prep.obs.meta['wave_obs']
ebv=0.101
r_v=4.05
a_v = ebv * r_v

# calzetti00_ext = extinction.calzetti00(wave, a_v, r_v)
# galaxy_dered = extinction.remove(calzetti00_ext, galaxy)
# noise_dered = extinction.remove(calzetti00_ext, noise)
plt.plot(wave, galaxy, label='observation')
# plt.plot(wave, galaxy_dered)
# plt.plot(wave, noise)
# plt.plot(wave, noise_dered)



galaxy_dered_fc_call = dered(galaxy, wave=wave, ebv=ebv)
# noise_dered_fc_call = dered(noise, wave=wave, ebv=ebv)

# plt.plot(wave, galaxy)
plt.plot(wave, galaxy_dered_fc_call, label='dered (calzetti+00)')
# plt.plot(wave, noise)
# plt.plot(wave, noise_dered_fc)
plt.legend()