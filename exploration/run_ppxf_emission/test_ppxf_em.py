#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:50:42 2022

@author: Luiz
"""

import matplotlib.pyplot as plt
import numpy as np
import ppxf.ppxf_util as util
import spectcube as sc
from astropy.io import fits
import json
from ppxf.ppxf import ppxf

C = 299792.458  # speed of light in km/s

data = ppxf_prep.data
#%%
# Observation
galaxy_filepath = '../../data_products/toy_100x100/MilesAgeMh/ppxf/galaxy.fits'
with fits.open(galaxy_filepath) as hdul:
    galaxy_full = hdul[0].data

noise_filepath = '../../data_products/toy_100x100/MilesAgeMh/ppxf/noise.fits'
with fits.open(noise_filepath) as hdul:
    noise_full = hdul[0].data

kinematics_filepath = '../../data_products/toy_100x100/MilesAgeMh/ppxf/sol.fits'
with fits.open(kinematics_filepath) as hdul:
    kinemtics_full = hdul[0].data

# wavelength
metadata_filepath = '../../data_products/toy_100x100/MilesAgeMh/ppxf/metadata.json'
with open(metadata_filepath) as f:
    metadata = json.load(f)
wave = np.array(metadata['obs']['wave_obs'])
    
# Stellar template
stellar_filepath = '../../data_products/toy_100x100/MilesAgeMh/ppxf/bestfit.fits'
with fits.open(stellar_filepath) as hdul:
    stellar_full = hdul[0].data

# Emission-lines template
lam_range_gal = [np.min(wave), np.max(wave)]
gas_templates, gas_names, line_wave = util.emission_lines(
    np.log(wave), lam_range_gal, 2.4)
#%% ppxf
data = ppxf_prep.data

i = 80

galaxy = data.obs.flux_grid[:, i]
noise = data.obs.flux_grid_unc[:, i]
star_continuum_template = data.stellar.flux_grid[:, i]
stellar_kinematics = data.stellar.stellar_kinematics

gas_templates = data.em_model.gas_templates
gas_names = data.em_model.gas_names
line_wave = data.em_model.line_wave

ngas_comp = 2
gas_templates = np.tile(gas_templates, ngas_comp)
gas_names = np.asarray([a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
line_wave = np.tile(line_wave, ngas_comp)

lam = data.obs.meta['wave_obs']

component = np.array([0] + [1]*7 + [2]*7)
gas_component = np.array(component) > 0
moments = [-2, 2, 2]

stars_gas_templates = np.column_stack([star_continuum_template, gas_templates])

velscale = C*np.diff(np.log(lam[-2:]))  # Smallest velocity step

start = [[0, 0],
         [stellar_kinematics[0, i], stellar_kinematics[1, i]],
         [stellar_kinematics[0, i], stellar_kinematics[1, i]]]

vlim = lambda x: stellar_kinematics[0, i] + x*np.array([-100, 100])
bounds = [[vlim(2), [20, 300]],       # Bounds are ignored for the stellar component=0 which has fixed kinematic
          [vlim(2), [20, 100]],       # I force the narrow component=1 to lie +/-200 km/s from the stellar velocity
          [vlim(6), [20, 400]]]      # I force the broad component=3 to lie +/-200 km/s from the stellar velocity

pp = ppxf(stars_gas_templates, galaxy, noise, velscale, start,
          plot=1, moments=moments, degree=2, mdegree=-1, component=component, 
          gas_component=gas_component, gas_names=gas_names,
          lam=lam, vsyst=0, global_search=True)