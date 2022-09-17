#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:14:07 2022

@author: chess-lin
"""
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage
import numpy as np
from skimage import restoration

file = '../../data_products/toy_trick/MilesAgeMh/ppxf_emission_line/sol.fits'

with fits.open(file) as hdul:
    data = hdul[0].data
    

#%% Velocity Component I

velocity_comp1_gaussian3_filter = ndimage.gaussian_filter(data[2], sigma=3)

fig, ax = plt.subplots(1,2)    
ax[0].imshow(data[2], origin='lower', cmap='Spectral')
ax[1].imshow(velocity_comp1_gaussian3_filter, origin='lower', cmap='Spectral')

# Dispersion Component I

dispersion_comp1_gaussian3_filter = ndimage.gaussian_filter(data[3], sigma=3)

fig, ax = plt.subplots(1,2)    
ax[0].imshow(data[3], origin='lower', cmap='Spectral')
ax[1].imshow(dispersion_comp1_gaussian3_filter, origin='lower', cmap='Spectral')

#%% Velocity Component II

velocity_comp2_gaussian3_filter = ndimage.gaussian_filter(data[4], sigma=3)

fig, ax = plt.subplots(1,2)    
ax[0].imshow(data[4], origin='lower', cmap='Spectral')
ax[1].imshow(velocity_comp2_gaussian3_filter, origin='lower', cmap='Spectral')

# Dispersion Component II

dispersion_comp2_gaussian3_filter = ndimage.gaussian_filter(data[5], sigma=3)

fig, ax = plt.subplots(1,2)    
ax[0].imshow(data[5], origin='lower', cmap='Spectral')
ax[1].imshow(dispersion_comp2_gaussian3_filter, origin='lower', cmap='Spectral')