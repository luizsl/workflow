#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:48:23 2022

@author: Luiz
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# cube_path = '../data/toy_100x100.fits'
cube_path = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits.gz'

with fits.open(cube_path) as hdul:
    image = np.nansum(hdul['data'].data, axis = 0)

levels = np.logspace(5.3, 7, 10)
alpha = 0.8
linewidths = 0.5

fig, ax = plt.subplots(1, 3)
ax[0].imshow(image)
ax[0].contour(image, colors = 'black', levels=levels, alpha=alpha, linewidths=linewidths)
ax[0].title.set_text('Original')

ds3 = image[::3, ::3]
ax[1].title.set_text('1:3 downsampling')
ax[1].imshow(ds3)
ax[1].contour(ds3, colors = 'red', levels=levels, alpha=alpha, linewidths=linewidths)

ds5 = image[::5, ::5]
ax[2].title.set_text('1:5 downsampling')
ax[2].imshow(ds5)
ax[2].contour(ds5, colors = 'blue', levels=levels, alpha=alpha, linewidths=linewidths)

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, ::3, ::3]
    hdul['stat'].data = hdul['stat'].data[:, ::3, ::3]
    hdul.writeto('../data/fov_sample_1_2.fits', overwrite = True)
    
with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, ::3, ::3]
    hdul['stat'].data = hdul['stat'].data[:, ::3, ::3]
    hdul.writeto('../data/fov_sample_1_3.fits', overwrite = True)

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, ::5, ::5]
    hdul['stat'].data = hdul['stat'].data[:, ::5, ::5]
    hdul.writeto('../data/fov_sample_1_5.fits', overwrite = True)
