#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:37:58 2022

@author: Luiz
"""

from astropy.io import fits

cube_path = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits.gz'

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 100:103, 100:103]
    hdul['stat'].data = hdul['stat'].data[:, 100:103, 100:103]
    hdul.writeto('../data/toy_3x3.fits', overwrite = True)
    
with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 100:120, 100:120]
    hdul['stat'].data = hdul['stat'].data[:, 100:120, 100:120]
    hdul.writeto('../data/toy_20x20.fits', overwrite = True)

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 100:110, 100:110]
    hdul['stat'].data = hdul['stat'].data[:, 100:110, 100:110]
    hdul.writeto('../data/toy_10x10.fits', overwrite = True)

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 150:200, 150:200]
    hdul['stat'].data = hdul['stat'].data[:, 150:200, 150:200]
    hdul.writeto('../data/toy_50x50.fits', overwrite = True)

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 100:200, 100:200]
    hdul['stat'].data = hdul['stat'].data[:, 100:200, 100:200]
    hdul.writeto('../data/toy_100x100.fits', overwrite = True)
    
with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 75:225, 75:225]
    hdul['stat'].data = hdul['stat'].data[:, 75:225, 75:225]
    hdul.writeto('../data/toy_150x150.fits', overwrite = True)
    
with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 50:250, 50:250]
    hdul['stat'].data = hdul['stat'].data[:, 50:250, 50:250]
    hdul.writeto('../data/toy_200x200.fits', overwrite = True)

