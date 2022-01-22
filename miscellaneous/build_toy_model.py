#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:37:58 2022

@author: Luiz
"""

from astropy.io import fits

cube_path = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits.gz'

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 100:120, 100:120]
    hdul['stat'].data = hdul['stat'].data[:, 100:120, 100:120]
    hdul.writeto('../data/toy_20x20.fits')

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 100:110, 100:110]
    hdul['stat'].data = hdul['stat'].data[:, 100:110, 100:110]
    hdul.writeto('../data/toy_10x10.fits')
