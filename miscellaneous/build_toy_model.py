#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:37:58 2022

@author: Luiz
"""

from astropy.io import fits

cube_path = '../data/NGC1068/Muse/NGC1068_DATACUBE_FINAL_clean.fits'
tag = 1068

# with fits.open(cube_path) as hdul:
#     hdul['data'].data = hdul['data'].data[:, 122:234, 125:214]
#     hdul['stat'].data = hdul['stat'].data[:, 122:234, 125:214]
#     hdul.writeto(f'../data/toy_trick_{tag}.fits', overwrite = True)
    
# with fits.open(cube_path) as hdul:
#     hdul['data'].data = hdul['data'].data[:, 100:103, 100:103]
#     hdul['stat'].data = hdul['stat'].data[:, 100:103, 100:103]
#     hdul.writeto(f'../data/toy_3x3_{tag}.fits', overwrite = True)
    
with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 100:120, 100:120]
    hdul['stat'].data = hdul['stat'].data[:, 100:120, 100:120]
    hdul.writeto(f'../data/toy_20x20_{tag}.fits', overwrite = True)

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 100:110, 100:110]
    hdul['stat'].data = hdul['stat'].data[:, 100:110, 100:110]
    hdul.writeto(f'../data/toy_10x10_{tag}.fits', overwrite = True)

with fits.open(cube_path) as hdul:
    hdul['data'].data = hdul['data'].data[:, 150:200, 150:200]
    hdul['stat'].data = hdul['stat'].data[:, 150:200, 150:200]
    hdul.writeto(f'../data/toy_50x50_{tag}.fits', overwrite = True)

# with fits.open(cube_path) as hdul:
#     hdul['data'].data = hdul['data'].data[:, 100:200, 100:200]
#     hdul['stat'].data = hdul['stat'].data[:, 100:200, 100:200]
#     hdul.writeto(f'../data/toy_100x100_{tag}.fits', overwrite = True)
    
# with fits.open(cube_path) as hdul:
#     hdul['data'].data = hdul['data'].data[:, 75:225, 75:225]
#     hdul['stat'].data = hdul['stat'].data[:, 75:225, 75:225]
#     hdul.writeto(f'../data/toy_150x150_{tag}.fits', overwrite = True)
    
# with fits.open(cube_path) as hdul:
#     hdul['data'].data = hdul['data'].data[:, 50:250, 50:250]
#     hdul['stat'].data = hdul['stat'].data[:, 50:250, 50:250]
#     hdul.writeto(f'../data/toy_200x200_{tag}.fits', overwrite = True)

