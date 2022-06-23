#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:43:48 2022

@author: Luiz

Collapse all the spectra into a single spectrum
"""

import numpy as np
from astropy.io import fits

cube_path = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits.gz'

with fits.open(cube_path, memmap=True) as hdul:
    hdul['data'].data = np.nansum(hdul['data'].data, axis=(1,2))
    hdul['stat'].data = np.nansum(hdul['stat'].data, axis=(1,2))
    hdul.writeto('../data/NGC0613_full_stacked_spectrum.fits', overwrite = True)
