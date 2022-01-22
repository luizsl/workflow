#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:36:02 2021

@author: Luiz
"""

from os import path

import matplotlib.pyplot as plt
import numpy as np
import vorbin
from astropy.io import fits
from vorbin.voronoi_2d_binning import voronoi_2d_binning

#%%
file = '../../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits.gz'
hdu = fits.open(file)
a = hdu['DATA'].data
noise = hdu['STAT'].data
del(hdu)

average_flux = np.nanmean(a, axis = 0).ravel()
average_noise = np.nanmean(noise, axis = 0).ravel()
XX, YY = np.meshgrid(np.arange(a.shape[2]),
                     np.arange(a.shape[1]))
XX = XX.ravel()
YY = YY.ravel()
#%%



# x, y, signal, noise = np.loadtxt(file_dir + '/example/voronoi_2d_binning_example_input.txt').T
targetSN = 50.0

# Perform the actual computation. The vectors
# (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
# are all generated in *output*
#
# binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
#     voronoi_2d_binning(x, y, signal, noise, targetSN, plot=1, quiet=0)

binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
    voronoi_2d_binning(x = XX, y = YY, 
                       signal = average_flux, noise = average_noise, 
                       targetSN = targetSN, plot=1, quiet=0)
    
# Save to a text file the initial coordinates of each pixel together
# with the corresponding bin number computed by this procedure.
# binNum uniquely specifies the bins and for this reason it is the only
# number required for any subsequent calculation on the bins.
#
# np.savetxt('voronoi_2d_binning_example_output.txt', np.column_stack([x, y, binNum]),
#            fmt=b'%10.6f %10.6f %8i')