#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:55:58 2021

@author: Luiz
"""

from data_preprocessing import DataPreprocessing
from ppxf_execution import ExecutePpxf
from post_processing import PostProcessing

if __name__ == '__main__':
    model_path = '../../data/models/tmpWzZ2t1/Mku1.30Z*.fits'
    obs_path = '../../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

    data = DataPreprocessing(model_path = model_path, obs_path = obs_path)
    # ppxf_output = ExecutePpxf(data)
    # PostProcessing(ppxf_output, data)

# import numpy as np
# from astropy.io import fits
# import matplotlib.pyplot as plt

# hdu = fits.open('NGC613_1/velocity.fits')
# plt.imshow(hdu[0].data)
# plt.show()

# hdu = fits.open('NGC613_1/bestfit.fits')
# plt.plot(hdu[0].data[:, 10, 10])

# result = np.memmap(filename = 'velocity.dat', dtype = float, mode='r+',
#                     shape = (data.shape_obs[0] * data.shape_obs[1],))