#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:04:39 2022

@author: Luiz
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

def read_fits_data(file, unit=0):
    with fits.open(file) as hdul:
        data = hdul[unit].data
        return np.asarray(data)
    
def find_regul(regul, 
               root_directory='../data_products/regularization_ngc613',
               filename_pattern=None): 
    name = filename_pattern.replace('[regul]', str(regul))
    for path in Path(root_directory).rglob(name):
        return path.parent.as_posix()

if __name__ == '__main__':
    reguls=[0,20,30,40,50,60,100]
    filename_pattern = 'sn100_regul[regul]_fov1x5.yaml'
    
    # Read the processed galaxy observation    
    galaxy_root_dir = find_regul(0, filename_pattern=filename_pattern)
    galaxy_filename = os.path.join(galaxy_root_dir, 'galaxy.fits')
    galaxy = read_fits_data(galaxy_filename)
    
    # Read bestfit of all tests
    for regul in reguls:
        root_dir = find_regul(regul, filename_pattern=filename_pattern)
        filename = os.path.join(root_dir, 'bestfit.fits')
        locals()[f'bestfit_regul{regul}'] = read_fits_data(filename)
    
    # # Compute mse (mean squared error)
    # for regul in reguls:
    #     filename = os.path.join(root_dir, 'bestfit.fits')
    #     locals()[f'mse_{regul}'] = np.nanmean(
    #         (locals()[f'bestfit_regul{regul}'] - galaxy)**2,
    #         axis=0)
    #     plt.subplots()
    #     plt.imshow(locals()[f'mse_{regul}'], vmin=0.001, vmax=0.2)
