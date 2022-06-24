#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:36:30 2022

@author: Luiz

Residuals analysis of Legendre polynomial test 
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def read_data(directory, file, ext=0):
    path = os.path.join(directory, file)
    with fits.open(path) as hdul:
        data = hdul[ext].data
        return data
    
def bestfit(order_range=None, directory=None, lib=None):
    assert order_range is not None
    assert directory is not None
    assert lib is not None
    
    # # Build variables
    # for i in range(1,length+1):
    #     locals()[f'm{i}'] = []
    
    out = []
    # Read and unpack data
    for i in range(len(order_range)):
        _directory = os.path.join(directory, lib) + f'_{i}/ppxf'
        file = 'bestfit.fits'

        # NOTE: Exception due to limitations on directory name assignment
        if i == 0:
            _directory = _directory.replace('_0','')
            
        d = read_data(_directory, file)
        out.append(d)
        # for i in range(1,length+1):
        #     if length > 1:
        #         locals()[f'm{i}'].append(d[i-1])
        #     else:
        #         locals()[f'm{i}'].append(d)

    # Convert to numpy arrays for sake of simplicity
    out = np.asarray(out).T
    return out
if __name__ == '__main__':
    degree_range = np.arange(-1, 21, 1)
    mdegree_range = np.arange(0, 21, 1)
    
    a = bestfit(order_range=degree_range, 
            directory='../data_products/polynomial_single/emiles/additive_polynomial/NGC0613_full_stacked_spectrum/',
            lib='MilesAgeMh')