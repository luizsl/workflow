#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 00:48:58 2025

@author: Luiz
"""

import sys
import numpy as np
from scipy.ndimage import convolve
from astropy.io import fits
from scipy.stats import multivariate_normal


## Define functions

def kernel(mu, sigma, d=1):
    pix = np.linspace(-3*sigma, 3*sigma, num=2*3*sigma+1, endpoint=True)
    
    x, y = np.meshgrid(pix, pix)
    pos = np.dstack((x, y))
    
    rv = multivariate_normal(mu*d*np.ones(2), sigma*d*np.identity(2))
    k = rv.pdf(pos)

    return k


def read_data(path_obs):   
    with fits.open(path_obs) as hdul:
        obs = hdul['data'].data
        unc = hdul['stat'].data
        obs_head = hdul['data'].header
        
    return obs, unc, obs_head


def smooth(obs, unc, k):
    smooth_obs = np.full_like(obs, np.nan)
    smooth_unc = np.full_like(obs, np.nan)
    
    for i in np.arange(obs.shape[0]):
        smooth_obs[i] = convolve(np.nan_to_num(obs[i],0), weights=k)
        smooth_unc[i] = convolve(np.nan_to_num(unc[i],0), weights=k**2)
    
    nan_footprint = np.isnan(obs)
    smooth_obs[nan_footprint] = 0
    smooth_unc[nan_footprint] = 0
    
    return smooth_obs, smooth_unc


def save_cube(smooth_obs,smooth_unc, path_obs):
    path_obs_smooth = path_obs.split('.')[0] + '_smooth.fits'

    with fits.open(path_obs) as hdul:
        hdul['data'].data = smooth_obs
        hdul['stat'].data = smooth_unc
        hdul.writeto(path_obs_smooth)


if __name__ == '__main__':
    ## Read data
    path_obs = sys.argv[1]
    obs, unc, obs_head = read_data(path_obs)

    ## Create kernel     
    mu = 0
    sigma = 1
    k = kernel(mu, sigma)
    
    ## Smooth observations
    smooth_obs, smooth_unc = smooth(obs, unc, k)
    
    ## Save smoothed cube
    save_cube(smooth_obs,smooth_unc, path_obs)
    
