"""
Created on Sat Aug 28 16:01:24 2021

@author: Luiz
"""
import numpy as np
from ppxf.ppxf_util import gaussian_filter1d


def _convolve(flux, sigma):
    flux = gaussian_filter1d(flux, sigma)
    return flux

def convolve(flux, sigma):
    if flux.ndim == 1:
        flux = _convolve(flux, sigma)
    elif flux.ndim == 2 or flux.ndim == 3:
        flux = np.apply_along_axis(_convolve, 0, flux, sigma)
    return flux
