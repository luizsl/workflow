""" 
Created on Tue Aug 31 14:04:00 2021

@author: Luiz
"""
import numpy as np


def _normalize_median(flux):
    flux = flux/np.nanmedian(flux)
    return flux

def normalize_median(flux):
    if flux.ndim == 1:
        flux = _normalize_median(flux)
    elif flux.ndim == 2 or flux.ndim == 3:
        flux = np.apply_along_axis(_normalize_median, 0, flux)
    return flux
