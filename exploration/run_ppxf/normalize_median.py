""" 
Created on Tue Aug 31 14:04:00 2021

@author: Luiz
"""
import numpy as np
from astropy.io import fits


def _normalize_median(flux):
    factor = np.nanmedian(flux)
    flux = flux/factor
    return flux, factor

def normalize_median(flux, save = False, directory = None):
    if flux.ndim == 1:
        flux, factor = _normalize_median(flux)
    elif flux.ndim == 2 or flux.ndim == 3:
        factor = np.nanmedian(flux, axis = 0)
        flux = flux/factor
        
    if save is True:
        save_factor(factor, directory)

    return flux, factor

def save_factor(factor, directory):
    hdu = fits.PrimaryHDU(data=np.atleast_1d(factor))
    hdul = fits.HDUList([hdu])
    hdul.writeto(f'{directory}/normalization_factor.fits', overwrite=True)

#%%
# flux1d = np.random.rand(10)
# flux2d = np.random.rand(10,3)
# flux3d = np.random.rand(10,4,5)

# res1, factor1 = normalize_median(flux1d)
# res2, factor2 = normalize_median(flux2d)
# res3, factor3 = normalize_median(flux3d, save = True, directory = '.')

# np.allclose(res1 * factor1, flux1d)
# np.allclose(res2 * factor2, flux2d)
# np.allclose(res3 * factor3, flux3d)
