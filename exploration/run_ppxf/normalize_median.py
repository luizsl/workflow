""" 
Created on Tue Aug 31 14:04:00 2021

@author: Luiz
"""
import numpy as np


def normalize_band(flux=None, wave=None, limits=[-np.inf, np.inf]):
    global band
    assert flux is not None

    if wave is not None:
        assert len(limits)==2
        band = (limits[0] < wave) & (wave < limits[-1])
    else:
        band = flux > 0

    factor = np.nanmedian(flux[band])
    flux = flux/factor
        
    return flux, factor

# def save_factor(factor, directory):
#     hdu = fits.PrimaryHDU(data=np.atleast_1d(factor))
#     hdul = fits.HDUList([hdu])
#     hdul.writeto(f'{directory}/normalization_factor.fits', overwrite=True)


# flux1d = np.random.rand(200)+5
# flux2d = np.random.rand(200,3)+5
# flux3d = np.random.rand(200,4,5)+5
# wave = np.linspace(1, 50, 200)

# res1, factor1 = normalize_band(flux1d, wave, [1, 2])
# res2, factor2 = normalize_band(flux2d, wave)
# res3, factor3 = normalize_band(flux3d, wave)

# plt.plot(wave, flux1d)
# plt.plot(wave, res1)
# # np.allclose(res2 * factor2, flux2d)
# # np.allclose(res3 * factor3, flux3d)
