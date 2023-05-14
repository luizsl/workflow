"""
Created on Tue Aug 31 14:04:00 2021

@author: Luiz
"""
import numpy as np


def normalize_band(flux=None, wave=None, limits=[-np.inf, np.inf],
                   weighting='light'):
    assert flux is not None

    if wave is not None:
        assert len(limits)==2
        band = (limits[0] < wave) & (wave < limits[-1])
    else:
        band = flux > 0

    weighting = weighting.lower()
    if weighting == 'mass' or weighting =='scalar':
        factor = np.nanmedian(flux[band])
    elif weighting == 'light' or weighting =='vector':
        factor = np.nanmedian(flux[band], 0)
    else:
        raise Exception

    flux = flux/factor

    return flux, factor, weighting

# flux = model.flux_grid
# wave = model.meta['wave_model']

# fig, ax = plt.subplots(3, 1)

# ax[0].plot(wave, flux[:, :])

# flux_l, factor_l, weighting_l =normalize_band(flux, wave, limits=[5450, 5550])

# flux_m, factor_m, weighting_m =normalize_band(flux, wave, limits=[5450, 5550],
#                                               weighting='scalar')

# ax[1].plot(wave, flux_l[:, :])
# ax[2].plot(wave, flux_m[:, :])