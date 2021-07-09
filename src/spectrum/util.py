"""
Created on Wed Jul  7 18:52:18 2021

@author: luiz
"""

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import spectres as spectres


def build_edges(wave, sampling_type):
    if sampling_type == 'linear':
        step = wave[1] - wave[0]
        edges = np.array([wave[0] - step/np.double(2)], dtype = np.double)
        edges = np.append(edges, wave + step/np.double(2))
    elif sampling_type == 'log':
        step = np.log10(wave[1]/wave[0])
        edges = np.array(wave[0]/np.double(10)**(step/np.double(2)), dtype = np.double)
        edges = np.append(edges, wave*np.double(10)**(step/np.double(2)))
    elif sampling_type == 'ln':
        step = np.log(wave[1]/wave[0])
        edges = np.array(wave[0]/np.e**(step/np.double(2)), dtype = np.double)
        edges = np.append(edges, wave*np.e**(step/np.double(2)))
    return edges


def resampling(flux, old_wave, old_sampling_type, new_wave, new_sampling_type,
               flux_err = None):
    # edges
    old_edges = build_edges(old_wave, old_sampling_type)
    new_edges = build_edges(new_wave, new_sampling_type)

    # intervals
    old_inter = np.ediff1d(old_edges)
    new_inter = np.ediff1d(new_edges)
    
    # integrate and resample the spectrum
    int_flux = np.append([0], np.cumsum(flux) * old_inter)

    f_interp = interpolate.interp1d(old_edges, int_flux, bounds_error = False)

    new_flux = f_interp(new_edges)
    new_flux = np.ediff1d(new_flux)
    new_flux = new_flux/new_inter

    # if the uncertainty is provided it's also processed
    if flux_err is not None:
        int_err = np.append([0], np.cumsum(flux_err) * old_inter)
        
        e_interp = interpolate.interp1d(old_edges, np.square(int_err),
                                        bounds_error = False)
        
        new_flux_err = np.sqrt(e_interp(new_edges))
        new_flux_err = np.ediff1d(new_flux_err)
        new_flux_err = new_flux_err/new_inter
    
        return new_flux, new_flux_err
    else:
        return new_flux


def rebinning(flux, old_wave, old_sampling_type, new_wave, new_sampling_type,
               flux_err = None):
    # todo: maybe use more x-axis points
    f_interp = interpolate.interp1d(self.wave, self.flux, kind = 'linear',
                                    bounds_error = False)
    reb_data = f_interp(new_wave)

    e_interp = interpolate.interp1d(self.wave, self.flux_unc, kind = 'linear',
                                    bounds_error = False)
    reb_error = e_interp(new_wave)

    self.sampling_type = new_sampling_type
    self.flux = reb_data
    self.wave = new_wave
    self.flux_unc = reb_error
    return NotImplemented


#%%
wave_ln = np.e**(np.arange(np.log(obs.wave[0]), 
                           np.log(obs.wave[-1]), 
                           np.log(obs.wave[1]/obs.wave[0])))

spec_res = spectres(wave_ln, obs.wave, flux, np.sqrt(obs.flux_unc))



our_res = resampling(flux,obs.wave, obs.sampling_type, wave_ln, 'ln',
                     flux_err=np.sqrt(obs.flux_unc))

plt.plot(obs.wave, obs.flux, label = 'muse_e')
plt.plot(wave_ln, spec_res[0], label = 'spectres')
plt.plot(wave_ln, our_res[0], label = 'our')


plt.plot(obs.wave, np.sqrt(obs.flux_unc), label = 'muse_e')
plt.plot(wave_ln, spec_res[1], label = 'spectres_e')
plt.plot(wave_ln, our_res[1], label = 'our_e')

plt.legend()
