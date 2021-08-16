"""
Created on Wed Jul  7 18:52:18 2021

@author: luiz
"""

import numpy as np
from scipy import interpolate

from matplotlib import pyplot as plt
from spectrum import Spectrum

from spectres import spectres
from astropy.io import fits
from time import perf_counter as clock

def _extend_dim(array):
    if array.ndim == 3:
        pass
    elif array.ndim == 1:
        array = array[..., np.newaxis, np.newaxis]
    elif array.ndim == 2:
        array = array[..., np.newaxis]
    else:
        raise TypeError

    return array

def _reduce_dim(array):

    if array.ndim == 3:
        if array.shape[1:] == (1, 1):
            array = array[:, 0, 0]
        elif array.shape[2:] == (1, ):
            array = array[:, :, 0]
    elif array.ndim == 2:
        if array.shape[1:] == (1,):
                array = array[:, 0]
    else:
        pass

    return array

def _build_edges(wave, sampling_type):
    if sampling_type == 'linear':
        step = wave[[1], ...] - wave[[0], ...]
        edges = wave[[0], ...] - step/2.
        edges = np.append(edges, wave + step/2., axis = 0)
    elif sampling_type == 'log':
        step = np.log10(wave[[1], ...]/wave[[0], ...])
        edges = wave[[0], ...] / 10.**(step/2.)
        edges = np.append(edges, wave * 10.**(step/2.), axis = 0)
    elif sampling_type == 'ln':
        step = np.log(wave[[1], ...]/wave[[0], ...])
        edges = wave[[0], ...] / np.e**(step/2.)
        edges = np.append(edges, wave * np.e**(step/2.), axis = 0)
    return edges

def _resampling_fast(flux, old_wave, old_sampling_type, new_wave, new_sampling_type,
               flux_err = None):

    # edges
    old_edges = _build_edges(old_wave, sampling_type = old_sampling_type)
    new_edges = _build_edges(new_wave, sampling_type = new_sampling_type)

    old_edges = _extend_dim(old_edges)
    new_edges = _extend_dim(new_edges)

    # intervals
    old_inter = np.diff(old_edges, axis = 0)
    new_inter = np.diff(new_edges, axis = 0)

    # integrate and resample the spectrum
    int_flux = np.cumsum(flux * old_inter, axis = 0)
    zeros_layer = np.zeros_like(flux[0:1, ...])
    int_flux = np.append(zeros_layer, int_flux, axis = 0)

    f_interp = interpolate.interp1d(_reduce_dim(old_edges), int_flux, bounds_error = False,
                                    axis = 0)

    # for i,j in np.ndindex(new_edges[0,...].shape):
    # new_flux[:, i, j] = f_interp(new_edges[:, i, j])

    new_flux = f_interp(new_edges[:, 0, 0])
    new_flux = np.diff(new_flux, axis = 0)
    new_flux = new_flux/new_inter

    # if the uncertainty is provided it's also processed
    if flux_err is not None:
        int_err = np.append([0], np.cumsum(flux_err * old_inter))

        e_interp = interpolate.interp1d(old_edges, np.square(int_err),
                                        bounds_error = False)

        new_flux_err = np.sqrt(e_interp(new_edges))
        new_flux_err = np.ediff1d(new_flux_err)
        new_flux_err = new_flux_err/new_inter

        return new_flux, new_flux_err
    else:
        return new_flux

def _resampling(flux, old_wave, old_sampling_type, new_wave, new_sampling_type,
               flux_err = None):
    # edges
    old_edges = _build_edges(old_wave, old_sampling_type)
    new_edges = _build_edges(new_wave, new_sampling_type)

    # intervals
    old_inter = np.ediff1d(old_edges)
    new_inter = np.ediff1d(new_edges)

    # integrate and resample the spectrum
    int_flux = np.append([0], np.cumsum(flux * old_inter))

    f_interp = interpolate.interp1d(old_edges, int_flux, bounds_error = False)

    new_flux = f_interp(new_edges)
    new_flux = np.ediff1d(new_flux)
    new_flux = new_flux/new_inter

    # if the uncertainty is provided it's also processed
    if flux_err is not None:
        int_err = np.append([0], np.cumsum(flux_err * old_inter))

        e_interp = interpolate.interp1d(old_edges, np.square(int_err),
                                        bounds_error = False)

        new_flux_err = np.sqrt(e_interp(new_edges))
        new_flux_err = np.ediff1d(new_flux_err)
        new_flux_err = new_flux_err/new_inter

        return new_flux, new_flux_err
    else:
        return new_flux


def _resampling_nd(flux, old_wave, old_sampling_type, new_wave,
                  new_sampling_type, flux_err = None):

    new_flux = np.zeros(shape = (new_wave.shape[0],) + flux[0,...].shape)

    if old_wave.shape[1:] == (1, 1) and new_wave.shape[1:] == (1, 1):
        if flux_err:
            new_flux_err = np.zeros(shape = (new_wave.shape[0],) + flux[0,...].shape)
            for i,j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j], new_flux_err[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave, old_sampling_type,
                                new_wave, new_sampling_type,
                                flux_err = error_d[:, i, j])
            return new_flux, new_flux_err
        else:
            for i, j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j] = \
                    _resampling(flux[:, i, j],
                                _reduce_dim(old_wave), old_sampling_type,
                                _reduce_dim(new_wave), new_sampling_type)
            return new_flux

    elif old_wave.shape[1:] == (1, 1) and new_wave.shape[1:] > (1, 1):
        if flux_err:
            new_flux_err = np.zeros(shape = (new_wave.shape[0],) + flux[0,...].shape)
            for i,j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j], new_flux_err[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave, old_sampling_type,
                                new_wave[:, i, j], new_sampling_type,
                                flux_err = error_d[:, i, j])
            return new_flux, new_flux_err
        else:
            for i, j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave, old_sampling_type,
                                new_wave[:, i, j], new_sampling_type)
            return new_flux

    elif old_wave.shape[1:] > (1, 1) and new_wave.shape[1:] == (1, 1):
        if flux_err:
            new_flux_err = np.zeros(shape = (new_wave.shape[0],) + flux[0,...].shape)
            for i,j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j], new_flux_err[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave[:, i, j], old_sampling_type,
                                new_wave, new_sampling_type,
                                flux_err = error_d[:, i, j])
            return new_flux, new_flux_err
        else:
            for i, j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave[:, i, j], old_sampling_type,
                                new_wave, new_sampling_type)
            return new_flux

    elif old_wave.shape[1:] > (1, 1) and new_wave.shape[1:] > (1, 1):
        if flux_err:
            new_flux_err = np.zeros(shape = (new_wave.shape[0],) + flux[0,...].shape)
            for i,j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j], new_flux_err[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave[:, i, j], old_sampling_type,
                                new_wave[:, i, j], new_sampling_type,
                                flux_err = error_d[:, i, j])
            return new_flux, new_flux_err
        else:
            for i, j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave[:, i, j], old_sampling_type,
                                new_wave[:, i, j], new_sampling_type)
            return new_flux

    else:
        raise TypeError

def build_wave_array(wave, sampling_type, size):
    if sampling_type == 'linear':
        wave_array = np.linspace(wave[0], wave[1], size, dtype = np.double)
    elif sampling_type == 'log':
        wave_array = np.logspace(np.log10(wave[0]), np.log10(wave[1]), size,
                                 np.double(10.), dtype = np.double)
    elif sampling_type == 'ln':
        wave_array = np.logspace(np.log(wave[0]), np.log(wave[1]), size, np.e,
                                 dtype = np.double)
    return wave_array

def resampling(flux, old_wave, old_sampling_type, new_wave = None,
               new_sampling_type = None, flux_err = None):

    assert old_sampling_type and new_sampling_type in ['linear', 'log', 'ln']

    if old_wave.shape[0] == 2:
        old_wave = build_wave_array(wave = old_wave,
                                    sampling_type = old_sampling_type,
                                    size = flux.shape[0])
    elif old_wave.shape[0] < 2:
        raise ValueError

    if new_wave is None:
        new_wave = build_wave_array(wave = [old_wave[0], old_wave[-1]],
                                    sampling_type = new_sampling_type,
                                    size = flux.shape[0])
    elif new_wave.shape[0] == 2:
        new_wave = build_wave_array(wave = new_wave,
                                    sampling_type = new_sampling_type,
                                    size = flux.shape[0])
    elif old_wave.shape[0] < 2:
        raise ValueError

    if flux.ndim < 3:
        flux = _extend_dim(flux)
    if old_wave.ndim < 3:
        old_wave = _extend_dim(old_wave)
    if new_wave.ndim < 3:
        new_wave = _extend_dim(new_wave)
    if flux_err is not None:
        if flux_err.ndim <3:
            flux_err = _extend_dim(flux_err)

    if flux_err is None:
        new_flux = _resampling_nd(flux,
                                  old_wave, old_sampling_type,
                                  new_wave, new_sampling_type)
        new_flux = _reduce_dim(new_flux)
        new_wave = _reduce_dim(new_wave)
        return new_flux, new_wave
    else:
        new_flux, new_flux_err = _resampling_nd(flux,
                                                old_wave, old_sampling_type,
                                                new_wave, new_sampling_type,
                                                flux_err = flux_err)
        new_flux = _reduce_dim(new_flux)
        new_wave = _reduce_dim(new_wave)
        new_flux_err = _reduce_dim(new_flux_err)
        return new_flux, new_wave, new_flux_err

#%% Reading data to tests
if __name__ == '__main__':
    
    #######################################################              test
    
    obs_file = '../../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'
    
    #########################################################################
    
    
    with fits.open(obs_file) as hdu:
        data = hdu['DATA'].data
        error_d = hdu['STAT'].data
    
        flux = data[:, 103:106, 103:105]
        error = error_d[:, 103:106, 103:105]
    
        first_wave = hdu['DATA'].header['CRVAL3']
        step_wave = hdu['DATA'].header['CD3_3']
    
        obs = Spectrum(flux, wave = [first_wave, step_wave],
                       medium = 'air', sampling_type = 'linear', flux_unc = error)
    
    
    wave_ln = np.e**(np.arange(np.log(obs.wave[0]),
                               np.log(obs.wave[-1]),
                               1*np.log(obs.wave[1]/obs.wave[0])))
    
    
    #%% single spectrum, simple old_wave, None new_wave
    
    # w = [4750, 9351]
    
    # flux = data[:, 103, 103]
    # error = error_d[:, 103, 103]
    
    
    # t1 = clock()
    # a,b = resampling(flux, [4750, 9000], 'linear', new_sampling_type = 'log')
    # print(clock() - t1)
    
    
    #%%
    flux = data[:, 10, 10]
    old_wave = obs.wave
    new_wave = wave_ln
    
    t1 = clock()
    _resampling(flux = flux,
                old_wave = old_wave, old_sampling_type = 'linear',
                new_wave = new_wave, new_sampling_type = 'log')
    print((clock() - t1)*1000)
    
    
    t2 = clock()
    spectres(_reduce_dim(flux), _reduce_dim(old_wave), _reduce_dim(new_wave))
    print((clock() - t2)*1000)
    
    
    # %%
    # plt.plot([1, 10, 100, 1000, 10000, 90000],
    #          [0.0005697300002793781, 0.003376420005224645, 0.029915755992988124,0.3128900789961335,3.7427781609949307,36.738016718001745])
    # x = [1, 10, 100, 1000, 10000, 90000]
    # plt.plot(x, np.log10(x))
    # plt.plot([1, 10, 100, 1000, 10000, 90000], x)