#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:54:52 2021

@author: chess-lin
"""

import numpy as np
import matplotlib.pyplot as plt
import ppxf.ppxf_util as util
from spectrum import Spectrum
from astropy.io import fits
from scipy import integrate
from scipy.constants import physical_constants


C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s
cube_file = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

with fits.open(cube_file) as hdu:
    header = hdu[1].header
    flux = np.array(hdu[1].data[:,150,150])
    err_f = np.array(hdu[2].data[:,150,150])

own = Spectrum(flux = flux,
               wave = [header['CRVAL3'], header['CD3_3']],
               medium = 'air', sampling_type = 'linear',
               flux_unc = err_f)

# getting wavelength array for the other methods
wave_lin = own.wave
wave_ln = np.e**(np.log(wave_lin[0]) 
                 + np.log(wave_lin[1]/wave_lin[0])*np.arange(2574))
wave_ln_over = np.e**(np.log(wave_lin[0]) 
                      + 0.5*np.log(wave_lin[1]/wave_lin[0])*np.arange(2574*2))


# Rebining with without oversampling
own.rebinning(wave_ln, new_sampling_type = 'ln')


# Rebinning with oversampling

own_over = Spectrum(flux = flux,
                    wave = [header['CRVAL3'], header['CD3_3']],
                    medium = 'air', sampling_type = 'linear',
                    flux_unc = err_f)
own_over.rebinning(wave_ln_over, new_sampling_type = 'ln')

# Plot
plt.style.use('plot_script/fig_conf.mplstyle')

ax = plt.subplot()
ax.step(wave_lin, flux, where = 'mid', label = r'MUSE spectrum')
ax.step(wave_ln, own.flux, where = 'mid', label = r'Without oversampling')
ax.step(wave_ln_over, own_over.flux, where = 'mid',
        label = r'With 2x oversampling')
ax.set_yscale('log')
ax.legend()

ax.set_xlabel(r'$\textbf{Wavelength} \mathbf{(\AA)}$')
ax.set_ylabel(r'$\textbf{Flux density} \, \mathbf{(10^{-20} \, \erg/\s/\cm^{2}/\AA)}$')
ax.set_title(r'Rebinning comparison (flux conservative)')

plt.savefig('../plots/rebinning_comparison.pdf')

#%% Plot residuals
x = np.linspace(wave_lin[0], wave_lin[-1], 10000)
muse = np.interp(x, wave_lin, flux)
signal_1over = np.interp(x, wave_ln, own.flux)
signal_2over = np.interp(x, wave_ln_over, own_over.flux)

fig, (ax1, ax2) = plt.subplots(2, sharey = True)
ax1.plot(x, (muse - signal_1over)/muse, label = 'Without oversampling',
         color = 'black')
ax1.set_ylabel(r'$\textbf{Residuals} \mathbf{(\%)}$')
ax1.legend(loc = 4)
ax2.plot(x, (muse - signal_2over)/muse, label = 'With 2x oversampling',
         color = 'black')
ax2.set_xlabel(r'$\textbf{Wavelength} \mathbf{(\AA)}$')
ax2.set_ylabel(r'$\textbf{Residuals} \mathbf{(\%)}$')
ax2.legend(loc = 4)

plt.savefig('../plots/rebinning_residuals.pdf')
