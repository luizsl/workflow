#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:16:21 2021

@author: Luiz
"""

import numpy as np
import matplotlib.pyplot as plt
import ppxf.ppxf_util as util
from spectres import spectres
from spectrum import Spectrum
from astropy.io import fits
from scipy import integrate
from scipy.constants import physical_constants
from time import perf_counter as clock

C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s
cube_file = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

def run_comparison():
    global t_flux_muse, t_flux_our, t_flux_ppxf, t_flux_spectres, t_our, t_spectres, t_ppxf
    
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
    
    # Resampling with our method
    own.resampling(wave_ln, new_sampling_type = 'ln')
    
    # Resampling with Spectres
    spectres_data, spectres_err = spectres(wave_ln, wave_lin, flux, err_f)
    
    # Resampling with ppxf util method
    # Computing sampling in km/s for logarithmic rebinning
    velscale = C * np.log(wave_lin[1]/wave_lin[0])
    ppxf_data, wave_ppxf = util.log_rebin([wave_lin[0], wave_lin[-1]], flux,
                                          flux = True, velscale = velscale)[0:2]
    
    # Plot
    plt.style.use('plot_script/fig_conf.mplstyle')
    
    ax = plt.subplot()
    ax.step(wave_lin, flux, where = 'mid', lw = 0.6, label = r'MUSE spectrum')
    ax.step(wave_ln, own.flux, where = 'mid', lw = 1.,
            label = r'Our implementation')
    ax.step(wave_ln, spectres_data, where = 'mid', lw = 0.6,
            label = r'\texttt{SpectRes} (\texttt{Astropy*})')
    ax.step(np.e**(wave_ppxf), ppxf_data, where = 'mid', lw = 0.6,
            label = r'\texttt{pPXF} util')
    ax.set_yscale('log')
    ax.legend()
    
    ax.set_xlabel(r'$\textbf{Wavelength} \mathbf{(\AA)}$')
    ax.set_ylabel(r'$\textbf{Flux density} \, \mathbf{(10^{-20} \, \erg/\s/\cm^{2}/\AA)}$')
    ax.set_title(r'Resampling comparison (flux conservative)')
    
    plt.savefig('../plots/resampling_comparison.pdf')

    # Flux Muse
    t_flux_muse = '{:.2e}'.format(np.trapz(flux, wave_lin)).split('e+')
    
    # Flux with our method
    t_flux_our = '{:.2e}'.format(integrate.trapezoid(own.flux, wave_ln)).split('e+')
    
    # Flux with Spectres
    t_flux_spectres = '{:.2e}'.format(integrate.trapezoid(spectres_data[1:], wave_ln[1:])).split('e+')
    
    # Flux with ppxf util method
    t_flux_ppxf = '{:.2e}'.format(integrate.trapezoid(ppxf_data, np.e**(wave_ppxf))).split('e+')

    n_rep = int(1e3)
    
    # Resampling with Spectres
    t_spectres = np.empty(n_rep)
    for i in range(n_rep):
        t1 = clock()
        spectres_data = spectres(wave_ln, wave_lin, flux)
        t_spectres[i] = clock() - t1
    t_spectres = round(t_spectres.mean()*1e3, 3)
    
    # Resampling with ppxf util method method
    t_ppxf = np.empty(n_rep)
    for i in range(n_rep):
        t1 = clock()
        ppxf_data, wave_ppxf = util.log_rebin([wave_lin[0], wave_lin[-1]], flux,
                                              flux = True, velscale = velscale)[0:2]
        t_ppxf[i] = clock() - t1
    t_ppxf = round(t_ppxf.mean()*1e3, 3)
    
    # Resampling with our method
    t_our = np.empty(n_rep)
    for i in range(n_rep):
        own = Spectrum(flux = flux,
                              wave = [header['CRVAL3'], header['CD3_3']],
                              medium = 'air', sampling_type = 'linear',
                              flux_unc = err_f)
        t1 = clock()
        own.resampling(wave_ln, new_sampling_type = 'ln')
        t_our[i] = clock() - t1
    t_our = round(t_our.mean()*1e3, 3)

    # Write latex table with results
    with open('../tables/table_resampling_comparison.tex', 'w') as t:
        t.write(f"""
                & Flux $(10^{{-14}} \\, \\erg/\\s/\\cm^{{2}})$ & Average runtime (ms) \\\\ \\hline
        MUSE                  & $ {t_flux_muse[0]}     $  & -             \\\\
        Our implementation    & $ {t_flux_our[0]}      $  & {t_our}       \\\\
        \\texttt{{SpectRes}}  & $ {t_flux_spectres[0]} $  & {t_spectres}  \\\\
        \\texttt{{pPXF}} util & $ {t_flux_ppxf[0]}     $  & {t_ppxf}      \\\\ \\hline
        """)
        
# Run comparison and 
if __name__ == '__main__':
    run_comparison()
