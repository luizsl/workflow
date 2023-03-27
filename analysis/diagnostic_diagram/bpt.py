#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:21:40 2023

@author: Luiz
"""

import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits


def k01_curve(x):
    y = 0.61/(x-0.47) + 1.19
    return y

    
def k03_curve(x):
    y = 0.61/(x-0.05) + 1.3
    return y


def k06_curve(x):
    y = 0.29/(x+0.2) + 0.96
    return y

def s07_curve(x):
    y = 1.05*x + 0.45
    return y

    
if __name__ == "__main__" :
    
    with plt.style.context(['science', 'nature']):
    
        # flux_path = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
        #              'XSLAgeMh_3/ppxf_emission_line_2components/'
        #              'corrected_flux.fits'
        #              )
       
        flux_path = ('../../data_products/NGC0613_DATACUBE_FINAL_clean/'
                      'XSLAgeMh_3/ppxf_emission_line_binned100_1components/'
                      'corrected_flux.fits'
                      )
       
        with fits.open(flux_path) as hdul:
            flux = hdul[0].data
        
        # NOTE: The fluxes are measured regarding the doublet. 
        # Taking the line ratio into account is required. 
        y = np.log10((3/4)*flux[0] / flux[8])
        x = np.log10((3/4)*flux[4] / flux[9])
        
        fig, ax = plt.subplots()
    
        ax.scatter(x, y, alpha=0.1, marker='o', s=5, linewidths=0.01)
    
        k01_x = np.linspace(-2, 0.3, 500)
        k01_y = k01_curve(k01_x)
        
        k03_x = np.linspace(-2, 0, 500)
        k03_y = k03_curve(k03_x)
    
        s07_x = np.linspace(-0.183807, 2, 500)
        s07_y = s07_curve(s07_x)
    
        ax.plot(k01_x, k01_y, color='k', ls=(1, (10, 2)))
        ax.plot(k03_x, k03_y, color='k', ls=(1, (10, 2, 1, 2)))
        ax.plot(s07_x, s07_y, color='k', ls=(1, (1, 0))) 
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.6, 1.8)
        
        ax.set_xlabel(r'$\log_{10} \left(\rm{[NII] \lambda6583} / \rm{H}\alpha\right)$')
        ax.set_ylabel(r'$\log_{10} \left(\rm{[OIII]\lambda5007} / \rm{H}\beta\right)$')
        # fig.xlim(-1.5, 1)
        # fig.ylim(-1.3, 1.5)
        
        fig.show()

#%%

        # NOTE: The fluxes are measured regarding the doublet. 
        # Taking the line ratio into account is required. 
        y = np.log10((3/4)*flux[0] / flux[8])#[120:200, 120:200]
        x = np.log10((3/4)*flux[4] / flux[9])#[120:200, 120:200]
        
        fig, ax = plt.subplots(1,2)
   
        ax[0].scatter(x, y, alpha=0.5, marker='o', s=5, linewidths=0.01, cmap='rainbow', c=x, vmin=-0.6, vmax=0.3)
    
        k01_x = np.linspace(-2, 0.3, 500)
        k01_y = k01_curve(k01_x)
        
        k03_x = np.linspace(-2, 0, 500)
        k03_y = k03_curve(k03_x)
    
        s07_x = np.linspace(-0.183807, 2, 500)
        s07_y = s07_curve(s07_x)
    
        ax[0].plot(k01_x, k01_y, color='k', ls=(1, (10, 2)))
        ax[0].plot(k03_x, k03_y, color='k', ls=(1, (10, 2, 1, 2)))
        ax[0].plot(s07_x, s07_y, color='k', ls=(1, (1, 0))) 
        
        ax[0].set_xlim(-0.8, 0.4)
        ax[0].set_ylim(-1.2, 1.0)
        
        ax[0].set_xlabel(r'$\log_{10} \left(\rm{[NII] \lambda6583} / \rm{H}\alpha\right)$')
        ax[0].set_ylabel(r'$\log_{10} \left(\rm{[OIII]\lambda5007} / \rm{H}\beta\right)$')
        # fig.xlim(-1.5, 1)
        # fig.ylim(-1.3, 1.5)
        
        a = ax[1].imshow(x, origin='lower', cmap='rainbow', vmin=-0.6, vmax=0.3)
        plt.colorbar(a)
        fig.show()