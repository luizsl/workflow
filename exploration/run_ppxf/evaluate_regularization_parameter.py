#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 08:17:19 2022

@author: Luiz
"""

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


class FactoryChi2Map(ABC):
    @abstractmethod
    def read_reduced_chi2(self):
        "Read fits file with reduced chi**2"
        pass
    
    @abstractmethod
    def read_dof(self):
        """Takes the degree of freedom (valid pixels in each spaxel)"""
        pass
        
    @property
    @abstractmethod
    def chi2(self):
        """Based on DOF and reduced chi**2, compute chi**2"""
        pass

    @property
    @abstractmethod
    def delta_chi2(self):
        """Estimate the target chi**2 value after regularization. According to 
        pPXF documentations, the employ of regularization should result in 
        an increase of chi**2 equal to sqrt(2*N_pix)"""
        pass


class Chi2Map(FactoryChi2Map):
    def __init__(self, filepath_reduced_chi2, filepath_dof):
        self.filepath_reduced_chi2 = filepath_reduced_chi2
        self.filepath_dof = filepath_dof
        self.reduced_chi2 = self.read_reduced_chi2()
        self.dof = self.read_dof()
    
    def read_reduced_chi2(self):
        with fits.open(self.filepath_reduced_chi2) as hdul:
            data = hdul[0].data
            return data
        
    def read_dof(self):
        with fits.open(self.filepath_dof) as hdul:
            data = hdul[0].data
            data = np.ma.masked_where(np.isnan(data), data)
            data = data.count(axis = 0)
            return data
        
    @property
    def chi2(self):
        return self.reduced_chi2 * self.dof

    @property
    def delta_chi2(self):
        target_chi2 = self.chi2 + np.sqrt(2 * self.dof)
        return target_chi2
    

class Chi2Comparison():
    def __init__(self, reg: FactoryChi2Map, unreg: FactoryChi2Map):
        self.reg = reg
        self.unreg = unreg
        
    @property
    def data(self):
        """Map the valid spaxels that fullfil the chi**2 increase requirement."""
        data = self.reg.chi2 / self.unreg.delta_chi2
        return data
    
    @property
    def fraction(self):
        data = self.reg.chi2 / self.unreg.delta_chi2
        fraction = len(data[data>=1]) / len(data)
        return fraction
    

def plot_red_chi2_reg(maps: list):
    n_col = len(maps)
    fig, ax = plt.subplots(1,n_col)
    for i in range(n_col):
        im = ax[i].imshow(maps[i],
                       vmin = 1 - .05, vmax = 1 + .05)

#%%     
            
if __name__ == "__main__":

    """
    Compare the effect of sigma in the dynamic mask
    """ 
    unreg0d00_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma0d0/ppxf/chi2.fits'
    unreg0d00_filepath_dof = '../../data_products//regularization_parameter/MilesAgeMh_unreg_sigma0d0/ppxf/goodpixels.fits'
    
    unreg3d00_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma3d0/ppxf/chi2.fits'
    unreg3d00_filepath_dof = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma3d0/ppxf/goodpixels.fits'

    unreg2d50_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d5/ppxf/chi2.fits'
    unreg2d50_filepath_dof = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d5/ppxf/goodpixels.fits'

    unreg2d30_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d3/ppxf/chi2.fits'
    unreg2d30_filepath_dof = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d3/ppxf/goodpixels.fits'
        
    unreg2d00_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d0/ppxf/chi2.fits'
    unreg2d00_filepath_dof = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d0/ppxf/goodpixels.fits'
    
    unreg0d00 = Chi2Map(unreg0d00_filepath_reduced_chi2, unreg0d00_filepath_dof)
    unreg3d00 = Chi2Map(unreg3d00_filepath_reduced_chi2, unreg3d00_filepath_dof)
    unreg2d50 = Chi2Map(unreg2d50_filepath_reduced_chi2, unreg2d50_filepath_dof)
    unreg2d30 = Chi2Map(unreg2d30_filepath_reduced_chi2, unreg2d30_filepath_dof)
    unreg2d00 = Chi2Map(unreg2d00_filepath_reduced_chi2, unreg2d00_filepath_dof)
    
    def plot_red_chi2_hist_residuals(maps: list, unreg=None, list_reg: list=None,
                                     bins = 10):
        n_col = len(maps)
        fig, ax = plt.subplots(1,n_col, figsize=(12,3), constrained_layout=True)
        for i in range(n_col):
            data = maps[i].ravel()
            median = np.nanmedian(data)
            ax[i].hist(data, bins=bins, range=(0.5,1.5))
            ax[i].axvline(median, color = 'k')
            ax[i].set_xlabel(r'$\chi_{\nu}^{2}$')
            if list_reg[i]:
                ax[i].set_title(f'$\sigma = {list_reg[i]}$', loc = 'left')
            if unreg:
                ax[i].hist(unreg.reduced_chi2.ravel(), range = (0.95, 1.5))
        fig.suptitle(r'Effect of $\sigma_{\rm res}$-based dynamic masking on the reduced $\chi^{2}$ for NGC 613')
        fig.supylabel('N')    
        
    plot_red_chi2_hist_residuals(
        maps = [unreg0d00.reduced_chi2, unreg3d00.reduced_chi2,
                unreg2d50.reduced_chi2, unreg2d30.reduced_chi2,
                unreg2d00.reduced_chi2],
        list_reg=[None, 3.0, 2.5, 2.3, 2.0],
        bins = 20)
        
#%% 
    """
    Increase of the chi**2 due to regularization
    """ 
    
    unreg_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d3/ppxf/chi2.fits'
    unreg_filepath_dof = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d3/ppxf/goodpixels.fits'
    
    reg0d1_filepath_reduced_chi2 = '../../data_products/fov_sample_1_5/MilesAgeMh_4/ppxf/chi2.fits'
    reg0d1_filepath_dof = '../../data_products/fov_sample_1_5/MilesAgeMh_4/ppxf/goodpixels.fits'
    
    reg0d2_filepath_reduced_chi2 = '../../data_products/fov_sample_1_5/MilesAgeMh_3/ppxf/chi2.fits'
    reg0d2_filepath_dof = '../../data_products/fov_sample_1_5/MilesAgeMh_3/ppxf/goodpixels.fits'
    
    reg0d3_filepath_reduced_chi2 = '../../data_products/fov_sample_1_5/MilesAgeMh_2/ppxf/chi2.fits'
    reg0d3_filepath_dof = '../../data_products/fov_sample_1_5/MilesAgeMh_2/ppxf/goodpixels.fits'
    
    reg0d4_filepath_reduced_chi2 = '../../data_products/fov_sample_1_5/MilesAgeMh_1/ppxf/chi2.fits'
    reg0d4_filepath_dof = '../../data_products/fov_sample_1_5/MilesAgeMh_1/ppxf/goodpixels.fits'
    
    reg0d5_filepath_reduced_chi2 = '../../data_products/fov_sample_1_5/MilesAgeMh/ppxf/chi2.fits'
    reg0d5_filepath_dof = '../../data_products/fov_sample_1_5/MilesAgeMh/ppxf/goodpixels.fits'
    
    unreg = Chi2Map(unreg_filepath_reduced_chi2, unreg_filepath_dof)
    reg0d1 = Chi2Map(reg0d1_filepath_reduced_chi2, reg0d1_filepath_dof)
    reg0d2 = Chi2Map(reg0d2_filepath_reduced_chi2, reg0d2_filepath_dof)
    reg0d3 = Chi2Map(reg0d3_filepath_reduced_chi2, reg0d3_filepath_dof)
    reg0d4 = Chi2Map(reg0d4_filepath_reduced_chi2, reg0d4_filepath_dof)
    reg0d5 = Chi2Map(reg0d5_filepath_reduced_chi2, reg0d5_filepath_dof)
    
    comp0d1 = Chi2Comparison(reg = reg0d1, unreg = unreg)
    comp0d2 = Chi2Comparison(reg = reg0d2, unreg = unreg)
    comp0d3 = Chi2Comparison(reg = reg0d3, unreg = unreg)
    comp0d4 = Chi2Comparison(reg = reg0d4, unreg = unreg)
    comp0d5 = Chi2Comparison(reg = reg0d5, unreg = unreg)
    
    plt.style.use('science')
    
    def plot_percentage(maps:list, list_reg:list=None):
        with plt.style.context('science'):
            n_col = len(maps)
            fig, ax = plt.subplots(1,n_col, figsize=(12,3), constrained_layout=True)
            for i in range(n_col):
                im = ax[i].imshow(maps[i], cmap='seismic',
                                  vmin = 1 - .05, vmax = 1 + .05,
                                  origin = 'lower')
                if list_reg[i]:
                    ax[i].set_title(f'regul = $1/{list_reg[i]}$', loc = 'left')
                    ax[i].tick_params(labelleft=False, labeltop=False,
                                      labelright=False, labelbottom=False)
                    ax[i].text(0, -.05, 'left top',
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=ax[i].transAxes)
            cbar = fig.colorbar(im, ax = ax[:])
            cbar.set_label(r'$ \dfrac{\chi^{2}_{\rm r}}{\chi^{2}_{\rm u} + \Delta \chi^{2}}$')
            
    plot_percentage(maps = [comp0d5.data, comp0d4.data, comp0d3.data,
                            comp0d2.data, comp0d1.data],
                    list_reg=[0.5, 0.4, 0.3, 0.2, 0.1])
    
 
    # def plot_red_chi2_hist(maps: list, unreg):
    #     n_col = len(maps)
    #     fig, ax = plt.subplots(2,n_col)
    #     for i in range(n_col):
    #         ax[0, i].hist(maps[i].ravel())
    #         ax[0, i].hist(unreg.reduced_chi2.ravel(), range = (0.95, 1.5))
            # fig.colorbar(im, ax = ax[:])

#%%
    def plot_red_chi2_map(maps: list, unreg, list_reg=None):
        n_col = len(maps)
        fig, ax = plt.subplots(1,n_col)
        for i in range(n_col):
            im = ax[i].imshow(maps[i],
                           vmin = 1 - .05, vmax = 1 + .05,
                           origin = 'lower')
            if list_reg:
                ax[i].set_title(list_reg[i])
        cbar = plt.colorbar(im, ax = ax[:])
        cbar.set_label(label = '$\chi_{red}^2$')

    plot_red_chi2_map(maps = [comp0d5.data, comp0d4.data, comp0d3.data,
                              comp0d2.data])
