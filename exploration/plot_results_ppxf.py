#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:56:56 2021

@author: Luiz
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from matplotlib import cm
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm

class PlotMap:

    def __init__(self, object_name):
        plt.style.use('fig_conf.mplstyle')
        self.dir = f'../data_products/{object_name}/miles/ppxf/'
        self.object_name = object_name
        self.get_header()
        
    def get_header(self):
        # file_path = os.path.join(self.dir, 'header' + '.fits')
        # file_path = '../../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'
        # with fits.open(file_path) as hdu:
        #     self.header = hdu[0].header
            
        file_path = '../data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'
        with fits.open(file_path) as hdu:
            self.header = hdu[1].header

    def get_data(self, param):
        # file_path = os.path.join(self.dir, param + '.fits')
        file_path = f"{self.dir}{param}.fits"
        hdu = fits.open(file_path)
        return hdu
      
    def save_figure(self, fig, param, add_name, ext = '.pdf'):
        if add_name is not None:
            fname = f'{self.object_name}_{param}_{add_name}{ext}'
        else:
            fname = f'{self.object_name}_{param}{ext}'
        plt.savefig(fname)

    def _plot_map(self, param_name, data, wcs, vmin, vmax, n_tick, unit, save = False,
                     get = False, add_name = None):
    
        ax = plt.subplot(projection=wcs[0])
        im = ax.imshow(data, vmin = vmin, vmax = vmax,
                       cmap = 'jet', origin = 'lower')
        
        cbar = plt.colorbar(im)
        ticks = np.linspace(vmin, vmax, n_tick, dtype = float)
        cbar.set_ticks(ticks)
        ticks = [f'$\\leqslant$ {str(round(vmin,2))}'] \
                + np.round(ticks[1:-1], 2).tolist() \
                + [f'$\\geqslant$ {str(round(vmax,2))}']
        cbar.ax.set_yticklabels(ticks, ha = 'center')
        cbar.ax.yaxis.set_tick_params(pad = 20)  # your number may vary
        if unit:
            cbar.set_label(r'$[\km\,\s^{-1}]$')

        ax.grid(color = 'white', ls = 'dotted')
        
        ax.set_xlabel(r'$\textbf{Right Ascension (J2000)}$')
        ax.set_ylabel(r'$\textbf{Declination (J2000)}$')
        ax.set_title(f'$\\textbf{{ {param_name} }}$')
        
        if add_name is not None:
            add_name = '_'.join(add_name)
            
        if save:
            current_fig = plt.gcf()
            self.save_figure(fig = current_fig,
                             param = param_name, add_name = add_name)
            
        if get:
            return plt.gcf()
    
    def moment(self, param_name, vmin, vmax, n_tick, unit, data_offset = None,
                save = False, get = False, add_name = None):
        # gather data
        assert param_name is not None
        
        wcs = WCS(self.header)
        hdu = self.get_data(param_name)
        data = np.array(hdu[0].data)
        
        if data_offset is not None:
            data = np.array(hdu[0].data) - data_offset
            
        if get:
            figure = self._plot_map(param_name = param_name,
                                       data= data, wcs = wcs, vmin = vmin,
                                       vmax = vmax, n_tick = n_tick, 
                                       unit = unit, save = save, get = get,
                                       add_name = add_name)
            return figure
        else:
            self._plot_map(param_name = param_name, data= data, wcs = wcs,
                              vmin = vmin, vmax = vmax, n_tick = n_tick, 
                              unit = unit, save = save, get = get,
                              add_name = add_name)

    def moment_rel_error(self, param_name, vmin, vmax, n_tick, unit,
                     data_offset = None, save = False, get = False,
                     add_name = None):
        # gather data
        assert param_name is not None
        
        wcs = WCS(self.header)
        hdu = self.get_data(param_name)
        data = np.array(hdu[0].data)
        
        hdu_chi2 = self.get_data('chi2')
        chi2 = np.array(hdu_chi2[0].data)
        
        data_error = data * np.sqrt(chi2)
        
        rel_error = data/data_error
        
        # if data_offset is not None:
        #     data = np.array(hdu[0].data) - data_offset
            
        if get:
            figure = self._plot_map(param_name = param_name,
                                       data = rel_error, wcs = wcs, vmin = vmin,
                                       vmax = vmax, n_tick = n_tick, 
                                       unit = unit, save = save, get = get,
                                       add_name = add_name)
            return figure
        else:
            self._plot_map(param_name = param_name, data = rel_error,
                              wcs = wcs, vmin = vmin, vmax = vmax, 
                              n_tick = n_tick, unit = unit, save = save,
                              get = get, add_name = add_name)
            
    def moment_abs_error(self, param_name, vmin, vmax, n_tick, unit,
                     data_offset = None, save = False, get = False,
                     add_name = None):
        # gather data
        assert param_name is not None
        
        wcs = WCS(self.header)
        hdu = self.get_data(param_name)
        data = np.array(hdu[0].data)
        
        hdu_chi2 = self.get_data('chi2')
        chi2 = np.array(hdu_chi2[0].data)
        
        data_error = data * np.sqrt(chi2)
        
        # if data_offset is not None:
        #     data = np.array(hdu[0].data) - data_offset
            
        if get:
            figure = self._plot_map(param_name = param_name,
                                       data = data_error, wcs = wcs, vmin = vmin,
                                       vmax = vmax, n_tick = n_tick, 
                                       unit = unit, save = save, get = get,
                                       add_name = add_name)
            return figure
        else:
            self._plot_map(param_name = param_name, data = data_error,
                              wcs = wcs, vmin = vmin, vmax = vmax, 
                              n_tick = n_tick, unit = unit, save = save,
                              get = get, add_name = add_name)
            
    def moment_masked(self, param_name, limit, vmin, vmax, n_tick, unit,
                      data_offset = None, save = False, get = False, 
                      add_name = None):
        # gather data
        assert param_name is not None
        
        wcs = WCS(self.header)
        hdu = self.get_data(param_name)
        data = np.array(hdu[0].data)
        
        if data_offset is not None:
            data = np.array(hdu[0].data) - data_offset
        
        mask = self.get_mask_w_velocity(limit = limit)
        
        data = np.where(mask == False, data, np.nan)
        
        self._plot_map(param_name = param_name, data = data, wcs = wcs,
                          vmin = vmin, vmax = vmax, n_tick = n_tick, 
                          unit = unit, save = save)
            
        if get:
            figure = self._plot_map(param_name = param_name,
                                       data= data, wcs = wcs, vmin = vmin,
                                       vmax = vmax, n_tick = n_tick, 
                                       unit = unit, save = save, get = get,
                                       add_name = add_name)
            return figure
        else:
            self._plot_map(param_name = param_name, data= data, wcs = wcs,
                              vmin = vmin, vmax = vmax, n_tick = n_tick, 
                              unit = unit, save = save, get = get,
                              add_name = add_name)
        
    # def reduced_chi2(selfparam_name, limit, vmin, vmax, n_tick, unit,
    #                   data_offset = None, save = False, get = False, 
    #                   add_name = None):
    #     # gather data
    #     assert param_name is not None
        
    #     wcs = WCS(self.header)
    #     hdu = self.get_data(param_name)
    #     data = np.array(hdu[0].data)
        
    #     hdu_chi2 = self.get_data('chi2')
    #     chi2 = np.array(hdu_chi2[0].data)
        
    #     if get:
    #         figure = self._plot_map(param_name = chi2,
    #                                    data= chi2, wcs = wcs, vmin = vmin,
    #                                    vmax = vmax, n_tick = n_tick, 
    #                                    unit = unit, save = save, get = get,
    #                                    add_name = add_name)
    #         return figure
    #     else:
    #         self._plot_map(param_name = chi2, data= chi2, wcs = wcs,
    #                           vmin = vmin, vmax = vmax, n_tick = n_tick, 
    #                           unit = unit, save = save, get = get,
    #                           add_name = add_name)

    def get_mask_w_velocity(self, limit = None, data_offset = None):
        
        hdu_vel = self.get_data('velocity')
        data_vel = np.array(hdu_vel[0].data)
        
        hdu_vel_error = self.get_data('velocity_error')
        data_vel_error = np.array(hdu_vel_error[0].data)
        
        hdu_chi2 = self.get_data('chi2')
        chi2 = np.array(hdu_chi2[0].data)

        if data_offset is None:
            pass
        else:
            data_vel = data_vel - data_offset 
            
        dv = data_vel_error * np.sqrt(chi2)
        
        # mask data with large uncertainty
        if limit is None:
            limit = 0

        mask = np.where(dv < limit, False, True)

        return mask
    
if __name__ == "__main__":
    ngc613_map = PlotMap('NGC613')

    # ngc613_map.moment('velocity', vmin = -120., vmax = 120., n_tick = 7,
    #                   data_offset = 1480, unit = True, save = True)
    # ngc613_map.moment('sigma', vmin = 50., vmax = 140., n_tick = 7,
    #                   unit = True, save = True)
    # ngc613_map.moment('h3', vmin = -0.13, vmax = 0.13, n_tick = 7, save = True,
    #                   unit = False)
    # ngc613_map.moment('h4', vmin = -0.14, vmax = 0.15, n_tick = 7,
    #                   unit = False, save = True)
    
    # ngc613_map.get_mask_w_velocity(limit = 20, data_offset=1480)
    
    ngc613_map.moment_masked('velocity', limit = 20, vmin = -120., vmax = 120.,
                              n_tick = 7, data_offset = 1480, unit = True, 
                              save = True, add_name = ['masked', 'v20'])
    ngc613_map.moment_masked('sigma', limit = 20, vmin = 50., vmax = 140., 
                              n_tick = 7, unit = True, save = True,
                              add_name = ['masked', 'v20'])
    ngc613_map.moment_masked('h3', limit = 20, vmin = -0.13, vmax = 0.13,
                              n_tick = 7, unit = False, save = True,
                              add_name = ['masked', 'v20'])
    ngc613_map.moment_masked('h4', limit = 20, vmin = -0.14, vmax = 0.15,
                              n_tick = 7, unit = False, save = True,
                              add_name = ['masked', 'v20'])
    
    # ngc613_map.moment_rel_error('h4', vmin = 0, vmax = 60, n_tick = 7,
    #                   unit = False, save = True)
    
    # ngc613_map.moment_abs_error('velocity', vmin = 0, vmax = 100, n_tick = 7,unit = False, save = True)

    # ngc613_map.moment('chi2', vmin = 0., vmax = 0.01, n_tick = 7, unit = True, save = False)

# from scipy.ndimage.filters import gaussian_filter

# smooth = gaussian_filter(data_vel_mask, 3)
# contour = plt.contour(smooth, 10, colors = 'black')
# plt.clabel(contour, inline=True, fontsize=10)

# hdu = fits.open('./NGC613_1/velocity_error.fits')
# data = hdu[0].data

# im = plt.imshow(data - 1480, vmin = -120, vmax = 120, cmap = 'Spectral', origin = 'lower')

#%%
# data = hdu[0].data
# data_hist = data.ravel()
#  # - 1473.24
# frac = 99
# lower_limit = np.nanquantile(data_hist, 0.01)
# upper_limit = np.nanquantile(data_hist, 0.999)
# data_hist = np.where(data_hist < upper_limit, data_hist, np.nan)
# data_hist = np.where(lower_limit < data_hist, data_hist, np.nan)
# plt.hist(data_hist - 1480, bins = 800)

# #%%
# data_map = np.where(data < upper_limit, data, np.nan)
# data_map = np.where(lower_limit < data_map, data_map, np.nan)
# im = plt.imshow(data_map - 1450, vmin = -121, vmax = 121, cmap = 'jet', origin = 'lower')
# cbar = plt.colorbar(im)

# np.nanquantile(data_hist, 0.05)
# np.nanquantile(data_hist, 0.5)
# np.nanquantile(data_hist, 0.99)
# data_hist = data.clip(data_hist, )

teste = np.array([1,2,3])
mas = np.array([True,False,True])

teste_out = np.where(mas == True, teste, np.nan)
