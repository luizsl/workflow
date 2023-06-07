#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 08:17:19 2022

@author: Luiz
"""
import os
import json
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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
    def __init__(self, filepath_reduced_chi2, filepath_dof, filepath_grid,
                 filepath_meta):
        self.filepath_reduced_chi2 = filepath_reduced_chi2
        self.filepath_meta = filepath_meta
        self.filepath_dof = filepath_dof
        self.filepath_grid = filepath_grid
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

    def read_grid(self):
        with (fits.open(self.filepath_grid) as hdul,
              open(self.filepath_meta) as f):
            data = hdul[0].data
            meta = json.load(f)

            return data, meta

    @property
    def chi2(self):
        return self.reduced_chi2 * self.dof

    @property
    def delta_chi2(self):
        target_chi2 = self.chi2 + np.sqrt(2 * self.dof)
        return target_chi2

    def weights(self, i, j):
        data, meta = self.read_grid()
        weights = data[:, i, j].reshape(meta['model']['reg_dim'])
        return weights


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

def read_fits_data(file, unit=0):
    with fits.open(file) as hdul:
        data = hdul[unit].data
        return np.asarray(data)

def find_regul(regul,
               root_directory=None,
               filename_pattern=None):
    name = filename_pattern.replace('[regul]', str(regul))
    # print(name)
    for path in Path(root_directory).rglob(name):
        return path.parent.as_posix()

def plot_percentage(maps:list, list_reg:list=None, save_title=None):
    plt.style.use('../src/fig_conf.mplstyle')
    n_col = len(maps)
    fig, ax = plt.subplots(1, n_col, figsize=(12,2.5), constrained_layout=True)
    for i in range(n_col):
        im = ax[i].imshow(maps[i], cmap='seismic',
                          vmin = 1 - .13, vmax = 1 + .13,
                          origin = 'lower')
        if list_reg[i]:
            ax[i].set_title(f'regul = $1/{list_reg[i]}$', loc='left')
            ax[i].tick_params(labelleft=False, labeltop=False,
                              labelright=False, labelbottom=False)
            # ax[i].text(0, -.05, 'left top',
            #         horizontalalignment='left',
            #         verticalalignment='top',
            #         transform=ax[i].transAxes)
    cbar = fig.colorbar(im, ax=ax[:], use_gridspec=True,  pad=0.01)
    cbar.set_label(r'$ \dfrac{\chi^{2}_{\rm r}}{\chi^{2}_{\rm u} + \Delta \chi^{2}}$')

    # plt.savefig(f'../plots/regularization/regul_param_{save_title}.pdf')

def plot_grid_param(maps: list, x, y, save_title=None):
    plt.style.use('../src/fig_conf.mplstyle')
    n_col = len(maps)
    fig, ax = plt.subplots(1, n_col, figsize=(12,2.5), constrained_layout=True, sharex=True, sharey=True)
    for i in range(n_col):
        _, meta = maps[i].read_grid()
        weights = maps[i].weights(i=x, j=y)
        x_age, y_mh = np.meshgrid(np.array(meta['model']['age_range']).round(2),
                                  meta['model']['mh_range'])
        grid = ax[i].pcolormesh(x_age, y_mh, weights.T, vmin=0, vmax=0.25)
        points = ax[i].scatter(x_age, y_mh, marker='.', color='white', s=1)
        ax[i].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax[i].tick_params(axis='x', rotation=90)
        ax[i].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    cbar = fig.colorbar(grid, ax = ax[:], use_gridspec=True,  pad = 0.01)
    cbar.set_label(r'weights')
    fig.supxlabel('$\log_{10}$ Age (yr)')
    ax[0].set_ylabel('[Fe/H]')

    # plt.savefig(f'../plots/regularization/grid_param_coord_{save_title}.pdf')

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

#%%

if __name__ == "__main__":
    pass


    # unreg2d50_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d5/ppxf/chi2.fits'
    # unreg2d50_filepath_dof = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d5/ppxf/goodpixels.fits'

    # unreg2d30_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d3/ppxf/chi2.fits'
    # unreg2d30_filepath_dof = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d3/ppxf/goodpixels.fits'

    # unreg2d00_filepath_reduced_chi2 = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d0/ppxf/chi2.fits'
    # unreg2d00_filepath_dof = '../../data_products/regularization_parameter/MilesAgeMh_unreg_sigma2d0/ppxf/goodpixels.fits'

    # unreg0d00 = Chi2Map(unreg0d00_filepath_reduced_chi2, unreg0d00_filepath_dof)
    # unreg3d00 = Chi2Map(unreg3d00_filepath_reduced_chi2, unreg3d00_filepath_dof)
    # unreg2d50 = Chi2Map(unreg2d50_filepath_reduced_chi2, unreg2d50_filepath_dof)
    # unreg2d30 = Chi2Map(unreg2d30_filepath_reduced_chi2, unreg2d30_filepath_dof)
    # unreg2d00 = Chi2Map(unreg2d00_filepath_reduced_chi2, unreg2d00_filepath_dof)

    # def plot_red_chi2_hist_residuals(maps: list, unreg=None, list_reg: list=None,
    #                                  bins = 10):
    #     n_col = len(maps)
    #     fig, ax = plt.subplots(1,n_col, figsize=(12,3), constrained_layout=True)
    #     for i in range(n_col):
    #         data = maps[i].ravel()
    #         median = np.nanmedian(data)
    #         ax[i].hist(data, bins=bins, range=(0.5,1.5))
    #         ax[i].axvline(median, color = 'k')
    #         ax[i].set_xlabel(r'$\chi_{\nu}^{2}$')
    #         if list_reg[i]:
    #             ax[i].set_title(f'$\sigma = {list_reg[i]}$', loc = 'left')
    #         if unreg:
    #             ax[i].hist(unreg.reduced_chi2.ravel(), range = (0.95, 1.5))
    #     fig.suptitle(r'Effect of $\sigma_{\rm res}$-based dynamic masking on the reduced $\chi^{2}$ for NGC 613')
    #     fig.supylabel('N')

    # plot_red_chi2_hist_residuals(
    #     maps = [unreg0d00.reduced_chi2, unreg3d00.reduced_chi2,
    #             unreg2d50.reduced_chi2, unreg2d30.reduced_chi2,
    #             unreg2d00.reduced_chi2],
    #     list_reg=[None, 3.0, 2.5, 2.3, 2.0],
    #     bins = 20)

    #%%     Increase of the chi**2 due to regularization
    filename_pattern = 'sn100_regul[regul]_fov1x3_norm_obs.yaml'
    # root_directory = '../data_products/regularization/sn100_fov_sample_1_3'
    root_directory = '../data_products/regularization/sn100_fov_sample_1_3_goodpixels'
    # root_directory = '../data_products/regularization/sn40_fov_sample_1_3'

    reguls = [0, 0.05, 0.04, 0.03, 0.02, 0.01]
    for i, regul in enumerate(reguls):
        regul = round(regul, 2)
        regul= f'{regul:0.2f}'
        regul= regul.replace('.', 'd')
        reguls[i] = regul

    # Read bestfit of all tests
    for regul in reguls:
        directory = find_regul(regul, root_directory, filename_pattern=filename_pattern)
        # print(directory)
        filename_rchi2 = os.path.join(directory, 'chi2.fits')
        filename_dof = os.path.join(directory, 'goodpixels.fits')
        filename_grid = os.path.join(directory, 'weights.fits')
        filename_meta = os.path.join(directory, 'metadata.json')

        print(filename_rchi2, filename_dof, filename_grid, filename_meta,
              sep='\n', end='\n\n')
        locals()[f'regul{regul}_rchi2_filepath'] = filename_rchi2
        locals()[f'regul{regul}_dof_filepath'] = filename_dof
        locals()[f'regul{regul}_grid_filepath'] = filename_grid
        locals()[f'regul{regul}_meta_filepath'] = filename_meta

        locals()[f'regul{regul}_chi2map'] = Chi2Map(
            locals()[f'regul{regul}_rchi2_filepath'],
            locals()[f'regul{regul}_dof_filepath'],
            locals()[f'regul{regul}_grid_filepath'],
            locals()[f'regul{regul}_meta_filepath'])

    for regul in reguls:
        locals()[f'comp{regul}'] = Chi2Comparison(
            reg=locals()[f'regul{regul}_chi2map'], unreg=locals()['regul0d00_chi2map'])

    #%% Increase of chi**2 with regularization

    plot_percentage(maps=[comp0d05.data, comp0d04.data, comp0d03.data,
                          comp0d02.data, comp0d01.data],
                    list_reg=[0.05, 0.04, 0.03, 0.02, 0.01],
                    save_title='sigma2d5')

    #%%

    plot_grid_param(maps=[regul0d00_chi2map, regul0d50_chi2map, regul0d40_chi2map,
                          regul0d30_chi2map, regul0d20_chi2map, regul0d10_chi2map,
                          regul0d05_chi2map],
                    x=25, y=20,
                    save_title='x25_y20')

    plot_grid_param(maps=[regul0d00_chi2map, regul0d50_chi2map, regul0d40_chi2map,
                          regul0d30_chi2map, regul0d20_chi2map, regul0d10_chi2map,
                          regul0d05_chi2map],
                    x=33, y=53,
                    save_title='x33_y53')

    plot_grid_param(maps=[regul0d00_chi2map, regul0d50_chi2map, regul0d40_chi2map,
                          regul0d30_chi2map, regul0d20_chi2map, regul0d10_chi2map,
                          regul0d05_chi2map],
                    x=32, y=33,
                    save_title='x32_y33')

    plot_grid_param(maps=[regul0d00_chi2map, regul0d50_chi2map, regul0d40_chi2map,
                          regul0d30_chi2map, regul0d20_chi2map, regul0d10_chi2map,
                          regul0d05_chi2map],
                    x=37, y=36,
                    save_title='x37_y36')


    # plot_red_chi2_map(maps = [comp0d50.data, comp0d40.data, comp0d30.data,
    #                           comp0d20.data])