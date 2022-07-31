# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:54:40 2021

@author: Luiz
"""
import sys
import io
import os
import tempfile
import logging
from logging import handlers
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter as clock

import extinction
import numpy as np
from astropy.io import fits
from ppxf.ppxf import ppxf, robust_sigma
from scipy.constants import physical_constants


@dataclass
class PpxfResults:
    pass


class ExecutePpxf:
    C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s
    meta = {}

    def __init__(self, data=None, metadata=None):
        assert data is not None
        assert metadata is not None
        
        self.data = data
        self.main_meta = metadata
        
        self.start_logging()
        
        # NOTE: Adding an exception to deal with a single spectrum
        # not neat but should work. <>
        if self.data.obs.flux_grid.ndim==1:
            self.data.obs.flux_grid = np.expand_dims(
                self.data.obs.flux_grid, axis=1)
            self.data.obs.flux_grid_unc = np.expand_dims(
                self.data.obs.flux_grid_unc, axis=1)

        self.storage = False
        self.size = self.data.obs.flux_grid[0, ...].size
        
        par = ['gas_reddening', 'reddening', 'status', 'gas_flux', 'gas_any',
               'gas_flux_error', 'gas_bestfit', 'phot_npix', 'gas_any_zero',
               'weights', 'bestfit','mpoly', 'gas_mpoly', 'dof', 'chi2',
               'sol', 'error', 'polyweights', 'apoly','goodpixels']

        # NOTE: Saving output unforeseen
        new_par = self.main_meta['output']['to_save']
        self.par = list(set(par) | set(new_par))

        self.run_all_data()
        
    def start_logging(self):
        name_log_file = os.path.join(
            self.main_meta['output_run_ppxf'],
            'log_ppxf_execution.log')
        
        formatter = logging.Formatter('%(message)s')
        loglevel = logging.INFO
        
        file_handler = logging.FileHandler(name_log_file )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(loglevel)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(loglevel)
        
        memory_handler = handlers.MemoryHandler(capacity=1024*1000,
                                                target=stream_handler)
        memory_handler.setFormatter(formatter)
        memory_handler.setLevel(loglevel)
        self.memory_handler = memory_handler
        
        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(loglevel)
        logger.addHandler(self.memory_handler)
        logger.addHandler(file_handler)
        
        self.logger = logger
        
    def run_all_data(self):
        self.logger.info('pPXF execution started')

        # keep start time
        self.meta['ppxf_start_time'] = datetime.now().strftime("%d/%m/%Y %H:%M")

        for i in range(self.size):
            pp = self.worker(i)
            if pp is not None:
                if not self.storage:
                    self.build_output_storage(out_obj=pp)
                self.store_output(out_obj=pp, index=i)

        # keep end time
        self.meta['ppxf_end_time'] = datetime.now().strftime("%d/%m/%Y %H:%M")

        self.logger.info('pPXF execution completed')

    def worker(self, i):
        stdout = sys.stdout
        sys.stdout = io.StringIO()

        print(70*'*', end='\n\n')
        print(f'{i+1}/{self.size}', end='\n\n')

        flux_obs_slice = self.data.obs.flux_grid[:, i]
        flux_obs_unc_slice = self.data.obs.flux_grid_unc[:, i]
        if np.any(np.isnan(flux_obs_unc_slice) | np.isnan(flux_obs_slice)):
            return None

        pp = None
        guess_goodpixels = self.data.obs.meta['guess_goodpixels']
        fixed_goodpixels = self.data.obs.meta['fixed_goodpixels']

        pp = self.execute_ppxf(
            galaxy = flux_obs_slice, noise = flux_obs_unc_slice,
            goodpixels=guess_goodpixels,
            fixed_goodpixels = fixed_goodpixels,
            pp=pp, conf=self.main_meta['ppxf'])
        print('*************', end='\n\n')

        if 'ppxf_dynamical_mask' in self.main_meta:
            print('Calling refit with new spectral mask', end='\n\n')
            # Determine actual goodpixels
            goodpixels = self.clip_outliers(
                pp.galaxy, pp.bestfit, pp.goodpixels, fixed_goodpixels,
                **self.main_meta['ppxf_refit']['mask'])

            pp = self.execute_ppxf(
                galaxy = flux_obs_slice, noise = flux_obs_unc_slice,
                goodpixels=goodpixels,
                fixed_goodpixels=fixed_goodpixels,
                pp=pp, conf=self.main_meta['ppxf_dynamical_mask'])
            print('*************', end='\n\n')

        if 'ppxf_fit_reddening' in self.main_meta:
            print('Calling fit of reddening', end='\n\n')
            pp = self.execute_ppxf(
                galaxy = flux_obs_slice, noise = flux_obs_unc_slice,
                goodpixels=goodpixels,
                fixed_goodpixels=fixed_goodpixels,
                pp=pp, conf=self.main_meta['ppxf_fit_reddening'])
            fly_reddening = pp.reddening

            print('\nDered observation on the fly')
            flux_obs_slice = self.dered(
                flux_obs_slice,
                wave=self.data.obs.meta['wave_obs'],
                ebv = fly_reddening)
            flux_obs_unc_slice = self.dered(
                flux_obs_unc_slice,
                wave=self.data.obs.meta['wave_obs'],
                ebv = fly_reddening)
            print('*************', end='\n\n')

        if 'ppxf_regularization' in self.main_meta:
            print('Calling refit with regulazired solution', end='\n\n')
            pp = self.execute_ppxf(
                galaxy = flux_obs_slice, noise = flux_obs_unc_slice,
                goodpixels=goodpixels,
                fixed_goodpixels=fixed_goodpixels,
                pp=pp, conf=self.main_meta['ppxf_regularization'])
            print(70*'*', end='\n\n')

        # Include reddening fitted on the fly if exists
        if fly_reddening:
            pp.reddening = fly_reddening
        
        output = sys.stdout.getvalue()
        sys.stdout = stdout
        
        self.logger.info(output)
        self.memory_handler.flush()
        return pp

    def execute_ppxf(self,
                     galaxy=None, noise=None,
                     goodpixels=None, fixed_goodpixels=None,
                     pp=None, conf=None):
        assert conf is not None
        assert galaxy is not None
        assert noise is not None

        t = clock()

        star_template = self.data.model.flux_grid
        template = star_template

        frac = self.data.obs.meta['wave_obs'][1]/self.data.obs.meta['wave_obs'][0]
        velscale = np.log(frac)*self.C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)

        if pp is None:
            start = [0., 2*velscale] # (km/s), starting guess for [V, sigma]
        else:
            start = pp.sol

        if goodpixels is None:
            goodpixels = np.arange(galaxy.size)
        if fixed_goodpixels is None:
            fixed_goodpixels = np.arange(galaxy.size)

        goodpixels = np.intersect1d(goodpixels, fixed_goodpixels)

        pp = ppxf(
            template, galaxy, noise, velscale, start,
            lam=self.data.obs.meta['wave_obs'],
            lam_temp=self.data.model.meta['wave_model'],
            reg_dim=self.data.model.meta['reg_dim'],
            goodpixels=goodpixels, **conf)

        print('Elapsed time in PPXF: %.2f s' % (clock() - t))
        return pp

    @staticmethod
    def dered(spectrum, wave=None, law='calzetti00', r_v=4.05, ebv=None):
        assert ebv is not None
        assert wave is not None
        a_v = ebv * r_v

        if law == 'fm07':
            ext_mag = extinction.__getattribute__(law)(wave=wave, a_v=a_v)
        else:
            ext_mag = extinction.__getattribute__(law)(wave=wave, a_v=a_v, r_v=r_v)

        dered_spectrum = extinction.remove(ext_mag, spectrum)

        return dered_spectrum

    @staticmethod
    def clip_outliers(galaxy, bestfit, goodpixels, fixed_goodpixels=None,
                      sigma=3):
        """
        Adapted from Michele Cappellari's example

        Repeat the fit after clipping bins deviants more than 3*sigma
        in relative error until the bad bins don't change any more.
        """
        if fixed_goodpixels is None:
            fixed_goodpixels = np.arange(galaxy.size)
        while True:
            goodpixels = np.intersect1d(goodpixels, fixed_goodpixels)
            scale = galaxy[goodpixels] @ bestfit[goodpixels]/np.sum(bestfit[goodpixels]**2)
            resid = scale*bestfit[goodpixels] - galaxy[goodpixels]
            err = robust_sigma(resid, zero=1)
            ok_old = goodpixels.copy()
            goodpixels = np.flatnonzero(np.abs(bestfit - galaxy) < sigma*err)
            goodpixels = np.intersect1d(goodpixels, fixed_goodpixels)
            if np.array_equal(goodpixels, ok_old):
                break
        return goodpixels

    def build_output_storage(self, out_obj=None):
        assert 'ppxf' not in dir(self)
        assert out_obj is not None

        self.ppxf = PpxfResults()
        n_obj = self.data.obs.flux_grid.shape[-1]

        for _p in self.par:
            assert _p in dir(out_obj), f"ppxf doesn't output {_p}"
            _obj = out_obj.__getattribute__(_p)

            if _obj is None:
                _shape = (n_obj,)
            elif isinstance(_obj, (float, int)):
                _shape = (n_obj,)
            elif _p == 'goodpixels':
            # NOTE: goodpixels array has a variable size. It's trick to deal with
            # this kind of object so I'm implementing a special case. <>
                _aux = out_obj.__getattribute__('galaxy')
                _shape = _aux.shape + (n_obj,)
            else:
                _shape = _obj.shape + (n_obj,)

            with tempfile.NamedTemporaryFile() as temp_file:
                arr = np.memmap(temp_file, dtype = float, shape = _shape)
                arr.fill(np.nan)
                arr.flush()
                self.ppxf.__setattr__(_p, arr)

        self.storage = True

    def store_output(self, out_obj=None, index=None):
        assert 'ppxf' in dir(self), 'execute self.build_output_storage'
        assert out_obj and index is not None

        for _p in self.par:
            _obj = out_obj.__getattribute__(_p)
            try:
                self.ppxf.__getattribute__(_p)[..., index] = _obj
                self.ppxf.__getattribute__(_p).flush()
            except ValueError:
                shape = _obj.shape[0]
                self.ppxf.__getattribute__(_p)[..., :shape, index] = _obj
                self.ppxf.__getattribute__(_p).flush()

    def reconstruct_map(self, data=None, parameter=[], save=True):
        assert data is not None

        for _p in parameter:
            if self.main_meta['vorbin']['apply']:
                if self.ppxf.__getattribute__(_p).ndim < 2:
                    map_shape = data.obs.meta['bin_num'].shape
                if self.ppxf.__getattribute__(_p).ndim >= 2:
                    map_shape = (self.ppxf.__getattribute__(_p).shape[:1]
                                 + data.obs.meta['bin_num'].shape)
                map_ = np.zeros(map_shape)
                for i in range(self.ppxf.__getattribute__(_p).shape[-1]):
                    match = data.obs.meta['bin_num'] == i
                    map_[..., match] = self.ppxf.__getattribute__(_p)[..., i:i+1]

                map_shape_full = np.array(data.obs.meta['shape_obs']).prod()
                if self.ppxf.__getattribute__(_p).ndim >= 2:
                    map_shape_full = (
                        self.ppxf.__getattribute__(_p).shape[:1]
                        + (map_shape_full,))

                map_full = np.full(map_shape_full, fill_value=np.nan)
                valid = data.obs.meta['valid']
                map_full[..., valid] = map_

                if map_full.ndim < 2:
                    new_shape = (data.obs.meta['shape_obs'])
                elif map_full.ndim >= 2:
                    new_shape = (-1,) + data.obs.meta['shape_obs']
                map_full = map_full.reshape(new_shape)

            else:
                map_shape_full = data.obs.meta['shape_obs']
                if self.ppxf.__getattribute__(_p).ndim >= 2:
                    map_shape_full = (
                        self.ppxf.__getattribute__(_p).shape[:1]
                        + map_shape_full)
                map_full = self.ppxf.__getattribute__(_p).reshape(map_shape_full)

            if save:
                if self.main_meta:
                    directory = self.main_meta['output_run_ppxf']
                self.save_fits(map_full, _p, directory)

    @staticmethod
    def save_fits(data_param, name, directory='.', overwrite=True):
        hdu = fits.PrimaryHDU(data=data_param)
        hdul = fits.HDUList([hdu])
        full_path = os.path.join(directory, f'{name}.fits')
        hdul.writeto(full_path, overwrite=overwrite)

if __name__ == '__main__':
    t = ExecutePpxf(ppxf_prep.data, ppxf_prep.data.main_meta)
    t.reconstruct_map(ppxf_prep.data, parameter = ['bestfit' , 'chi2'])
