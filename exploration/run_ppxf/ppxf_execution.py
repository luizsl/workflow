#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:54:40 2021

@author: Luiz
"""
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter as clock

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
        print('pPXF execution started')
        self.main_meta = metadata
        self.run_all_data(data)
        print('pPXF execution completed')

    def run_all_data(self, data=None, par=[]):
        assert data is not None

        # keep start time
        self.meta['ppxf_start_time'] = datetime.now().strftime("%d/%m/%Y %H:%M")

        par = ['gas_reddening', 'reddening', 'status', 'gas_flux', 'gas_any', 
               'gas_flux_error', 'gas_bestfit', 'phot_npix', 'gas_any_zero', 
               'weights', 'bestfit','mpoly', 'gas_mpoly', 'dof', 'chi2', 
               'sol', 'error', 'polyweights', 'apoly','goodpixels']
        
        # NOTE: Saving output unforeseen
        new_par = self.main_meta['output']['to_save']
        par = list(set(par) | set(new_par))
 
        storage = False
        size = data.obs.flux_grid[0, ...].size
        
        if data.obs.flux_grid.ndim==1:
            # NOBUG: Adding an exception to deal with a single spectrum
            # not neat but shoul work. <>
            data.obs.flux_grid = np.expand_dims(data.obs.flux_grid, axis=1)
            data.obs.flux_grid_unc = np.expand_dims(data.obs.flux_grid_unc, axis=1)
            
        for i in range(size):
            print(70*'*', end='\n\n')
            print(f'{i+1}/{size}', end='\n\n')
            
            flux_obs_slice = data.obs.flux_grid[:, i]
            flux_obs_unc_slice = data.obs.flux_grid_unc[:, i]
            if np.any(np.isnan(flux_obs_unc_slice) | np.isnan(flux_obs_slice)):
                print('nan')
            else:
                guess_goodpixels = data.obs.meta['guess_goodpixels']
                fixed_goodpixels = data.obs.meta['fixed_goodpixels']
                
                pp = self.execute_ppxf(obs=data.obs, model=data.model,index=i,
                                       goodpixels=guess_goodpixels,
                                       fixed_goodpixels = fixed_goodpixels,
                                       pp=None, conf=self.main_meta['ppxf'])
                
                if 'ppxf_dynamical_mask' in self.main_meta:
                    print('\n*************')
                    print('Calling refit with new spectral mask', end='\n\n')
                    # Determine actual goodpixels
                    goodpixels = self.clip_outliers(
                        pp.galaxy, pp.bestfit, pp.goodpixels, fixed_goodpixels,
                        **self.main_meta['ppxf_refit']['mask'])
                    
                    pp = self.execute_ppxf(
                        obs=data.obs, model=data.model,index=i,
                        goodpixels=goodpixels,
                        fixed_goodpixels=fixed_goodpixels,
                        pp=pp, conf=self.main_meta['ppxf_dynamical_mask'])
                    
                if 'ppxf_fixed_kinematics' in self.main_meta:
                    print('\n*************')
                    print('Calling refit with fixed kinematics', end='\n\n')
                    pp = self.execute_ppxf(
                        obs=data.obs, model=data.model,index=i,
                        goodpixels=goodpixels,
                        fixed_goodpixels=fixed_goodpixels,
                        pp=pp, conf=self.main_meta['ppxf_fixed_kinematics'])
                    
                if not storage:
                    n_obj = data.obs.flux_grid.shape[-1]
                    self.build_output_storage(par=par, out_obj=pp, n_obj=n_obj)
                    storage = True
                self.store_output(par=par, out_obj=pp, index=i)
            print(70*'*', end='\n\n')
            
        # keep end time
        self.meta['ppxf_end_time'] = datetime.now().strftime("%d/%m/%Y %H:%M")

    def execute_ppxf(self, obs=None, model=None, index=None, 
                     goodpixels=None, fixed_goodpixels=None,
                     pp=None, conf=None):
        assert conf is not None
        assert obs is not None
        assert model is not None

        t = clock()

        galaxy = obs.flux_grid[:, index]
        noise = obs.flux_grid_unc[:, index]

        star_template = model.flux_grid
        template = star_template

        frac = obs.meta['wave_obs'][1]/obs.meta['wave_obs'][0]
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
            template, galaxy, noise, velscale, start, lam=obs.meta['wave_obs'],
            lam_temp=model.meta['wave_model'], reg_dim=model.meta['reg_dim'],
            goodpixels=goodpixels, **conf)
        
        print('Elapsed time in PPXF: %.2f s' % (clock() - t))
        return pp
    
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

    def build_output_storage(self, par=[], out_obj=None, n_obj=None):
        assert 'ppxf' not in dir(self)
        assert n_obj and out_obj is not None

        self.ppxf = PpxfResults()

        for _p in par:
            assert _p in dir(out_obj), f"ppxf doesn't output {_p}"
            _obj = out_obj.__getattribute__(_p)

            if _obj is None:
                _shape = (n_obj,)
            elif isinstance(_obj, (float, int)):
                _shape = (n_obj,)
            elif _p == 'goodpixels':
            # goodpixels array has a variable size. It's trick to deal with 
            # this kind of object so I'm implementing a special case
                _aux = out_obj.__getattribute__('galaxy')
                _shape = _aux.shape + (n_obj,)
            else:
                _shape = _obj.shape + (n_obj,)

            with tempfile.NamedTemporaryFile() as temp_file:
                arr = np.memmap(temp_file, dtype = float, shape = _shape)
                arr.fill(np.nan)
                arr.flush()
                self.ppxf.__setattr__(_p, arr)

    def store_output(self, par=[], out_obj=None, index=None):
        assert 'ppxf' in dir(self), 'execute self.build_output_storage'
        assert out_obj and index is not None

        for _p in par:
            _obj = out_obj.__getattribute__(_p)
            try:
                self.ppxf.__getattribute__(_p)[..., index] = _obj
                self.ppxf.__getattribute__(_p).flush()
            except ValueError:
                shape = _obj.shape[0]
                self.ppxf.__getattribute__(_p)[..., :shape, index] = _obj
                self.ppxf.__getattribute__(_p).flush()
                
    def reconstruct_map(self, data=None, par=[], save=True):
        assert data is not None

        for _p in par:
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
    t.reconstruct_map(ppxf_prep.data, par = ['bestfit' , 'chi2'])
