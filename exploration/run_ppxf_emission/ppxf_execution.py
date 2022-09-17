# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:54:40 2021

@author: Luiz
"""
import io
import logging
import multiprocessing as mp
import os
import pickle
import tempfile
from contextlib import redirect_stdout
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
        self.storage = False
        self.process_manager = mp.Manager()
        
        self.start_logging()
        # self.ppxf = PpxfResults()
        # NOTE: Adding an exception to deal with a single spectrum
        # not neat but should work. <>
        if self.data.obs.flux_grid.ndim==1:
            self.data.obs.flux_grid = np.expand_dims(
                self.data.obs.flux_grid, axis=1)
            self.data.obs.flux_grid_unc = np.expand_dims(
                self.data.obs.flux_grid_unc, axis=1)

        self.size = self.data.obs.flux_grid[0, ...].size

        # par = ['gas_reddening', 'reddening', 'status', 'gas_flux', 'gas_any',
        #        'gas_flux_error', 'gas_bestfit', 'phot_npix', 'gas_any_zero',
        #        'weights', 'bestfit','mpoly', 'gas_mpoly', 'dof', 'chi2',
        #        'sol', 'error', 'polyweights', 'apoly','goodpixels']
        
        par =[]
        
        # NOTE: Saving output unforeseen
        new_par = self.main_meta['output']['to_save']
        self.par = list(set(par) | set(new_par))

        self.run_all_data()
        
    def start_logging(self):
        name_log_file = os.path.join(
            self.main_meta['output_run'],
            'log_ppxf_execution.log')
        
        formatter = logging.Formatter('%(message)s')
        loglevel = logging.INFO
        
        file_handler = logging.FileHandler(name_log_file )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(loglevel)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(loglevel)

        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(loglevel)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        self.logger = logger
        
    def run_all_data(self):
        self.logger.info('pPXF execution started')

        # keep start time
        start_time = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.meta['ppxf_start_time'] = start_time
        self.logger.info(start_time)
        
        if 'n_process' in self.main_meta['common']:
            N_PROCESS = self.main_meta['common']['n_process']
        else:
            N_PROCESS = mp.cpu_count()
        
        input_queue = self.process_manager.Queue()
        output_queue = self.process_manager.Queue(maxsize=10)
        
        ps = [mp.Process(target=self.worker, args=[input_queue, output_queue]) 
              for _ in range(N_PROCESS)]

        for p in ps: 
            p.start()
            self.logger.debug('Start multiprocessing')
        for i in range(self.size):
            input_queue.put(i)
        for _ in range(N_PROCESS): 
            input_queue.put(None)
        
        # return_dict = self.process_manager.dict()
        p_out = mp.Process(target=self.store_output, args=[output_queue])
        p_out.start()
        
        for p in ps: 
            p.join()
            
        output_queue.put(None)
        p_out.join()
        
        # for p in ps:
        #     p.join()
            
        # keep end time
        end_time = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.meta['ppxf_end_time'] = end_time
        self.logger.info(end_time)
        
        self.logger.info('pPXF execution completed')

    def worker(self, input_queue, output_queue):
        for i in iter(input_queue.get, None):
            pp = None
            with redirect_stdout(io.StringIO()) as f:
                id_ = f'{i+1}/{self.size}'
                print(70*'*')
                
                galaxy = self.data.obs.flux_grid[:, i]
                noise = self.data.obs.flux_grid_unc[:, i]
                stellar = self.data.stellar.flux_grid[:, i]
                goodpixels = self.data.obs.meta['fixed_goodpixels']

                if np.any(np.isnan(galaxy) | np.isnan(noise) | np.isnan(stellar)):
                    print('nan')
                    pack = [i, pp]
                    data_out = pickle.dumps(pack)
                    continue
                
                if 'ppxf_emission_fit' in self.main_meta:
                    print(id_, 'Fitting emission', end='\n\n')
                    pp = self.execute_ppxf(
                        galaxy=galaxy, noise=noise,
                        stellar=stellar,
                        index=i,
                        goodpixels=goodpixels,
                        pp=pp, conf=self.main_meta['ppxf_emission_fit'])

                out_log = f.getvalue()
                
            self.logger.info(out_log)
            pack = [i, pp]
            data_out = pickle.dumps(pack)
            output_queue.put(data_out)
            # print(output_queue.qsize())
            
    def execute_ppxf(self,
                     galaxy=None, noise=None, stellar=None,
                     index=None,
                     goodpixels=None,
                     pp=None, 
                     conf=None):
        assert conf is not None
        assert galaxy is not None
        assert noise is not None
        
        t = clock()
        
        vlim = lambda x: stellar_kinematics[0, index] + x*np.array([-100, 100])

        stellar_kinematics = self.data.stellar_kinematics.kinematics_grid
        gas_templates = self.data.em_model.gas_templates
        gas_names = self.data.em_model.gas_names
        line_wave = self.data.em_model.line_wave
        lam = self.data.obs.meta['wave_obs']
        velscale = self.C*np.diff(np.log(lam[-2:]))

        ### 1 Component
        ngas_comp = 1
        component = np.array([0] + [1]*7)
        moments = [-2, 2]
        
        start = [[0, 0],
                  [stellar_kinematics[0, index], stellar_kinematics[1, index]]]
        
        bounds = [[vlim(1), [20, 100]],       # Bounds are ignored for the stellar component=0 which has fixed kinematic
                  [vlim(6), [20, 300]],]     # I force the component=2 to lie +/-600 km/s from the stellar velocity
          
        A_ineq_kin = np.array([[0, 0, 0, 0]])
        b_ineq_kin = np.array([0])/velscale

        # NOTE: It seem the minimisation functions used on ppxf have some
        # limitation on the adoption of constraits for multiple components <>.
        A_ineq_templ = np.array([[0, 0, -1, 0.42, 0, 0, 0, 0],
                                  [0, 0, 1, -1.45, 0, 0, 0, 0],])
        b_ineq_templ = np.array([0, 0])
        
        ### 2 Component
        # ngas_comp = 2
        # component = np.array([0] + [1]*7 + [2]*7)
        # moments = [-2, 2, 2]

        # start = [[0, 0],
        #           [stellar_kinematics[0, index], stellar_kinematics[1, index]],
        #           [stellar_kinematics[0, index], stellar_kinematics[1, index]]]

        # bounds = [[vlim(1), [20, 300]],       # Bounds are ignored for the stellar component=0 which has fixed kinematic
        #           [vlim(2), [20, 100]],       # I force the component=1 to lie +/-200 km/s from the stellar velocity
        #           [vlim(6), [20, 400]],]       # I force the component=2 to lie +/-600 km/s from the stellar velocity
          
        # A_ineq_kin = np.array([[0, 0, 0, 1, 0, -1]])
        # b_ineq_kin = np.array([0])/velscale

        #                           #star  #Hb   Ha  [SII]   [SII] [OIII] [OI] [NII]   Hb   Ha [SII]  [SII] [OIII] [OI] [NII]
        # A_ineq_templ = np.array([[0,      0,    0,  -1,    0.42,   0,    0,     0,   0,   0,   0,     0,    0,    0,    0],
        #                           [0,      0,    0,   1,   -1.45,   0,    0,     0,   0,   0,   0,     0,    0,    0,    0],
        #                           [0,      0,    0,   0,       0,   0,    0,     0,   0,   0,  -1,  0.42,    0,    0,    0],
        #                           [0,      0,    0,   0,       0,   0,    0,     0,   0,   0,   1, -1.45,    0,    0,    0],
        #                           ])
        # b_ineq_templ = np.array([0, 0, 0, 0])
        
        # ## 3 Component
        # ngas_comp = 3
        # component = np.array([0] + [1]*7 + [2]*7 + [3]*7)
        # moments = [-2, 2, 2, 2]

        # start = [[0, 0],
        #           [stellar_kinematics[0, index], stellar_kinematics[1, index]],
        #           [stellar_kinematics[0, index], stellar_kinematics[1, index]],
        #           [stellar_kinematics[0, index], stellar_kinematics[1, index]]]

        # bounds = [[vlim(1), [20, 100]],       # Bounds are ignored for the stellar component=0 which has fixed kinematic
        #           [vlim(2), [20, 100]],       # I force the component=1 to lie +/-200 km/s from the stellar velocity
        #           [vlim(10), [20, 300]],  
        #           [vlim(10), [20, 300]],]       # I force the component=2 to lie +/-600 km/s from the stellar velocity
          
        # A_ineq_kin = np.array([[0, 0, 0, 1, 0, -1, 0, 0],
        #                       [0, 0, 0, 0, 0, 1, 0, -1]])
        # b_ineq_kin = np.array([0, 0])/velscale

        #                           #star  #Hb   Ha  [SII]   [SII] [OIII] [OI] [NII]   Hb   Ha [SII]  [SII] [OIII] [OI] [NII]  Hb   Ha [SII]  [SII] [OIII] [OI] [NII]
        # A_ineq_templ = np.array([[0,      0,    0,  -1,    0.42,   0,    0,     0,   0,   0,   0,     0,    0,    0,    0,    0,   0,   0,     0,    0,    0,    0],
        #                          [0,      0,    0,   1,   -1.45,   0,    0,     0,   0,   0,   0,     0,    0,    0,    0,    0,   0,   0,     0,    0,    0,    0],
        #                          [0,      0,    0,   0,       0,   0,    0,     0,   0,   0,  -1,  0.42,    0,    0,    0,    0,   0,   0,     0,    0,    0,    0],
        #                          [0,      0,    0,   0,       0,   0,    0,     0,   0,   0,   1, -1.45,    0,    0,    0,    0,   0,   0,     0,    0,    0,    0],
        #                          # [0,      0,    0,   0,       0,   0,    0,     0,   0,   0,   0,     0,    0,    0,    0,    0,   0,  -1,  0.42,    0,    0,    0],
        #                          # [0,      0,    0,   0,       0,   0,    0,     0,   0,   0,   0,     0,    0,    0,    0,    0,   0,   1, -1.45,    0,    0,    0],
        #                          ])
        # b_ineq_templ = np.array([0, 0, 0, 0])        

        ###
        
        constr_kinem = {"A_ineq": A_ineq_kin, "b_ineq": b_ineq_kin}    
        constr_templ = {"A_ineq": A_ineq_templ, "b_ineq": b_ineq_templ}

        gas_templates = np.tile(gas_templates, ngas_comp)
        gas_names = np.asarray([a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
        line_wave = np.tile(line_wave, ngas_comp)
        gas_component = np.array(component) > 0
        stars_gas_templates = np.column_stack([stellar, gas_templates])
        
        pp = ppxf(stars_gas_templates, galaxy, noise, velscale, start,
                  plot=False, moments=moments, degree=2, mdegree=-1,
                  component=component,
                  gas_component=gas_component, gas_names=gas_names,
                  lam=lam, vsyst=0,
                  goodpixels=goodpixels,
                  bounds=bounds,
                  constr_kinem=constr_kinem,
                  # constr_templ=constr_templ,
                  global_search=True)
        
        print('Elapsed time in PPXF: %.2f s' % (clock() - t))
        return pp

    def build_output_storage(self, out_obj=None):
        assert out_obj is not None
        
        self.logger.info('Building storage')
        self.ppxf = PpxfResults()
        n_obj = self.data.obs.flux_grid.shape[-1]
    
        for _p in self.par:
            assert _p in dir(out_obj), f"ppxf doesn't output {_p}"
            _obj = out_obj.__getattribute__(_p)
    
            if _obj is None:
                _shape = (n_obj,)
            elif isinstance(_obj, (float, int, bool)):
                _shape = (n_obj,)
            elif _p == 'goodpixels':
            # NOTE: goodpixels array has a variable size. It's trick to deal with
            # this kind of object so I'm implementing a special case. <>
                _aux = out_obj.__getattribute__('galaxy')
                _shape = _aux.shape + (n_obj,)
            elif isinstance(_obj, list):
                _obj = np.asarray(_obj).ravel()
                _shape = _obj.shape + (n_obj,)
            else:
                _shape = _obj.shape + (n_obj,)
    
            with tempfile.NamedTemporaryFile() as temp_file:
                arr = np.memmap(temp_file, dtype = float, shape = _shape)
                arr.fill(np.nan)
                arr.flush()
                self.ppxf.__setattr__(_p, arr)
        self.storage = True

    def store_output(self, output_queue):
        for serial_out in iter(output_queue.get, None):
            # t = clock()
            index, out_obj = pickle.loads(serial_out)
            
            if out_obj is None:
                continue
            
            if self.storage is False:
                self.build_output_storage(out_obj)
                
            for _p in self.par:
                _obj = out_obj.__getattribute__(_p)
                
                if isinstance(_obj, list):
                    _obj = np.asarray(_obj).ravel()
                
                try:
                    self.ppxf.__getattribute__(_p)[..., index] = _obj
                except ValueError:
                    shape = _obj.shape[0]
                    self.ppxf.__getattribute__(_p)[..., :shape, index] = _obj
                    
            self.ppxf.__getattribute__(_p).flush()   
            # print('Elapsed time to save: %.5f s' % (clock() - t))
            
        self.reconstruct_map(data=self.data, 
                              parameter=self.main_meta['output']['to_save'])
        
    def reconstruct_map(self, data=None, parameter=[], save=True):
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
                    directory = self.main_meta['output_run']
                self.save_fits(map_full, _p, directory)

    @staticmethod
    def save_fits(data_param, name, directory='.', overwrite=True):
        data_param = np.array(data_param, dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data_param)
        hdul = fits.HDUList([hdu])
        full_path = os.path.join(directory, f'{name}.fits')
        hdul.writeto(full_path, overwrite=overwrite)

if __name__ == '__main__':
    t = ExecutePpxf(ppxf_prep.data, ppxf_prep.data.main_meta)
