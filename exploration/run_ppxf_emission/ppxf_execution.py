# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:54:40 2021

@author: Luiz
"""
import ctypes
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

import numpy as np
from astropy.io import fits
from ppxf.ppxf import ppxf, robust_sigma
from scipy.constants import physical_constants

from bounds_processing import build_bounds


class ExecutePpxf:
    C = physical_constants['speed of light in vacuum'][0]/1e3  # km/s
    meta = {}

    def __init__(self, data=None, metadata=None):
        assert data is not None
        assert metadata is not None

        self.data = data
        self.main_meta = metadata
        self.storage = mp.Value(ctypes.c_bool, False)
        self.process_manager = mp.Manager()
        self.lock = mp.Lock()

        self.start_logging()
        self.out_ppxf = self.process_manager.dict()

        # NOTE: Adding an exception to deal with a single spectrum
        # not neat but should work. <>
        if self.data.obs.flux_grid.ndim == 1:
            self.data.obs.flux_grid = np.expand_dims(
                self.data.obs.flux_grid, axis=1)
            self.data.obs.flux_grid_unc = np.expand_dims(
                self.data.obs.flux_grid_unc, axis=1)

        self.size = self.data.obs.flux_grid[0, ...].size

        # par = ['gas_reddening', 'reddening', 'status', 'gas_flux', 'gas_any',
        #        'gas_flux_error', 'gas_bestfit', 'phot_npix', 'gas_any_zero',
        #        'weights', 'bestfit','mpoly', 'gas_mpoly', 'dof', 'chi2',
        #        'sol', 'error', 'polyweights', 'apoly','goodpixels']

        par = []

        # NOTE: Saving output unforeseen
        new_par = self.main_meta['output']['to_save'] \
            + self.main_meta['output']['to_map']
        self.par = list(set(par) | set(new_par))

        self.run_all_data()

    def start_logging(self):
        name_log_file = os.path.join(
            self.main_meta['output_run'],
            'log_ppxf_execution.log')

        formatter = logging.Formatter('%(message)s')
        loglevel = logging.INFO
        # loglevel = logging.DEBUG

        file_handler = logging.FileHandler(name_log_file)
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

        ps_out = [mp.Process(target=self.store_output, args=[output_queue])
                  for _ in range(N_PROCESS)]

        for p_out in ps_out:
            p_out.start()
            self.logger.debug('Start multiprocessing of output')

        for p in ps:
            p.join()

        # output_queue.put(None)
        # p_out.join()

        for p_out in ps_out:
            output_queue.put(None)

        for p_out in ps_out:
            p_out.join()

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

                if np.any(np.isnan(galaxy)
                          | np.isnan(noise)
                          | np.isnan(stellar)):
                    print('nan')
                    pack = [i, pp]
                    data_out = pickle.dumps(pack)
                    continue

                if 'ppxf_stellar_continuum' in self.main_meta:
                    print(id_, 'Stellar continuum fine-tunning', end='\n\n')
                    pp = self.execute_ppxf_continuum(
                        galaxy=galaxy, noise=noise,
                        stellar=stellar,
                        index=i,
                        goodpixels=goodpixels,
                        pp=pp, conf=self.main_meta['ppxf_stellar_continuum'])
                    print('*************', end='\n\n')

                if 'ppxf_emission_fit' in self.main_meta:
                    print(id_, 'Emission-line fitting', end='\n\n')
                    pp = self.execute_ppxf(
                        galaxy=galaxy, noise=noise,
                        stellar=stellar,
                        index=i,
                        goodpixels=goodpixels,
                        pp=pp, conf=self.main_meta['ppxf_emission_fit'])
                    print('*************', end='\n\n')

                out_log = f.getvalue()

            self.logger.info(out_log)
            pack = [i, pp]
            data_out = pickle.dumps(pack)
            output_queue.put(data_out)

    def execute_ppxf_continuum(self,
                               galaxy=None, noise=None, stellar=None,
                               index=None,
                               goodpixels=None,
                               pp=None,
                               conf=None):
        assert conf is not None
        assert galaxy is not None
        assert noise is not None
        t = clock()

        lam = self.data.obs.meta['wave_obs']
        velscale = self.C*np.diff(np.log(lam[-2:]))
        start_stellar_kinematics = [0, 1, 0, 0]
        component = np.array([0])
        moments = [-4]

        pp = ppxf(stellar, galaxy, noise, velscale, start_stellar_kinematics,
                  moments=moments, component=component, lam=lam,
                  **self.main_meta['ppxf_stellar_continuum'])

        print('Elapsed time in PPXF: %.2f s' % (clock() - t))
        return pp

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

        stellar_kinematics = \
            self.data.stellar_kinematics.kinematics_grid[:, index]
        self.logger.debug(stellar_kinematics)

        gas_kinematics = self.data.gas_kinematics.kinematics_grid[:, index]
        self.logger.debug(gas_kinematics)

        gas_templates = self.data.em_model.template
        gas_names = self.data.em_model.label
        label_wave = self.data.em_model.label_wave
        lam = self.data.obs.meta['wave_obs']
        velscale = self.C*np.diff(np.log(lam[-2:]))
        gas_moments = self.data.main_meta['gas_template']['moments']
        ngas_comp = self.data.main_meta['gas_template']['components']
        em_shape = self.data.em_model.size

        comp = [1] + em_shape * ngas_comp
        component = np.concatenate(
            [[i]*value for i, value in enumerate(comp)]
            )

        stellar_moments = stellar_kinematics.shape[0]
        moments = [-stellar_moments] + [gas_moments] * ngas_comp*len(em_shape)

        gas_templates = np.tile(gas_templates, ngas_comp)
        gas_names = np.asarray(
            [a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
        label_wave = np.tile(label_wave, ngas_comp)
        gas_component = np.array(component) > 0
        stars_gas_templates = np.column_stack([pp.bestfit, gas_templates])

        start_stellar_kinematics = np.array([0, 1, 0, 0])
        start_gas_kinematics = gas_kinematics.reshape(-1, gas_moments)
        start = [start_stellar_kinematics.tolist()] \
            + start_gas_kinematics.tolist()
        self.logger.debug(start)

        try:
            bounds_rule = self.data.bounds_rule
            bounds_gas = np.array(build_bounds(gas_kinematics, bounds_rule))
            bounds_gas = bounds_gas.reshape(
                ngas_comp * len(em_shape), -1, gas_moments)
            bounds_gas = bounds_gas.tolist()
            bounds_stellar = [[-1, 1], [1, 2], [-1, 1], [-1, 1]]
            bounds = [bounds_stellar] + bounds_gas
        except Exception:
            bounds = None
        self.logger.debug(bounds)

        try:
            A_ineq_kin = self.data.main_meta['gas_template']['A_ineq_kin']
            b_ineq_kin = self.data.main_meta['gas_template']['b_ineq_kin']
            A_ineq_kin = np.asarray(A_ineq_kin, dtype=float)
            b_ineq_kin = np.asarray(b_ineq_kin, dtype=float)
            b_ineq_kin = b_ineq_kin / velscale
            constr_kinem = {"A_ineq": A_ineq_kin, "b_ineq": b_ineq_kin}
        except Exception:
            constr_kinem = None
        self.logger.debug(constr_kinem)

        pp = ppxf(stars_gas_templates, galaxy, noise, velscale, start,
                  plot=False, moments=moments, component=component,
                  gas_component=gas_component, gas_names=gas_names,
                  lam=lam, vsyst=0,
                  bounds=bounds,
                  constr_kinem=constr_kinem,
                  **self.main_meta['ppxf_emission_fit']
                  )

        pp.sol[0] = stellar_kinematics

        corrected_flux = np.full_like(gas_names, np.nan, dtype=float)
        amplitude_rms = np.full_like(gas_names, np.nan, dtype=float)
        rms = robust_sigma(pp.galaxy - pp.bestfit, zero=1)
        for p, name in enumerate(gas_names):
            kk = gas_names == name
            # Angstrom per pixel at line wavelength (dlam/lam = dv/c)
            dlam = label_wave[kk]*velscale/self.C
            # Convert to ergs/(cm^2 s)
            corrected_flux[p] = (pp.gas_flux[kk]*dlam)[0]
            amplitude_rms[p] = np.max(pp.gas_bestfit_templates[:, kk])/rms
            print(f"{name:20s}",
                  f"-Amp/Res: {amplitude_rms[p]:6.2f};",
                  f"flux: {corrected_flux[p]:6.0f} ergs/(cm^2 s)")

        pp.corrected_flux = corrected_flux
        pp.amplitude_rms = amplitude_rms
        pp.gas_names = gas_names

        print('Elapsed time in PPXF: %.2f s' % (clock() - t))
        return pp

    def build_output_storage(self, out_obj=None):
        assert out_obj is not None

        self.logger.info('Building storage')
        n_obj = self.data.obs.flux_grid.shape[-1]

        for _p in self.par:
            assert _p in dir(out_obj), f"ppxf doesn't output {_p}"
            _obj = out_obj.__getattribute__(_p)

            if _obj is None:
                _shape = (n_obj,)
            elif isinstance(_obj, (float, int, bool)):
                _shape = (n_obj,)
            elif _p == 'goodpixels':
                # NOTE: goodpixels array has a variable size. It's trick to
                # deal with this kind of object so I'm implementing a
                # special case. <>
                _aux = out_obj.__getattribute__('galaxy')
                _shape = _aux.shape + (n_obj,)
            elif isinstance(_obj, list):
                _obj = np.concatenate(_obj)
                _obj = _obj.ravel()
                _shape = _obj.shape + (n_obj,)
            else:
                _shape = _obj.shape + (n_obj,)

            try:
                dtype = _obj.dtype
            except Exception:
                dtype = type(_obj)
            print(dtype)

            with tempfile.NamedTemporaryFile() as temp_file:
                arr = np.memmap(temp_file, dtype=dtype, shape=_shape)
                arr.fill(np.nan)
                arr.flush()
                self.out_ppxf.__setitem__(_p, arr)

        self.storage.value = True
        self.logger.info('Storage built')

    def store_output(self, output_queue):
        for serial_out in iter(output_queue.get, None):
            # t = clock()

            index, out_obj = pickle.loads(serial_out)

            if out_obj is None:
                continue

            with self.lock:
                if self.storage.value is False:
                    self.build_output_storage(out_obj)

            for _p in self.par:
                _obj = out_obj.__getattribute__(_p)

                if isinstance(_obj, list):
                    _obj = np.concatenate(_obj)
                    _obj = _obj.ravel()

                with self.lock:
                    try:
                        item = self.out_ppxf.__getitem__(_p)
                        item[..., index] = _obj
                    except ValueError:
                        shape = _obj.shape[0]
                        item = self.out_ppxf.__getitem__(_p)
                        item[..., :shape, index] = _obj
                    finally:
                        self.out_ppxf.__setitem__(_p, item)
                    # print(self.out_ppxf.__getitem__(_p))

                    self.out_ppxf.__getitem__(_p).flush()
            # print(f'Elapsed time to save {index}: %.5f s' % (clock() - t))


if __name__ == '__main__':
    t = ExecutePpxf(ppxf_prep.data, ppxf_prep.data.main_meta)
