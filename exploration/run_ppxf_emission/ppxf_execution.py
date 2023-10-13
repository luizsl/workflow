# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:54:40 2021

@author: Luiz
"""
import ctypes
import json
import io
import logging
import multiprocessing as mp
import os
import pickle
import tempfile
from contextlib import redirect_stdout
from datetime import datetime
from time import perf_counter as clock

import numpy as np
import xarray as xr
from ppxf.ppxf import ppxf, robust_sigma
from scipy.constants import physical_constants
from concurrent.futures import ProcessPoolExecutor
from mpi4py.futures import MPIPoolExecutor
from astropy.utils.misc import JsonCustomEncoder

from bounds_processing import build_bounds


class ExecutePpxf:
    def __init__(self, data=None, metadata=None):
        assert data is not None
        assert metadata is not None

        self.meta = {}
        self.data = data
        self.main_meta = metadata
        self.storage_flag = mp.Value(ctypes.c_bool, False)
        self.lock = mp.Lock()

        self.start_logging()
        self.out_ppxf = xr.Dataset()

        # NOTE: Adding an exception to deal with a single spectrum
        # not neat but should work. <>
        if self.data.obs.flux_grid.ndim == 1:
            self.data.obs.flux_grid = np.expand_dims(
                self.data.obs.flux_grid, axis=1)
            self.data.obs.flux_grid_unc = np.expand_dims(
                self.data.obs.flux_grid_unc, axis=1)

        self.size = self.data.obs.flux_grid[0, ...].size

        try:
            to_save = self.main_meta['output']['to_save']
        except:
            to_save = []

        try:
            to_map = self.main_meta['output']['to_map']
        except:
            to_map = []

        # NOTE: Saving output unforeseen
        par = to_save + to_map
        self.par = list(set(par))

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

        try:
            self.N_PROCESS = self.main_meta['common']['n_process']
        except Exception:
            self.N_PROCESS = mp.cpu_count()

        with MPIPoolExecutor() as executor:
        # with ProcessPoolExecutor(self.N_PROCESS) as executor:
            self.storage_flag.value = False

            n_obj = self.data.obs.flux_grid.shape[-1]
            futures = []

            for index in np.arange(self.size):
                self.logger.debug(index, end='\n')
                fit = executor.submit(
                        worker,
                        index,
                        self.data.obs.flux_grid[:, index],
                        self.data.obs.flux_grid_unc[:, index],
                        models=None,
                        stellar_bestfit=self.data.stellar.flux_grid[:, index],
                        em_model=self.data.em_model,
                        logger=self.logger, size=self.size,
                        gas_kinematics_slice=self.data.gas_kinematics.kinematics_grid[:, index],
                        stellar_kinematics_slice=self.data.stellar_kinematics.kinematics_grid[:, index],
                        bounds_rule=self.data.bounds_rule,
                        main_meta=self.data.main_meta,
                        obs_meta=self.data.obs.meta,
                        model_meta=self.data.stellar.meta)
                futures.append(fit)

                with self.lock:
                    if self.storage_flag.value is False:
                        future = futures.pop(0)
                        out_obj = future.result()
                        out_obj = pickle.loads(out_obj)
                        try:
                            build_output_storage(
                                out_obj=out_obj, out_dataset=self.out_ppxf,
                                logger=self.logger, n_obj=n_obj, par=self.par)
                            self.storage_flag.value = True
                            self.logger.info('Storage built')
                            futures.append(future)
                        except Exception as e:
                            if str(e) == 'Invalid data':
                                pass
                            else:
                                raise Exception

            while len(futures) > 0:
                future = futures.pop(0)
                try:
                    out_obj = future.result()
                    out_obj = pickle.loads(out_obj)
                    store_output(out_obj, par=self.par, logger=self.logger,
                        out_dataset=self.out_ppxf)
                    if out_obj is not None:
                        self.logger.info(out_obj.out_log)
                except:
                    continue

        # keep end time
        end_time = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.meta['ppxf_end_time'] = end_time
        self.logger.info(end_time)
        self.logger.info('pPXF execution completed\n\n')


def worker(i, flux_obs_slice=None, flux_obs_unc_slice=None, models=None,
           em_model=None, stellar_bestfit=None, logger=None, size=None,
           stellar_kinematics_slice=None, gas_kinematics_slice=None,
           bounds_rule=None, main_meta=None, obs_meta=None, model_meta=None):

    if logger is None:
        logger = logging.getLogger(__name__)

    with redirect_stdout(io.StringIO()) as f:
        id_ = f'{i+1}/{size}'
        print(70*'*')

        if np.any(np.isnan(flux_obs_unc_slice) | np.isnan(flux_obs_slice) | np.isnan(stellar_bestfit)):
            return pickle.dumps(None)

        pp = None

        guess_goodpixels = obs_meta['guess_goodpixels']
        fixed_goodpixels = obs_meta['fixed_goodpixels']
        goodpixels = np.intersect1d(guess_goodpixels, fixed_goodpixels)

        if 'ppxf_stellar_continuum' in main_meta:
            print(id_, 'Stellar continuum fine-tunning', end='\n\n')
            pp = execute_ppxf_continuum(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, em_model=em_model,
                stellar_bestfit=stellar_bestfit,
                goodpixels=goodpixels, bounds_rule=bounds_rule,
                main_meta=main_meta, obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_stellar_continuum'])
            print('*************', end='\n\n')

        if 'ppxf_emission_fit' in main_meta:
            print(id_, 'Emission-line fitting', end='\n\n')
            pp = execute_ppxf(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, em_model=em_model,
                goodpixels=goodpixels,
                stellar_kinematics_slice=stellar_kinematics_slice,
                gas_kinematics_slice=gas_kinematics_slice,
                bounds_rule=bounds_rule,
                main_meta=main_meta, obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_emission_fit'])
            print('*************', end='\n\n')

        # Include index
        pp.i = i

        # Include log
        pp.out_log = f.getvalue()

        data_out = pickle.dumps(pp)
        out_log = f.getvalue()
        logger.info(out_log)

        return data_out


def execute_ppxf_continuum(galaxy=None, noise=None, models=None, em_model=None,
                           stellar_bestfit=None, goodpixels=None,
                           obs_meta=None, model_meta=None,
                           main_meta=None, bounds_rule=None, pp=None,
                           kwargs_ppxf=None, logger=None):
    assert kwargs_ppxf is not None
    assert galaxy is not None
    assert noise is not None

    if logger is None:
        logger = logging.getLogger(__name__)

    t = clock()

    C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

    star_templates = stellar_bestfit

    frac = obs_meta['wave_obs'][1]/obs_meta['wave_obs'][0]
    velscale = np.log(frac)*C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)

    start_stellar_kinematics = np.array([0, 1, 0, 0])
    component = np.array([0])
    moments = [-4]

    pp = ppxf(star_templates, galaxy, noise, velscale,
              start_stellar_kinematics,
              moments=moments, component=component, goodpixels=goodpixels,
              lam=obs_meta['wave_obs'], #lam_temp=model_meta['wave'],
              **main_meta['ppxf_stellar_continuum'])

    print('Elapsed time in PPXF: %.2f s' % (clock() - t))
    return pp


def execute_ppxf(galaxy=None, noise=None, models=None, em_model=None,
                 goodpixels=None, obs_meta=None, model_meta=None,
                 main_meta=None, bounds_rule=None,
                 stellar_kinematics_slice=None, gas_kinematics_slice=None,
                 pp=None, kwargs_ppxf=None, logger=None):
    assert kwargs_ppxf is not None
    assert galaxy is not None
    assert noise is not None

    if logger is None:
        logger = logging.getLogger(__name__)

    t = clock()

    C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

    stellar_kinematics = stellar_kinematics_slice
    logger.debug(stellar_kinematics)

    gas_kinematics = gas_kinematics_slice
    logger.debug(gas_kinematics)

    gas_templates = em_model.template
    gas_names = em_model.label
    label_wave = em_model.label_wave
    lam = obs_meta['wave_obs']
    velscale = C * np.diff(np.log(lam[-2:]))[0]
    gas_moments = main_meta['gas_template']['moments']
    ngas_comp = main_meta['gas_template']['components']
    em_shape = em_model.size

    comp = [1] + em_shape * ngas_comp
    component = np.concatenate(
        [[i]*value for i, value in enumerate(comp)]
        )

    stellar_moments = stellar_kinematics[:4].shape[0]
    moments = [-stellar_moments] + [gas_moments] * ngas_comp*len(em_shape)

    gas_templates = np.tile(gas_templates, ngas_comp)
    gas_names = np.asarray(
        [gas + f"_({p+1})" for p in range(ngas_comp) for gas in gas_names])
    label_wave = np.tile(label_wave, ngas_comp)
    gas_component = np.array(component) > 0
    stars_gas_templates = np.column_stack([pp.bestfit, gas_templates])

    start_stellar_kinematics = np.array([0, 1, 0, 0])

    # start_gas_kinematics = np.zeros(gas_moments)
    # if gas_moments > 1:
    #     start_gas_kinematics[:2] = [0., 2*velscale]
    # start_gas_kinematics = [start_gas_kinematics.tolist()]
    # start_gas_kinematics = start_gas_kinematics * ngas_comp * len(em_shape)

    # start = [start_stellar_kinematics.tolist()] + start_gas_kinematics

    start_gas_kinematics = gas_kinematics.reshape([-1, gas_moments])

    start = [start_stellar_kinematics.tolist()] + start_gas_kinematics.tolist()

    if len(start) == 1:
        start = start[0]

    try:
        aux = np.asarray(start_gas_kinematics).ravel()
        bounds_gas = np.array(build_bounds(aux, bounds_rule))
        bounds_gas = bounds_gas.reshape(
            ngas_comp * len(em_shape), -1, 2)
        bounds_gas = bounds_gas.tolist()

        bounds_stellar = [[-200, 200], [1, 200], [-1, 1], [-1, 1]]

        bounds = [bounds_stellar] + bounds_gas
    except Exception:
        print('Could not build bounds')
        bounds = None

    try:
        A_ineq_kin = main_meta['gas_template']['A_ineq_kin']
        b_ineq_kin = main_meta['gas_template']['b_ineq_kin']
        A_ineq_kin = np.asarray(A_ineq_kin, dtype=float)
        b_ineq_kin = np.asarray(b_ineq_kin, dtype=float)

        # Adjust Constraint conditioning
        p = np.concatenate(start)
        if not np.all(A_ineq_kin.dot(p) <= b_ineq_kin):
            print('Try to get well-posed constraint and initial guess')
            try:
                p = constr_cond(A_ineq_kin, b_ineq_kin, p)
                star = p[:start_stellar_kinematics.size].reshape(start_stellar_kinematics.shape)
                gas =p[-start_gas_kinematics.size:].reshape(start_gas_kinematics.shape)
                start = [star.tolist()] + gas.tolist()
            except Exception:
                raise Exception

        b_ineq_kin = b_ineq_kin / velscale
        constr_kinem = {"A_ineq": A_ineq_kin, "b_ineq": b_ineq_kin}

    except Exception:
        print('Could not apply constraints on the kinematics')
        constr_kinem = None

    logger.debug(constr_kinem)
    # print(constr_kinem)
    # print(start)
    # print(A_ineq_kin)
    # print(b_ineq_kin)
    pp = ppxf(stars_gas_templates, galaxy, noise, velscale, start,
              plot=False, moments=moments, component=component,
              gas_component=gas_component, gas_names=gas_names,
              lam=lam, vsyst=0,
              bounds=bounds, goodpixels=goodpixels,
              constr_kinem=constr_kinem,
              **main_meta['ppxf_emission_fit'],
              )

    pp.sol[0] = stellar_kinematics

    corrected_flux = np.full_like(gas_names, np.nan, dtype=float)
    amplitude_rms = np.full_like(gas_names, np.nan, dtype=float)
    rms = robust_sigma(pp.galaxy - pp.bestfit, zero=1)
    for p, name in enumerate(gas_names):
        kk = gas_names == name
        # Angstrom per pixel at line wavelength (dlam/lam = dv/c)
        dlam = label_wave[kk]*velscale/C
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


def store_output(serial_out=None, par=None, logger=None,
                 out_dataset=None, lock=None):
    assert par is not None
    assert out_dataset is not None

    total_time = clock()

    if logger is None:
        logger = logging.getLogger(__name__)

    if serial_out is None:
        return True

    # add_param = self.main_meta['output']['additional_param']
    # self.keep_add_param(out_obj, parameters=add_param)

    [_store(out_obj=serial_out, _p=_p, out_dataset=out_dataset, logger=logger)
     for _p in par]

    logger.debug(
        f'Elapsed time to save {serial_out.i}: %.5f s' % (clock()-total_time))

    return True


def build_output_storage(out_obj=None, out_dataset=None, logger=None,
                         n_obj=None, par=None):
    assert par is not None
    assert out_dataset is not None
    assert n_obj is not None

    if logger is None:
        logger = logging.getLogger(__name__)
    if n_obj is None:
        n_obj = 1

    if out_obj is None:
       raise Exception('Invalid data')

    logger.info('Building storage')

    for _p in par:
        assert _p in dir(out_obj), f"{_p} is not available"
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
            _obj = np.hstack(_obj)
            _obj = _obj.ravel()
            _shape = _obj.shape + (n_obj,)
        else:
            _shape = _obj.shape + (n_obj,)

        try:
            dtype = _obj.dtype
        except Exception:
            dtype = type(_obj)
        finally:
            if dtype == float:
                dtype = np.float32
            if dtype == type(None):
                dtype = np.float32
            logger.debug(dtype)

        if _p == 'goodpixels':
            # chunks = list(_shape)
            # chunks[1] = 100*n_process

            data_axis = np.arange(_aux.shape[0])
            index_axis = np.arange(n_obj)
            coords = [data_axis, index_axis]
            dims = [f'{_p}_data', 'index']

        elif len(_shape) == 1:
            # chunks = 100*n_process

            index_axis = np.arange(n_obj)
            coords = [index_axis]
            dims = ['index']

        elif len(_shape) == 2:
            # chunks = list(_shape)
            # chunks[1] = 100*n_process

            data_axis = np.arange(_obj.shape[0])
            index_axis = np.arange(n_obj)
            coords = [data_axis, index_axis]
            dims = [f'{_p}_data', 'index']

        # logger.debug(chunks)

        # empty_arr = np.empty(dtype=dtype, shape=_shape)
        # empty_arr = da.empty(dtype=dtype, shape=_shape, chunks=chunks)
        with tempfile.NamedTemporaryFile() as temp_file:
            empty_arr = np.memmap(temp_file, dtype = float, shape = _shape)
            empty_arr.fill(np.nan)
            empty_arr.flush()

        logger.debug(empty_arr)
        logger.debug(_p)
        logger.debug(empty_arr.dtype)
        logger.debug(empty_arr.shape)
        logger.debug(coords)
        logger.debug(dims)
        logger.debug('\n')

        data_array = xr.DataArray(
            empty_arr, coords=coords, dims=dims, name=_p)
        logger.debug(data_array)

        out_dataset[_p] = data_array
        logger.debug(out_dataset)

    return True


def _store(out_obj, _p, out_dataset=None, logger=None):
    assert out_dataset is not None
    if logger is None:
        logger = logging.getLogger(__name__)

    t = clock()
    index = out_obj.i
    _obj = out_obj.__getattribute__(_p)

    if isinstance(_obj, list):
        _obj = np.hstack(_obj)
        _obj = _obj.ravel()

    try:
        out_dataset[_p][..., index] = _obj
    except ValueError:
        shape = _obj.shape[0]
        out_dataset[_p][..., :shape, index] = _obj

    # try:
    #     self.out_ppxf[_p].flush()
    # except Exception:
    #     pass

    logger.debug(
        f'Time to save {_p: >15}_{index}: %.5f s' % (clock() - t))


def keep_add_param(out_obj=None, parameters=[], main_meta=None):
    assert out_obj is not None
    assert main_meta is not None

    for parameter in parameters:
        data = out_obj.__getattribute__(parameter)
        path = os.path.join(main_meta['output_run'],
                            f'{parameter}.json')

        with open(path, 'w') as out:
            json.dump(data, fp=out, indent=4, cls=JsonCustomEncoder)


def constr_cond(A, b, p):
    A_new = A.copy()
    b_new = b.copy()
    n_iter = 0
    while not np.all(A_new.dot(p) <= b_new):
        if n_iter > 1_000:
            raise StopIteration
        else:
            where = np.argwhere(A_new.dot(p) > b_new)
            for w in where:
                p[A_new[w[0]] < 0] = p[A_new[w[0]] < 0] * 1.01
    return p

# test

if __name__ == '__main__':
    t = ExecutePpxf(ppxf_control.data, ppxf_control.data.main_meta)
    # t.run_all_data()


#%% Test single spectrum

    fits = []
    i = 5
    fit = worker(i,
                t.data.obs.flux_grid[:, i],
                t.data.obs.flux_grid_unc[:, i],
                models=None,
                stellar_bestfit=t.data.stellar.flux_grid[:, i],
                em_model=t.data.em_model,
                logger=t.logger, size=t.size,
                gas_kinematics_slice=t.data.gas_kinematics.kinematics_grid[:, i],
                stellar_kinematics_slice=t.data.stellar_kinematics.kinematics_grid[:, i],
                bounds_rule=t.data.bounds_rule,
                main_meta=t.data.main_meta, obs_meta=t.data.obs.meta,
                model_meta=t.data.stellar.meta)
    fits.append(fit)

    pp = pickle.loads(fit)

#%% test with mpi

    fits = []
    with MPIPoolExecutor() as executor:
        # executor =  MPIPoolExecutor(1)
        # i = 0
        for i in range(100):
            print(i)
            fit = executor.submit(
                    worker,
                    i,
                    t.data.obs.flux_grid[:, i],
                    t.data.obs.flux_grid_unc[:, i],
                    models=None,
                    stellar_bestfit=t.data.stellar.flux_grid[:, i],
                    em_model=t.data.em_model,
                    logger=t.logger, size=t.size,
                    gas_kinematics_slice=t.data.gas_kinematics.kinematics_grid[:, i],
                    stellar_kinematics_slice=t.data.stellar_kinematics.kinematics_grid[:, i],
                    bounds_rule=t.data.bounds_rule,
                    main_meta=t.data.main_meta, obs_meta=t.data.obs.meta,
                    model_meta=t.data.stellar.meta)
            fits.append(fit)

        for _ in fits:
            import matplotlib.pyplot as plt
            a = pickle.loads(_.result())
            try:
                print(a.out_log)
                if a is not None:
                    pass
                    # fig, ax = plt.subplots()
                    # a.plot()
            except:
                pass

# %%
    n_obj = t.data.obs.flux_grid.shape[-1]
    futures = []
    with MPIPoolExecutor() as executor:
        t.storage_flag.value = False
        for index in np.arange(100):
            t.logger.debug(index, end='\n')
            fit = executor.submit(
                    worker,
                    index,
                    t.data.obs.flux_grid[:, index],
                    t.data.obs.flux_grid_unc[:, index],
                    models=None,
                    stellar_bestfit=t.data.stellar.flux_grid[:, index],
                    em_model=t.data.em_model,
                    logger=t.logger, size=t.size,
                    gas_kinematics_slice=t.data.gas_kinematics.kinematics_grid[:, index],
                    stellar_kinematics_slice=t.data.stellar_kinematics.kinematics_grid[:, index],
                    bounds_rule=t.data.bounds_rule,
                    main_meta=t.data.main_meta,
                    obs_meta=t.data.obs.meta,
                    model_meta=t.data.stellar.meta)
            futures.append(fit)

            with t.lock:
                if t.storage_flag.value is False:
                    future = futures.pop(0)
                    out_obj = future.result()
                    out_obj = pickle.loads(out_obj)
                    try:
                        build_output_storage(
                            out_obj=out_obj, out_dataset=t.out_ppxf,
                            logger=t.logger, n_obj=n_obj, par=t.par)
                        t.storage_flag.value = True
                        t.logger.info('Storage built')
                        futures.append(future)
                    except Exception as e:
                        if str(e) == 'Invalid data':
                            pass
                        else:
                            raise Exception

        while len(futures) > 0:
            future = futures.pop(0)
            out_obj = future.result()
            out_obj = pickle.loads(out_obj)
            store_output(out_obj, par=t.par, logger=t.logger,
                out_dataset=t.out_ppxf)
            if out_obj is not None:
                t.logger.info(out_obj.out_log)
            t.logger.debug('saving')

# %%
    i = 5
    start_stellar_kinematics = np.array([0, 1, 0, 0])
    gas_moments = t.data.main_meta['gas_template']['moments']

    gas_kinematics_slice = t.data.gas_kinematics.kinematics_grid[:, i]
    gas_kinematics = gas_kinematics_slice

    gas_templates = t.data.em_model.template
    gas_names = t.data.em_model.label
    label_wave = t.data.em_model.label_wave
    lam = t.data.obs.meta['wave_obs']
    velscale = 50
    gas_moments = t.data.main_meta['gas_template']['moments']
    ngas_comp = t.data.main_meta['gas_template']['components']
    em_shape = t.data.em_model.size

    # start_gas_kinematics = np.zeros(gas_moments)
    # # start_gas_kinematics = gas_kinematics_slice

    # if gas_moments > 1:
    #     start_gas_kinematics[:2] = [0., 2*velscale]
    # start_gas_kinematics = [start_gas_kinematics.tolist()]
    # start_gas_kinematics = start_gas_kinematics * ngas_comp * len(em_shape)

    start_gas_kinematics = gas_kinematics.reshape([-1, gas_moments])

    start = [start_stellar_kinematics.tolist()] + start_gas_kinematics.tolist()

#
    bounds_rule = t.data.bounds_rule

    aux = np.asarray(start_gas_kinematics).ravel()
    bounds_gas = np.array(build_bounds(aux, bounds_rule))
    bounds_gas = bounds_gas.reshape(
        ngas_comp * len(em_shape), -1, 2)
    bounds_gas = bounds_gas.tolist()

    bounds_stellar = [[-200, 200], [1, 200], [-1, 1], [-1, 1]]

    bounds = [bounds_stellar] + bounds_gas

#%%
    from scipy import linalg

    def constr_cond(A, b, p):
        A_new = A.copy()
        b_new = b.copy()
        n_iter = 0
        while not np.all(A_new.dot(p) <= b_new):
            if n_iter > 1_000:
                raise StopIteration
            else:
                where = np.argwhere(A_new.dot(p) > b_new)
                for w in where:
                    p[A_new[w[0]] < 0] = p[A_new[w[0]] < 0] * 1.01
        return A_new, b_new

    main_meta = t.data.main_meta
    A_ineq_kin = main_meta['gas_template']['A_ineq_kin']
    b_ineq_kin = main_meta['gas_template']['b_ineq_kin']
    A_ineq_kin = np.asarray(A_ineq_kin, dtype=float)
    b_ineq_kin = np.asarray(b_ineq_kin, dtype=float)

    # Adjust Constraint conditioning
    p = np.concatenate(start)
    if not np.all(A_ineq_kin.dot(p) <= b_ineq_kin):
        print('Try to get well-posed constraint and initial guess')
        try:
            A_ineq_kin, b_ineq_kin = constr_cond(
                A_ineq_kin, b_ineq_kin, p)
            star = p[:start_stellar_kinematics.size].reshape(start_stellar_kinematics.shape)
            gas =p[-start_gas_kinematics.size:].reshape(start_gas_kinematics.shape)
            start = [star.tolist()] + gas.tolist()
        except Exception:
            raise Exception


    b_ineq_kin = b_ineq_kin / velscale
    constr_kinem = {"A_ineq": A_ineq_kin, "b_ineq": b_ineq_kin}

    #%%
    p = np.concatenate(start)
    A_new = A_ineq_kin.copy()
    b_new = b_ineq_kin.copy()
    n_iter = 0
    while not np.all(A_new.dot(p) <= b_new):
        if n_iter > 1_000:
            raise StopIteration
        else:
            where = np.argwhere(A_new.dot(p) > b_new)
            for w in where:
                p[A_new[w[0]] < 0] = p[A_new[w[0]] < 0] * 1.01
            n_iter += 1

    #%%
    where = np.argwhere(A_new.dot(p) > b_new)
    for w in where:
        p[A_new[w[0]] < 0] = p[A_new[w[0]] < 0] * 1.01
