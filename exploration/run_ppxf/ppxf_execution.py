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
from datetime import datetime
from time import perf_counter as clock

import numpy as np
import ppxf as ppxf_package
import xarray as xr
from packaging import version
from ppxf.ppxf import ppxf, robust_sigma
from scipy.constants import physical_constants

from bounds_processing import build_bounds
from reddening import dered

from concurrent.futures import ProcessPoolExecutor
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


class ExecutePpxf:
    def __init__(self, data=None, metadata=None):
        assert data is not None
        assert metadata is not None

        self.meta = {}
        self.data = data
        self.main_meta = metadata
        self.storage_flag = mp.Value(ctypes.c_bool, False)
        self.process_manager = mp.Manager()
        self.lock = mp.Lock()

        self.start_logging()

        self.out_ppxf = xr.Dataset()
        # NOTE: Adding an exception to deal with a single spectrum
        # not neat but should work. <>
        if self.data.obs.flux_grid.ndim==1:
            self.data.obs.flux_grid = np.expand_dims(
                self.data.obs.flux_grid, axis=1)
            self.data.obs.flux_grid_unc = np.expand_dims(
                self.data.obs.flux_grid_unc, axis=1)

        self.size = self.data.obs.flux_grid[0, ...].size

        par = []

        # NOTE: Saving output unforeseen
        new_par = self.main_meta['output']['to_save']
        self.par = list(set(par) | set(new_par))

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

        with MPIPoolExecutor(self.N_PROCESS) as executor:
        # with ProcessPoolExecutor(self.N_PROCESS, max_tasks_per_child=5) as executor:
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
                        models=self.data.model,
                        em_model=self.data.em_model,
                        logger=self.logger, size=self.size,
                        main_meta=self.data.main_meta,
                        obs_meta=self.data.obs.meta,
                        model_meta=self.data.model.meta)
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
                out_obj = future.result()
                out_obj = pickle.loads(out_obj)
                store_output(out_obj, par=self.par, logger=self.logger,
                    out_dataset=self.out_ppxf)
                self.logger.info(out_obj.out_log)
                self.logger.debug('saving')

        # keep end time
        end_time = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.meta['ppxf_end_time'] = end_time
        self.logger.info(end_time)
        self.logger.info('pPXF execution completed\n\n')


def worker(i, flux_obs_slice=None, flux_obs_unc_slice=None, models=None,
           em_model=None, logger=None, size=None,
           main_meta=None, obs_meta=None, model_meta=None):
    assert em_model is not None

    if logger is None:
        logger = logging.getLogger(__name__)

    with redirect_stdout(io.StringIO()) as f:
        id_ = f'{i+1}/{size}'
        print(70*'*')

        if np.any(np.isnan(flux_obs_unc_slice) | np.isnan(flux_obs_slice)):
            return pickle.dumps(None)

        error_corr = None
        pp = None
        a_v = None
        ebv = None

        guess_goodpixels = obs_meta['guess_goodpixels']
        fixed_goodpixels = obs_meta['fixed_goodpixels']
        goodpixels = np.intersect1d(guess_goodpixels, fixed_goodpixels)

        if 'ppxf_optimise_mask' in main_meta:
            print(id_, 'Trying to optimise mask', end='\n\n')
            pp = execute_ppxf(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, em_model=em_model, goodpixels=goodpixels,
                main_meta=main_meta, obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_optimise_mask'])

            # Determine actual goodpixels
            goodpixels = clip_outliers(
                pp.galaxy, pp.bestfit, pp.goodpixels,
                **main_meta['ppxf_refit']['mask'])
            goodpixels = np.intersect1d(goodpixels, fixed_goodpixels)
            print('*************', end='\n\n')

        if 'ppxf_fit_reddening' in main_meta:
            print(id_, 'Fit of reddening', end='\n\n')
            pp = execute_ppxf(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, em_model=em_model, goodpixels=goodpixels,
                main_meta=main_meta, obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_fit_reddening'])

            # Note: From the version 8.2.1 ppxf return A_V instead of
            # E(B-V) in pp.reddening <>
            if version.parse(ppxf_package.__version__) \
                >= version.parse('8.2.1'):
                a_v = pp.reddening
                print(f'A_V = {a_v: .2f}')
            else:
                ebv = pp.reddening
                print(f'E(B-V) = {ebv: .2f}')

            print('\nDered observation on the fly', end='\n\n')
            flux_obs_slice, a_v, ebv = dered(
                flux_obs_slice,
                wave=obs_meta['wave_obs'],
                ebv=ebv, a_v=a_v)

            flux_obs_unc_slice, _, _ = dered(
                flux_obs_unc_slice,
                wave=obs_meta['wave_obs'],
                ebv=ebv, a_v=a_v)
            print('*************', end='\n\n')

        if 'ppxf_kinematics' in main_meta:
            print(id_, 'Fit of kinematics', end='\n\n')
            pp = execute_ppxf(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, em_model=em_model,  goodpixels=goodpixels,
                main_meta=main_meta, obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_kinematics'])
            print('*************', end='\n\n')

        error_corr = np.hstack(pp.error) * np.sqrt(pp.chi2)

        if 'ppxf_regularization' in main_meta:
            print(id_, 'Fit with regulazired solution', end='\n\n')

            flux_obs_unc_slice = flux_obs_unc_slice*np.sqrt(pp.chi2)

            pp = execute_ppxf(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, em_model=em_model, goodpixels=goodpixels,
                main_meta=main_meta, obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_regularization'])
            print('*************', end='\n\n')

        # Try to compute average properties of stellar populations
        ranges = []
        try:
            ranges.append(model_meta['age_range'])
        except Exception:
            ranges.append([np.nan])
        try:
            ranges.append(model_meta['mh_range'])
        except Exception:
            ranges.append([np.nan])
        try:
            ranges.append(model_meta['alpha_range'])
        except Exception:
            ranges.append([np.nan])

        try:
            weights = pp.weights[pp.component == 0]
            weights = weights / weights.sum()
            av_age, av_mh, av_alpha = average(
                weights, model_meta['reg_dim'], *ranges,
                age_log10=main_meta['model']['age_log10'],
                age_gyr=main_meta['model']['age_gyr'])
        except Exception:
            av_age = np.nan
            av_mh = np.nan
            av_alpha = np.nan
        finally:
            print(f'<age>: {av_age: .2f} Gyr')
            print(f'<metallicity>: {av_mh: .2f} dex')
            print(f'<alpha>: {av_alpha: .2f} dex')

            weighting = main_meta['model']['weighting']
            weighting = weighting.lower()
            pp.average_age = av_age
            pp.average_metallicity = av_mh
            pp.average_alpha = av_alpha

        # Include corrected kinematics uncertainty
        pp.error_corr = error_corr

        # Include reddening fitted on the fly if exists
        pp.a_v = a_v
        pp.ebv = ebv

        # Include index
        pp.i = i

        # Include log
        pp.out_log = f.getvalue()

        data_out = pickle.dumps(pp)
        out_log = f.getvalue()
        logger.info(out_log)

        return data_out


def average(weights, reg_dim, age_range=np.asarray([np.nan]),
            mh_range=np.asarray([np.nan]),
            alpha_range=np.asarray([np.nan]),
            age_log10=False, age_gyr=False):

    light_weights = weights
    light_weights = light_weights.reshape(reg_dim)

    if light_weights.ndim == 1:
        light_weights = light_weights[:, None, None]
    elif light_weights.ndim == 2:
        light_weights = light_weights[:, :, None]

    light_weights /= light_weights.sum()

    age_grid, mh_grid, alpha_grid = np.meshgrid(
        age_range, mh_range, alpha_range,
        indexing='ij')

    try:
        av_age = np.average(age_grid, weights=light_weights)
    except Exception:
        av_age = np.nan

    try:
        av_mh = np.average(mh_grid, weights=light_weights)
    except Exception:
        av_mh = np.nan

    try:
        av_alpha = np.average(alpha_grid, weights=light_weights)
    except Exception:
        av_alpha = np.nan

    if age_log10:
        av_age = 10**av_age
    if not age_gyr:
        av_age = av_age / 1e9

    return av_age, av_mh, av_alpha


def execute_ppxf(galaxy=None, noise=None, models=None, em_model=None,
                 goodpixels=None, obs_meta=None, model_meta=None,
                 main_meta=None, bounds_rule=None, pp=None, kwargs_ppxf=None):
    assert kwargs_ppxf is not None
    assert galaxy is not None
    assert noise is not None

    t = clock()

    C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

    star_templates = models.flux_grid

    frac = obs_meta['wave_obs'][1]/obs_meta['wave_obs'][0]
    velscale = np.log(frac)*C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)

    try:
        gas_templates = em_model.template
        gas_names = em_model.label
        label_wave = em_model.label_wave
        em_shape = em_model.size
        gas_moments = main_meta['gas_template']['moments']
        ngas_comp = main_meta['gas_template']['components']
    except Exception:
        gas_templates = None
        gas_names = None
        label_wave = None
        em_shape = []
        gas_moments = 0
        ngas_comp = 0

    comp = [star_templates.shape[-1]] + em_shape * ngas_comp
    component = np.concatenate([[i]*value for i, value in enumerate(comp)])

    if pp is None:
        start_stellar_kinematics = np.zeros(4)
        start_stellar_kinematics[:2] = [0., 2*velscale]

        start_gas_kinematics = np.zeros(gas_moments)
        if gas_moments > 1:
            start_gas_kinematics[:2] = [0., 2*velscale]
        start_gas_kinematics = [start_gas_kinematics.tolist()]
        start_gas_kinematics = start_gas_kinematics * ngas_comp * len(em_shape)

        start = [start_stellar_kinematics.tolist()] + start_gas_kinematics
    else:
        start = pp.sol

    if len(start) == 1:
        start = start[0]

    gas_templates = np.tile(gas_templates, ngas_comp)
    gas_names = np.asarray(
        [gas + f"_({p+1})" for p in range(ngas_comp) for gas in gas_names])
    label_wave = np.tile(label_wave, ngas_comp)

    gas_component = np.array(component) > 0
    if np.any(gas_component) == False:
        gas_component = None

    try:
        template = np.column_stack([star_templates, gas_templates])
    except:
        template = star_templates

    try:
        aux = np.asarray(start_gas_kinematics).ravel()
        bounds_gas = np.array(build_bounds(aux, bounds_rule))
        bounds_gas = bounds_gas.reshape(
            ngas_comp * len(em_shape), -1, 2)
        bounds_gas = bounds_gas.tolist()
        bounds_stellar = [[-200, 200], [1, 200], [-1, 1], [-1, 1]]
        bounds = [bounds_stellar] + bounds_gas
    except Exception:
        bounds = None

    try:
        A_ineq_kin = main_meta['gas_template']['A_ineq_kin']
        b_ineq_kin = main_meta['gas_template']['b_ineq_kin']
        A_ineq_kin = np.asarray(A_ineq_kin, dtype=float)
        b_ineq_kin = np.asarray(b_ineq_kin, dtype=float)

        # Adjust Constraint conditioning
        p = np.concatenate(start)
        if not A_ineq_kin.dot(p) < b_ineq_kin:
            print('Try to get obtain well-posed constraint')
            try:
                A_ineq_kin, b_ineq_kin = constr_cond(
                    A_ineq_kin, b_ineq_kin, p)
            except Exception:
                raise Exception

        b_ineq_kin = b_ineq_kin / velscale
        constr_kinem = {"A_ineq": A_ineq_kin, "b_ineq": b_ineq_kin}

    except Exception:
        constr_kinem = None

    pp = ppxf(
        template, galaxy, noise, velscale, start,
        lam=obs_meta['wave_obs'],
        lam_temp=model_meta['wave_model'],
        reg_dim=model_meta['reg_dim'],
        component=component, gas_component=gas_component, gas_names=gas_names,
        constr_kinem=constr_kinem,
        bounds=bounds,
        goodpixels=goodpixels,
        **kwargs_ppxf)

    pp.stellar_bestfit = pp.bestfit - pp.gas_bestfit

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


def clip_outliers(galaxy, bestfit, goodpixels, sigma=3):
    """
    Adapted from Michele Cappellari's example

    Repeat the fit after clipping bins deviants more than 3*sigma
    in relative error until the bad bins don't change any more.
    """
    while True:
        scale = galaxy[goodpixels] @ bestfit[goodpixels]/np.sum(bestfit[goodpixels]**2)
        resid = scale*bestfit[goodpixels] - galaxy[goodpixels]
        err = robust_sigma(resid, zero=1)
        ok_old = goodpixels
        goodpixels = np.flatnonzero(np.abs(bestfit - galaxy) < sigma*err)
        if np.array_equal(goodpixels, ok_old):
            break

    return goodpixels


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


def constr_cond(A, b, p):
    A_new = A.copy()
    b_new = b.copy()
    n_iter = 0
    while np.any(A_new.dot(p) > b_new):
        if n_iter > 1_000:
            raise StopIteration
        else:
            A_new[A < 0] = A_new[A < 0] * 1.01
            n_iter += 1
    return A_new, b_new


if __name__ == '__main__':
    t = ExecutePpxf(ppxf_control.data, ppxf_control.data.main_meta)

    t.run_all_data()

#%%    APPLY_ASYNC

    with mp.Pool(4) as pool:
        t.storage_flag.value = False

        n_obj = t.data.obs.flux_grid.shape[-1]
        futures = []
        test = []

        # def f():
        #     return pickle.dumps(None)

        # res = pool.apply_async(f)
        # futures.append(res)

        for i in np.arange(t.size):
            t.logger.debug(i)
            fit = pool.apply_async(
                    worker,
                    (i,
                    t.data.obs.flux_grid[:, i],
                    t.data.obs.flux_grid_unc[:, i]),
                    {'models':t.data.model, 'em_model':t.data.em_model,
                    'logger':t.logger, 'size':t.size,
                    'main_meta':t.data.main_meta, 'obs_meta':t.data.obs.meta,
                    'model_meta':t.data.model.meta})
            futures.append(fit)

            with t.lock:
                if t.storage_flag.value is False:
                    future = futures.pop(0)
                    out_obj = future.get()
                    out_obj = pickle.loads(out_obj)
                    try:
                        build_output_storage(
                            out_obj=out_obj, out_dataset=t.out_ppxf, logger=t.logger,
                            n_obj=n_obj, par=t.par)
                        t.storage_flag.value = True
                        t.logger.info('Storage built')
                        futures.append(future)
                    except Exception as e:
                        if str(e) == ('Invalid data'):
                            pass
                        else:
                            raise Exception

        while len(futures) > 0:
            future = futures.pop(0)
            out_obj = future.get()
            out_obj = pickle.loads(out_obj)
            store = store_output(out_obj, par=t.par, logger=t.logger,
                out_dataset=t.out_ppxf)
            test.append(out_obj)
            t.logger.debug('saving')

#%%    SUBMIT MPI

    from concurrent.futures import ProcessPoolExecutor
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor

    # n_procs = MPI.COMM_WORLD.Get_size()  # Size of communicator
    # print(n_procs)
    with MPIPoolExecutor() as executor:
    # with ProcessPoolExecutor(4) as executor:
        t.storage_flag.value = False

        n_obj = t.data.obs.flux_grid.shape[-1]
        futures = []
        test = []

        # def f():
        #     return pickle.dumps(None)

        # res = pool.apply_async(f)
        # futures.append(res)

        for i in np.arange(t.size):
            t.logger.debug(i)
            fit = executor.submit(
                    worker,
                    i,
                    t.data.obs.flux_grid[:, i],
                    t.data.obs.flux_grid_unc[:, i],
                    models=t.data.model, em_model=t.data.em_model,
                    logger=t.logger, size=t.size,
                    main_meta=t.data.main_meta, obs_meta=t.data.obs.meta,
                    model_meta=t.data.model.meta)
            futures.append(fit)

            with t.lock:
                if t.storage_flag.value is False:
                    future = futures.pop(0)
                    out_obj = future.result()
                    out_obj = pickle.loads(out_obj)
                    try:
                        build_output_storage(
                            out_obj=out_obj, out_dataset=t.out_ppxf, logger=t.logger,
                            n_obj=n_obj, par=t.par)
                        t.storage_flag.value = True
                        t.logger.info('Storage built')
                        futures.append(future)
                    except Exception as e:
                        if str(e) == ('Invalid data'):
                            pass
                        else:
                            raise Exception

        while len(futures) > 0:
            future = futures.pop(0)
            out_obj = future.result()
            out_obj = pickle.loads(out_obj)
            store = store_output(out_obj, par=t.par, logger=t.logger,
                out_dataset=t.out_ppxf)
            test.append(out_obj)
            t.logger.info(out_obj.out_log)
            t.logger.debug('saving')
#%%
    i = 0
    galaxy = ppxf_control.data.obs.flux_grid[:, i]
    noise = ppxf_control.data.obs.flux_grid_unc[:, i]
    goodpixels=None

    pp = None
    bounds_rule = ppxf_control.data.bounds_rule
    models = ppxf_control.data.model
    em_model = ppxf_control.data.em_model
    obs_meta = ppxf_control.data.obs.meta
    model_meta = ppxf_control.data.model.meta
    main_meta = ppxf_control.data.main_meta
    kwargs_ppxf=main_meta['ppxf_optimise_mask']
    t = clock()

    C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

    star_templates = models.flux_grid

    frac = obs_meta['wave_obs'][1]/obs_meta['wave_obs'][0]
    velscale = np.log(frac)*C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)

    try:
        gas_templates = em_model.template
        gas_names = em_model.label
        label_wave = em_model.label_wave
        em_shape = em_model.size
        gas_moments = main_meta['gas_template']['moments']
        ngas_comp = main_meta['gas_template']['components']
    except Exception:
        gas_templates = None
        gas_names = None
        label_wave = None
        em_shape = []
        gas_moments = 0
        ngas_comp = 0

    comp = [star_templates.shape[-1]] + em_shape * ngas_comp
    component = np.concatenate([[i]*value for i, value in enumerate(comp)])

    if pp is None:
        start_stellar_kinematics = np.zeros(4)
        start_stellar_kinematics[:2] = [0., 2*velscale]

        start_gas_kinematics = np.zeros(gas_moments)
        if gas_moments > 1:
            start_gas_kinematics[:2] = [0., 2*velscale]
        start_gas_kinematics = [start_gas_kinematics.tolist()]
        start_gas_kinematics = start_gas_kinematics * ngas_comp * len(em_shape)

        start = [start_stellar_kinematics.tolist()] + start_gas_kinematics
    else:
        start = pp.sol

    if len(start) == 1:
        start = start[0]

    gas_templates = np.tile(gas_templates, ngas_comp)
    gas_names = np.asarray(
        [gas + f"_({p+1})" for p in range(ngas_comp) for gas in gas_names])
    label_wave = np.tile(label_wave, ngas_comp)

    gas_component = np.array(component) > 0
    if np.any(gas_component) == False:
        gas_component = None

    try:
        template = np.column_stack([star_templates, gas_templates])
    except:
        template = star_templates

    try:
        # bounds_rule = bounds_rule
        aux = np.asarray(start_gas_kinematics).ravel()
        bounds_gas = np.array(build_bounds(aux, bounds_rule))
        bounds_gas = bounds_gas.reshape(
            ngas_comp * len(em_shape), -1, 2)
        bounds_gas = bounds_gas.tolist()
        bounds_stellar = [[-200, 200], [1, 200], [-1, 1], [-1, 1]]
        bounds = [bounds_stellar] + bounds_gas
    except Exception:
        bounds = None

    try:
        A_ineq_kin = main_meta['gas_template']['A_ineq_kin']
        b_ineq_kin = main_meta['gas_template']['b_ineq_kin']
        A_ineq_kin = np.asarray(A_ineq_kin, dtype=float)
        b_ineq_kin = np.asarray(b_ineq_kin, dtype=float)

        # Adjust Constraint conditioning
        p = np.concatenate(start)
        if not A_ineq_kin.dot(p) < b_ineq_kin:
            print('Try to get obtain well-posed constraint')
            try:
                A_ineq_kin, b_ineq_kin = constr_cond(
                    A_ineq_kin, b_ineq_kin, p)
            except Exception:
                raise Exception

        b_ineq_kin = b_ineq_kin / velscale
        constr_kinem = {"A_ineq": A_ineq_kin, "b_ineq": b_ineq_kin}

    except Exception:
        constr_kinem = None

    pp = ppxf(
        template, galaxy, noise, velscale, start,
        lam=obs_meta['wave_obs'],
        lam_temp=model_meta['wave_model'],
        reg_dim=model_meta['reg_dim'],
        component=component, gas_component=gas_component, gas_names=gas_names,
        constr_kinem=constr_kinem,
        bounds=bounds,
        goodpixels=goodpixels,
        **kwargs_ppxf
        )

    pp.plot()


#%%
    i = 0
    fit = worker(
        i,
        t.data.obs.flux_grid[:, i],
        t.data.obs.flux_grid_unc[:, i],
        models=t.data.model, em_model=t.data.em_model,
        size=t.size,
        main_meta=t.data.main_meta, obs_meta=t.data.obs.meta,
        model_meta=t.data.model.meta)

    import matplotlib.pyplot as plt
    a = pickle.loads(fit)
    fig, ax = plt.subplots()
    a.plot()

#%%
    from mpi4py.futures import MPIPoolExecutor

    fits = []
    with MPIPoolExecutor(3) as executor:
        # executor =  MPIPoolExecutor(1)
        # i = 0
        for i in range(3):
            fit = executor.submit(
                    worker,
                    i,
                    t.data.obs.flux_grid[:, i],
                    t.data.obs.flux_grid_unc[:, i],
                    models=t.data.model, em_model=t.data.em_model,
                    logger=t.logger, size=t.size,
                    main_meta=t.data.main_meta, obs_meta=t.data.obs.meta,
                    model_meta=t.data.model.meta)
            fits.append(fit)

        for _ in fits:
            import matplotlib.pyplot as plt
            a = pickle.loads(_.result())
            fig, ax = plt.subplots()
            a.plot()

#%% I/O

    with redirect_stdout(io.StringIO()) as f:
        print('a')
        print('b')

        out_log = f.getvalue()
