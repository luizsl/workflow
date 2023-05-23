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

import extinction
import numpy as np
import ppxf as ppxf_package
import xarray as xr
# import dask.array as da
from packaging import version
from ppxf.ppxf import ppxf, robust_sigma
from scipy.constants import physical_constants


class ExecutePpxf:
    def __init__(self, data=None, metadata=None):
        assert data is not None
        assert metadata is not None

        self.meta = {}
        self.data = data
        self.main_meta = metadata
        self.storage = mp.Value(ctypes.c_bool, False)
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

        # self.run_all_data()

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
        # self.logger.info('pPXF execution started')

        # keep start time
        # start_time = datetime.now().strftime("%d/%m/%Y %H:%M")
        # self.meta['ppxf_start_time'] = start_time
        # self.logger.info(start_time)

        # try:
        #     self.N_PROCESS = self.main_meta['common']['n_process']
        # except Exception:
        #     self.N_PROCESS = mp.cpu_count()

        # input_queue = self.process_manager.Queue()
        output_queue = self.process_manager.Queue(maxsize=self.N_PROCESS)

        ps = [mp.Process(target=self.worker, args=[input_queue, output_queue])
              for _ in range(self.N_PROCESS)]

        # for p in ps:
        #     p.start()
        #     self.logger.debug('Start multiprocessing of fitting')
        # for i in range(self.size):
        #     input_queue.put(i)
        # for _ in range(self.N_PROCESS):
        #     input_queue.put(None)

        # self.store_output(input_queue, output_queue)

        # for p in ps:
        #     p.join()

        # keep end time
        # end_time = datetime.now().strftime("%d/%m/%Y %H:%M")
        # self.meta['ppxf_end_time'] = end_time
        # self.logger.info(end_time)

        # self.logger.info('pPXF execution completed\n\n')

    def build_output_storage(self, out_obj=None):
        assert out_obj is not None

        self.logger.info('Building storage')
        n_obj = self.data.obs.flux_grid.shape[-1]

        for _p in self.par:
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
                _obj = np.concatenate(_obj)
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
                self.logger.debug(dtype)

            if _p == 'goodpixels':
                chunks = list(_shape)
                chunks[1] = 100*self.N_PROCESS

                data_axis = np.arange(_aux.shape[0])
                index_axis = np.arange(n_obj)
                coords = [data_axis, index_axis]
                dims = [f'{_p}_data', 'index']

            elif len(_shape) == 1:
                chunks = 100*self.N_PROCESS

                index_axis = np.arange(n_obj)
                coords = [index_axis]
                dims = ['index']

            elif len(_shape) == 2:
                chunks = list(_shape)
                chunks[1] = 100*self.N_PROCESS

                data_axis = np.arange(_obj.shape[0])
                index_axis = np.arange(n_obj)
                coords = [data_axis, index_axis]
                dims = [f'{_p}_data', 'index']

            self.logger.debug(chunks)

            # empty_arr = np.empty(dtype=dtype, shape=_shape)
            # empty_arr = da.empty(1dtype=dtype, shape=_shape, chunks=chunks)
            with tempfile.NamedTemporaryFile() as temp_file:
                empty_arr = np.memmap(temp_file, dtype = float, shape = _shape)
                # empty_arr.fill(np.nan)
                empty_arr.flush()

            self.logger.debug(empty_arr)
            self.logger.debug(_p)
            self.logger.debug(empty_arr.dtype)
            self.logger.debug(empty_arr.shape)
            self.logger.debug(coords)
            self.logger.debug(dims)
            self.logger.debug('\n')

            data_array = xr.DataArray(
                empty_arr, coords=coords, dims=dims, name=_p)
            self.logger.debug(data_array)

            self.out_ppxf[_p] = data_array
            self.logger.debug(self.out_ppxf)

        self.storage.value = True
        self.logger.info('Storage built')

    def store_output(self, input_queue, output_queue):
        while not all([output_queue.empty(), input_queue.empty()]):
            serial_out = output_queue.get()
            self.logger.debug(output_queue.qsize())
            total_time = clock()

            index, out_obj = pickle.loads(serial_out)

            if out_obj is None:
                continue

            with self.lock:
                if self.storage.value is False:
                    self.build_output_storage(out_obj)

                    # add_param = self.main_meta['output']['additional_param']
                    # self.keep_add_param(out_obj, parameters=add_param)

            [self._store(index=index, out_obj=out_obj, _p=_p)
               for _p in self.par]

            self.logger.debug(
                f'Elapsed time to save {index}: %.5f s' % (clock()-total_time))

    def _store(self, index, out_obj, _p):
        t = clock()
        _obj = out_obj.__getattribute__(_p)

        if isinstance(_obj, list):
            _obj = np.concatenate(_obj)
            _obj = _obj.ravel()

        try:
            self.out_ppxf[_p][..., index] = _obj
        except ValueError:
            shape = _obj.shape[0]
            self.out_ppxf[_p][..., :shape, index] = _obj

        # try:
        #     self.out_ppxf[_p].flush()
        # except Exception:
        #     pass

        self.logger.debug(
            f'Time to save {_p: >15}_{index}: %.5f s' % (clock() - t))


def worker(i, flux_obs_slice=None, flux_obs_unc_slice=None, models=None,
           logger=None, size=None,
           main_meta=None, obs_meta=None, model_meta=None):
    # for i in iter(input_queue.get, None):
    # with redirect_stdout(io.StringIO()) as f:
        id_ = f'{i+1}/{size}'
        print(70*'*')

        if np.any(np.isnan(flux_obs_unc_slice) | np.isnan(flux_obs_slice)):
            return None

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
                models=models, goodpixels=goodpixels,
                obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_optimise_mask'])

            # Determine actual goodpixels
            goodpixels = clip_outliers(
                pp.galaxy, pp.bestfit, pp.goodpixels,
                **main_meta['ppxf_refit']['mask'])
            goodpixels = np.intersect1d(goodpixels, fixed_goodpixels)
            print('*************', end='\n\n')

        if 'ppxf_fit_reddening' in main_meta:
            print(i, 'Fit of reddening', end='\n\n')
            pp = execute_ppxf(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, goodpixels=goodpixels,
                obs_meta=obs_meta, model_meta=model_meta,
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
            print(i, 'Fit of kinematics', end='\n\n')
            pp = execute_ppxf(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, goodpixels=goodpixels,
                obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_kinematics'])
            print('*************', end='\n\n')

        if 'ppxf_regularization' in main_meta:
            print(i, 'Fit with regulazired solution', end='\n\n')

            flux_obs_unc_slice = flux_obs_unc_slice*np.sqrt(pp.chi2)

            pp = execute_ppxf(
                galaxy=flux_obs_slice, noise=flux_obs_unc_slice,
                models=models, goodpixels=goodpixels,
                obs_meta=obs_meta, model_meta=model_meta,
                pp=pp, kwargs_ppxf=main_meta['ppxf_regularization'])
            print('*************', end='\n\n')

        # Try to compute average properties of stellar populations
        ranges = []
        try:
            ranges.append(model_meta['age_range'])
        except Exception:
            pass
        try:
            ranges.append(model_meta['mh_range'])
        except Exception:
            pass
        try:
            ranges.append(model_meta['alpha_range'])
        except Exception:
            pass

        try:
            weights = pp.weights/ pp.weights.sum()
            av_age, av_mh, av_alpha = average(
                weights, model_meta['reg_dim'],
                *ranges,
                age_log10=main_meta['model']['age_log10'],
                age_gyr=main_meta['model']['age_gyr'])
        except Exception:
            av_age = np.nan
            av_mh = np.nan
            av_alpha = np.nan
        finally:
            print(f'<Age>: {av_age: .2f} Gyr')
            print(f'<metallicity>: {av_mh: .2f} dex')
            print(f'<alpha>: {av_alpha: .2f} dex')

            weighting = main_meta['model']['weighting']
            weighting = weighting.lower()
            pp.__setattr__('average_age', av_age)
            pp.__setattr__('average_metallicity', av_mh)
            pp.__setattr__('average_alpha', av_alpha)

        # Include reddening fitted on the fly if exists
        pp.a_v = a_v
        pp.ebv = ebv

        # out_log = f.getvalue()

        # logger.info(out_log)
        pp.i = i
        data_out = pickle.dumps(pp)

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


def execute_ppxf(galaxy=None, noise=None, models=None,
                 goodpixels=None, obs_meta=None, model_meta=None,
                 pp=None, kwargs_ppxf=None):
    assert kwargs_ppxf is not None
    assert galaxy is not None
    assert noise is not None

    C = physical_constants['speed of light in vacuum'][0]/1e3  #km/s

    t = clock()

    star_template = models.flux_grid
    template = star_template

    frac = obs_meta['wave_obs'][1]/obs_meta['wave_obs'][0]
    velscale = np.log(frac)*C       # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)

    if pp is None:
        start = [0., 2*velscale] # (km/s), starting guess for [V, sigma]
    else:
        start = pp.sol

    pp = ppxf(
        template, galaxy, noise, velscale, start,
        lam=obs_meta['wave_obs'],
        lam_temp=model_meta['wave_model'],
        reg_dim=model_meta['reg_dim'],
        goodpixels=goodpixels, **kwargs_ppxf)

    print('Elapsed time in PPXF: %.2f s' % (clock() - t))
    return pp

def dered(spectrum, wave=None, law='calzetti00', r_v=4.05, ebv=None,
          a_v=None):
    assert (a_v, ebv) != (None, None)
    assert wave is not None

    if a_v is None:
        a_v = ebv * r_v

    if ebv is None:
        ebv = a_v / r_v

    if law == 'fm07':
        ext_mag = extinction.__getattribute__(law)(wave=wave, a_v=a_v)
    else:
        ext_mag = extinction.__getattribute__(law)(wave=wave, a_v=a_v, r_v=r_v)

    dered_spectrum = extinction.remove(ext_mag, spectrum)

    return dered_spectrum, a_v, ebv

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


if __name__ == '__main__':
    t = ExecutePpxf(ppxf_control.data, ppxf_control.data.main_meta)

    #%%
    from multiprocessing import Pool
    i = 1
    res =\
    worker(i,
            t.data.obs.flux_grid[:, i],
            t.data.obs.flux_grid_unc[:, i],
            models=t.data.model,
            logger=t.logger, size=t.size,
            main_meta=t.data.main_meta, obs_meta=t.data.obs.meta,
            model_meta=t.data.model.meta)
#%%
    with Pool(4) as pool:
        # res = pool.apply_async(
        #     worker,
        #     (i,
        #     t.data.obs.flux_grid[:, i],
        #     t.data.obs.flux_grid_unc[:, i]),
        #     {'models':t.data.model,
        #     'logger':t.logger, 'size':t.size,
        #     'main_meta':t.data.main_meta, 'obs_meta':t.data.obs.meta,
        #     'model_meta':t.data.model.meta})
        # # res.get()
        # res.wait()
        # print('a')

        n_obj = t.data.obs.flux_grid.shape[-1]
        # lock = Lock('flag')
        futures = []
        output_queue = t.process_manager.Queue(maxsize=10)
        stores = []
        for i in np.arange(t.size):
            x = pool.apply_async(
                    worker,
                    (i,
                    t.data.obs.flux_grid[:, i],
                    t.data.obs.flux_grid_unc[:, i]),
                    {'models':t.data.model,
                    'logger':t.logger, 'size':t.size,
                    'main_meta':t.data.main_meta, 'obs_meta':t.data.obs.meta,
                    'model_meta':t.data.model.meta})
            # futures.append(x)
            output_queue.put(x)

            # for completed in as_completed(futures):
        #     if t.storage_flag.value is False:
        #         try:
        #             # out_obj = pickle.loads(future.result())
        #             out_obj = x.result()
        #             if build_output_storage(out_obj=out_obj, out_dataset=t.out_ppxf, logger=None,
        #                 n_obj=n_obj, par=t.par, n_process=t.N_PROCESS):
        #                     t.storage_flag.value = True
        #                     t.logger.info('Storage built')
        #                     # print('storage built')
        #         except Exception:
        #             raise Exception
        #         finally:
        #             # print(future)
        #             # print('saving')
        #             store = client.submit(
        #                 store_output, x, par=t.par, logger=t.logger,
        #                 out_dataset=t.out_ppxf)
        #             stores.append(store)
        #     else:
        #         # print('saving')
        #         store = client.submit(
        #             store_output, x, par=t.par, logger=t.logger,
        #             out_dataset=t.out_ppxf, priority=10)
        #         stores.append(store)

        # # store.result()
        # [store.result() for store in stores]