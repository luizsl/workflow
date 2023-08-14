"""
Created on Tue Aug 31 16:52:36 2021

@author: Luiz
"""

import glob
import io
import logging
import os
from functools import partial
from contextlib import redirect_stdout
from time import perf_counter as clock

import numpy as np

from bounds_processing import build_bounds
from model_processing import Model
from observation_processing import Muse, sn_function
from compute_muse_lsf import equation_lsf
from emission_modelling import EmissionModel


class DataPreprocessing:
    def __init__(self, metadata={}):
        assert metadata

        self.main_meta = metadata
        self.start_logging()
        self.logger.info('\nStarting\n--------\n')
        self.pre_prepare()
        self.prepare_observation()
        self.prepare_model()
        self.prepare_emission_model()
        self.prepare_bound()
        self.logger.info('\nFinished\n--------\n')

    def start_logging(self):
        name_log_file = os.path.join(
            self.main_meta['output_run_ppxf'],
            'log_ppxf_preprocessing.log')

        formatter = logging.Formatter('%(message)s')
        loglevel = logging.INFO

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

    def pre_prepare(self):
        self.logger.info('''Gathering Information\n*********************''')

        # Reading a single model to gather information
        self.logger.info('Reading model data')
        factory = Model()
        self.model = factory.get_model(self.main_meta['model']['class_'])
        self.logger.info(f"--{self.main_meta['model']['class_']}")
        self.model.load(self.main_meta['resources']['model'])

        try:
            age_log10 = self.main_meta['model']['age_log10']
            self.logger.info('--log10 age: %s', age_log10)
        except Exception:
            self.logger.info(
                'Could not identify whether the age is in a log scale')

        try:
            age_gyr = self.main_meta['model']['age_gyr']
            self.logger.info('--age Gyr: %s', age_gyr)
        except Exception:
            self.logger.info(
                'Could not identify whether the age is in Gyr')

        try:
            for key, value in self.main_meta['model']['remove'].items():
                self.model.remove_param(key, value)
        except Exception:
            pass

        # Reading observations to gather information
        self.logger.info('Reading observations data')
        path = os.path.join(self.main_meta['resources']['observation'],
                            self.main_meta['observation']['obs_name'])
        files = glob.glob(path + '*')
        self.logger.info('--' + f'{files[0]}')
        assert len(files) == 1, "Multiple files match the observation name"

        self.obs = Muse(
            files[0],
            self.main_meta['observation']['redshift'])

    def prepare_model(self):
        t = clock()
        self.logger.info('''\nModel preparation\n*****************''')
        self.model.build()
        self.model.reshape()

        try:
            if self.main_meta['model']['convolve'] is True:
                z = self.main_meta['observation']['redshift']
                self.model.convolve(self.obs.meta['limit_obs'], z=z)
                self.logger.info('--Broadening templates')
            elif self.main_meta['model']['convolve'] is False:
                self.logger.info('--Not broadening templates')
        except KeyError:
            self.logger.warning(
                '--Not broadening templates, keyword not found'
            )

        oversample = 1
        for key in self.main_meta.keys():
            if 'velscale_ratio' in self.main_meta[key]:
                oversample = self.main_meta[key]['velscale_ratio']
                break

        log_step = np.log(
            self.obs.meta['wave_obs'][1]/self.obs.meta['wave_obs'][0])
        wave = np.exp(np.arange(
            np.log(self.model.meta['o_wave_model'][0]),
            np.log(self.model.meta['o_wave_model'][-1]),
            log_step/oversample))
        self.model.resample(wave)

        try:
            limits = self.main_meta['model']['normalization']
            if limits == []:
                limits = [-np.inf, np.inf]
        except Exception:
            limits = [-np.inf, np.inf]
        self.logger.info(f'--scaling band: {limits}')

        try:
            weighting = self.main_meta['model']['weighting']
        except:
            weighting = 'light'
        self.logger.info(f'--weighting: {weighting}')

        self.model.normalize(limits=limits, weighting=weighting)

        try:
            limits = self.main_meta['model']['trim']
            if limits == []:
                limits = [-np.inf, np.inf]
        except Exception:
            limits = [-np.inf, np.inf]
        finally:
            self.model.trim_spectral_axis(*limits)
        self.logger.info(f'--trimming spectral axis: {limits}')

        self.model.convert_to_mmap()
        self.logger.info(f'{round(clock()-t,2)} s')

    def prepare_observation(self):
        t = clock()
        self.logger.info('''\nObservation preparation\n***********************''')
        self.obs.build_grid(
            min_valid_sn=self.main_meta['observation']['snr']['min'],
            snr_window=self.main_meta['observation']['snr']['window'])

        self.logger.info('--Foreground extinction')
        try:
            ext_conf = self.main_meta['observation']['foreground_extinction']

            map_name = ext_conf['dust_map']
            self.logger.info(f'\tdust_map: {map_name}')

            law = ext_conf['law']
            self.logger.info(f'\tlaw: {law}')

            r_v = ext_conf['r_v']
            self.logger.info(f'\tr_v: {r_v}')

            parameter = ext_conf['parameter']
            self.logger.info(f'\tparam: {parameter}')

            mode = ext_conf['mode']
            self.logger.info(f'\tmode: {mode}')

            self.obs.foreground_extinction(
                law=law, r_v=r_v, map_parameter=parameter, mode=mode,
                map_name=map_name)
        except:
            self.logger.info('\tNot applying')

        try:
            limits = self.main_meta['observation']['trim']
            if limits == []:
                limits = [-np.inf, np.inf]
        except Exception:
            limits = [-np.inf, np.inf]
        finally:
            self.obs.trim_spectral_axis(*limits)
        self.logger.info(f'--trimming spectral axis: {limits}')

        if (self.model.meta['o_limit_model'][0] > self.obs.meta['limit_obs'][0] - 100
            or self.model.meta['o_limit_model'][1] < self.obs.meta['limit_obs'][1] + 100):
            self.logger.info("--Observation's spectral axis needs to be trimmed")
            lower, upper = self.model.meta['o_limit_model']
            lower+=100
            upper-=100
            self.obs.trim_spectral_axis(lower, upper)

        self.obs.resample()

        if self.main_meta['vorbin']['apply'] is True:

            os.environ["MKL_NUM_THREADS"]     = "2"
            os.environ["NUMEXPR_NUM_THREADS"] = "2"
            os.environ["OMP_NUM_THREADS"]     = "2"

            target_sn = self.main_meta['vorbin']['target_sn']

            try:
                covar_a = self.main_meta['vorbin']['covar_sn_a']
            except Exception:
                covar_a = 0

            try:
                covar_b = self.main_meta['vorbin']['covar_sn_b']
            except Exception:
                covar_b = 1

            sn_func = partial(
                sn_function, covar_sn_a=covar_a, covar_sn_b=covar_b
            )

            self.logger.info(
                '--Voronoi binning with target SNR:{}'.format(target_sn)
            )
            self.obs.vorbin(target_sn=target_sn, sn_func=sn_func)

            os.environ.pop("MKL_NUM_THREADS")
            os.environ.pop("NUMEXPR_NUM_THREADS")
            os.environ.pop("OMP_NUM_THREADS")

        try:
            limits = self.main_meta['observation']['normalization']
            if limits == []:
                limits = [-np.inf, np.inf]
        except Exception:
            limits = [-np.inf, np.inf]
        self.logger.info(f'--scaling band: {limits}')

        try:
            weighting = self.main_meta['observation']['scaling']
        except:
            weighting = 'scalar'
        self.logger.info(f'--normalisation scale: {weighting}')

        self.obs.normalize(limits=limits, weighting=weighting)

        self.obs.convert_to_mmap()

        if 'spectral_mask' in self.main_meta['observation']:
            self.logger.info('--Ansatz for masked pixels')
            mask_list = self.main_meta['observation']['spectral_mask']
            self.obs.mask_spectral_axis(mask_list, kind='guess')
        else:
            self.logger.info('--Ansatz for masked pixels not found')
            mask_list = []
            self.obs.mask_spectral_axis(mask_list, kind='guess')

        if 'fixed_spectral_mask' in self.main_meta['observation']:
            self.logger.info('--Fixed masked pixels')
            fixed_mask_list = self.main_meta['observation']['fixed_spectral_mask']
            self.obs.mask_spectral_axis(fixed_mask_list, kind='fixed')
        else:
            self.logger.info('--Fixed masked pixels not found')
            fixed_mask_list = []
            self.obs.mask_spectral_axis(fixed_mask_list, kind='fixed')

        self.logger.info(f'{round(clock()-t,2)} s')

    def prepare_emission_model(self):
        t = clock()
        self.logger.info('''\nEmission model preparation\n**************************''')

        wave = self.model.meta['wave_model']
        lower_lamb = np.min(self.obs.meta['wave_obs'])
        upper_lamb = np.max(self.obs.meta['wave_obs'])

        # wave = np.array(self.stellar_fit_metadata['obs']['wave_obs'])
        z = self.main_meta['observation']['redshift']
        path_line_list = self.main_meta['resources']['emission_line_list']
        lsf_lamb = partial(equation_lsf, lower_lamb=lower_lamb,
                           upper_lamb=upper_lamb, z=z)

        with redirect_stdout(io.StringIO()) as f:
            self.em_model = EmissionModel()
            self.em_model.from_file(path=path_line_list, spectral_axis=wave,
                                    fwhm=lsf_lamb)
            out_log = f.getvalue()
        self.logger.info(out_log)

        self.logger.info(f'{round(clock()-t,2)} s')

    def prepare_gas_kinematics(self):
        t = clock()

        self.logger.info('''\nGas kinematics preparation\n**************************''')
        self.gas_kinematics = None

        self.logger.info(f'{round(clock()-t,2)} s\n')

    def prepare_bound(self):
        t = clock()
        self.logger.info('--Build bounds rule')
        try:
            self.bounds_rule = self.main_meta['gas_template']['bounds']
        except:
            self.bounds_rule = None
        self.logger.info(f'{round(clock()-t,2)} s\n')

if __name__ == '__main__':
    data = DataPreprocessing(ppxf_control.meta)

    # plt.plot(data.model.meta['wave_model'], data.model.flux_grid[:, 0])
    # plt.plot(data.obs.meta['wave_obs'], data.obs.flux_grid[:, 0])

#%%

