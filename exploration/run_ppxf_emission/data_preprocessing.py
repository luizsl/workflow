"""
Created on Tue Aug 31 16:52:36 2021

@author: Luiz
"""
import glob
import io
import json
import logging
import os
from contextlib import redirect_stdout
from functools import partial
from time import perf_counter as clock

import numpy as np

from bounds_processing import build_bounds
from compute_muse_lsf import equation_lsf
from emission_modelling import EmissionModel
from kinematics_processing import (AnomalyDetection, Field, FieldInferece,
                                   Kinematics)
from observation_processing import Muse, StellarContinuum, sn_function


class DataPreprocessing:
    def __init__(self, metadata: dict):
        assert isinstance(metadata, dict)
        self.main_meta = metadata
        self.start_logging()

        self.logger.info('\nStarting\n--------\n')
        self.pre_prepare()
        self.prepare_emission_model()
        self.prepare_observation()
        self.prepare_stellar_continuum()

        self.stellar_kinematics = self.prepare_stellar_kinematics()

        if 'ppxf_emission_dir' in self.main_meta['resources']:
            self.prepare_gas_kinematics()
        else:
            self.gas_kinematics = self.gas_kinematics_from_stellar()

        self.prepare_bound()

        # self.validate()
        self.logger.info('\nFinished\n--------\n')

    # def validate(self):
    #     assert self.obs.flux_grid.shape == self.stellar.flux_grid.shape

    def start_logging(self):
        name_log_file = os.path.join(
            self.main_meta['output_run'],
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
        t = clock()

        # Reading metadata of pPXF execution
        self.logger.info('''Gathering Information\n*********************''')
        path_stellar_fit_metadata = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'],
            'metadata.json')

        with open(path_stellar_fit_metadata) as f:
            self.stellar_fit_metadata = json.load(f)

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

        self.logger.info(f'{round(clock()-t,2)} s')

    def prepare_emission_model(self):
        t = clock()
        self.logger.info('''\nEmission model preparation\n**************************''')

        wave = np.array(self.stellar_fit_metadata['obs']['wave_obs'])
        z = self.main_meta['observation']['redshift']
        path_line_list = self.main_meta['resources']['emission_line_list']
        lsf_lamb = partial(equation_lsf, lower_lamb=None, upper_lamb=None, z=z)

        self.em_model = EmissionModel()
        self.em_model.from_file(
            path=path_line_list, spectral_axis=wave, fwhm=lsf_lamb
            )

        # with redirect_stdout(io.StringIO()) as f:
        #     self.em_model.build_model(wave, z=0)
        #     out_log = f.getvalue()

        # self.logger.info(out_log)
        self.logger.info(f'{round(clock()-t,2)} s')

    def prepare_observation(self):
        t = clock()
        self.logger.info('''\nObservation preparation\n***********************''')
        self.obs.build_grid(
            min_valid_sn=self.main_meta['observation']['snr']['min'],
            snr_window=self.main_meta['observation']['snr']['window'])

        # if (self.model.meta['o_limit_model'][0] > self.obs.meta['limit_obs'][0] - 100
        #     or self.model.meta['o_limit_model'][1] < self.obs.meta['limit_obs'][1] + 100):
        #     self.logger.info("--Observation's spectral axis needs to be trimmed")
        #     lower, upper = self.model.meta['o_limit_model']
        #     lower+=100
        #     upper-=100
        #     self.obs.trim_spectral_axis(lower, upper)

        wave = np.array(self.stellar_fit_metadata['obs']['wave_obs'])
        self.obs.resample(wave)

        if self.main_meta['vorbin']['apply'] is True:

            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"

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


        # if 'normalization' in self.main_meta['common']:
        #     limits = self.main_meta['common']['normalization']
        #     self.obs.normalize(limits=limits)

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

    def prepare_stellar_continuum(self):
        t = clock()

        self.logger.info('''\nStellar continuum preparation\n**************************''')
        # Path stellar continuum
        path_stellar_continuum = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'], 'stellar_bestfit.fits')

        wave = np.array(self.stellar_fit_metadata['obs']['wave_obs'])

        self.stellar = StellarContinuum(path_stellar_continuum, wave)
        self.stellar.build_grid()

        if self.main_meta['vorbin']['apply'] is True:
            self.logger.info('--Applying same binning of observations')
            valid = self.obs.meta['valid']
            npixels = self.obs.meta['nPixels']
            bin_num = self.obs.meta['bin_num']
            self.stellar.apply_binning(bin_num, npixels, valid)

        if 'normalization' in self.main_meta['common']:
            limits = self.main_meta['common']['normalization']
            wave = self.stellar.meta['wave']
            self.logger.info('--Normalising')
            self.stellar.normalize(limits=limits)

            self.logger.info('--Rescaling')
            self.stellar.flux_grid = self.stellar.rescale(
                scale_template = self.obs.flux_grid,
                wave=wave,
                limits=limits)

        self.stellar.convert_to_mmap()

        self.logger.info(f'{round(clock()-t,2)} s')

    def prepare_stellar_kinematics(self):
        t = clock()

        self.logger.info('''\nStellar kinematics preparation\n**************************''')

        # Path stellar kinematics directory
        stellar_dir = self.main_meta['resources']['ppxf_stellar_dir']
        stellar_kinematics_file = \
            self.main_meta['resources']['stellar_kinematics_file']
        path_kinematics = os.path.join(stellar_dir, stellar_kinematics_file)

        # Instantiate stellar Kinematics
        stellar_kinematics = Kinematics()
        stellar_kinematics.from_file(path_kinematics)

        self.logger.info('--Reshape kinematics grid')
        stellar_kinematics.reshape()

        if self.main_meta['vorbin']['apply'] is True:
            self.logger.info(
                '--Average with the same binning of observations')
            valid = self.obs.meta['valid']
            npixels = self.obs.meta['nPixels']
            bin_num = self.obs.meta['bin_num']
            xbin = self.obs.meta['xbin']
            ybin = self.obs.meta['ybin']

            stellar_kin_field = stellar_kinematics.mean_binning_field(
                bin_num, npixels, valid, xbin, ybin)

            stellar_kinematics = stellar_kin_field

        self.logger.info(f'{round(clock()-t,2)} s\n')

        return stellar_kinematics

    def gas_kinematics_from_stellar(self):
        t = clock()

        self.logger.info('''\nGas kinematics preparation from stellar kinematics\n**************************''')

        n_moments = self.main_meta['gas_template']['moments']
        n_components = self.main_meta['gas_template']['components']
        em_shape = self.em_model.size

        # Path stellar kinematics directory
        dir_ = self.main_meta['resources']['ppxf_stellar_dir']
        file = self.main_meta['resources']['stellar_kinematics_file']
        path_kinematics = os.path.join(dir_, file)

        # Instantiate Kinematics
        kinematics = Kinematics()

        kinematics.from_file(path_kinematics, clip=(None, n_moments))

        self.logger.info('--Reshape kinematics grid')
        kinematics.reshape()

        self.logger.info('--Tile kinematics grid')
        kinematics.kinematics_grid = np.tile(
            kinematics.kinematics_grid, (n_components * len(em_shape), 1))

        if self.main_meta['vorbin']['apply'] is True:
            self.logger.info(
                '--Average with the same binning of observations')
            valid = self.obs.meta['valid']
            npixels = self.obs.meta['nPixels']
            bin_num = self.obs.meta['bin_num']
            xbin = self.obs.meta['xbin']
            ybin = self.obs.meta['ybin']

            kin_field = kinematics.mean_binning_field(
                bin_num, npixels, valid, xbin, ybin)

            kinematics = kin_field

        self.logger.info(f'{round(clock()-t,2)} s\n')

        return kinematics

    def prepare_gas_kinematics(self):
        t = clock()

        self.logger.info('''\nGas kinematics preparation\n**************************''')

        # Path gas kinematics directory
        gas_emission_dir = self.main_meta['resources']['ppxf_emission_dir']
        gas_kinematics_file = \
            self.main_meta['resources']['emission_kinematics_file']
        path_kinematics = os.path.join(gas_emission_dir, gas_kinematics_file)

        # Instantiate gas kinematics
        n_stellar_moments = \
            self.main_meta['guess_handling']['n_stellar_moments']
        # field.read_data(clip=[n_stellar_moments, None])

        self.gas_kinematics = Kinematics()
        self.gas_kinematics.from_file(path_kinematics,
                                      clip=[n_stellar_moments, None])
        self.gas_kinematics.reshape()

        if 'guess_handling' in self.main_meta:
            guess_handling = self.main_meta['guess_handling']

            self.logger.info('Processing ansatz for kinematics')
            self.logger.info(path_kinematics)

            emission_kinematics_metadata = \
                self.main_meta['resources']['emission_kinematics_metadata']

            metadatapath_kinematics = os.path.join(
                gas_emission_dir, emission_kinematics_metadata
                )

            with open(metadatapath_kinematics) as f:
                metadata = json.load(f)
                bin_num = np.array(metadata['obs']['bin_num'])
                npixels = np.array(metadata['obs']['nPixels'])
                valid = np.array(metadata['obs']['valid'])
                xbin = np.array(metadata['obs']['xbin'])
                ybin = np.array(metadata['obs']['ybin'])
                x_full = np.array(metadata['obs']['x_full'])
                y_full = np.array(metadata['obs']['y_full'])

            field = self.gas_kinematics.mean_binning_field(
                bin_num, npixels, valid, xbin, ybin)

            try:
                method = guess_handling['inference']['method']
            except Exception:
                raise ValueError

            try:
                conf = guess_handling['inference']['conf']
            except Exception:
                conf = {}

            # Processing maps one by one
            grid = np.asarray((x_full, y_full)).T
            data_map = np.full(
                (field.kinematics_grid.shape[0], grid.shape[0]),
                0, dtype=float)
            range_ = range(field.kinematics_grid.shape[0])
            for index in range_:
                self.logger.info(f'\nProcessing map {1+index}')

                coordinates = np.vstack([field.xbin, field.ybin]).T
                values = field.kinematics_grid[index]

                # Detect stuck
                try:
                    remove_stuck = guess_handling['remove_stuck']
                except KeyError:
                    remove_stuck = False
                finally:
                    if remove_stuck is True:
                        self.logger.info(
                            '--Removing guesses possibly stuck at boundaries'
                            )
                        stuck = field.filter_stuck(values)
                    else:
                        self.logger.info(
                            '--Not removing guesses possibly'
                            ' stuck at boundaries'
                            )
                        stuck = np.full_like(values, False, dtype=bool)

                # Anomaly detection
                if 'anomaly_detection' in guess_handling:
                    self.logger.info('--Employing anomaly detector')
                    anomaly = guess_handling['anomaly_detection']

                    try:
                        n_neighbors = anomaly['neighbours']
                        iterations = anomaly['iterations']
                    except Exception:
                        n_neighbors = 5
                        iterations = 1

                    outliers = np.full_like(values, False, dtype=bool)
                    for i in range(iterations):
                        valid = np.logical_and(~stuck, ~outliers)
                        self.logger.info(
                            f'\tIteration: {i + 1}\n'
                            f'\t\t#valid entries: {valid.sum()}'
                            )
                        detector_data = values[valid]
                        detector_coordinates = coordinates[valid]

                        detector = AnomalyDetection(
                            detector_coordinates, detector_data
                            )

                        outliers[valid] = detector.outliers_lof(
                            n_neighbors=n_neighbors
                            )
                else:
                    self.logger.info('--Not employing anomaly detector')
                    outliers = np.full_like(values, False, dtype=bool)

                # Field inference
                inference = FieldInferece()
                self.logger.info('--Starting field inference')
                self.logger.info(f'\tMethod: {method}')

                valid = np.logical_and(~stuck, ~outliers)
                grid = np.asarray((x_full, y_full)).T
                points = coordinates[valid]
                data = values[valid]

                res_filter = inference.__getattribute__(method)(
                    data, points, grid, conf=conf)
                res_filter = np.clip(res_filter, data.min(), data.max())
                data_map[index] = res_filter.T

            new_shape = (-1,) + (len(set(y_full)), len(set(x_full)))
            data_map = data_map.reshape(new_shape)
            self.gas_kinematics = Kinematics(data_map)
            self.gas_kinematics.reshape()

        if self.main_meta['vorbin']['apply'] is True:
            self.logger.info(
                '\n--Average with the same binning of observations')
            valid = self.obs.meta['valid']
            npixels = self.obs.meta['nPixels']
            bin_num = self.obs.meta['bin_num']
            self.gas_kinematics.mean_binning_field(bin_num, npixels, valid,
                                                   xbin, ybin)

        self.logger.info(f'{round(clock()-t,2)} s\n')

    def prepare_bound(self):
        t = clock()
        self.logger.info('--Build bounds rule')

        self.bounds_rule = self.main_meta['gas_template']['bounds']

        self.logger.info(f'{round(clock()-t,2)} s\n')


if __name__ == '__main__':
    data = DataPreprocessing(ppxf_prep.meta)
