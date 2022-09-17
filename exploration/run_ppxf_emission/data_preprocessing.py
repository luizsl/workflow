"""
Created on Tue Aug 31 16:52:36 2021

@author: Luiz
"""

import glob
import os
import json
import logging
from time import perf_counter as clock

import numpy as np

from emission_modelling import Model
from observation_processing import Muse, StellarContinuum, StellarKinematics


class DataPreprocessing:
    def __init__(self, metadata={}):
        # assert metadata

        self.main_meta = metadata
        self.start_logging()
        self.logger.info('\nStarting\n--------\n')
        self.pre_prepare()
        self.prepare_observation()
        self.prepare_stellar_continuum()
        self.prepare_stellar_kinematics()
        self.prepare_emission_model()
        # self.validate()
        self.logger.info('\nFinished\n--------\n')
    
    def validate(self):
        assert self.obs.flux_grid.shape == self.stellar.flux_grid.shape
        
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
                                          
        self.em_model = Model()
        self.em_model.build_model(wave, z=0)
        
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
            target_sn=self.main_meta['vorbin']['target_sn']
            self.logger.info('--Voronoi binning with target SNR:{}'.format(target_sn))
            self.obs.vorbin(target_sn=target_sn)
            
        # if 'normalization' in self.main_meta['common']:
        #     limits = self.main_meta['common']['normalization']
        #     self.obs.normalize(limits=limits)
        
        self.obs.convert_to_mmap()
            
        # if 'spectral_negative_mask' in self.main_meta['observation']:
        #     self.logger.info('--Ansatz for masked pixels')
        #     mask_list = self.main_meta['observation']['spectral_mask']
        #     self.obs.mask_spectral_axis(mask_list, kind='guess')
        # else:
        #     self.logger.info('--Ansatz for masked pixels not found')
        #     mask_list = []
        #     self.obs.mask_spectral_axis(mask_list, kind='guess')
            
        if 'spectral_negative_mask' in self.main_meta['observation']:
            self.logger.info('--Fixed masked pixels')
            fixed_mask_list = self.main_meta['observation']['spectral_negative_mask']
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
            self.main_meta['resources']['ppxf_stellar_dir'], 'bestfit.fits')
        # Path stellar kinematics
        path_stellar_kinematics = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'], 'sol.fits')
        
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

        # Path stellar kinematics
        path_stellar_kinematics = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'], 'sol.fits')
        
        # Instantiate stellar Kinematics
        self.stellar_kinematics = StellarKinematics(path_stellar_kinematics)
        self.stellar_kinematics.gather_kinematics()
        
        if self.main_meta['vorbin']['apply'] is True:
            self.logger.info('--Average with the same binning of observations')
            valid = self.obs.meta['valid']
            npixels = self.obs.meta['nPixels']
            bin_num = self.obs.meta['bin_num']
            self.stellar_kinematics.apply_mean_binning(bin_num, npixels, valid)
        
        self.logger.info(f'{round(clock()-t,2)} s')
        
if __name__ == '__main__':
    data = DataPreprocessing(ppxf_prep.meta)
    
    # plt.plot(data.model.meta['wave_model'], data.model.flux_grid[:, 0])
    # plt.plot(data.obs.meta['wave_obs'], data.obs.flux_grid[:, 0])
