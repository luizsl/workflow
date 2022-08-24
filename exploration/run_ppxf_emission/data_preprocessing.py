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
from observation_processing import Muse, StellarContinuum


class DataPreprocessing:
    def __init__(self, metadata={}):
        # assert metadata

        self.main_meta = metadata
        self.start_logging()
        self.logger.info('\nStarting\n--------\n')
        self.pre_prepare()
        self.prepare_observation()
        self.prepare_model()
        # self.logger.info('\nFinished\n--------\n')
        
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
        self.logger.info('''Gathering Information\n*********************''')
        path_stellar_fit_metadata = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'],
            'metadata.json')
        
        with open(path_stellar_fit_metadata) as f:
            self.stellar_fit_metadata = json.load(f)
            
    def prepare_model(self):
        t = clock()
        self.logger.info('''\nModel preparation\n*****************''')

        wave = np.array(self.stellar_fit_metadata['obs']['wave_obs'])
        # z = self.main_meta['observation']['redshift']
                                          
        self.em_model = Model()
        self.em_model.build_model(wave, z=0)

        self.logger.info(f'{round(clock()-t,2)} s')
        
    def prepare_observation(self):
        t = clock()
        self.logger.info('''\nObservation preparation\n***********************''')

        # Read observation as used in ppxf input
        path_obs_flux = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'], 'galaxy.fits')
        path_obs_flux_unc = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'], 'noise.fits')
        
        wave = np.array(self.stellar_fit_metadata['obs']['wave_obs'])
        
        self.obs = Muse(path_obs_flux, path_obs_flux_unc, wave)
        self.obs.build_grid(min_valid_sn=3, snr_window=[5450, 5550])
        self.obs.convert_to_mmap()
        
        # Read stellar continuum and stellar kinematics 
        path_stellar_continuum = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'], 'bestfit.fits')
        path_stellar_kinematics = os.path.join(
            self.main_meta['resources']['ppxf_stellar_dir'], 'sol.fits')
        
        self.stellar = StellarContinuum(path_stellar_continuum, wave)
        self.stellar.build_grid()
        self.stellar.convert_to_mmap()
        self.stellar.gather_kinematics(path_stellar_kinematics)
            
        self.logger.info(f'{round(clock()-t,2)} s')

        
if __name__ == '__main__':
    data = DataPreprocessing(ppxf_prep.meta)
    
    # plt.plot(data.model.meta['wave_model'], data.model.flux_grid[:, 0])
    # plt.plot(data.obs.meta['wave_obs'], data.obs.flux_grid[:, 0])
