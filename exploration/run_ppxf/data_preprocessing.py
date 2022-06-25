"""
Created on Tue Aug 31 16:52:36 2021

@author: Luiz
"""

import glob
import os
from time import perf_counter as clock

import numpy as np

from model_processing import Model
from observation_processing import Muse


class DataPreprocessing:
    def __init__(self, metadata={}):
        assert metadata
        print('\nStarting\n--------\n')
        self.main_meta = metadata
        self.pre_prepare()
        self.prepare_observation()
        self.prepare_model()
        print('\nFinished\n--------\n')

    def pre_prepare(self):
        print('''Gathering Information\n*********************''')

        # Reading a single model to gather information
        print('Reading model data')
        factory = Model()
        self.model = factory.get_model(self.main_meta['model']['class_'])
        print(f"--{self.main_meta['model']['class_']}")
        self.model.load(self.main_meta['resources']['model'])
        
        if 'remove' in self.main_meta['model']:
            for key, value in self.main_meta['model']['remove'].items():
                self.model.remove_param(key, value)
        
        # Reading observations to gather information
        print('Reading observations data')
        path = os.path.join(self.main_meta['resources']['observation'],
                            self.main_meta['observation']['obs_name'])
        files = glob.glob(path + '*')
        print('--', files[0], sep='')
        assert len(files) == 1, "Multiple files match the observation name"
        
        self.obs = Muse(
            files[0],
            self.main_meta['observation']['redshift'])

    def prepare_model(self):
        t = clock()
        print('''\nModel preparation\n*****************''')
        self.model.build()
        self.model.reshape()
        
        try:
            if self.main_meta['model']['convolve'] is True:
                z = self.main_meta['observation']['redshift']
                self.model.convolve(self.obs.meta['limit_obs'], z=z)
            else:
                print('--Not broadening templates')
        except:
            print('--Not broadening templates, keyword not found')
        
        oversample = self.main_meta['ppxf']['velscale_ratio']
        log_step = np.log(
            self.obs.meta['wave_obs'][1]/self.obs.meta['wave_obs'][0])
        wave = np.exp(np.arange(
            np.log(self.model.meta['o_wave_model'][0]),
            np.log(self.model.meta['o_wave_model'][-1]),
            log_step/oversample))
        self.model.resample(wave)
        
        if 'normalization' in self.main_meta['common']:
            limits = self.main_meta['common']['normalization']
        self.model.normalize(limits=limits)
        
        self.model.convert_to_mmap()
        print(f'{round(clock()-t,2)} s')
        
    def prepare_observation(self):
        t = clock()
        print('''\nObservation preparation\n***********************''')
        self.obs.build_grid(
            min_valid_sn=self.main_meta['observation']['snr']['min'], 
            snr_window=self.main_meta['observation']['snr']['window'])

        if (self.model.meta['o_limit_model'][0] > self.obs.meta['limit_obs'][0] - 100
            or self.model.meta['o_limit_model'][1] < self.obs.meta['limit_obs'][1] + 100):
            print("--Observation's spectral axis needs to be trimmed")
            lower, upper = self.model.meta['o_limit_model']
            lower+=100
            upper-=100
            self.obs.trim_spectral_axis(lower, upper)
            
        self.obs.resample()
        if self.main_meta['vorbin']['apply'] is True:
            target_sn=self.main_meta['vorbin']['target_sn']
            print('--Voronoi binning with target SNR:{}'.format(target_sn))
            self.obs.vorbin(target_sn=target_sn)
            
        if 'normalization' in self.main_meta['common']:
            limits = self.main_meta['common']['normalization']
        self.obs.normalize(limits=limits)
        
        self.obs.convert_to_mmap()
            
        if 'spectral_mask' in self.main_meta['observation']:
            print('--Ansatz for masked pixels')
            mask_list = self.main_meta['observation']['spectral_mask']
            self.obs.mask_spectral_axis(mask_list, kind='guess')
        else:
            print('--Ansatz for masked pixels not found')
            mask_list = []
            self.obs.mask_spectral_axis(mask_list, kind='guess')
            
        if 'fixed_spectral_mask' in self.main_meta['observation']:
            print('--Fixed masked pixels')
            fixed_mask_list = self.main_meta['observation']['fixed_spectral_mask']
            self.obs.mask_spectral_axis(fixed_mask_list, kind='fixed')
        else:
            print('--Fixed masked pixels not found')
            fixed_mask_list = []
            self.obs.mask_spectral_axis(fixed_mask_list, kind='fixed')
            
        print(f'{round(clock()-t,2)} s')

        
if __name__ == '__main__':
    data = DataPreprocessing(ppxf_prep.meta)
    
    # plt.plot(data.model.meta['wave_model'], data.model.flux_grid[:, 0])
    # plt.plot(data.obs.meta['wave_obs'], data.obs.flux_grid[:, 0])
