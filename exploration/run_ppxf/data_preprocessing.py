"""
Created on Tue Aug 31 16:52:36 2021

@author: Luiz
"""

from time import perf_counter as clock

import _pickle as pickle
import numpy as np

from model_processing import XSLAgeMh
from observation_processing import Muse


class DataPreprocessing:
    def __init__(self, metadata_path):
        print('\nStarting\n--------\n')

        with open(metadata_path, 'rb') as inp:
            self.meta = pickle.load(inp)

        self.pre_prepare()
        self.prepare_observation()
        self.prepare_model()

        with open(metadata_path, 'wb') as out:
            pickle.dump(self.meta, out)

        print('\nFinished\n--------')

    def pre_prepare(self):
        print('''Gathering Information\n*********************''')

        # Reading a single model to gather information
        print('Reading model data')
        self.model = XSLAgeMh(self.meta.model_path)
        self.model.remove_param('age_range', values = [10.2])
        self.model.remove_param('mh_range', values = [-0.1, 0.1])
        
        # Reading observations to gather information
        print('Reading observations data')
        self.obs = Muse(self.meta.obs_path, 0.004951)

    def prepare_model(self):
        t = clock()
        print('''\nModel preparation\n*****************''')
        self.model.build()
        self.model.convolve()
        
        log_step = np.log(self.obs.meta['wave_obs'][1]/self.obs.meta['wave_obs'][0])
        wave = np.exp(np.arange(
            np.log(self.model.meta['o_wave_model'][0]),
            np.log(self.model.meta['o_wave_model'][-1]),
            log_step))

        self.model.resample(wave)
        self.model.normalize()
        self.model.convert_to_mmap()
        print(f'{round(clock()-t,2)} s')
        

    def prepare_observation(self):
        t = clock()
        print('''\nObservation preparation\n***********************''')
        self.obs.build_grid()
        self.obs.resample()
        self.obs.normalize()
        self.obs.reshape()
        self.obs.convert_to_mmap()
        print(f'{round(clock()-t,2)} s')

    def match_wavelength_range(self):
        pass
        # mask_match = (self.meta.wave_model >= self.meta.o_obs_limit[0]) & \
        #     (self.meta.wave_model <= self.meta.o_obs_limit[1])
        # flux_model = flux_model[mask_match, ...]
        # self.meta.wave_model = self.meta.wave_model[mask_match, ...]
        # print(f'{round(clock()-t,2)} s\n')


    def trim_wavelength_range(self):
        pass
        # Trimming with offset value to remove zeros added during convolution
        # print('Trimming to remove trailing and leading zeros:')
        # t = clock()
        # mask_zeros = (self.meta.wave_model >= self.meta.wave_model[0] + self.meta.model_wave_trim[0]) & \
        #     (self.meta.wave_model <= self.meta.wave_model[-1] - self.meta.model_wave_trim[1])

        # flux_model = flux_model[mask_zeros, ...]
        # self.meta.wave_model = self.meta.wave_model[mask_zeros, ...]
        # print(f'{round(clock()-t,2)} s\n')
        
        
if __name__ == '__main__':
    prep = DataPreprocessing('../../data_products/toy_10x10/miles/ppxf/metadata.pkl')
    
    # flux = prep.obs.mmap_flux_obs[:,]
    
    # flux = flux.sum(axis=1)
    # factor = np.nanmedian(flux)
    # galaxy = flux/factor
    
    # flux_unc = prep.obs.mmap_flux_obs[:,]**2
    # flux_unc = flux_unc.sum(axis=1)
    # noise= np.sqrt(flux_unc)
    # noise= noise/factor

    # import matplotlib.pyplot as plt
    # plt.step(prep.obs.meta['wave_obs'],  prep.obs.mmap_flux_obs[:,0], where = 'mid')
    # plt.step(prep.model.meta['wave_model'], prep.model.mmap_flux_model[:,20, 10], where = 'mid')



