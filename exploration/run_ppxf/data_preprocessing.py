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

        # with open(metadata_path, 'wb') as out:
        #     pickle.dump(self.meta, out)

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
        self.model.reshape()
        self.model.convolve(self.obs.meta['limit_obs'])
        
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
        self.obs.vorbin(target_sn=20)
        self.obs.normalize()
        self.obs.convert_to_mmap()
        print(f'{round(clock()-t,2)} s')

        
if __name__ == '__main__':
    prep = DataPreprocessing('../../data_products/toy_10x10/miles/ppxf/metadata.pkl')
