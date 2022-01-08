"""
Created on Tue Aug 31 16:52:36 2021

@author: Luiz
"""

import glob
import numpy as np
import spectcube as sc
import _pickle as pickle
import compute_muse_lsf as lsf
from astropy.io import fits
from normalize_median import normalize_median
from convolve import convolve
from time import perf_counter as clock


class DataPreprocessing:


    def __init__(self, metadata_path):
        with open(metadata_path, 'rb') as inp:
            self.meta = pickle.load(inp)

        self.pre_prepare()
        self.prepare_model()
        self.prepare_observation()
        self.data_to_mmap()

        with open(metadata_path, 'wb') as out:
            pickle.dump(self.meta, out)

    def pre_prepare(self):
        print('''
              Model pre-preparation
              *********************''')
        print('Starting\n--------\n')

        # Reading a single model to gather information
        print('Reading model data:')
        t = clock()

        model_files = glob.glob(self.meta.model_path)
        self.meta.model_files = model_files
        with fits.open(model_files[0]) as hdu:
            self.meta.o_first_wave_model = np.double(hdu['PRIMARY'].header['CRVAL1'])
            self.meta.o_step_wave_model = np.double(hdu['PRIMARY'].header['CDELT1'])
            self.meta.o_n_pixel_model = hdu['PRIMARY'].header['NAXIS1']

        self.meta.o_n_model = len(model_files)

        self.meta.o_wave_model = \
            sc.util.build_wave_array([self.meta.o_first_wave_model, self.meta.o_step_wave_model],
                                      sampling_type = 'linear',
                                      size = self.meta.o_n_pixel_model)

        self.meta.o_limit_model = self.meta.o_wave_model[[0,-1]]
        print(f'{round(clock()-t,2)} s\n')


        # Reading observations to gather information
        print('Reading observations data:')
        t = clock()

        with fits.open(self.meta.obs_path) as hdu:
            self.meta.o_first_wave_obs = np.double(hdu['DATA'].header['CRVAL3'])
            self.meta.o_step_wave_obs = np.double(hdu['DATA'].header['CD3_3'])
            self.meta.o_n_pixel_obs = hdu['DATA'].header['NAXIS3']
        self.meta.o_wave_obs = \
            sc.util.build_wave_array([self.meta.o_first_wave_obs, self.meta.o_step_wave_obs],
                                      sampling_type = 'linear',
                                      size = self.meta.o_n_pixel_obs)

        self.meta.o_obs_limit = self.meta.o_wave_obs[[0,-1]]
        print(f'{round(clock()-t,2)} s\n')

    def prepare_model(self, dtype = float):
        print('''
              Model preparation
              *****************''')
        print('Starting\n--------\n')

        # Reading model data
        print('Reading data:')
        t = clock()

        model_files = glob.glob(self.meta.model_path)

        self.meta.wave_model = \
            sc.util.fit_wave_interval(self.meta.o_wave_model,
                                      old_sampling = self.meta.o_model_sampling,
                                      new_sampling = 'log',
                                      new_size = self.meta.o_n_pixel_model)

        flux_model = np.zeros((self.meta.o_n_pixel_model, self.meta.o_n_model))
        for j, file in enumerate(model_files):
            with fits.open(model_files[j]) as hdu:
                data = hdu['PRIMARY'].data
                flux_model[:, j] = data
        print(f'{round(clock()-t,2)} s\n')

        # Convolution
        print('Convolution:')
        t = clock()

        fwhm_obs = lsf.equation_lsf(self.meta.wave_model,
                                    self.meta.o_obs_limit[0],
                                    self.meta.o_obs_limit[1])
        fwhm_dif = np.sqrt((fwhm_obs**2 - self.meta.model_lsf**2).clip(0))
        sigma = fwhm_dif / (2.355 * self.meta.o_step_wave_model) # Sigma difference in pixels
        flux_model = convolve(flux = flux_model, sigma = sigma)
        print(f'{round(clock()-t,2)} s\n')

        # Resampling
        print('Resampling:')
        t = clock()

        flux_model, _, _ = sc.resampling(flux = flux_model,
                                         old_wave = self.meta.o_wave_model,
                                         old_sampling_type = 'linear',
                                         new_wave = self.meta.wave_model,
                                         new_sampling_type ='log')
        print(f'{round(clock()-t,2)} s\n')


        # Trimming wavelength range
        print('Trimming to match wavelength range of the observations:')
        t = clock()

        mask_match = (self.meta.wave_model >= self.meta.o_obs_limit[0]) & \
            (self.meta.wave_model <= self.meta.o_obs_limit[1])
        flux_model = flux_model[mask_match, ...]
        self.meta.wave_model = self.meta.wave_model[mask_match, ...]
        print(f'{round(clock()-t,2)} s\n')

        # Trimming with offset value to remove zeros added during convolution
        print('Trimming to remove trailing and leading zeros:')
        t = clock()
        mask_zeros = (self.meta.wave_model >= self.meta.wave_model[0] + self.meta.model_wave_trim[0]) & \
            (self.meta.wave_model <= self.meta.wave_model[-1] - self.meta.model_wave_trim[1])

        flux_model = flux_model[mask_zeros, ...]
        self.meta.wave_model = self.meta.wave_model[mask_zeros, ...]
        print(f'{round(clock()-t,2)} s\n')

        #Normalisation
        t = clock()
        print('Normalisation:')
        flux_model, _ = normalize_median(flux_model)
        print(f'{round(clock()-t,2)} s\n')

        # Recording metadata
        print('Recording metadata\n')
        self.meta.n_pixel_model = flux_model.shape[0]
        self.meta.limit_model = self.meta.wave_model[[0,-1]]
        print('Finished\n--------\n')

        self.flux_model = np.array(flux_model, dtype = dtype)

    def prepare_observation(self, dtype = float):
        print('''
              Observation preparation
              ***********************
              ''')
        print('Starting\n--------\n')

        # Reading Data
        print('Reading data:')
        t = clock()
        with fits.open(self.meta.obs_path, memmap = True, lazy_load_hdus = True,
                       cache = False) as hdu:
            flux_obs = np.array(hdu['DATA'].data,
                                dtype = dtype)#[:, 110:120, 110:120]
            del hdu['DATA'].data
            flux_obs_unc = np.array(hdu['STAT'].data,
                                    dtype = dtype)#[:, 110:120, 110:120]
            flux_obs_unc = np.sqrt(flux_obs_unc)
            del hdu['STAT'].data
        print(f'{round(clock()-t,2)} s\n')

        # Resampling
        print('Resampling data:')
        t = clock()

        flux_obs, self.meta.wave_obs, flux_obs_unc = \
            sc.resampling(flux = flux_obs,
                          old_wave = self.meta.o_wave_obs,
                          old_sampling_type = 'linear',
                          new_wave = self.meta.wave_model,
                          new_sampling_type = 'log',
                          flux_err = flux_obs_unc)

        flux_obs = np.array(flux_obs, dtype = dtype)
        flux_obs_unc = np.array(flux_obs_unc, dtype = dtype)
        print(f'{round(clock()-t,2)} s\n')

        # Normalising
        print('Data normalisation:')
        t = clock()

        flux_obs, factor = normalize_median(flux_obs, save = True,
                                            directory=self.meta.output_dir)
        flux_obs_unc = flux_obs_unc/factor
        print(f'{round(clock()-t,2)} s\n')

        # Reshaping
        print('Data reshaping:')
        t = clock()

        self.meta.shape_obs = flux_obs.shape[1:]

        self.flux_obs = flux_obs.reshape((-1, np.array(flux_obs.shape[1:]).prod()))
        self.flux_obs_unc = flux_obs_unc.reshape((-1, np.array(flux_obs_unc.shape[1:]).prod()))
        print(f'{round(clock()-t,2)} s\n')

        # Recording metadata
        print('Recording metadata\n')
        self.meta.n_pixel_obs = len(self.meta.wave_obs)
        self.meta.limit_obs = self.meta.wave_obs[[0,-1]]
        print('Finished\n--------\n')
        
        
    def data_to_mmap(self):
        
        temp_input = self.meta.temp_input_dir
            
        mmap_flux_obs = np.memmap(f'{temp_input}/flux_obs.dat',
                                  dtype='float32', mode='w+',
                                  shape= self.flux_obs.shape)
        mmap_flux_obs[:] = self.flux_obs[:]

        mmap_flux_obs_unc = np.memmap(f'{temp_input}/flux_obs_unc.dat', 
                                      dtype='float32', mode='w+',
                                      shape= self.flux_obs_unc.shape)
        mmap_flux_obs_unc[:] = self.flux_obs_unc[:]

        mmap_flux_model = np.memmap(f'{temp_input}/flux_model.dat', 
                                    dtype='float32', mode='w+',
                                    shape= self.flux_model.shape)
        mmap_flux_model[:] = self.flux_model[:]