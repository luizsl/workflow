"""
Created on Tue Aug 31 16:52:36 2021

@author: Luiz
"""
import glob
import numpy as np
import spectcube as sc
import compute_muse_lsf as lsf
from astropy.io import fits
from normalize_median import normalize_median
# from dataclasses import dataclass
from convolve import convolve
from time import perf_counter as clock

# @dataclass
# class Meta:

#     flux: np.ndarray = np.array([])
#     flux_unc: np.ndarray = np.array([])
#     flux_: np.ndarray = np.array([])

#     wave_model: np.ndarray = np.array([])
#     wave_obs: np.ndarray = np.array([])

#     # step_wave_obs: np.ndarray = np.array([])
#     # step_wave_model: np.ndarray = np.array([])

#     limit_model: np.ndarray = np.array([])
#     limit_obs: np.ndarray = np.array([])

#     # wave_trim_offset: np.ndarray = np.array([0, 1.5])

#     z: float = 0.003129



# class ExecuteProcess:
class DataPreprocessing:


    def __init__(self, model_path, obs_path):
        self.model_path = model_path
        self.obs_path = obs_path

        self.wave_trim_offset = np.array([0, 1.5])
        self.z = 0.003129

        self.pre_prepare()
        self.prepare_model()
        self.prepare_observation()
        self.data_to_mmap(flux_obs, flux_obs_unc, flux_model)

    def pre_prepare(self):
        print('''
              Model pre-preparation
              *********************''')
        print('Starting\n--------\n')

        # Reading a single model to gather information
        print('Reading model data:')
        t = clock()

        model_files = glob.glob(self.model_path)
        with fits.open(model_files[0]) as hdu:
            self.o_first_wave_model = np.double(hdu['PRIMARY'].header['CRVAL1'])
            self.o_step_wave_model = np.double(hdu['PRIMARY'].header['CDELT1'])
            self.o_n_pixel_model = hdu['PRIMARY'].header['NAXIS1']

        self.o_n_model = len(model_files)

        self.o_wave_model = \
            sc.util.build_wave_array([self.o_first_wave_model, self.o_step_wave_model],
                                     sampling_type = 'linear',
                                     size = self.o_n_pixel_model)

        self.o_limit_model = self.o_wave_model[[0,-1]]
        print(f'{round(clock()-t,2)} s\n')


        # Reading observations to gather information
        print('Reading observations data:')
        t = clock()

        with fits.open(self.obs_path) as hdu:
            self.o_first_wave_obs = np.double(hdu['DATA'].header['CRVAL3'])
            self.o_step_wave_obs = np.double(hdu['DATA'].header['CD3_3'])
            self.o_n_pixel_obs = hdu['DATA'].header['NAXIS3']
        self.o_wave_obs = \
            sc.util.build_wave_array([self.o_first_wave_obs, self.o_step_wave_obs],
                                     sampling_type = 'linear',
                                     size = self.o_n_pixel_obs)

        self.o_obs_limit = self.o_wave_obs[[0,-1]]
        print(f'{round(clock()-t,2)} s\n')

    def prepare_model(self, dtype = float):
        global flux_model
        print('''
              Model preparation
              *****************''')
        print('Starting\n--------\n')

        # Reading model data
        print('Reading data:')
        t = clock()

        model_files = glob.glob(self.model_path)

        self.wave_model = \
            sc.util.fit_wave_interval(self.o_wave_model,
                                      old_sampling = 'linear',
                                      new_sampling = 'log',
                                      new_size = self.o_n_pixel_model)

        flux_model = np.zeros((self.o_n_pixel_model, self.o_n_model))
        for j, file in enumerate(model_files):
            with fits.open(model_files[j]) as hdu:
                data = hdu['PRIMARY'].data
                flux_model[:, j] = data
        print(f'{round(clock()-t,2)} s\n')

        # # Convolution
        print('Convolution:')
        t = clock()

        fwhm_model = 2.51 #Vazdekis+10
        fwhm_obs = lsf.equation_lsf(self.wave_model, self.o_obs_limit[0], self.o_obs_limit[1])
        fwhm_dif = np.sqrt((fwhm_obs**2 - fwhm_model**2).clip(0))
        sigma = fwhm_dif / (2.355 * self.o_step_wave_model) # Sigma difference in pixels
        flux_model = convolve(flux = flux_model, sigma = sigma)
        print(f'{round(clock()-t,2)} s\n')

        # Resampling
        print('Resampling:')
        t = clock()

        flux_model, _, _ = sc.resampling(flux = flux_model,
                                         old_wave = self.o_wave_model,
                                         old_sampling_type = 'linear',
                                         new_wave = self.wave_model,
                                         new_sampling_type ='log')
        print(f'{round(clock()-t,2)} s\n')


        # Trimming wavelength range
        print('Trimming to match wavelength range of the observations:')
        t = clock()

        mask_match = (self.wave_model >= self.o_obs_limit[0]) & (self.wave_model <= self.o_obs_limit[1])
        flux_model = flux_model[mask_match, ...]
        self.wave_model = self.wave_model[mask_match, ...]
        print(f'{round(clock()-t,2)} s\n')

        # Trimming with offset value to remove zeros added during convolution
        print('Trimming to remove trailing and leading zeros:')
        t = clock()
        mask_zeros = (self.wave_model >= self.wave_model[0] + self.wave_trim_offset[0]) & \
            (self.wave_model <= self.wave_model[-1] - self.wave_trim_offset[1])

        flux_model = flux_model[mask_zeros, ...]
        self.wave_model = self.wave_model[mask_zeros, ...]
        print(f'{round(clock()-t,2)} s\n')

        #Normalisation
        t = clock()
        print('Normalisation:')
        flux_model = normalize_median(flux_model)
        print(f'{round(clock()-t,2)} s\n')

        # Recording metadata
        print('Recording metadata\n')
        self.n_pixel_model = flux_model.shape[0]
        self.limit_model = self.wave_model[[0,-1]]
        print('Finished\n--------\n')

        flux_model = np.array(flux_model, dtype = dtype)

    def prepare_observation(self, dtype = float):
        global flux_obs, flux_obs_unc
        print('''
              Observation preparation
              ***********************
              ''')
        print('Starting\n--------\n')

        # Reading Data
        print('Reading data:')
        t = clock()
        with fits.open(self.obs_path, memmap = True, lazy_load_hdus = True,
                       cache = False) as hdu:
            flux_obs = np.array(hdu['DATA'].data,
                                dtype = dtype)#[:, 0:20, 0:20]
            del hdu['DATA'].data
            flux_obs_unc = np.array(hdu['STAT'].data,
                                    dtype = dtype)#[:, 0:20, 0:20]
            flux_obs_unc = np.sqrt(flux_obs_unc)
            del hdu['STAT'].data
        print(f'{round(clock()-t,2)} s\n')

        # Resampling
        print('Resampling data:')
        t = clock()

        flux_obs, self.wave_obs, flux_obs_unc = \
            sc.resampling(flux = flux_obs,
                          old_wave = self.o_wave_obs,
                          old_sampling_type = 'linear',
                          new_wave = self.wave_model,
                          new_sampling_type = 'log',
                          flux_err = flux_obs_unc)

        flux_obs = np.array(flux_obs, dtype = dtype)
        flux_obs_unc = np.array(flux_obs_unc, dtype = dtype)
        print(f'{round(clock()-t,2)} s\n')

        # Normalising
        print('Data normalisation:')
        t = clock()

        flux_obs = normalize_median(flux_obs)
        flux_obs_unc = normalize_median(flux_obs_unc)
        print(f'{round(clock()-t,2)} s\n')

        # Reshaping
        print('Data reshaping:')
        t = clock()

        self.shape_obs = flux_obs.shape[1:]

        flux_obs = flux_obs.reshape((-1, np.array(flux_obs.shape[1:]).prod()))
        flux_obs_unc = flux_obs_unc.reshape((-1, np.array(flux_obs_unc.shape[1:]).prod()))
        print(f'{round(clock()-t,2)} s\n')

        # Recording metadata
        print('Recording metadata\n')
        self.n_pixel_obs = len(self.wave_obs)
        self.limit_obs = self.wave_obs[[0,-1]]
        print('Finished\n--------\n')

    def data_to_mmap(self, flux_obs, flux_obs_unc, flux_model):

        mmap_flux_obs = np.memmap('flux_obs.dat', dtype='float32', mode='w+',
                                  shape= flux_obs.shape)
        mmap_flux_obs[:] = flux_obs[:]
        del flux_obs

        mmap_flux_obs_unc = np.memmap('flux_obs_unc.dat', dtype='float32', mode='w+',
                                      shape= flux_obs_unc.shape)
        mmap_flux_obs_unc[:] = flux_obs_unc[:]
        del flux_obs_unc

        mmap_flux_model = np.memmap('flux_model.dat', dtype='float32', mode='w+',
                                    shape= flux_model.shape)
        mmap_flux_model[:] = flux_model[:]
        del flux_model
