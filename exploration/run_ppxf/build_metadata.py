#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 19:23:13 2021

@author: Luiz
"""

import numpy as np
import _pickle as pickle
from dataclasses import dataclass
from configparser import ConfigParser
from configparser import ExtendedInterpolation

@dataclass
class Meta:
    # configuration file
    conf_file: str
    
    #data path
    model_path: str = ''
    obs_path: str = ''
    
    # model lsf
    model_lsf: float = 0
    
    # original model sampling
    o_model_sampling: str = ''
    
    #number of models
    n_model: int = 0

    # original wavelength steps of model and data 
    o_step_wave_model: float = 0
    o_step_wave_obs: float = 0

    # original and new number of pixels of model and data 
    o_n_pixel_model: int = 0
    o_n_pixel_obs: int = 0
    n_pixel_model: int = 0
    n_pixel_obs: int = 0
    
    # arrays with original and new wavelengths
    o_wave_model: np.ndarray = np.array([])
    o_wave_obs: np.ndarray = np.array([])
    wave_model: np.ndarray = np.array([])
    wave_obs: np.ndarray = np.array([])
    
    # redundant information
    o_limit_model: np.ndarray = np.array([])
    o_limit_obs: np.ndarray = np.array([])
    limit_model: np.ndarray = np.array([])
    limit_obs: np.ndarray = np.array([])
    o_first_wave_model = float = 0
    o_first_wave_obs = float = 0
    
    # original shape of observational data structure
    shape_obs: tuple = ()
    # Trim to remove leading and trailing zeros
    model_wave_trim: np.ndarray = np.array([])
    
    # observation redshift
    z = float
    
    # temporary directories
    temp_input_dir = str = ''
    temp_output_dir = str = ''

    def __post_init__(self):
        configur = ConfigParser(interpolation=ExtendedInterpolation())
        configur.read(self.conf_file)

        self.model_path = configur.get('resources', 'model')
        self.obs_path = configur.get('resources', 'observation')
        
        model_wave_trim = configur.get('model', 'wave_trim').split(' ')
        self.model_wave_trim = np.array(model_wave_trim, dtype = float)
    
        self.model_lsf = configur.getfloat('model', 'lsf')
        self.o_model_sampling = configur.get('model', 'sampling')
        
        self.z = configur.getfloat('observation', 'redshift')
        self.output_dir = configur.get('output', 'output_dir')
