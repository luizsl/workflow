#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 19:23:13 2021

@author: Luiz
"""

from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass

import numpy as np


@dataclass
class Meta:
    # configuration file
    conf_file: str

    #data path
    model_path: str = ''
    obs_path: str = ''

    def __post_init__(self):
        configur = ConfigParser(interpolation=ExtendedInterpolation())
        configur.read(self.conf_file)

        self.model_path = configur.get('resources', 'model')
        self.obs_path = configur.get('resources', 'observation')

        self.model_name = configur.get('model', 'model_name')
        self.obs_name = configur.get('observation', 'obs_name')

        model_wave_trim = configur.get('model', 'wave_trim').split(' ')
        self.model_wave_trim = np.array(model_wave_trim, dtype = float)

        self.model_lsf = configur.getfloat('model', 'lsf')
        self.o_model_sampling = configur.get('model', 'sampling')

        self.o_obs_sampling = configur.get('observation', 'sampling')
        self.z = configur.getfloat('observation', 'redshift')
        self.output_root = configur.get('output', 'output_root')
        self.output_dir = configur.get('output', 'output_dir')

        self.ppxf_moments = int(configur.get('ppxf', 'moments'))
        self.ppxf_degree = int(configur.get('ppxf', 'degree'))
        self.ppxf_clean = configur.getboolean('ppxf', 'clean')
            



# class MyParser(ConfigParser):

#     def as_dict(self):
#         d = dict(self._sections)
#         for k in d:
#             d[k] = dict(self._defaults, **d[k])
#             d[k].pop('__name__', None)
#         return d

# parser = ConfigParser()
# parser.read('test.ini')
# confdict = {section: dict(parser.items(section)) for section in parser.sections()}

# MyParser('test.ini')
