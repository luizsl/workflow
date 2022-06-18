#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:55:42 2022

@author: Luiz

Assess the impact of the degree (N_a) of the Legendre polynomial in the 
kinematics measurement.
"""

import os
import yaml
import numpy as np


class Conf:
    def __init__(self, file):
        self.file = file
        self.read_conf()
        
    def read_conf(self):
        "Read the configuration file"
        with open(self.file) as f:
            self.data = yaml.load(f, yaml.Loader)
           
    def update_and_save(self, degree=-1, mdegree=0):
        "Update configuration file with the new polynomial degree"
        self.data['ppxf']['degree'] = int(degree)
        self.data['ppxf_dynamical_mask']['degree'] = int(degree)
        
        self.data['ppxf']['mdegree'] = int(mdegree)
        self.data['ppxf_dynamical_mask']['mdegree'] = int(mdegree)
        
        assert self.data['ppxf']['degree'] == self.data['ppxf_dynamical_mask']['degree']
        assert self.data['ppxf']['mdegree'] == self.data['ppxf_dynamical_mask']['mdegree']
    
        print(f"Additive degree:{self.data['ppxf']['degree']} and multiplicative degree:{self.data['ppxf']['mdegree']}")
        
        with open(self.file,'w+') as f:
            yaml.dump(self.data, f)


class TestDegree:
    def __init__(self, config_file=None, degree_range=[-1], mdegree_range=[0],
                 script_dir=None):
        assert config_file is not None
        assert script_dir is not None 
        
        self.degree_range = degree_range
        self.mdegree_range = mdegree_range
        self.script_dir = script_dir
        self.config_file = config_file
        self.conf = Conf(config_file)
        
        for degree in degree_range:
            for mdegree in mdegree_range:
                self.conf.update_and_save(degree=degree, mdegree=mdegree)
                self.execute_ppxf()
    
    def execute_ppxf(self, script_name='main.py'):
        "Execute ppxf with the updated configuration file"
        try:
            script_path = os.path.join(self.script_dir, script_name)
            os.system(f"nohup python {script_path} {self.config_file}")
        except:
            raise Exception


if __name__ == "__main__":
    degree_range = np.arange(-1, 21, 1)
    mdegree_range = np.arange(0, 21, 1)
    
    config_file_additive = 'assess_additive_polynomial_degree.yaml'
    config_file_multiplicative = 'assess_multiplicative_polynomial_degree.yaml'
    script_dir = '../exploration/run_ppxf'
    
    # Run for a range of additive Legendre polynomial degree
    TestDegree(config_file=config_file_additive, degree_range=degree_range, 
               script_dir=script_dir)
    
    # Run for a range of multiplicative Legendre polynomial degree
    TestDegree(config_file=config_file_multiplicative, 
                mdegree_range=mdegree_range, script_dir=script_dir)
    