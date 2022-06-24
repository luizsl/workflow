#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:55:42 2022

@author: Luiz

Assess the impact of the degree (N_a) of the Legendre polynomial in the
kinematics measurement.
"""

import os

import numpy as np
import yaml


class Conf:
    def __init__(self, file):
        self.file = file
        self.read_conf()

    def read_conf(self):
        "Read the configuration file"
        with open(self.file) as f:
            self.data = yaml.load(f, yaml.Loader)

    def update_and_save(self, degree=-1, mdegree=0, fixed_degree=None,
                        fixed_mdegree=None):
        "Update configuration file with the new polynomial degree"

        if fixed_degree is None:
            fixed_degree = degree
        if fixed_mdegree is None:
            fixed_mdegree = mdegree

        self.data['ppxf']['degree'] = int(fixed_degree)
        self.data['ppxf_dynamical_mask']['degree'] = int(degree)

        self.data['ppxf']['mdegree'] = int(fixed_mdegree)
        self.data['ppxf_dynamical_mask']['mdegree'] = int(mdegree)


        print(f"Additive fixed degree:{self.data['ppxf']['degree']} and multiplicative fixed degree:{self.data['ppxf']['mdegree']}")

        print(f"Additive degree:{self.data['ppxf_dynamical_mask']['degree']} and multiplicative degree:{self.data['ppxf_dynamical_mask']['mdegree']}")

        with open(self.file,'w+') as f:
            yaml.dump(self.data, f)


class TestDegree:
    def __init__(self, config_file=None, degree_range=[-1], mdegree_range=[0],
                 script_dir=None, fixed_degree=-1, fixed_mdegree=0):
        assert config_file is not None
        assert script_dir is not None

        self.degree_range = degree_range
        self.mdegree_range = mdegree_range
        self.script_dir = script_dir
        self.config_file = config_file
        self.conf = Conf(config_file)

        for degree in degree_range:
            for mdegree in mdegree_range:
                self.conf.update_and_save(degree=degree, mdegree=mdegree,
                                          fixed_degree=fixed_degree,
                                          fixed_mdegree=fixed_mdegree)
                self.execute_ppxf()

    def execute_ppxf(self, script_name='main.py'):
        "Execute ppxf with the updated configuration file"
        try:
            script_path = os.path.join(self.script_dir, script_name)
            os.system(f"nohup python {script_path} {self.config_file}")
        except:
            raise Exception


def ifu_data(degree_range, mdegree_range):
    '''Testing the effects of polynomial degree on pPXF for IFU data'''

    script_dir = '../exploration/run_ppxf'

    # E-MILES
    config_file_add_emiles = 'polynomial_degree_conf/assess_add_pol_degree_emiles.yaml'
    config_file_mlt_emiles = 'polynomial_degree_conf/assess_mlt_pol_degree_emiles.yaml'


    # Run for a range of additive Legendre polynomial degree
    TestDegree(config_file=config_file_add_emiles, degree_range=degree_range,
               script_dir=script_dir,
               fixed_degree=8, fixed_mdegree=0)

    # Run for a range of multiplicative Legendre polynomial degree
    TestDegree(config_file=config_file_mlt_emiles,
               mdegree_range=mdegree_range, script_dir=script_dir,
               fixed_degree=-1, fixed_mdegree=8)

    # MILES
    config_file_add_miles = 'polynomial_degree_conf/assess_add_pol_degree_miles.yaml'
    config_file_mlt_miles = 'polynomial_degree_conf/assess_mlt_pol_degree_miles.yaml'


    # Run for a range of additive Legendre polynomial degree
    TestDegree(config_file=config_file_add_miles, degree_range=degree_range,
               script_dir=script_dir,
               fixed_degree=8, fixed_mdegree=0)

    # Run for a range of multiplicative Legendre polynomial degree
    TestDegree(config_file=config_file_mlt_miles,
               mdegree_range=mdegree_range, script_dir=script_dir,
               fixed_degree=-1, fixed_mdegree=8)

    # XSL
    config_file_add_xsl = 'polynomial_degree_conf/assess_add_pol_degree_xsl.yaml'
    config_file_mlt_xsl = 'polynomial_degree_conf/assess_mlt_pol_degree_xsl.yaml'


    # Run for a range of additive Legendre polynomial degree
    TestDegree(config_file=config_file_add_xsl, degree_range=degree_range,
               script_dir=script_dir,
               fixed_degree=8, fixed_mdegree=0)

    # Run for a range of multiplicative Legendre polynomial degree
    TestDegree(config_file=config_file_mlt_xsl,
               mdegree_range=mdegree_range, script_dir=script_dir,
               fixed_degree=-1, fixed_mdegree=8)

def integrated_spectrum(degree_range, mdegree_range):
    '''Testing the effects of polynomial degree on pPXF for IFU data
    collapsed into a single spectrum'''

    script_dir = '../exploration/run_ppxf'

    # E-MILES
    config_file_add_emiles = 'polynomial_degree_conf/assess_add_pol_degree_emiles_single.yaml'
    config_file_mlt_emiles = 'polynomial_degree_conf/assess_mlt_pol_degree_emiles_single.yaml'


    # Run for a range of additive Legendre polynomial degree
    TestDegree(config_file=config_file_add_emiles, degree_range=degree_range,
               script_dir=script_dir,
               fixed_degree=8, fixed_mdegree=0)

    # Run for a range of multiplicative Legendre polynomial degree
    TestDegree(config_file=config_file_mlt_emiles,
               mdegree_range=mdegree_range, script_dir=script_dir,
               fixed_degree=-1, fixed_mdegree=8)

    # MILES
    config_file_add_miles = 'polynomial_degree_conf/assess_add_pol_degree_miles_single.yaml'
    config_file_mlt_miles = 'polynomial_degree_conf/assess_mlt_pol_degree_miles_single.yaml'


    # Run for a range of additive Legendre polynomial degree
    TestDegree(config_file=config_file_add_miles, degree_range=degree_range,
               script_dir=script_dir,
               fixed_degree=8, fixed_mdegree=0)

    # Run for a range of multiplicative Legendre polynomial degree
    TestDegree(config_file=config_file_mlt_miles,
               mdegree_range=mdegree_range, script_dir=script_dir,
               fixed_degree=-1, fixed_mdegree=8)

    # XSL
    config_file_add_xsl = 'polynomial_degree_conf/assess_add_pol_degree_xsl_single.yaml'
    config_file_mlt_xsl = 'polynomial_degree_conf/assess_mlt_pol_degree_xsl_single.yaml'


    # Run for a range of additive Legendre polynomial degree
    TestDegree(config_file=config_file_add_xsl, degree_range=degree_range,
               script_dir=script_dir,
               fixed_degree=8, fixed_mdegree=0)

    # Run for a range of multiplicative Legendre polynomial degree
    TestDegree(config_file=config_file_mlt_xsl,
               mdegree_range=mdegree_range, script_dir=script_dir,
               fixed_degree=-1, fixed_mdegree=8)

if __name__ == "__main__":
    degree_range = np.arange(-1, 21, 1)
    mdegree_range = np.arange(0, 21, 1)

    # ifu_data(degree_range, mdegree_range)

    # os.system("nohup python stack_on_a_single_spectrum.py")
    integrated_spectrum(degree_range, mdegree_range)
