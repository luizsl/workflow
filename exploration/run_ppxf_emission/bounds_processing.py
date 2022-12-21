#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:56:06 2022

@author: chess-lin
"""

from functools import partial
from pprint import pprint
from time import perf_counter as clock

import numpy as np
import yaml


def _bounds_interval(value=None, intervals=None):
    intervals = np.asarray(intervals)
    bounds = value + intervals
    return bounds


def bounds_interval(intervals):
    func = partial(_bounds_interval, intervals=intervals)
    return func


def bounds_interval_constructor(loader, node):
    return bounds_interval(loader.construct_sequence(node))


def _bounds_fixed(value=None, intervals=None):
    intervals = np.asarray(intervals)
    return intervals


def bounds_fixed(intervals):
    func = partial(_bounds_fixed, intervals=intervals)
    return func


def bounds_fixed_constructor(loader, node):
    return bounds_fixed(loader.construct_sequence(node))


def build_bounds(values, rule):
    bounds = [func(d) for func, d in zip(rule, values)]
    return bounds


if __name__ == '__main__':
    t = clock()

    conf = ('''
        'bounds': [
            !Interval [-100, -100],
            !Fixed    [  20,  100],
            !Interval [-100,  100],
            !Fixed    [  20,  100],
            !Interval [-500,  500],
            !Fixed    [ 400,   20],
            !Interval [-500,  500],
            !Fixed    [ 400,   20]]
        '''
    )

    loader = yaml.Loader
    loader.add_constructor(tag='!Interval',
                           constructor=bounds_interval_constructor)
    loader.add_constructor(tag='!Fixed',
                           constructor=bounds_fixed_constructor)
    meta = yaml.load(conf, Loader=loader)

    rule = meta['bounds']
    values = np.array([-56.296314, 101.330086, -56.296314, 101.330086,
                       -56.296314, 101.330086, -56.296314, 101.330086])

    bounds = build_bounds(values, rule)
    pprint(bounds)

    print(f'Runtime: {clock() - t} s')
