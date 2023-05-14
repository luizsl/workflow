#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:47:28 2022

@author: Luiz
"""

from abc import ABC, abstractmethod
from itertools import product
from time import perf_counter as clock

import matplotlib.pyplot as plt
import numpy as np
import ppxf.ppxf_util as util
import yaml


class LineFactory(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def _apply_ratio(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass


class Line(LineFactory):
    def __init__(self, line_wave, line_name, label, spectral_axis, fwhm,
                 ratios=None):
        min_ = spectral_axis.min() < line_wave
        max_ = line_wave < spectral_axis.max()
        assert np.all(min_ & max_), 'Emission-line outside spectral axis'

        self.line_wave = np.asarray(line_wave)
        self.line_name = np.asarray(line_name)
        self.label = np.asarray(label)
        self.label_wave = np.mean(self.line_wave)
        self.spectral_axis = np.asarray(spectral_axis)
        self.fwhm = fwhm
        self.ratios = np.asarray(ratios)

        self.template = util.gaussian(
            ln_lam_temp=np.log(self.spectral_axis),
            line_wave=self.line_wave,
            FWHM_gal=fwhm)

        if ratios is not None:
            self._apply_ratio()

    def _apply_ratio(self):
        assert self.template.shape[1] == self.ratios.shape[0]

        self.template = self.template @ self.ratios

        if self.ratios.ndim == 2:
            range_ = range(1, self.ratios.shape[1] + 1)
            iter_name = product(range_, self.line_name)
            names = [name + f'_({n})' for n, name in iter_name]
            self.line_name = np.asarray(names)

            labels = [str(self.label) + f'_({n})' for n in range_]
            self.label = np.asarray(labels)

            label_wave = np.tile(self.label_wave, (self.ratios.shape[1], 1))
            self.label_wave = label_wave

            self.spectral_axis = np.tile(
                self.spectral_axis, (self.ratios.shape[1], 1)).T

    def __str__(self):
        _str = (
            f"label: {self.label}\n"
            f"\tline_name: {self.line_name}\n"
            f"\tline_wave: {self.line_wave}"
            )
        return _str

    def __repr__(self):
        _str = (
            "Line(line_wave={}, line_name={}, label={}, spectral_axis={},"
            "fwhm={}, ratios={})"
            )
        _str = _str.format(
            self.line_wave, self.line_name, self.label,
            self.spectral_axis, self.fwhm, self.ratios)
        return _str


class KinematicsGroup:
    def __init__(self):
        self.lines = []

    def add(self, line_list):
        assert isinstance(line_list, list)
        self.lines.extend(line_list)

    @property
    def line_name(self):
        group = np.array([])
        for line in self.lines:
            group = np.append(group, line.line_name)
        return group

    @property
    def line_wave(self):
        group = np.array([])
        for line in self.lines:
            group = np.append(group, line.line_wave)
        return group

    @property
    def label(self):
        group = np.array([])
        for line in self.lines:
            group = np.append(group, line.label)
        return group

    @property
    def label_wave(self):
        group = np.array([])
        for line in self.lines:
            group = np.append(group, line.label_wave)
        return group

    @property
    def template(self):
        group_template = [line.template for line in self.lines]
        group_template = np.column_stack(group_template)
        return group_template

    @property
    def spectral_axis(self):
        group_spectral_axis = [line.spectral_axis for line in self.lines]
        group_spectral_axis = np.column_stack(group_spectral_axis)
        return group_spectral_axis

    def __str__(self):
        str_ = '\n'.join(line.__str__() for line in self.lines)
        return str_

    @property
    def size(self):
        len_ = len(self.label)
        return len_


class EmissionModel:
    def __init__(self):
        self.names = []

    def _add(self, name, object_):
        self.names.append(name)
        self.__setattr__(name, object_)

    def from_file(self, path, spectral_axis, fwhm):
        with open(path) as f:
            self._line_list = yaml.full_load(f)

        for key, value in self._line_list.items():
            if 'line_name' and 'line_wave' in value:
                # build_line
                try:
                    line = Line(**value, label=key,
                                spectral_axis=spectral_axis, fwhm=fwhm)
                    self._add(key, line)
                except Exception:
                    print(f'{key} not included')
            elif 'line_name' and 'line_wave' not in value:
                # build group
                group = KinematicsGroup()
                for subkey, subvalue in value.items():
                    if 'line_name' and 'line_wave' in subvalue:
                        try:
                            # build line
                            line = Line(**subvalue, label=subkey,
                                        spectral_axis=spectral_axis, fwhm=fwhm)
                            # add to group
                            group.add([line])
                        except Exception:
                            print(f'{subkey} not included'.center(80, '-'))
                    else:
                        raise Exception
                self._add(key, group)
                del group

    @property
    def template(self):
        _ = [self.__getattribute__(name).template for name in self.names]
        array = np.column_stack(_)
        return array

    @property
    def spectral_axis(self):
        _ = [self.__getattribute__(name).spectral_axis for name in self.names]
        array = np.column_stack(_)
        return array

    @property
    def line_name(self):
        _ = [self.__getattribute__(name).line_name for name in self.names]
        array = np.concatenate(_)
        return array

    @property
    def line_wave(self):
        _ = [self.__getattribute__(name).line_wave for name in self.names]
        array = np.concatenate(_)
        return array

    @property
    def label(self):
        _ = [self.__getattribute__(name).label for name in self.names]
        array = np.concatenate(_)
        return array

    @property
    def label_wave(self):
        _ = [self.__getattribute__(name).label_wave for name in self.names]
        array = np.concatenate(_)
        return array

    @property
    def size(self):
        len_ = [self.__getattribute__(name).size for name in self.names]
        return len_


if __name__ == '__main__':
    _t = clock()

    path = 'emission_line_list.yaml'
    spectral_axis = 4750 + np.arange(0, 3680, 1.25)

    from functools import partial

    from compute_muse_lsf import equation_lsf
    lsf_lamb = partial(equation_lsf, lower_lamb=None, upper_lamb=None, z=0.005)
    # lsf_muse = 0

    em_model = EmissionModel()
    em_model.from_file(path, spectral_axis, lsf_lamb)

    for name in em_model.names:
        print(f"Group: {name}")
        print(em_model.__getattribute__(name), end='\n\n')

    print(f'Runtime: {clock() - _t: .3f} s')
    fig, ax = plt.subplots()
    ax.plot(em_model.spectral_axis, em_model.template, color='k')

    # # Comparing with ppxf tool for template building
    # gas_templates, gas_names, line_wave = \
    #     util.emission_lines(
    #         ln_lam_temp=np.log(spectral_axis),
    #         lam_range_gal=[spectral_axis.min(), spectral_axis.max()],
    #         FWHM_gal=lsf_lamb,
    #         pixel=True, tie_balmer=False,
    #         limit_doublets=True, vacuum=False)

    # fig, ax = plt.subplots()
    # ax.plot(spectral_axis, gas_templates, color='red')
