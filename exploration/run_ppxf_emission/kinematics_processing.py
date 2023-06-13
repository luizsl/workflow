#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:18:38 2022

@author: Luiz
"""

import json
# import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import interpolate
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# from abc import ABC, abstractmethod



class Kinematics:
    def __init__(self, kinematics_grid=None, kinematics_unc_grid=None):
        self.kinematics_grid = kinematics_grid
        self.kinematics_unc_grid = kinematics_unc_grid

    def from_file(self, datapath, datapath_unc=None, clip=(None, None)):
        with fits.open(datapath, memmap=True, lazy_load_hdus=True
                       ) as hdul:
            self.kinematics_grid = np.array(hdul[0].data[clip[0]: clip[1]])

        try:
            with fits.open(datapath_unc, memmap=True, lazy_load_hdus=True
                           ) as hdul:
                self.kinematics_unc_grid = \
                    np.array(hdul[0].data[clip[0]: clip[1]])
        except Exception:
            self.kinematics_unc_grid = None

    def reshape(self):
        if self.kinematics_grid.ndim == 3:
            self.shape_kinematics = self.kinematics_grid.shape[1:]
            new_shape = (-1, np.array(self.shape_kinematics).prod())
        elif self.kinematics_grid.ndim == 2:
            new_shape = (-1,) + self.shape_kinematics

        self.kinematics_grid = self.kinematics_grid.reshape(new_shape)

        try:
            self.kinematics_unc_grid = \
                self.kinematics_unc_grid.reshape(new_shape)
        except Exception:
            pass

    def mean_binning(self, bin_num, npixels, valid):
        if self.kinematics_grid.ndim == 3:
            self.reshape()

        self.bin_num = bin_num
        self.npixels = npixels
        self.valid = valid

        with tempfile.TemporaryFile() as temp_kinematics_file, \
                tempfile.TemporaryFile() as temp_kinematics_bin_file:

            kinematicsvalid = np.memmap(
                temp_kinematics_file,
                dtype='float32', mode='w+',
                shape=self.kinematics_grid[:, self.valid].shape)

            kinematicsvalid[:] = self.kinematics_grid[:, self.valid]

            kinematics_bin = np.memmap(
                temp_kinematics_bin_file,
                dtype='float32', mode='w+',
                shape=(self.kinematics_grid.shape[0], self.npixels.size))

            for j in range(self.npixels.size):
                w = self.bin_num == j
                kinematics_bin[:, j] = np.nanmean(kinematicsvalid[:, w],
                                                  axis=1)

            self.kinematics_grid = kinematics_bin

            return kinematics_bin

    def mean_binning_field(self, bin_num, npixels, valid, xbin, ybin):
        binned = self.mean_binning(bin_num, npixels, valid)

        field = Field(kinematics_grid=binned, xbin=xbin, ybin=ybin,
                      shape_obs=self.shape_kinematics, bin_num=bin_num,
                      npixels=npixels, valid=valid)
        return field


class Field:
    def __init__(self, kinematics_grid, xbin, ybin, shape_obs=None,
                 # x_full=None, y_full=None,
                 bin_num=None, npixels=None,
                 valid=None):
        self.kinematics_grid = np.asarray(kinematics_grid)
        self.shape_obs = np.asarray(shape_obs)
        self.bin_num = np.asarray(bin_num)
        self.npixels = np.asarray(npixels)
        self.valid = np.asarray(valid)

        self.xbin = np.asarray(xbin)
        self.ybin = np.asarray(ybin)
        # self.x_full = np.asarray(x_full)
        # self.y_full = np.asarray(y_full)

    @staticmethod
    def filter_stuck(data):
        upper_value = data.max()
        lower_value = data.min()
        mask = np.asarray((data <= lower_value) | (data >= upper_value))
        return mask


class FieldInferece:
    def rbf(self, data, points, grid, units=None, conf=None):
        if data.ndim == 1:
            data = data[None, :]

        if units is None:
            units = np.arange(data.shape[0])
        if conf is None:
            conf = {}

        output_shape = grid.shape[:1] + data.shape[:1]
        rbf_data = np.full(output_shape, fill_value=np.nan)
        for unit in units:
            rbf = interpolate.RBFInterpolator(points, data[unit], **conf)
            rbf_data[:, unit] = rbf(grid)
        return rbf_data

    def inla(self, data, points, grid, units=None, conf=None):
        return NotImplemented


class AnomalyDetection:
    def __init__(self, coordinates, values):
        self.coordinates = coordinates
        self.values = values

    def find_neighbours(self, n_neighbors=None):
        assert n_neighbors is not None
        neighbours = NearestNeighbors(n_neighbors=n_neighbors)
        neighbours.fit(self.coordinates)

        neighbour_graph = neighbours.kneighbors_graph().toarray()
        neighbour_graph = np.asarray(neighbour_graph, bool)

        return neighbour_graph

    def outliers_lof(self, n_neighbors=None, conf={}):
        neighbour_graph = self.find_neighbours(n_neighbors)

        inliers = np.full(self.values.shape, 0)
        for index, value in enumerate(self.values):
            small_neigh = self.values[neighbour_graph[index, :]]
            clf = LocalOutlierFactor(n_neighbors, novelty=True, **conf)
            clf.fit(small_neigh[:, None])

            value = np.asarray(value).reshape(-1, 1)
            pred = clf.predict(value)
            inliers[index] = pred[0]

        outliers = np.full_like(inliers, 0)
        outliers[inliers == -1] = 1
        outliers = outliers.astype(bool)

        return outliers


class Bounds:
    def build(self):
        pass

if __name__ == '__main__':

    # Stellar kinematics

    path_stellar_kinematics = (
        '../../data_products/toy_trick/MilesAgeMh/'
        'ppxf/sol.fits'
        )

    # path_stellar_kinematics_unc = (
    #     '../../data_products/toy_trick/MilesAgeMh/'
    #     'ppxf/sol.fits'
    #     )

    metadatapath = ('../../data_products/toy_trick/MilesAgeMh/ppxf/'
                    'metadata.json')

    with open(metadatapath) as f:
        metadata = json.load(f)
        bin_num = np.array(metadata['obs']['bin_num'])
        npixels = np.array(metadata['obs']['nPixels'])
        valid = np.array(metadata['obs']['valid'])
        xbin = np.array(metadata['obs']['xbin'])
        ybin = np.array(metadata['obs']['ybin'])

    star_kin = Kinematics()
    star_kin.from_file(path_stellar_kinematics)

    star_kin = star_kin.mean_binning_field(bin_num, npixels, valid, xbin, ybin)

    # Gas kinematics

    datapath = ('../../data_products/toy_trick/MilesAgeMh/'
                'ppxf_emission_line_1/sol.fits'
                )
    metadatapath = ('../../data_products/toy_trick/MilesAgeMh'
                    '/ppxf_emission_line_1/metadata.json'
                    )

    with open(metadatapath) as f:
        metadata = json.load(f)
        bin_num = np.array(metadata['obs']['bin_num'])
        npixels = np.array(metadata['obs']['nPixels'])
        valid = np.array(metadata['obs']['valid'])
        xbin = np.array(metadata['obs']['xbin'])
        ybin = np.array(metadata['obs']['ybin'])
        x_full = np.array(metadata['obs']['x_full'])
        y_full = np.array(metadata['obs']['y_full'])

    gas_kin = Kinematics()
    gas_kin.from_file(path_stellar_kinematics)
    gas_kin.reshape()

    gas_field = gas_kin.mean_binning_field(bin_num, npixels, valid, xbin, ybin)

    #  Find neighbour
    n_neighbors = 3
    iteractions = 3
    values = gas_field.kinematics_grid[0]
    coordinates = np.vstack([gas_field.xbin, gas_field.ybin]).T

    # Stuck detector
    stuck = gas_field.filter_stuck(values)

    # Anomaly detector test
    outliers = np.full_like(values, False, dtype=bool)
    for i in range(iteractions):
        valid = np.logical_and(~stuck, ~outliers)
        # print(valid.sum())
        detector_data = values[valid]
        detector_coordinates = coordinates[valid]
        detector = AnomalyDetection(detector_coordinates, detector_data)

        outliers[valid] = detector.outliers_lof(n_neighbors=n_neighbors)

    # Field inference
    # without filtering
    inference = FieldInferece()
    grid = np.asarray((x_full, y_full)).T
    points = coordinates
    data = values

    res_no_filter = inference.rbf(data, points, grid)
    res_no_filter = res_no_filter.reshape(gas_kin.shape_kinematics)
    res_no_filter = np.clip(res_no_filter, data.min(), data.max())

    # with filtering
    valid = ~np.logical_or(stuck, outliers)
    inference = FieldInferece()
    grid = np.asarray((x_full, y_full)).T
    points = coordinates[valid]
    data = values[valid]

    rbf_dict = {'smoothing': 0.1}
    res_filter = inference.rbf(data, points, grid, conf=rbf_dict)
    res_filter = np.clip(res_filter, data.min(), data.max())
    res_filter = res_filter.reshape(gas_kin.shape_kinematics)


    # plot

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    cmap = 'inferno'
    common = {'vmin': data.min(), 'vmax': data.max(), 'alpha': 0.8}

    base = axs[0, 0].scatter(gas_field.xbin, gas_field.ybin, c=values.data,
                             cmap=cmap, s=2, **common)
    axs[0, 0].set_aspect(1)
    axs[0, 0].set_facecolor('#dfe9e3')

    axs[0, 1].scatter(gas_field.xbin[~outliers], gas_field.ybin[~outliers],
                      c=values[~outliers],
                      cmap=cmap, s=2,
                      **common)
    axs[0, 1].scatter(gas_field.xbin[stuck], gas_field.ybin[stuck],
                      c=values[stuck],
                      cmap=cmap, s=2, **common)
    axs[0, 1].scatter(gas_field.xbin[outliers], gas_field.ybin[outliers],
                      c=values[outliers],
                      cmap=cmap, s=2, **common)
    axs[0, 1].scatter(gas_field.xbin[outliers], gas_field.ybin[outliers],
                      s=5, edgecolors="red", facecolors='none',
                      label='Local anomaly', **common)
    axs[0, 1].scatter(gas_field.xbin[stuck], gas_field.ybin[stuck],
                      s=5, edgecolors='green', facecolors='none',
                      label='Stuck at bounds', **common)
    axs[0, 1].set_aspect(1)
    axs[0, 1].set_facecolor('#dfe9e3')

    extent = [x_full.min(), x_full.max(),
              y_full.min(), y_full.max()]
    axs[1, 0].imshow(res_no_filter, origin='lower', cmap=cmap, extent=extent,
                     **common)
    axs[1, 1].imshow(res_filter, origin='lower', cmap=cmap, extent=extent,
                     **common)

    axs[0, 0].set_ylabel('arcsec')
    axs[1, 0].set_ylabel('arcsec')
    axs[1, 0].set_xlabel('arcsec')
    axs[1, 1].set_xlabel('arcsec')

    fig.legend()

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(base, cax = cbar_ax)

    fig.tight_layout()
