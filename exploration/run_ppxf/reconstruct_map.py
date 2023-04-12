#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:02:43 2022

@author: Luiz
"""
import json
import os

import numpy as np
from astropy.io import fits


def reconstruct_map(data=None, out_metadata=None, parameter=None,
                    binned=None, save=True, directory=None):
    assert binned is not None
    assert isinstance(parameter, str)

    try:
        valid = np.asarray(out_metadata['obs']['valid'])
        bin_num = np.asarray(out_metadata['obs']['bin_num'])
    except Exception:
        valid = None
        bin_num = None
    finally:
        shape_obs = tuple(out_metadata['obs']['shape_obs'])

    data = np.asarray(data)

    if data.dtype.kind in ('biufcmM'):
        pass
    else:
        raise Exception (f'{parameter} cannot be converted into a map')

    if binned is True:
        if data.ndim < 2:
            map_shape = bin_num.shape
        elif data.ndim >= 2:
            map_shape = (data.shape[:1] + bin_num.shape)

        map_ = np.zeros(map_shape)
        for i in range(data.shape[-1]):
            match = bin_num == i
            map_[..., match] = data[..., i:i+1]

        map_shape_full = np.array(shape_obs).prod()
        if data.ndim >= 2:
            map_shape_full = (data.shape[:1] + (map_shape_full,))

        map_full = np.full(map_shape_full, fill_value=np.nan)
        map_full[..., valid] = map_

        if map_full.ndim < 2:
            new_shape = shape_obs
        elif map_full.ndim >= 2:
            new_shape = (-1,) + shape_obs
        map_full = map_full.reshape(new_shape)

    else:
        map_shape_full = shape_obs
        if data.ndim >= 2:
            map_shape_full = (data.shape[:1] + map_shape_full)
        map_full = data.reshape(map_shape_full)

    if save:
        # directory = out_metadata['conf']['output_run']
        save_fits(map_full, parameter, directory)


def save_fits(data_param, name, directory='.', overwrite=True):
    data_param = np.array(data_param, dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data_param)
    hdul = fits.HDUList([hdu])
    full_path = os.path.join(directory, f'{name}.fits')
    hdul.writeto(full_path, overwrite=overwrite)


if __name__ == '__main__':
    file_metadata = ('../../data_products/toy_trick/MilesAgeMh/'
                     'ppxf_emission_line_binned300_2components/'
                     'metadata.json'
                     )

    file_output = ('../../data_products/toy_trick/MilesAgeMh/'
                   'ppxf_emission_line_binned300_2components/'
                   'ppxf_output.json'
                   )

    with open(file_output) as fp:
        out_ppxf = json.load(fp)

    with open(file_metadata) as fp:
        out_metadata = json.load(fp)

    reconstruct_map(data=out_ppxf, out_metadata=out_metadata,
                    parameter='sol', binned=True, save=False)
