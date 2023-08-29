#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:34:21 2023

@author: Luiz
"""

import extinction
import numpy as np
import importlib
from astropy.coordinates import SkyCoord


def dered(spectrum, wave=None, law='calzetti00', r_v=4.05, ebv=None, a_v=None):
    assert any(_ is not None for _ in [a_v, ebv])
    assert wave is not None

    if a_v is None:
        a_v = ebv * r_v

    if ebv is None:
        ebv = a_v / r_v

    if law == 'fm07':
        ext_mag = extinction.__getattribute__(law)(wave=wave, a_v=a_v)
    else:
        ext_mag = extinction.__getattribute__(law)(wave=wave, a_v=a_v, r_v=r_v)

    dered_spectrum = extinction.remove(ext_mag, spectrum)

    return dered_spectrum, a_v, ebv


def get_dust_map(map_name=None, mode='map', obs_wcs=None):
    assert map_name is not None
    assert mode in ['map', 'single']
    assert obs_wcs is not None

    query = _get_dust_query(map_name)

    if mode == 'map':
        coords = _map_coord(obs_wcs)
    elif mode == 'single':
        coords = _single_coord(obs_wcs)
    else:
        raise Exception

    dust_map = query(coords)

    return dust_map


def _get_slice(obs_wcs):
    slices = np.array([], dtype=bool)
    for n, axis in enumerate(obs_wcs.world_axis_object_components):
        if axis[0] == 'celestial':
            slices = np.append(slices, True)
        else:
            slices = np.append(slices, False)

    return slices


def _get_unit(obs_wcs):
    unit = set()
    for n, axis in enumerate(obs_wcs.world_axis_object_components):
        if axis[0] == 'celestial':
            unit.add(obs_wcs.world_axis_units[n])
    unit = list(unit)[0]

    return unit

def _single_coord(obs_wcs):
    slices = _get_slice(obs_wcs)
    unit = _get_unit(obs_wcs)
    x, y = obs_wcs.wcs.crpix[slices]
    z = 1

    pix_coords = np.array([x, y, z])
    world_coords = obs_wcs.all_pix2world(*pix_coords, 1)

    sky_coords = SkyCoord(world_coords[0], world_coords[1], unit=unit)

    return sky_coords


def _map_coord(obs_wcs):
    slices = _get_slice(obs_wcs)
    unit = _get_unit(obs_wcs)
    axes = np.array(obs_wcs.pixel_shape)[slices]
    xp, yp = np.meshgrid(*list(map(np.arange , axes)))

    xpr = xp.ravel()
    ypr = yp.ravel()
    zpr = np.zeros_like(xpr)

    pix_coords = np.array([xpr, ypr, zpr]).T
    world_coords = obs_wcs.all_pix2world(pix_coords, 0)

    sky_coords = SkyCoord(world_coords[:, 0], world_coords[:, 1], unit=unit)
    sky_coords = sky_coords.reshape(xp.shape)

    return sky_coords


def _get_dust_query(map_name):
    module = importlib.import_module(f'dustmaps.{map_name}')
    name = module.__name__.split('.')[-1] + 'query'

    for i in dir(module):
        if name.lower() == i.lower():
            query_class = getattr(module, i)
            break
    query = query_class()

    return query


if __name__ == '__main__':
    pass
#%%
    a_v = 'a_v'
    ebv = None
    assert (a_v, ebv) != (None, None)
