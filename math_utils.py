#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : math_utils.py
#   Author       : Yufeng Liu
#   Date         : 2021-07-19
#   Description  : 
#
#================================================================

import numpy as np

def calc_included_angles_from_vectors(vecs1, vecs2, return_rad=False, epsilon=1e-7, spacing=None, return_cos=False):
    if vecs1.ndim == 1:
        vecs1 = vecs1.reshape((1,-1))
    if vecs2.ndim == 1:
        vecs2 = vecs2.reshape((1,-1))

    if spacing is not None:
        spacing_reshape = np.array(spacing).reshape(1,-1)
        # rescale vectors according to spacing
        vecs1 = vecs1 * spacing_reshape
        vecs2 = vecs2 * spacing_reshape

    inner = (vecs1 * vecs2).sum(axis=1)
    norms = np.linalg.norm(vecs1, axis=1) * np.linalg.norm(vecs2, axis=1)
    cos_ang = inner / (norms + epsilon)

    if return_cos:
        return_val = cos_ang
    else:
        rads = np.arccos(np.clip(cos_ang, -1, 1))
        if return_rad:
            return_val = rads
        else:
            return_val = np.rad2deg(rads)
    return return_val    

def calc_included_angles_from_coords(anchor_coords, coords1, coords2, return_rad=False, epsilon=1e-7, spacing=None, return_cos=False):
    anchor_coords = np.array(anchor_coords)
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    v1 = coords1 - anchor_coords
    v2 = coords2 - anchor_coords
    angs = calc_included_angles_from_vectors(
                v1, v2, return_rad=return_rad, 
                epsilon=epsilon, spacing=spacing,
                return_cos=return_cos)
    return angs

