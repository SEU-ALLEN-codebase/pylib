#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : image_utility.py
#   Author       : Yufeng Liu
#   Date         : 2021-07-03
#   Description  : 
#
#================================================================

from image_io import load_image, save_image


def get_mip_image(img3d, axis=0, mode='MAX'):
    if mode == 'MAX':
        img2d = img3d.max(axis=axis)
    elif mode = 'MIN':
        img2d = img3d.min(axis=axis)
    else:
        raise ValueError

    return img2d


class AbastractCropImage:
    def __init__(self):
        pass

