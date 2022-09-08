#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : image_io.py
#   Author       : Yufeng Liu
#   Date         : 2021-05-17
#   Description  : 
#
#================================================================
import SimpleITK as sitk
import pickle
from v3d.io import *
from pathlib import Path

def load_image(img_file):
    img_file = Path(img_file)
    if img_file.suffix in ['.v3draw', '.V3DRAW']:
        return load_v3draw(img_file)
    if img_file.suffix in ['.v3dpbd', '.V3DPBD']:
        return PBD().load_image(img_file)
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
    if img_file.suffix in ['.TIF', '.TIFF', '.tif', '.tiff']:
        img = np.flip(img, axis=-2)
    return img


def save_image(outfile, img: np.ndarray):
    outfile = Path(outfile)
    if outfile.suffix in ['.v3draw', '.V3DRAW']:
        save_v3draw(img, outfile)
    elif outfile.suffix in ['.TIF', '.TIFF', '.tif', '.tiff']:
        sitk.WriteImage(sitk.GetImageFromArray(np.flip(img, axis=-2)), outfile)
    else:
        sitk.WriteImage(sitk.GetImageFromArray(img), outfile)
    return True


def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as fp:
        data = pickle.load(fp)
    return data


def save_pickle(obj, outfile):
    with open(outfile, 'wb') as fp:
        pickle.dump(obj, outfile)


def save_markers(outfile, markers, radius=0, shape=0, name='', comment='', c=(0,0,255)):
    with open(outfile, 'w') as fp:
        fp.write('##x,y,z,radius,shape,name,comment, color_r,color_g,color_b\n')
        for marker in markers:
            x, y, z = marker
            fp.write(f'{x:3f}, {y:.3f}, {z:.3f}, {radius},{shape}, {name}, {comment},0,0,255\n')

