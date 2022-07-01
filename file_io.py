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
# import subprocess
# import platform
from v3d.io import *


def load_image(img_file: str, vaa3d="vaa3d", temp_dir=None):
    # if img_file.endswith("pbd"):
    #     temp_file = img_file + "_temp.v3draw"
    #     if platform.system() == "Windows":
    #         subprocess.call(" ".join([vaa3d, "/x", "qc200k", "/f", "img_convert",
    #                                   "/i", img_file, "/o", temp_file]))
    #         img = load_v3draw(temp_file)
    #         subprocess.call(" ".join(["rm", temp_file]))
    #     else:
    #         subprocess.call(" ".join([vaa3d, "-x", "qc200k", "-f", "img_convert",
    #                                   "-i", img_file, "-o", temp_file]))
    #         img = load_v3draw(temp_file)
    #         subprocess.call(" ".join("del", temp_file))
    if img_file.endswith(".v3draw"):
        return load_v3draw(img_file)
    if img_file.endswith(".v3dpbd"):
        return PBD().load_image(img_file)
    return sitk.GetArrayFromImage(sitk.ReadImage(img_file))


def save_image(outfile: str, img: np.ndarray):
    if outfile.endswith(".v3draw"):
        save_v3draw(img, outfile)
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
            
