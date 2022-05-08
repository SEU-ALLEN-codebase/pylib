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
import os
import sys
import struct
import numpy as np


def load_v3draw(path):
    """
    by Zuohan Zhao
    from basic_c_fun/stackutils.cpp
    2022/5/8
    """
    assert(os.path.exists(path))
    formatkey = "raw_image_stack_by_hpeng"
    with open(path, "rb") as f:
        filesize = os.path.getsize(path)
        assert(filesize >= len(formatkey) + 2 + 4*4 + 1)
        format = f.read(len(formatkey)).decode('utf-8')
        assert(format == formatkey)
        endianCodeData = f.read(1).decode('utf-8')
        if endianCodeData == 'B':
            endian = '>'
        elif endianCodeData == 'L':
            endian = '<'
        else:
            raise Exception('endian be big/little')
        datatype = struct.unpack(endian+'h', f.read(2))[0]
        if datatype == 1:
            dt = np.uint8
        elif datatype == 2:
            dt = np.uint16
        elif datatype == 4:
            dt = np.float32
        else:
            raise Exception('datatype be 1/2/4')
        sz = struct.unpack(endian+'iiii', f.read(4*4))
        tot = sz[0] * sz[1] * sz[2] * sz[3]
        if tot * datatype + 4*4 + 2 + 1 + len(formatkey) != filesize:
            f.seek(-4*2, 1)
            tot = sz[0] * sz[1] * sz[2] * sz[3]
            assert(tot * datatype + 4 * 4 + 2 + 1 + len(formatkey) == filesize)
        img = np.frombuffer(f.read(tot), dt)
        return img.reshape(sz[-1:-5:-1])


def load_image(imgfile):
    return sitk.GetArrayFromImage(sitk.ReadImage(imgfile))

def save_image(outfile, img):
    sitk.WriteImage(sitk.GetImageFromArray(img), outfile)
    return True

def load_pickle(pklfile):
    with open(pklfile, 'rb') as fp:
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
            

