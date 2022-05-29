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
import struct
import numpy as np
import sys


    MDATA_BIN_FILE_NAME  = "mdata.bin"       # name of binary metadata file
    MDATA_BIN_FILE_VERSION = 2               # version of binary metadata file
    MC_MDATA_BIN_FILE_NAME = "cmap.bin"      # name of binary metadata file for multichannel volumes
    FORMAT_MDATA_FILE_NAME = ".iim.format"   # name of format metadata file
    CHANNEL_PREFIX = "CH_"                   # prefix identifying a folder containing data of a certain channel
    TIME_FRAME_PREFIX = "T_"                 # prefix identifying a folder containing data of a certain time frame
    DEF_IMG_DEPTH = 8                        # default image depth
    NUL_IMG_DEPTH = 0                        # invalid image depth
    NATIVE_RTYPE  = 0                        # loadVolume returns the same bytes per channel as in the input image
    DEF_IMG_FORMAT = "tif"                   # default image format
    STATIC_STRINGS_SIZE = 1024               # size of static C-strings
    RAW_FORMAT            = "Vaa3D raw"                  # unique ID for the RawVolume class
    SIMPLE_RAW_FORMAT     = "Vaa3D raw (series, 2D)"     # unique ID for the SimpleVolumeRaw class
    STACKED_RAW_FORMAT    = "Vaa3D raw (tiled, 2D)"      # unique ID for the StackedVolume class
    TILED_FORMAT          = "Vaa3D raw (tiled, 3D)"      # unique ID for the TiledVolume class
    TILED_MC_FORMAT       = "Vaa3D raw (tiled, 4D)"      # unique ID for the TiledMCVolume class
    TIF3D_FORMAT          = "TIFF (3D)"                  # unique ID for multipage TIFF format (nontiled)
    SIMPLE_FORMAT         = "TIFF (series, 2D)"          # unique ID for the SimpleVolume class
    STACKED_FORMAT        = "TIFF (tiled, 2D)"           # unique ID for the StackedVolume class
    TILED_TIF3D_FORMAT    = "TIFF (tiled, 3D)"           # unique ID for multipage TIFF format (tiled)
    TILED_MC_TIF3D_FORMAT = "TIFF (tiled, 4D)"           # unique ID for multipage TIFF format (nontiled, 4D)
    UNST_TIF3D_FORMAT     = "TIFF (unstitched, 3D)"      # unique ID for multipage TIFF format (nontiled, 4D)
    BDV_HDF5_FORMAT       = "HDF5 (BigDataViewer)"       # unique ID for BDV HDF5
    IMS_HDF5_FORMAT       = "HDF5 (Imaris IMS)"          # unique ID for IMS HDF5
    MAPPED_FORMAT         = "Mapped Volume"              # unique ID for mapped volumes
    MULTISLICE_FORMAT     = "MultiSlice Volume"          # unique ID for multi-slice volumes
    MULTICYCLE_FORMAT     = "MultiCycle Volume"			# unique ID for multi-cycle volumes
    VOLATILE_FORMAT       = "Volatile Volume"            # unique ID for volatile volumes
    TIME_SERIES           = "Time series"                # unique ID for the TimeSeries class


def load_v3draw(path: str):
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
            dt = 'u1'
        elif datatype == 2:
            dt = 'u2'
        elif datatype == 4:
            dt = 'f4'
        else:
            raise Exception('datatype be 1/2/4')
        sz = struct.unpack(endian+'iiii', f.read(4*4))
        tot = sz[0] * sz[1] * sz[2] * sz[3]
        if tot * datatype + 4*4 + 2 + 1 + len(formatkey) != filesize:
            f.seek(-4*2, 1)
            tot = sz[0] * sz[1] * sz[2] * sz[3]
            assert(tot * datatype + 4 * 4 + 2 + 1 + len(formatkey) == filesize)
        img = np.frombuffer(f.read(tot), endian+dt)
        return img.reshape(sz[-1:-5:-1])


def save_v3draw(img: np.ndarray, path: str):
    """
    by Zuohan Zhao
    from basic_c_fun/stackutils.cpp
    2022/5/8
    """
    with open(path, 'wb') as f:
        formatkey = "raw_image_stack_by_hpeng"
        f.write(formatkey.encode())
        if img.dtype.byteorder == '>':
            endian = 'B'
        elif img.dtype.byteorder == '<':
            endian = 'L'
        elif img.dtype == '|':
            endian = 'B'
        else:
            if sys.byteorder == 'little':
                endian = 'L'
            else:
                endian = 'B'
        f.write(endian.encode())
        if img.dtype == np.uint8:
            datatype = 1
        elif img.dtype == np.uint16:
            datatype = 2
        else:
            datatype = 4
        endian = '>' if endian == 'B' else '<'
        f.write(struct.pack(endian+'h', datatype))
        sz = list(img.shape)
        sz.extend([0] * (4 - len(sz)))
        sz.reverse()
        f.write(struct.pack(endian+'iiii', *sz))
        f.write(img.tobytes())


def load_image(imgfile: str):
    if imgfile.endswith(".v3draw"):
        return load_v3draw(imgfile)
    return sitk.GetArrayFromImage(sitk.ReadImage(imgfile))


def save_image(outfile: str, img: np.ndarray):
    if outfile.endswith(".v3draw"):
        save_v3draw(img, outfile)
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
            

def getSubVolumeTerafly(dir: str, coord_from, coord_to):
    # open dir
    # try getting format
    with open(os.path.join(dir, ".iim.format")) as f:
        format = f.readline()
        if format == "Vaa3D raw (tiled, 4D)":
            pass
        elif format == "TIFF (tiled, 2D)":
            pass
        elif format == "Vaa3D raw (tiled, 3D)":
            pass
        elif format == "":
            pass
        elif format == "":
            pass
        elif format == "":
            pass
        else:
            print()
    # or try each format
    pass


def getDimTerafly(dir: str):
    pass