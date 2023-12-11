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
import os
import glob
import SimpleITK as sitk
import pickle
from v3d.io import *
from pathlib import Path
from swc_handler import parse_swc, write_swc


def load_image(img_file, flip_tif=True):
    img_file = Path(img_file)
    if img_file.suffix in ['.v3draw', '.V3DRAW', '.raw', '.RAW']:
        return load_v3draw(img_file)
    if img_file.suffix in ['.v3dpbd', '.V3DPBD']:
        return PBD().load_image(img_file)
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_file)))
    if flip_tif and img_file.suffix in ['.TIF', '.TIFF', '.tif', '.tiff']:
        img = np.flip(img, axis=-2)
    return img


def save_image(outfile, img: np.ndarray, flip_tif=True, useCompression=False):
    outfile = Path(outfile)
    if outfile.suffix in ['.v3draw', '.V3DRAW']:
        save_v3draw(img, outfile)
    elif outfile.suffix in ['.TIF', '.TIFF', '.tif', '.tiff']:
        if flip_tif:
            img = np.flip(img, axis=-2)
        sitk.WriteImage(sitk.GetImageFromArray(img), str(outfile), useCompression=useCompression)
    else:
        sitk.WriteImage(sitk.GetImageFromArray(img), str(outfile), useCompression=useCompression)
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

def generate_ano_file(swcfile, outdir=None):
    tree = parse_swc(swcfile)
    swcname = os.path.split(swcfile)[-1]

    if outdir is None:
        apofile = f'{swcname}.apo'
        anofile = f'{swcname}.ano'
    else:
        apofile = os.path.join(outdir, f'{swcname}.apo')
        anofile = os.path.join(outdir, f'{swcname}.ano')
    
    fapo = open(apofile, 'w')
    fapo.write('##n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b\n')

    with open(anofile, 'w') as fp:
        fp.write(f'APOFILE={apofile}\n')
        fp.write(f'SWCFILE={swcname}\n')

    new_tree = []
    for node in tree:
        idx, type_, x, y, z, r, pid = node[:7]
        if pid == -1:
            fapo.write(f'{idx}, , ,, {z},{x},{y},0.000,0.000,0.000,314.159,0.000,,,,0,255,255\n')
        new_node = (idx, type_, x, y, z, r, pid)
        new_tree.append(new_node)
    write_swc(new_tree, swcname)

    # generate apo file
    fapo.close()

def get_tera_res_path(tera_dir, res_ids=None, bracket_escape=True):
    '''
    res_ids: int or tuple
    - if int: it represents the resolution level, starting from the lowest resolution. The value -1
        means the highest resolution
    - if tuple: multiple levels 
    '''
    
    resfiles = list(glob.glob(os.path.join(tera_dir, 'RES*')))
    if len(resfiles) == 0:
        print(f'Error: the brain {os.path.split(tera_dir)[-1]} is not found!')
        return 
    
    # sort by resolutions
    ress = sorted(resfiles, key=lambda x: int(x.split('/')[-1][4:-1].split('x')[0]))
    if type(res_ids) is int:
        res_path = ress[res_ids]
        if bracket_escape:
            res_path = res_path.replace('(', '\(')
            res_path = res_path.replace(')', '\)')
        return res_path
    elif type(res_ids) is tuple:
        res_pathes = []
        for idx in res_ids:
            res_path = ress[idx]
            if bracket_escape:
                res_path = res_path.replace('(', '\(')
                res_path = res_path.replace(')', '\)')
            res_pathes.append(res_path)
        return res_pathes

        

