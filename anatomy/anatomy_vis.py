#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : anatomy_vis.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-22
#   Description  : 
#
#================================================================

import numpy as np
from image_utils import get_mip_image

def detect_edges2d(img2d):
    img = img2d.astype(float)
    gx, gy = np.gradient(img)
    edges = (gx*gx + gy*gy) != 0
    return edges

def get_section_boundary(mask, axis=0, c=None, v=255):
    if c == None:
        c = mask.shape[axis] // 2
    if axis == 0:
        section = mask[c]
    elif axis == 1:
        section = mask[:,c]
    elif axis == 2:
        section = mask[:,:,c]
    else:
        raise NotImplementedError(f'Argument axis supports only 0/1/2, but got {axis}!')

    boundary = detect_edges2d(section)
    if v == 255:
        boundary = boundary.astype(np.uint8) * v
    return boundary

def get_brain_outline2d(mask, axis=0, v=255):
    mask = mask > 0
    mask2d = get_mip_image(mask, axis=axis)
    outline = detect_edges2d(mask2d)
    if v == 255:
        outline = outline.astype(np.uint8) * v
    return outline

def get_section_boundary_with_outline(mask, axis=0, sectionX=None, v=255):
    boundary = get_section_boundary(mask, axis=axis, c=sectionX, v=v)
    outline = get_brain_outline2d(mask, axis=axis, v=v)
    return boundary, outline


if __name__ == '__main__':
    from file_io import load_image
    import cv2
    
    mask_file = './resources/annotation_25.nrrd'
    axis = 0
    c = 100
    v = 255
    
    mask = load_image(mask_file)
    boundary, outline = get_section_boundary_with_outline(mask, axis=0, sectionX=c)
    bo = np.maximum(boundary, outline)
    cv2.imwrite(f'boundary_axis{axis}.png', bo)

