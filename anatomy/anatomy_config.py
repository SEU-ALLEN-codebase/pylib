#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : __init__.py
#   Author       : Yufeng Liu
#   Date         : 2022-11-10
#   Description  : 
#
#================================================================

import os

__CUR_ABS_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

ANATOMY_TREE_FILE = os.path.join(__CUR_ABS_PATH, 'resources/tree.json')
ID_MAP_RES25_FILE = os.path.join(__CUR_ABS_PATH, 'resources/annotation25_16bit_id_mapping.json')
MASK_CCF25_FILE = os.path.join(__CUR_ABS_PATH, 'resources/annotation_25.nrrd')



