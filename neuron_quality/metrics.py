#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : metrics.py
#   Author       : Yufeng Liu
#   Date         : 2023-04-04
#   Description  : 
#
#================================================================

import numpy as np

from swc_handler import tree_to_voxels, parse_swc
from math_utils import min_distances_between_two_sets

class DistanceEvaluation(object):
    def __init__(self, dsa_thr=2., esa_thr=2.):
        self.dsa_thr = dsa_thr

    def calc_dist(self, voxels1, voxels2):
        ds = {
            'ESA': None,
            'DSA': None,
            'PDS': None,
        }

        dists1, dists2 = min_distances_between_two_sets(voxels1, voxels2, reciprocal=True, return_index=False)
        for key in ds.keys():
            if key == 'DSA':
                dists1_ = dists1[dists1 > self.dsa_thr]
                dists2_ = dists2[dists2 > self.dsa_thr]
                if dists1_.shape[0] == 0:
                    dists1_ = np.array([0.])
                if dists2_.shape[0] == 0:
                    dists2_ = np.array([0.])
            elif key == 'PDS':
                dists1_ = (dists1 > self.dsa_thr).astype(np.float32)
                dists2_ = (dists2 > self.dsa_thr).astype(np.float32)
            elif key == 'ESA':
                dists1_ = dists1
                dists2_ = dists2
            ds[key] = dists1_.mean(), dists2_.mean(), (dists1_.sum() + dists2_.sum()) / (len(dists1) + len(dists2))
        return ds

    def run(self, reconfile, gsfile):
        tree1 = parse_swc(reconfile)
        tree2 = parse_swc(gsfile)
        print(f'#nodes for recon and gs: {len(tree1)}, {len(tree2)}')

        voxels1 = tree_to_voxels(tree1, crop_box=(10000,10000,10000))
        voxels2 = tree_to_voxels(tree2, crop_box=(10000,10000,10000))
        if len(voxels1) == 0 or len(voxels2) == 0:
            print(len(voxels1), len(voxels2))
            return False
        
        ds = self.calc_dist(voxels1, voxels2)
        return ds


if __name__ == '__main__':
    gsfile = '/home/lyf/Research/cloud_paper/micro_environ/benchmark/gs_crop/18452_4536_x11274_y21067.swc'
    reconfile = '/home/lyf/Research/cloud_paper/micro_environ/benchmark/recon1891_weak/18452/5642_10537_2271.swc'
    de = DistanceEvaluation()
    ds = de.run(reconfile, gsfile)
    print(ds)


