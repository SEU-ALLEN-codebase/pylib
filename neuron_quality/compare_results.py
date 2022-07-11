#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : compare_results.py
#   Author       : Yufeng Liu
#   Date         : 2021-12-30
#   Description  : 
#
#================================================================

import numpy as np
import pickle

def compare(pklfile1, pklfile2):
    with open(pklfile1, 'rb') as f1:
        d1 = pickle.load(f1)
    with open(pklfile2, 'rb') as f2:
        d2 = pickle.load(f2)

    dc1 = {
        'ESA': [0,0,0],
        'DSA': [0,0,0],
        'PDS': [0,0,0]
    }

    dc2 = {
        'ESA': [0,0,0],
        'DSA': [0,0,0],
        'PDS': [0,0,0]
    }

    nc = 0
    nc1, nc2 = 0, 0
    for k1, v1 in d1.items():
        is_common = True

        if v1['ESA'] is None:
            is_common = False
        else:
            nc1 += 1
        if d2[k1]['ESA'] is None:
            is_common = False
        else:
            nc2 += 1

        if not is_common:
            continue

        nc += 1
        for kk1, vv1 in v1.items():
            for i in range(3):
                dc1[kk1][i] += vv1[i]
                dc2[kk1][i] += d2[k1][kk1][i]

    # averaging
    for k, v1 in dc1.items():
        v1 = np.array(v1) / nc
        v2 = np.array(dc2[k]) / nc
        print(k, v1, v2)

    print(f'#Neurons are: {nc1} / {nc2} / {nc}')

if __name__ == '__main__':
    pklfile1 = './results/segLenThresh5.pkl'
    pklfile2 = './results/standardized.pkl'
    compare(pklfile1, pklfile2)
            

