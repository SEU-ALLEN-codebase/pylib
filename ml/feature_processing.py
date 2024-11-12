#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : feature_processing.py
#   Author       : Yufeng Liu
#   Date         : 2022-09-01
#   Description  : 
#
#================================================================
import numpy as np
import pandas as pd

def whitening(features, epsilon=1e-9):
    # do whitening
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + epsilon)
    return features

def clip_outliners(df, col_ids=None):
    if col_ids is None:
        col_ids = np.arange(df.shape[1])
    
    q25 = np.percentile(df.iloc[:, col_ids], q=25, axis=0)
    q75 = np.percentile(df.iloc[:, col_ids], q=75, axis=0)
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    lower_thres = dict(zip(df.columns[col_ids], lower))
    upper_thres = dict(zip(df.columns[col_ids], upper))
    dft = df.iloc[:,col_ids]
    dft.clip(lower=lower_thres, upper=upper_thres, axis=1, inplace=True)


