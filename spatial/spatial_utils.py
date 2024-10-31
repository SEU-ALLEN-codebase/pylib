##########################################################
#Author:          Yufeng Liu
#Create time:     2024-10-31
#Description:               
##########################################################
import numpy as np
import pysal.lib as pslib
from esda.moran import Moran

def moranI_score(coords, feats, eval_ids=None, reduce_type='average', weight_type='distance', threshold=0.5, k=5):
    """
    The coordinates should be in `mm`, and as type of numpy.array
    The feats should be standardized
    """
    # spatial coherence
    if weight_type == 'distance':
        weights = pslib.weights.DistanceBand.from_array(coords, threshold=threshold)
    elif weight_type == 'knn':
        weights = pslib.weights.KNN.from_array(coords, k=k)
    else:
        raise NotImplementedError

    avgI = []
    if eval_ids is None:
        if feats.ndim == 1:
            feats = feats.reshape((-1,1))
        eval_ids = range(feats.shape[1])
    for i in eval_ids:
        moran = Moran(feats[:,i], weights)
        avgI.append(moran.I)

    if reduce_type == 'average':
        avgI = np.mean(avgI)
    elif reduce_type == 'max':
        avgI = np.max(avgI)
    elif reduce_type == 'all':
        return avgI
    else:
        raise NotImplementedError
    return avgI

