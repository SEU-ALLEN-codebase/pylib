##########################################################
#Author:          Yufeng Liu
#Create time:     2024-10-31
#Description:               
##########################################################
import numpy as np
import pysal.lib as pslib
from esda.moran import Moran

def moranI_score(coords, feats, eval_ids=None, reduce_type='average', threshold=0.5):
    """
    The coordinates should be in `mm`, and as type of numpy.array
    The feats should be standardized
    """
    # spatial coherence
    weights = pslib.weights.DistanceBand.from_array(coords, threshold=threshold)
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

