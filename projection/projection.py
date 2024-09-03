##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-01
#Description:               
##########################################################
import os
import time
import numpy as np
import pandas as pd

from anatomy.anatomy_config import MASK_CCF25_FILE, SALIENT_REGIONS
from anatomy.anatomy_core import parse_ana_tree
from file_io import load_image

class Projection:

    def __init__(self, use_two_hemispheres=True, resample_scale=8, atlas_file=None):
        # make sure the swc are uniformly sampled, otherwise the estimation
        # should be changed
        self.resample_scale = resample_scale
        # Get the atlas
        if atlas_file is None:
            atlas = load_image(MASK_CCF25_FILE)
            self.ccf_atlas = True
        else:
            atlas = load_image(atlas_file)
            self.ccf_atlas = False

        if use_two_hemispheres:
            # get the new atlas with differentiation of left-right hemisphere
            zdim, ydim, xdim = atlas.shape
            atlas_lr = np.zeros(atlas.shape, dtype=np.int64)
            atlas_lr[:zdim//2] = atlas[:zdim//2]
            atlas_lr[zdim//2:] = -atlas[zdim//2:].astype(np.int64)
            self.atlas_lr = atlas_lr
        else:
            self.atlas_lr = atlas

    def calc_proj_matrix(self, axon_files, proj_csv='temp.csv'):
        zdim, ydim, xdim = self.atlas_lr.shape
        
        # vector
        regids = np.unique(self.atlas_lr[self.atlas_lr != 0])
        rdict = dict(zip(regids, range(len(regids))))

        fnames = [os.path.split(fname)[-1][:-4] for fname in axon_files]
        projs = pd.DataFrame(np.zeros((len(axon_files), len(regids))), index=fnames, columns=regids)

        t0 = time.time()
        for iaxon, axon_file in enumerate(axon_files):
            ncoords = pd.read_csv(axon_file, sep=' ', usecols=(2,3,4,6), comment='#', header=None).values
            # flipping
            smask = ncoords[:,-1] == -1
            if smask.sum() == 0:
                print(axon_file)
            # convert to CCF-25um
            ncoords[:,:-1] = ncoords[:,:-1] / 25.
            soma_coord = ncoords[smask][0,:-1]
            ncoords = ncoords[~smask][:,:-1]
            if soma_coord[2] > zdim/2:
                ncoords[:,2] = zdim - ncoords[:,2]
            # make sure no out-of-mask points
            ncoords = np.round(ncoords).astype(int)
            ncoords[:,0] = np.clip(ncoords[:,0], 0, xdim-1)
            ncoords[:,1] = np.clip(ncoords[:,1], 0, ydim-1)
            ncoords[:,2] = np.clip(ncoords[:,2], 0, zdim-1)
            # get the projected regions
            proj = self.atlas_lr[ncoords[:,2], ncoords[:,1], ncoords[:,0]]
            # to project matrix
            rids, rcnts = np.unique(proj, return_counts=True)
            # Occasionally, there are some nodes located outside of the atlas, due to
            # the registration error
            nzm = rids != 0
            rids = rids[nzm]
            rcnts = rcnts[nzm]
            rindices = np.array([rdict[rid] for rid in rids])
            projs.iloc[iaxon, rindices] = rcnts

            if (iaxon + 1) % 10 == 0:
                print(f'--> finished {iaxon+1} in {time.time()-t0:.2f} seconds')

        projs *= self.resample_scale # to um scale

        # zeroing non-salient regions
        if self.ccf_atlas:
            salient_mask = np.array([True if np.fabs(int(col)) in SALIENT_REGIONS else False for col in projs.columns])
            #keep_mask = (projs.sum() > 0) & salient_mask
            keep_mask = salient_mask
            # filter the neurons not in target regions
            projs = projs.loc[:, keep_mask]
        
        projs.to_csv(proj_csv)

        return projs

