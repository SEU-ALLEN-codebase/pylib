#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : image_utility.py
#   Author       : Yufeng Liu
#   Date         : 2021-07-03
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from scipy.ndimage import convolve
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import KDTree
from skimage import morphology
from skimage.draw import line_nd
from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize

from file_io import load_image, save_image

def get_mip_image(img3d, axis=0, mode='MAX'):
    if mode == 'MAX':
        img2d = img3d.max(axis=axis)
    elif mode == 'MIN':
        img2d = img3d.min(axis=axis)
    else:
        raise ValueError

    return img2d

def crop_nonzero_mask(mask3d, pad=0):
    # get the boundary of region
    nzcoords = mask3d.nonzero()
    nzcoords_t = np.array(nzcoords).transpose()
    zmin, ymin, xmin = nzcoords_t.min(axis=0)
    zmax, ymax, xmax = nzcoords_t.max(axis=0)

    sz, sy, sx = mask3d.shape
    zs = max(0, zmin-pad)
    ze = min(sz, zmax+pad+1)
    ys = max(0, ymin-pad)
    ye = min(sy, ymax+pad+1)
    xs = max(0, xmin-pad)
    xe = min(sx, xmax+pad+1)
    sub_mask = mask3d[zs:ze, ys:ye, xs:xe]
    return sub_mask, (zs, ze, ys, ye, xs, xe)

def image_histeq(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def montage_images_for_folder(img_dir, sw, sh, prefix=''):
    imgfiles = list(glob.glob(os.path.join(img_dir, '*.png')))
    swh = sw * sh
    for i in range(0, len(imgfiles), swh):
        subset = imgfiles[i : i + swh]
        args_str = f'montage {" ".join(subset)} -tile {sw}x{sh} montage_{prefix}_{i:04d}.png'
        os.system(args_str)

def extend_skel_to_boundary(boundaries, pcoords, is_start=True):
    """
    boundaries: all points on the mask boundary
    pcoords: coordinates of skeletonal points
    is_start: if extending from the start point
    """

    if is_start:
        pt = pcoords[0]
        pts_neighbor = pcoords[:10]
        vref = pts_neighbor[-1] - pts_neighbor[0]
    else:
        pt = pcoords[-1]
        pts_neighbor = pcoords[-10:]
        vref = pts_neighbor[0] - pts_neighbor[-1]

    # find the boundary point align well with the principal axis of skelenton
    pca = PCA()
    pca.fit(pts_neighbor)
    pc1 = pca.components_[0]
    if vref.dot(pc1) < 0:
        pc1 = -pc1

    # estimate the direction matchness
    vb = (pt - boundaries).astype(np.float64)
    vb /= (np.linalg.norm(vb, axis=1).reshape((-1,1)) + 1e-10)
    cos_dist = pc1.dot(vb.transpose())
    max_id = np.argmax(cos_dist)
    pta = boundaries[max_id]

    lpts = np.array(line_nd(pt, pta, endpoint=True)).transpose()[1:]
    if is_start:
        pcoords = np.vstack((lpts[::-1], pcoords))
    else:
        pcoords = np.vstack((pcoords, lpts))

    return pcoords

def get_longest_skeleton(mask, is_3D=True, extend_to_boundary=True, smoothing=True):
    mask = mask > 0 # only binary mask supported
    if smoothing:
        mask = morphology.closing(mask, morphology.square(5), mode='constant')

    # get the skeleton
    skel = skeletonize(mask, method='lee')
    skel[skel > 0] = 1

    # get the critical points: tip and multi-furcations
    summ = summarize(Skeleton(skel))
    summ_bak = summ.copy()
    summ = summ[['node-id-src', 'node-id-dst', 'branch-distance', 'branch-type']]
    nid_keys = ['node-id-src', 'node-id-dst']
    # iterative prune
    while summ.shape[0] >= 2:
        print(summ.shape[0])
        dcnts = dict(zip(*np.unique(summ[['node-id-src', 'node-id-dst']], return_counts = True)))
        for nid, cnt in dcnts.items():
            if cnt == 1: continue

            # remove possible circles
            ids, cnts = np.unique(summ[nid_keys].values, axis=0, return_counts=True)
            if (cnts > 1).sum() != 0:
                lcnts = cnts > 1
                for lids in ids[lcnts]:
                    dup_ones = (summ[nid_keys] == lids).sum(axis=1) == 2
                    nzi = np.nonzero(dup_ones)[0]
                    sub_summ = summ[dup_ones]
                    max_d_id = np.argmax(sub_summ['branch-distance'])
                    max_d_index = sub_summ.index[max_d_id]
                    to_drop = []
                    for idx in range(len(nzi)):
                        if idx != max_d_id:
                            to_drop.append(sub_summ.index[idx])
                    summ.drop(index=to_drop, inplace=True)
                    # check the type of current branch
                    nc_dict = dict(zip(*np.unique(summ[nid_keys], return_counts=True)))
                    if (nc_dict[lids[0]] != 1) and (nc_dict[lids[1]] != 1):
                        summ.loc[max_d_index, 3] = 2
                    else:
                        summ.loc[max_d_index, 3] = 1


            con0 = (summ['node-id-src']==nid) | (summ['node-id-dst']==nid)
            con = con0 & (summ['branch-type']==1)
            con1 = con0 & (summ['branch-type'] != 1)
            #if con.sum() <= 1: continue # process only

            if con1.sum() == 0:
                # keep the top two branches
                to_del = summ.index[np.argsort(summ['branch-distance'].values)[:-2]]
                summ.drop(index=to_del, inplace=True)
                # merge the last two segments
                nids, ncnts = np.unique(summ[nid_keys], return_counts=True)
                final_nids = nids[ncnts == 1]
                final_dist = summ['branch-distance'].sum()
                summ.drop(index=summ.index[0], inplace=True)
                summ.loc[summ.index, nid_keys] = final_nids
                summ.loc[summ.index, 'branch-distance'] = final_dist
            else:
                cur = summ[con]
                max_id = np.argmax(cur['branch-distance'])
                idx_max = cur.index[max_id]
                # remove several points
                to_del = [k for k in cur.index if k != idx_max]
                # remove items from dataframe
                summ.drop(index=to_del, inplace=True)
                # modify their features
                #import ipdb; ipdb.set_trace()
                if con0.sum() - con.sum() == 1:
                    # the current node is now a non-critical point, remove it
                    idx = con1[con1].index[0]
                    tt = summ.loc[idx]
                    tr = summ.loc[idx_max]
                    summ.loc[idx, 'branch-type'] = 1
                    summ.loc[idx, 'branch-distance'] = tt['branch-distance'] + tr['branch-distance']
                    stacks = np.hstack((tr[['node-id-src', 'node-id-dst']], tt[['node-id-src', 'node-id-dst']]))
                    ids = [idx for idx in stacks if idx != nid]
                    summ.loc[idx, ['node-id-src', 'node-id-dst']] = ids
                    summ.drop(index=idx_max, inplace=True)
                elif con0.sum() - con.sum() == 0:
                    print('WARNING: ')


    # get the original information
    if is_3D:
        src_key = ['node-id-src', 'image-coord-src-0', 'image-coord-src-1', 'image-coord-src-2']
        dst_key = ['node-id-dst', 'image-coord-dst-0', 'image-coord-dst-1', 'image-coord-dst-2']
        vm = 3
    else:
        src_key = ['node-id-src', 'image-coord-src-0', 'image-coord-src-1']
        dst_key = ['node-id-dst', 'image-coord-dst-0', 'image-coord-dst-1']
        vm = 2

    # the two terminal points
    p1 = summ_bak[src_key].values
    p2 = summ_bak[dst_key].values
    pts_all = np.vstack((p1, p2))
    node1, node2 = summ[nid_keys].values[0]
    coords1 = pts_all[pts_all[:,0] == node1][0][1:]
    coords2 = pts_all[pts_all[:,0] == node2][0][1:]

    # get the path
    pgraph, coordinates = skeleton_to_csgraph(skel)
    coordinates = np.array(coordinates).transpose()
    id1 = np.nonzero((coordinates == coords1).sum(axis=1) == vm)[0][0]
    id2 = np.nonzero((coordinates == coords2).sum(axis=1) == vm)[0][0]
    # The skeletonization may result small circular points!
    dij = dijkstra(pgraph, directed=True, indices=[id1], return_predecessors=True)
    parents = dij[1][0]
    # transverse to the full path
    pids = []
    pid = id2
    while pid != -9999:
        pids.append(pid)
        pid = parents[pid]
    pcoords = coordinates[pids]

    new_skel = skel.copy()
    new_skel.fill(0)
    if is_3D:
        new_skel[pcoords[:,0], pcoords[:,1], pcoords[:,2]] = 1
    else:
        new_skel[pcoords[:,0], pcoords[:,1]] = 1

    if extend_to_boundary:
        #---- extend the skeleton to boundary of image ----#
        from anatomy.anatomy_vis import detect_edges3d, detect_edges2d
        if is_3D:
            edges = detect_edges3d(mask)
        else:
            edges = detect_edges2d(mask)
        ecoords = np.array(edges.nonzero()).transpose()

        pcoords = extend_skel_to_boundary(ecoords, pcoords, is_start=True)
        pcoords = extend_skel_to_boundary(ecoords, pcoords, is_start=False)
        # udate skeleton
        if is_3D:
            new_skel[pcoords[:,0], pcoords[:,1], pcoords[:,2]] = 1
        else:
            new_skel[pcoords[:,0], pcoords[:,1]] = 1

    return new_skel, pcoords


class AbastractCropImage:
    def __init__(self):
        pass

