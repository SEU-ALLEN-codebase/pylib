#!/usr/bin/env python

# ================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : neuron_dist_vaa3d.py
#   Author       : Yufeng Liu
#   Date         : 2021-10-28
#   Description  : 
#
# ================================================================

import os, sys, glob

sys.path.append(sys.path[0] + "/..")
import math
import numpy as np
import subprocess
from skimage.draw import line_nd
from scipy.spatial import distance_matrix

from swc_handler import parse_swc, write_swc, scale_swc, is_in_box


def tree_to_voxels(tree, crop_box):
    """
    Turn an swc tree to voxel coordinates. Fragments (segment between nodes) are interpolated.
    The crop_box enables choosing a sub region
    """
    # initialize position dict
    pos_dict = {}
    new_tree = []
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        leaf_new = (*leaf, is_in_box(x, y, z, crop_box))
        pos_dict[leaf[0]] = leaf_new
        new_tree.append(leaf_new)
    tree = new_tree

    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p, ib = leaf
        if p == -1: continue  # soma

        if p not in pos_dict:
            continue

        parent_leaf = pos_dict[p]
        if (not ib) and (not parent_leaf[ib]):
            print('All points are out of box! do trim_swc before!')
            raise ValueError

        # draw line connecting each pair
        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(cur_pos[::-1], par_pos[::-1], endpoint=True)
        xl.extend(list(lin[2]))
        yl.extend(list(lin[1]))
        zl.extend(list(lin[0]))

    voxels = []
    for (xi, yi, zi) in zip(xl, yl, zl):
        if is_in_box(xi, yi, zi, crop_box):
            voxels.append((xi, yi, zi))
    # remove duplicate points
    voxels = np.array(list(set(voxels)), dtype=np.float32)
    return voxels


def get_specific_neurite(tree, type_id):
    if (not isinstance(type_id, list)) and (not isinstance(type_id, tuple)):
        type_id = (type_id,)

    new_tree = []
    for leaf in tree:
        if leaf[1] in type_id:
            new_tree.append(leaf)
    return new_tree


class DistanceEvaluation(object):
    def __init__(self, crop_box, neurite_type='all', diff_dist=2.0, abs=False, pct_by_self=True, soma_radius=0, spacing=(1, 1, 1)):
        """
        dists between neuron1 and neuron2 are different when you use one of them as background and the other as foreground,
        we define dist1 as neuron1 against neuron2 and vice versa. The trees are voxelized to maximize precision, and the
        dists and quantities of different structures are measured in voxels.

        Here are some instructions on the input arguments.
        crop_box: the range of voxelization
        pct_by_self: flag to set if pds is divided by its own size or the other when abs is True. Default as True.
                    You may want to turn it False when the 2
        diff_dist: the minimum distance for different structures, default as 2.0
        abs: whether to output the absolute #voxels of each metric, default as False
        soma_radius: not doing calculation for region within. Default as 0
        spacing: that of the image the swc is reconstructed from
        """
        self.crop_box = crop_box
        self.neurite_type = neurite_type
        self.diff_dist = diff_dist
        self.abs = abs
        self.pct_by_self = pct_by_self
        self.soma_radius = soma_radius
        self.spacing = spacing


    def memory_safe_min_distances(self, voxels1, voxels2, num_thresh=50000):
        """
        output the distance matrix between 2 sets of voxel coordinates
        """
        # verified
        nv1 = len(voxels1)
        nv2 = len(voxels2)
        if (nv1 > num_thresh) or (nv2 > num_thresh):
            # use block wise calculation
            vq1 = [voxels1[i * num_thresh:(i + 1) * num_thresh] for i in range(int(math.ceil(nv1 / num_thresh)))]
            vq2 = [voxels2[i * num_thresh:(i + 1) * num_thresh] for i in range(int(math.ceil(nv2 / num_thresh)))]

            dists1 = np.ones(nv1) * 1000000.
            dists2 = np.ones(nv2) * 1000000.
            for i, v1 in enumerate(vq1):
                idx00 = i * num_thresh
                idx01 = i * num_thresh + len(v1)
                for j, v2 in enumerate(vq2):
                    idx10 = j * num_thresh
                    idx11 = j * num_thresh + len(v2)

                    d = distance_matrix(v1, v2)
                    dists1[idx00:idx01] = np.minimum(d.min(axis=1), dists1[idx00:idx01])
                    dists2[idx10:idx11] = np.minimum(d.min(axis=0), dists2[idx10:idx11])
        else:
            pdist = distance_matrix(voxels1, voxels2)
            dists1 = pdist.min(axis=1)
            dists2 = pdist.min(axis=0)
        return dists1, dists2

    def calc_DMs(self, voxels1, voxels2):
        dist_results = {
            'ESA': None,
            'DSA': None,
            'PDS': None
        }
        # exceptions
        if len(voxels1) > 500000 or len(voxels2) > 500000:
            dist_results['ESA'] = (99999, 99999, 99999)
            dist_results['DSA'] = (99999, 99999, 99999)
            dist_results['PDS'] = (1.0, 1.0, 1.0)
            return dist_results
        elif len(voxels1) == 0 or len(voxels2) == 0:
            return dist_results
        # distace estimation
        dists1, dists2 = self.memory_safe_min_distances(voxels1, voxels2)

        for key in dist_results:
            if key == 'DSA':
                dists1_ = dists1[dists1 > self.diff_dist]
                dists2_ = dists2[dists2 > self.diff_dist]
                if dists1_.shape[0] == 0:
                    dists1_ = np.array([0.])
                if dists2_.shape[0] == 0:
                    dists2_ = np.array([0.])
            elif key == 'PDS':
                dists1_ = (dists1 > self.diff_dist).astype(np.float32)
                dists2_ = (dists2 > self.diff_dist).astype(np.float32)
            elif key == 'ESA':
                dists1_ = dists1
                dists2_ = dists2
            if self.abs:
                dist_results[key] = dists1_.sum(), dists2_.sum(), dists1_.sum() + dists2_.sum()
            else:
                if self.pct_by_self:
                    dist_results[key] = dists1_.mean(), dists2_.mean(), (dists1_.sum() + dists2_.sum()) / (len(dists1) + len(dists2))
                else:
                    dist_results[key] = dists1_.sum() / len(dists2), dists2_.sum() / len(dists1), (dists1_.sum() + dists2_.sum()) / (len(dists1) + len(dists2))
        return dist_results

    def calc_DIADEM(self, swc_file1, swc_file2, jar_path='/home/lyf/Softwares/packages/Diadem/DiademMetric.jar'):
        exec_str = f'java -jar {jar_path} -G {swc_file1} -T {swc_file2} -x 6 -R 3 -z 2 --xyPathThresh 0.08 --zPathThresh 0.20 --excess-nodes false'
        # print(exec_str)
        output = subprocess.check_output(exec_str, shell=True)
        # print(output)
        score1 = float(output.split()[-1])

        exec_str = f'java -jar {jar_path} -G {swc_file2} -T {swc_file1} -x 6 -R 3 -z 2 --xyPathThresh 0.08 --zPathThresh 0.20 --excess-nodes false -r 17'
        output = subprocess.check_output(exec_str, shell=True)
        print(output)
        score2 = float(output.split()[-1])

        score = (score1 + score2) / 2.
        return score1, score2, score

    def calc_distance(self, swc_file1, swc_file2, dist_type='DM', downsampling=True):
        """
        output distance matrix between 2 swc files
        """
        if dist_type == 'DM':
            tree1 = parse_swc(swc_file1)
            tree2 = parse_swc(swc_file2)
            if downsampling:
                # downsampling the swc by scale 2, that is, to secondary Resolution
                tree1 = scale_swc(tree1, 0.5)
                tree2 = scale_swc(tree2, 0.5)

            print(f'Length of nodes in tree1 and tree2: {len(tree1)}, {len(tree2)}')
            if self.neurite_type == 'all':
                pass
            elif self.neurite_type == 'dendrite':
                type_id = (3, 4)
                tree1 = get_specific_neurite(tree1, type_id)
            elif self.neurite_type == 'axon':
                type_id = 2
                tree1 = get_specific_neurite(tree1, type_id)
            else:
                raise NotImplementedError

            # to successive voxels
            voxels1 = tree_to_voxels(tree1, self.crop_box)
            voxels2 = tree_to_voxels(tree2, self.crop_box)
            dist_t = self.calc_DMs(voxels1, voxels2)
            print(dist_t)
            if self.soma_radius > 0:
                soma1 = [i for i in tree1 if i[6] == -1]
                soma1 = soma1[0][2:5]
                soma2 = [i for i in tree2 if i[6] == -1]
                soma2 = soma2[0][2:5]
                voxels1 = voxels1[np.linalg.norm((voxels1 - soma1) * self.spacing, axis=-1) > self.soma_radius]
                voxels2 = voxels2[np.linalg.norm((voxels2 - soma2) * self.spacing, axis=-1) > self.soma_radius]
            dist = self.calc_DMs(voxels1, voxels2)
            print(dist)
        elif dist_type == 'DIADEM':
            dist = self.calc_DIADEM(swc_file1, swc_file2)
        else:
            raise NotImplementedError

        return dist

    def calc_distance_triple(self, swc_gt, swc_cmp1, swc_cmp2, dist_type='DM'):
        """
        with gold standard or ground truth swc, you can compare another 2 swc's difference against it.
        For instance, comparing a neuron tree's error rate before and after a pruning process.
        return: swc1's dist to gt, swc2's dist to gt
        """
        if dist_type == 'DM':
            tree_gt = parse_swc(swc_gt)
            tree_cmp1 = parse_swc(swc_cmp1)
            tree_cmp2 = parse_swc(swc_cmp2)

            if self.neurite_type == 'all':
                pass
            elif self.neurite_type == 'dendrite':
                type_id = (3, 4)
                tree_gt = get_specific_neurite(tree_gt, type_id)
            elif self.neurite_type == 'axon':
                type_id = 2
                tree_gt = get_specific_neurite(tree_gt, type_id)
            else:
                raise NotImplementedError
            print(f'Length of nodes for gt, cmp1 and cmp2: {len(tree_gt)}, {len(tree_cmp1)}, {len(tree_cmp2)}')

            # to successive voxels
            voxels_gt = tree_to_voxels(tree_gt, self.crop_box).astype(np.float32)
            voxels_cmp1 = tree_to_voxels(tree_cmp1, self.crop_box).astype(np.float32)
            voxels_cmp2 = tree_to_voxels(tree_cmp2, self.crop_box).astype(np.float32)
            dist1 = self.calc_DMs(voxels_gt, voxels_cmp1)
            dist2 = self.calc_DMs(voxels_gt, voxels_cmp2)
        elif dist_type == 'DIADEM':
            dist1 = self.calc_DIADEM(swc_gt, swc_cmp1)
            dist2 = self.calc_DIADEM(swc_gt, swc_cmp2)
        else:
            raise NotImplementedError

        return dist1, dist2


def parse_files(test_list_file='./datalist/par_set_singleSoma.list'):
    # filter the files
    fsets = []
    with open(test_list_file, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            fsets.append(line)
    fsets = set(fsets)
    return fsets


if __name__ == '__main__':
    import time
    import os, glob
    import pickle

    neurite_type = 'all'
    ut_dir = f'./results/standardized_dark1.0std'
    outfile = f'./results/temp.pkl'
    gt_dir = '/PBshare/lyf/transtation/seu_mouse/crop_data/dendriteImageSecR/swc_new'
    crop_box = (512, 512, 256)
    downsampling = False  # downsampling to SecRes to speedup calculation

    # initialize the matrices
    avg = {
        'ESA': [0, 0, 0],
        'DSA': [0, 0, 0],
        'PDS': [0, 0, 0]
    }

    fsets = parse_files()
    de = DistanceEvaluation(crop_box, neurite_type=neurite_type)
    dr_dict = {}
    ns = 0
    for br_dir in glob.glob(os.path.join(ut_dir, '*')):
        br_id = os.path.split(br_dir)[-1]
        for tr_file in glob.glob(os.path.join(br_dir, '*.swc')):
            t0 = time.time()
            tr_name = os.path.split(tr_file)[-1]
            prefix = os.path.splitext(tr_name)[0]
            if prefix not in fsets:
                continue
            if prefix == '11700_8319_2991':
                continue

            gt_file = os.path.join(gt_dir, br_id, f'{tr_name}')

            dr = de.calc_distance(gt_file, tr_file, downsampling=downsampling)
            dr_dict[prefix] = dr
            print(f'{tr_name} in {time.time() - t0} seconds')
            for key in dr:
                if dr[key] is None: continue
                ns += 1
                for kk in range(3):
                    avg[key][kk] = avg[key][kk] + dr[key][kk]
                print(f'{key}: {dr[key][0]}, {dr[key][1]}, {dr[key][2]}')
            print('\n')

    ns /= 3
    for k1, va in avg.items():
        va1, va2, va3 = va
        va1 /= ns
        va2 /= ns
        va3 /= ns
        print(f'{k1}: {va1}, {va2}, {va3}')
    print(f'{ns} files left!')

    # save result to file
    with open(outfile, 'wb') as fp:
        pickle.dump(dr_dict, fp)
