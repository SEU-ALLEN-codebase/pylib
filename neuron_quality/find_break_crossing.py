#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : find_break_crossing.py
#   Author       : Yufeng Liu
#   Date         : 2022-04-01
#   Description  : 
#
#================================================================
import numpy as np
from scipy.spatial import distance_matrix

from swc_handler import parse_swc
from math_utils import calc_included_angles_from_coords, calc_included_angles_from_vectors

def find_point_by_distance(pt, anchor_idx, is_parent, morph, dist, return_center_point=True, epsilon=1e-7):
    """ 
    Find the point of exact `dist` to the start pt on tree structure. args are:
    - pt: the start point, [coordinate]
    - anchor_idx: the first node on swc tree to trace, first child or parent node
    - is_parent: whether the anchor_idx is the parent of `pt`, otherwise child. 
                 if an furcation points encounted, then break
    - morph: Morphology object for current tree
    - dist: distance threshold
    - return_center_point: whether to return the point with exact distance or
                 geometric point of all traced nodes
    - epsilon: small float to avoid zero-division error 
    """

    d = 0 
    ci = pt
    pts = [pt]
    while d < dist:
        try:
            cc = np.array(morph.pos_dict[anchor_idx][2:5])
        except KeyError:
            print(f"Parent/Child node not found within distance: {dist}")
            break
        d0 = np.linalg.norm(ci - cc)
        d += d0
        if d < dist:
            ci = cc  # update coordinates
            pts.append(cc)

            if is_parent:
                anchor_idx = morph.pos_dict[anchor_idx][6]
                if len(morph.child_dict[anchor_idx]) > 1:
                    break
            else:
                if (anchor_idx not in morph.child_dict) or (len(morph.child_dict[anchor_idx]) > 1):
                    break
                else:
                    anchor_idx = morph.child_dict[anchor_idx][0]

    # interpolate to find the exact point
    dd = d - dist
    if dd < 0:
        pt_a = cc
    else:
        dcur = np.linalg.norm(cc - ci)
        assert(dcur - dd >= 0)
        pt_a = ci + (cc - ci) * (dcur - dd) / (dcur + epsilon)
        pts.append(pt_a)
        
    if return_center_point:
        pt_a = np.mean(pts, axis=0)

    return pt_a


class BreakFinder(object):
    def __init__(self, morph, soma_radius=30, dist_thresh=4.0, line_length=5.0, angle_thresh=90., spacing=np.array([1.,1.,1.])):
        self.morph = morph

        self.soma_radius = soma_radius
        self.dist_thresh = dist_thresh
        self.line_length = line_length
        self.angle_thresh = angle_thresh
        self.spacing = spacing

    def find_break_pairs(self):
        ntips = len(self.morph.tips)

        # filter tips near soma
        tip_list = []
        sc = np.array(self.morph.pos_dict[self.morph.idx_soma][2:5])
        for tip in self.morph.tips:
            ci = np.array(self.morph.pos_dict[tip][2:5])
            dist = np.linalg.norm((sc - ci) * self.spacing)
            if dist < self.soma_radius:
                continue
            tip_list.append(tip)

        ret = {}
        for idx1, tip1 in enumerate(tip_list):
            c1 = np.array(self.morph.pos_dict[tip1][2:5])
            for idx2 in range(idx1+1, len(tip_list)):
                tip2 = tip_list[idx2]
                c2 = np.array(self.morph.pos_dict[tip2][2:5])
                # distance criterion
                dist = np.linalg.norm(c1 - c2)
                if dist > self.dist_thresh: continue

                # estimate the fiber distance
                pid1 = self.morph.pos_dict[tip1][6] # parent id for tip1
                pt1 = find_point_by_distance(c1, pid1, True, self.morph, self.line_length, False)
                pid2 = self.morph.pos_dict[tip2][6]
                pt2 = find_point_by_distance(c2, pid2, True, self.morph, self.line_length, False)
                # angle 
                v1 = pt1 - c1
                v2 = pt2 - c2
                ang = calc_included_angles_from_vectors(v1, v2)[0]
                if ang > self.angle_thresh:
                    ret[(tip1, tip2)] = (ang, dist)
                #print(tip1, tip2, ang, dist)

        return ret

    def find_break_pairs_by_distances(self):
        """
        find potential tip pair based on pair-wise distance only, so it can be accelerated
        with more compact vectorization
        """
        sc = np.array(self.morph.pos_dict[self.morph.idx_soma][2:5])
        tip_coords = np.array([self.morph.pos_dict[tip][2:5] for tip in self.morph.tips])
        tip_indices = np.array([self.morph.pos_dict[tip][0] for tip in self.morph.tips])
        ts_vec = (tip_coords - sc) * self.spacing
        ts_dists = np.linalg.norm(ts_vec, axis=1)

        # filter out tips near soma
        tip_out_mask = ts_dists > self.soma_radius
        tip_out_indices = tip_indices[tip_out_mask]
        tip_out_coords = tip_coords[tip_out_mask]

        # pairwise distances
        pdists = distance_matrix(tip_out_coords, tip_out_coords)
        ids1, ids2 = np.nonzero(pdists > self.dist_thresh)
        ret = {}
        for i, j in zip(ids1, ids2):
            try:
                ret[tip_out_indices[i]].append(tip_out_indices[j])
            except KeyError:
                ret[tip_out_indices[i]] = [tip_out_indices[j]]
        return ret
        
        
class CrossingFinder(object):
    def __init__(self, morph, soma_radius=30., dist_thresh=3., spacing=np.array([1.,1.,4.])):
        self.morph = morph
        self.soma_radius = soma_radius
        self.dist_thresh = dist_thresh
        self.spacing = spacing

    def find_crossing_pairs(self):
        pairs = []
        points = []
        morph = self.morph

        cs = np.array(morph.pos_dict[morph.idx_soma][2:5])
        pset = set([])
        visited = set([])
        for tid in morph.tips:
            idx = tid 
            pre_tip_id = None

            while idx != morph.idx_soma:
                if idx == -1: 
                    break
                if idx in morph.child_dict:
                    n_child = len(morph.child_dict[idx])
    
                    if n_child == 2:
                        cur_tip_id = idx 
                        if cur_tip_id in visited: 
                            idx = morph.pos_dict[idx][6]
                            break

                        if pre_tip_id is not None:
                            c0 = np.array(morph.pos_dict[cur_tip_id][2:5])
                            c1 = np.array(morph.pos_dict[pre_tip_id][2:5])
                            if np.linalg.norm((c0 - cs) * self.spacing) < self.soma_radius:
                                break
                            dist = np.linalg.norm(c0 - c1) 
                            if (dist < self.dist_thresh) and ((pre_tip_id, cur_tip_id) not in pset):
                                #print(f'{pre_tip_id}, {cur_tip_id}')
                                pairs.append((pre_tip_id, cur_tip_id, dist))
                                pset.add((pre_tip_id, cur_tip_id))
                        #update tip
                        pre_tip_id = cur_tip_id
                        visited.add(pre_tip_id)
                    elif n_child > 2:
                        c = np.array(morph.pos_dict[idx][2:5])
                        if np.linalg.norm((c - cs) * self.spacing) < self.soma_radius:
                            break
                        if idx in visited:
                            idx = morph.pos_dict[idx][6]
                            break
                        else:
                            points.append(idx)
                            pre_tip_id = idx
                            visited.add(idx)
                        
    
                idx = morph.pos_dict[idx][6]

        #print(f'Dist: {dists.mean():.2f}, {dists.std():.2f}, {dists.max():.2f}, {dists.min():.2f}')
        #for pair in pairs:
        #    print(f'idx1 / idx2 and dist: {pair[0]} / {pair[1]} / {pair[2]}')
        print(f'Number of crossing and crossing-like: {len(points)} / {len(pairs)}')

        return points, pairs


