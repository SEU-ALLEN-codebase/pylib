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

from swc_handler import parse_swc
from math_utils import calc_included_angles_from_coords, calc_included_angles_from_vectors

def find_point_by_distance(pt, anchor_idx, is_parent, morph, dist, return_center_point=True, epsilon=1e-7):
    """ 
    Find the point of exact `dist` to the start pt on tree structure. args are:
    - pt: the start point, [coordinate]
    - anchor_idx: the first node on swc tree to trace, first child or parent node
    - is_parent: whether the anchor_idx is the parent of `pt`, otherwise child. 
                 if the node has several child, a random one is selected
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
                anchor_idx = morph.pos_dict[anchor_idx][-1]
            else:
                if anchor_idx not in morph.child_dict:
                    break
                else:
                    anchor_idxs = morph.child_dict[anchor_idx]
                    anchor_idx = anchor_idxs[np.random.randint(0, len(anchor_idxs))]

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


class breakFinder(object):
    def __init__(self, morph, soma_radius=10, dist_thresh=4.0, line_length=5.0, angle_thresh=90.):
        self.morph = morph

        self.soma_radius = soma_radius
        self.dist_thresh = dist_thresh
        self.line_length = line_length
        self.angle_thresh = angle_thresh

    def find_break_pairs(self):
        ntips = len(self.morph.tips)

        # filter tips near soma
        tip_list = []
        sc = np.array(self.morph.pos_dict[self.morph.idx_soma][2:5])
        for tip in self.morph.tips:
            ci = np.array(self.morph.pos_dict[tip][2:5])
            dist = np.linalg.norm(sc - ci)
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
                v1 = (pt1 - c1).reshape((1,-1))
                v2 = (pt2 - c2).reshape((1,-1))
                ang = calc_included_angles_from_vectors(v1, v2)[0]
                if ang > self.angle_thresh:
                    ret[(tip1, tip2)] = (ang, dist)
                #print(tip1, tip2, ang, dist)

        return ret
        
        


