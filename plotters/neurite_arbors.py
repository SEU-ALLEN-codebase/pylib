#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : neurite_arbors.py
#   Author       : Yufeng Liu
#   Date         : 2022-09-26
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np

from skimage.draw import line_nd
import matplotlib.pyplot as plt

from swc_handler import parse_swc, write_swc, get_specific_neurite, NEURITE_TYPES
from morph_topo import morphology


class NeuriteArbors:
    def __init__(self, swcfile):
        tree = parse_swc(swcfile)
        self.morph = morphology.Morphology(tree)
        self.morph.get_critical_points()

    def get_paths_of_specific_neurite(self, type_id=None, mip='z'):
        """
        Tip: set
        """
        if mip == 'z':
            idx1, idx2 = 2, 3
        elif mip == 'x':
            idx1, idx2 = 3, 4
        elif mip == 'y':
            idx1, idx2 = 2, 4
        else:
            raise NotImplementedError

        paths = []
        for tip in self.morph.tips:
            path = []
            node = self.morph.pos_dict[tip]
            if type_id is not None:
                if node[1] not in type_id: continue
            path.append([node[idx1], node[idx2]])
            while node[6] in self.morph.pos_dict:
                pid = node[6]
                pnode = self.morph.pos_dict[pid]
                path.append([pnode[idx1], pnode[idx2]])
                if (type_id is not None) and (pnode[1] not in type_id):
                    break
                node = self.morph.pos_dict[pid]

            paths.append(np.array(path))

        return paths

       
    def plot_morph_mip(self, type_id, xxyy=None, mip='z', color='r', figname='temp.png', out_dir='.', show_name=False, linewidth=2):
        paths = self.get_paths_of_specific_neurite(type_id, mip=mip)
        
        plt.figure(figsize=(8,8))
        for path in paths:
            plt.plot(path[:,0], path[:,1], color=color, lw=linewidth)
            
        try:
            all_paths = np.vstack(paths)
            if xxyy is None:
                xxyy = (all_paths.min(axis=0)[0], all_paths.max(axis=0)[0], all_paths.min(axis=0)[1], all_paths.max(axis=0)[1])

            plt.xlim([xxyy[0], xxyy[1]])
            plt.ylim([xxyy[2], xxyy[3]])
        except ValueError:
            pass

        plt.tick_params(left = False, right = False, labelleft = False ,
                labelbottom = False, bottom = False)
        # Iterating over all the axes in the figure
        # and make the Spines Visibility as False
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)
        
        # title
        if show_name:
            plt.title(figname)
        #plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{figname}.png'), dpi=200)
        plt.close()
        

if __name__ == '__main__':
    #swcfile = '/PBshare/SEU-ALLEN/Users/yfliu/transtation/Research/platform/lyf_mac/morphology_conservation/axon_bouton_ccf_sorted/18457_00116.swc_sorted.swc'
    swc_dir = '/PBshare/SEU-ALLEN/Users/yfliu/transtation/1741_All'
    neurite = 'axon'
    colors = {
        'basal dendrite': 'b',
        'apical dendrite': 'm',
        'axon': 'r', 
    }

    out_dir = neurite.split()[0]
    for swcfile in glob.glob(os.path.join(swc_dir, '*.swc')):
        type_id = set(NEURITE_TYPES[neurite])
        na = NeuriteArbors(swcfile)
        figname = os.path.split(swcfile)[-1].split('.')[0]
        print(f'--> Processing for file: {figname}')
        na.plot_morph_mip(type_id, color=colors[neurite], figname=figname, out_dir=out_dir, show_name=True)
    


