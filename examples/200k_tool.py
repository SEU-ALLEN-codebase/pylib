import sys

sys.path.append(sys.path[0] + "/..")
import fire
from concurrent.futures import ProcessPoolExecutor, wait
from itertools import compress
from functools import reduce
import glob
import os
from file_io import load_image, save_markers
import swc_handler
from neuron_quality.find_break_crossing import CrossingFinder
from morph_topo.morphology import Morphology
import numpy as np
from scipy.interpolate import CubicSpline
import math
from neuron_quality.find_break_crossing import find_point_by_distance


def anchor_angles(center: np.ndarray, p: np.ndarray, ch, spacing=(1, 1, 4)):
    """
    modified from Jingzhou's code
    :param center: coordinate for the multifurcation center
    :param p: coordinate for the parent anchor
    :param ch: coordinates for the furcation anchors
    :param spacing: scale factor for each dimension
    """
    vec_p = (p - center) * spacing
    vec_ch = [(coord - center) * spacing for coord in ch]
    cos_ch = [np.sum(vec_p * vec) / (np.linalg.norm(vec_p) * np.linalg.norm(vec)) for vec in vec_ch]
    return [*map(lambda x: math.acos(x) * 180 / math.pi, cos_ch)]


def get_anchors(morph: Morphology, ind: list, dist: float):
    """
    get anchors for a set of swc nodes to calculate angles, suppose they are one, their center is their mean coordinate,
    getting anchors requires removing redundant protrudes
    :param dist: path distance thr
    :param ind: array of coordinates
    """
    coords = [morph.pos_dict[i][2:5] for i in ind]
    center = np.mean(coords, axis=0)
    protrude = set()
    for i in ind:
        if morph.child_dict.get(i) is not None:
            protrude = protrude.union(morph.child_dict[i])
    com_line = None
    for i in ind:
        line = [i]
        while line[-1] != -1:
            line.append(morph.pos_dict[line[-1]][-1])
        line.reverse()
        protrude = protrude - set(line[:-1])
        if com_line is None:
            com_line = line
        else:
            for j, t in enumerate(com_line):
                if t != line[j]:
                    com_line = com_line[:j]
                    break
    # nearest common parent of all indices
    com_node = com_line[-1]
    protrude = list(protrude)
    anchor_p = find_point_by_distance(center, com_node, True, morph, dist, False, stop_by_branch=False)
    anchor_ch = [find_point_by_distance(center, i, False, morph, dist, False) for i in protrude]
    return center, anchor_p, anchor_ch, protrude, com_node


def prune(morph: Morphology, ind_set: set):
    tree = morph.tree.copy()
    for i in ind_set:
        q = []
        ind = morph.index_dict[i]
        if tree[ind] is None:
            continue
        tree[ind] = None
        if morph.child_dict.get(i) is not None:
            q.extend(morph.child_dict[i])
        while len(q) > 0:
            head = q.pop(0)
            ind = morph.index_dict[head]
            if tree[ind] is None:
                continue
            tree[ind] = None
            if morph.child_dict.get(head) is not None:
                q.extend(morph.child_dict[head])
    return [t for t in tree if t is not None]


def angle_prune(args):
    tree, angle_thr, anchor_dist, soma_radius, dist_thr, spacing = args
    morph = Morphology(tree)
    cf = CrossingFinder(morph, soma_radius, dist_thr)
    tri, db = cf.find_crossing_pairs()
    rm_ind = set()
    for i in tri:
        # angle
        center, anchor_p, anchor_ch, protrude, com_node = get_anchors(morph, [i], anchor_dist)
        angles = anchor_angles(center, np.array(anchor_p), np.array(anchor_ch), spacing=spacing)
        angles = np.array(angles)
        to = np.argmin(abs(angles - 180))
        rm_ind |= (set(protrude) - {protrude[to]})
    for down, up, dist in db:
        # angle
        center, anchor_p, anchor_ch, protrude, com_node = get_anchors(morph, [up, down], anchor_dist)
        angles = anchor_angles(center, np.array(anchor_p), np.array(anchor_ch), spacing=spacing)
        angles = np.array(angles)
        to = np.argmin(abs(angles - 180))
        j = protrude[to]
        jj = morph.pos_dict[j][-1]
        while j != com_node:
            for k in morph.child_dict[jj]:
                if k != j:
                    rm_ind.add(k)
            j = jj
            jj = morph.pos_dict[j][-1]
    awry_angle = set()
    for n, l in morph.pos_dict.items():
        if morph.child_dict.get(n) is not None and len(morph.child_dict[n]) > 1 and morph.pos_dict[n][-1] != -1:
            center, anchor_p, anchor_ch, protrude, com_node = get_anchors(morph, [n], anchor_dist)
            angles = anchor_angles(center, np.array(anchor_p), np.array(anchor_ch), spacing=spacing)
            angles = np.array(angles)
            protrude = np.array(protrude)
            awry_angle |= set(protrude[angles < angle_thr])

    rm_ind |= awry_angle
    return prune(morph, rm_ind)


class CLI200k:
    """
    BICCN 200K SWC post-processing CLI tools
    """

    def __init__(self, files=None, dir=None, apo=None, jobs=1, chunk_size=1000):
        """
        file comes first than apo,
        a wild card matching is available, or give a list like [p1,p2,p3],
        if dir is specified, anything from files or apo will be joined as a new path
        also, this dir can decide the output folder structure, if not given, will be inferred as the top folder
        for all input files
        """
        if files is not None:
            if type(files) is str:
                if dir is not None:
                    files = os.path.join(dir, files)
                self.files = glob.glob(files)
            elif type(files) is list:
                if dir is not None:
                    files = [os.path.join(dir, f) for f in files]
                self.files = files
            else:
                raise "match should be list or str"
        elif apo is not None:
            with open(apo) as f:
                files = [i.rstrip().split('SWCFILE=')[1] for i in f.readlines()]
                if dir is not None:
                    files = [os.path.join(dir, f) for f in files]
                self.files = files
        else:
            raise "either match or apo should be specified"
        if dir is not None:
            self.root = dir
        else:
            self.root = os.path.commonpath(self.files)
        self.jobs = jobs
        self.chunk = chunk_size
        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            self.trees = list(pool.map(swc_handler.parse_swc, self.files, chunksize=self.chunk))
        print('swc loading finished.')
        print('loaded number of files: {}'.format(len(self.trees)))

    def node_limit_filter(self, downer=1, upper=-1):
        """
        given a closed interval, return a list of swc paths that meet the range
        upper can be negative(default) to disable
        """
        bool_list = [*map(lambda tree: len(tree) >= downer and (upper < 0 or len(tree) <= upper), self.trees)]
        self.files = list(compress(self.files, bool_list))
        self.trees = list(compress(self.trees, bool_list))
        print('swc filtering finished.')
        print('remaining number of swc: {}'.format(len(self.trees)))
        return self

    def angle_prune(self, angle_thr=90, anchor_dist=15, soma_radius=15, dist_thr=1.5, spacing=(1, 1, 4)):
        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            self.trees = list(
                pool.map(angle_prune,
                         [(tree, angle_thr, anchor_dist, soma_radius, dist_thr, spacing) for tree in self.trees],
                         chunksize=self.chunk)
            )
            print('angle prune done.')
        return self

    def write(self, dir=None, suffix="_filtered"):
        """
        add suffix between file name and file type
        """
        if dir is not None and self.root == "":
            print('warning: no common dir for all the swc, will save to the original folder.')
            dir = None
        if dir is None and suffix == "":
            ok = input('warning: swc writing will replace the original file. proceed? (y/yes/[enter])')
            if not (ok.lower() == 'y' or ok == '' or ok.lower() == 'yes'):
                print('job terminated.')
                return
        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            for f, t in zip(self.files, self.trees):
                path = f.split('.swc')[0] + suffix + '.swc'
                if dir is not None:
                    path = os.path.join(dir, path.split(self.root)[1].lstrip('/').lstrip('\\'))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                swc_handler.write_swc(t, path)
        print('swc writing finished.')


if __name__ == '__main__':
    fire.Fire(CLI200k)
