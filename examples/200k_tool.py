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


class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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
    cos_ch = [vec_p.dot(vec) / (np.linalg.norm(vec_p) * np.linalg.norm(vec)) for vec in vec_ch]
    out = [*map(lambda x: math.acos(max(min(x, 1), -1)) * 180 / math.pi, cos_ch)]
    return out


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
        if i in morph.child_dict:
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
        if i in morph.child_dict:
            q.extend(morph.child_dict[i])
        while len(q) > 0:
            head = q.pop(0)
            ind = morph.index_dict[head]
            if tree[ind] is None:
                continue
            tree[ind] = None
            if head in morph.child_dict:
                q.extend(morph.child_dict[head])
    return [t for t in tree if t is not None]


def crossing_prune(args):
    tree, img_path, anchor_dist, soma_radius, dist_thr, spacing, sampling = args
    morph = Morphology(tree)
    cf = CrossingFinder(morph, soma_radius, dist_thr)
    img = load_image(img_path)
    with HidePrint():
        tri, db = cf.find_crossing_pairs()
    rm_ind = set()
    for i in tri:
        # angle
        with HidePrint():
            center, anchor_p, anchor_ch, protrude, com_node = get_anchors(morph, [i], anchor_dist)
        angles = anchor_angles(center, np.array(anchor_p), np.array(anchor_ch), spacing=spacing)
        angles = np.array(angles)
        to = np.argmin(abs(angles - 180))
        rm_ind |= (set(protrude) - {protrude[to]})
    for down, up, dist in db:
        # angle
        with HidePrint():
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
    return prune(morph, rm_ind)


def branch_prune(args):
    tree, angle_thr, anchor_dist, soma_radius, spacing = args
    morph = Morphology(tree)
    awry_node = set()
    cs = np.array(morph.pos_dict[morph.idx_soma][2:5])
    for n, l in morph.pos_dict.items():
        if n in morph.child_dict and len(morph.child_dict[n]) > 1 and \
                morph.pos_dict[n][-1] != -1 and np.linalg.norm(morph.pos_dict[n][2:5] - cs) * spacing <= soma_radius:
            # branch, no soma, away from soma
            with HidePrint():
                center, anchor_p, anchor_ch, protrude, com_node = get_anchors(morph, [n], anchor_dist)
            angles = anchor_angles(center, np.array(anchor_p), np.array(anchor_ch), spacing=spacing)
            angles = np.array(angles)
            protrude = np.array(protrude)
            awry_node |= set(protrude[angles < angle_thr])
    return prune(morph, awry_node)


class CLI200k:
    """
    BICCN 200K SWC post-processing CLI tools
    """

    def __init__(self, swc_files=None, img_files=None, ano=None, jobs=1, chunk_size=1000, spacing=(1, 1, 4), soma_radius=15):
        """
        The script allows 3 input manners for swc and img files, which can work in parallel.

        1. specify swc-files and img-files as 2 lists like [a,b,c], which should have the same length to map 1 by 1.

        2. specify swc-files ad img-files as 2 wildcards. This assumes swc and images share their number as well as
        their path pattern, meaning the file lists are generated and sorted by name.

        3. specify ano an ano with same number of lines of SWCFILE and RAWIMAGE for everything, and they are matched by
        the order of input, or it can be a wildcard for a set of ano with the same requirement.

        Whichever way, the output paths are inferred based on the swc-files.

        soma-radius: nodes within soma-radius are not pruned
        spacing: swc and image scaling over the dimensions
        jobs: number of processes
        """
        assert type(swc_files) == type(img_files)
        self.spacing = spacing
        self.soma_radius = soma_radius
        self.swc_files = []
        self.img_files = []
        if swc_files is not None:
            if type(swc_files) is str:
                self.swc_files.extend(sorted(glob.glob(swc_files, recursive=True)))
                self.img_files.extend(sorted(glob.glob(img_files, recursive=True)))
            elif type(swc_files) is list:
                self.swc_files.extend(swc_files)
                self.img_files.extend(img_files)
            else:
                raise "swc-files and img-files should be specified as either lists or strings"
        assert len(self.swc_files) > 0 and len(self.swc_files) == len(self.img_files)
        if ano is not None:
            if type(ano) is str:
                ano_files = glob.glob(ano, recursive=True)
            elif type(ano) is list:
                ano_files = ano
            else:
                raise "ano should be specified as either a list or a string"
            for a in ano_files:
                with open(a) as f:
                    ano_dir = os.path.dirname(a)
                    lines = [i.rstrip().split('=') for i in f.readlines()]
                    swc_files = [i[1] if os.path.isabs(i[1]) else os.path.abspath(os.path.join(ano_dir, i[1]))
                                 for i in lines if i[0] == "SWCFILE"]
                    img_files = [i[1] if os.path.isabs(i[1]) else os.path.abspath(os.path.join(ano_dir, i[1]))
                                 for i in lines if i[0] == "RAWIMAGE"]
                    assert len(swc_files) > 0 and len(swc_files) == len(img_files)
                    self.swc_files.extend(swc_files)
                    self.img_files.extend(img_files)
        self.root = os.path.commonpath([os.path.dirname(f) for f in self.swc_files])
        self.jobs = jobs
        self.chunk = chunk_size
        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            self.trees = list(pool.map(swc_handler.parse_swc, self.swc_files, chunksize=self.chunk))
        print('swc loading finished.')
        print('loaded number of files: {}'.format(len(self.trees)))

    def node_limit_filter(self, downer=1, upper=-1):
        """
        given a closed interval, return a list of swc paths that meet the range
        upper can be negative(default) to disable
        """
        bool_list = [*map(lambda tree: len(tree) >= downer and (upper < 0 or len(tree) <= upper), self.trees)]
        self.swc_files = list(compress(self.swc_files, bool_list))
        self.trees = list(compress(self.trees, bool_list))
        print('swc filtering finished.')
        print('remaining number of swc: {}'.format(len(self.trees)))
        return self

    def crossing_prune(self, anchor_dist=15, dist_thr=3, sampling=10):
        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            self.trees = list(
                pool.map(crossing_prune,
                         [(tree, img_path, anchor_dist, self.soma_radius, dist_thr, self.spacing, sampling)
                          for tree, img_path in zip(self.trees, self.img_files)],
                         chunksize=self.chunk)
            )
            print('angle prune done.')
        return self

    def branch_prune(self, angle_thr=90, anchor_dist=15):
        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            self.trees = list(
                pool.map(branch_prune,
                         [(tree, angle_thr, anchor_dist, self.soma_radius, self.spacing) for tree in self.trees],
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
            for f, t in zip(self.swc_files, self.trees):
                path = f.split('.swc')[0] + suffix + '.swc'
                if dir is not None:
                    path = os.path.join(dir, path.split(self.root)[1].lstrip('/').lstrip('\\'))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                swc_handler.write_swc(t, path)
        print('swc writing finished.')


if __name__ == '__main__':
    fire.Fire(CLI200k)
