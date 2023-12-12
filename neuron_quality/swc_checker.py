#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : swc_checker.py
#   Author       : Yufeng Liu
#   Date         : 2022-08-04
#   Description  : The script tries to check the common errors of swc file
#
#================================================================

from itertools import groupby
from collections import Counter

from swc_handler import parse_swc
from morph_topo.morphology import Morphology

class AbstractErrorChecker(object):
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self):
        pass

class MultiSomaChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(MultiSomaChecker, self).__init__(debug)

    def __call__(self, morph):
        num_soma = 0
        for node in morph.tree:
            if node[6] == -1:
                num_soma += 1
        if num_soma > 1:
            if self.debug:
                print(f'Warning: the swc contains {num_soma} somata')
            return False
        else:
            return True

class NoSomaChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(NoSomaChecker, self).__init__(debug)

    def __call__(self, morph):
        num_soma = 0
        for node in morph.tree:
            if node[6] == -1:
                num_soma += 1
        if num_soma == 0:
            if self.debug:
                print(f'Warning: the swc has no soma!')
            return False
        else:
            return True

class ParentZeroIndexChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(ParentZeroIndexChecker, self).__init__(debug)

    def __call__(self, morph):
        for node in morph.tree:
            if node[6] == 0:
                if self.debug:
                    print('Warning: the swc has node with parent index 0!')
                return False
        return True

class MultifurcationChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(MultifurcationChecker, self).__init__(debug)

    def __call__(self, morph):
        if not hasattr(morph, 'multifurcation'):
            morph.get_critical_points()
        no_multifur = len(morph.multifurcation) == 0
        if self.debug and not no_multifur:
            print('Warning: the swc has multifurcation!')

        return no_multifur

class TypeErrorChecker(AbstractErrorChecker):
    def __init__(self, debug, ignore_3_4=True):
        super(TypeErrorChecker, self).__init__(debug)
        self.ignore_3_4 = ignore_3_4

    def __call__(self, morph):
        paths = morph.get_all_paths()
        for path in paths.values():
            # get all types, except for the soma
            types = [morph.pos_dict[node][1] for node in path[:-1]]
            types_set = set(types)
            if len(types_set) == 1:
                continue
            else:
                if self.ignore_3_4:
                    new_types = [3 if v == 4 else v for v in types]
                    types_group = list(groupby(new_types))
                else:
                    types_group = list(groupby(types))

                num_switch = len(types_group) - 1
                if num_switch > 1:
                    if self.debug:
                        print('Too many type switches:')
                        for v in types_group:
                            print(f'  --> {v[0]}')
                        for t, p in zip(types, path[:-1]):
                            print(f'({t}, {p})', end=" ")
                        print('\n')
                    return False
                elif num_switch == 1:
                    v1, v2 = types_group[0][0], types_group[1][0]
                    if (v1 == 2) and (v2 in [3,4]): 
                        continue
                    else:
                        print(f'Wrong type switch: {v2} --> {v1}')
                        for t, p in zip(types, path[:-1]):
                            print(f'({t}, {p})', end=" ")
                        print('\n')
                        return False
        return True        

class LoopChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(LoopChecker, self).__init__(debug)

    def __call__(self, morph):
        coords = [node[2:5] for node in morph.tree]
        coords_set = set(coords)
        if len(coords) != len(coords_set):
            if self.debug:
                # for debug only
                print('Warning: duplicated nodes error! Duplicate coordinates are: ')
                counter = Counter(coords)
                for c, cc in counter.items():
                    if cc > 1:
                        print(c, cc)
            return False
        return True

class SingleTreeChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(SingleTreeChecker, self).__init__(debug)

    def __call__(self, morph):
        childs_l = [node[0] for node in morph.tree]
        parents_l = [node[6] for node in morph.tree]
        childs = set(childs_l)
        parents = set(parents_l)

        pc = parents - childs
        cp = childs - parents

        if len(childs) != len(morph.tree):
            return False

        vpc = len(pc) - 1
        if vpc != 0:
            print(vpc, pc)
            return False

        nos = sum([idx == -1 for idx in parents_l]) == 1
        if nos != 1:
            print('--> No. of somas is not 1', nos)
            return False

        niso = 0
        for idx1, idx2 in (childs_l, parents_l)::
            if idx1 == idx2:
                niso += 1
        niso == 0
        if not niso:
            print('==> incorrent node index', niso)
            return False

        return True

class SWCChecker(object):
    """
    Check the common errors of swc file
    """
    
    ERROR_TYPES = {
        'MultiSoma': 0,
        'NoSoma': 1,
        'ParentZeroIndex': 2,
        'Multifurcation': 3,
        'TypeError': 4,
        'Loop': 5,  # detect nodes with identical coordinates
        'SingleTree',
    }

    def __init__(self, error_types=(), debug=False, ignore_3_4=False):
        if not error_types:
            error_types = self.ERROR_TYPES
        
        self.checkers = []
        gvs = globals()
        for error_type in error_types:
            check_name = error_type + 'Checker'
            if error_type == 'TypeError':
                checker = gvs[check_name](debug=debug, ignore_3_4=ignore_3_4)
            else:
                checker = gvs[check_name](debug=debug)
            self.checkers.append(checker)
        
    def run(self, swcfile):
        if type(swcfile) is str:
            # load swc
            tree = parse_swc(swcfile)
        elif type(swcfile) is list:
            tree = swcfile
        morph = Morphology(tree)
        errors = []
        for checker in self.checkers:
            err = checker(morph)
            errors.append(err)
        return errors
        

if __name__ == '__main__':
    swcfile = '/home/lyf/test.swc'
    #error_types = ('MultiSoma', )
    
    swc_checker = SWCChecker(debug=True)
    print(swc_checker.run(swcfile))

