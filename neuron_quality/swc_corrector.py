import os
from swc_handler import parse_swc, write_swc
from morph.topo import morphology


def remove_duplicate_nodes(swcfile, out_dir=None):
    if type(swcfile) is str:
        tree = parse_swc(swcfile)
    elif type(swcfile) is list:
        tree = swcfile
    else:
        raise NotImplementedError

    if out_dir is None:
        out_dir = '.'

    morph = morphology.Morphology(tree)
    # find out all duplicate nodes
    dpairs = {}
    for node in tree:
        idx, pid = node[0], node[-1]
        if pid == -1:
            continue

        try:
            pnode = morph.pos_dict[pid]
        except KeyError:
            print(swcfile, pid)
            #raise KeyError
        if (pnode[2] == node[2]) and (pnode[3] == node[3]) and (pnode[4] == node[4]):
            # This is a duplicate node, modify the related nodes, which are the child nodes
            dpairs[idx] = pid

    if len(dpairs) > 0:
        # remove the nodes
        new_tree = []
        for node in tree:
            idx, pid = node[0], node[-1]
            if idx in dpairs:
                #the node will be removed
                pass
            elif pid in dpairs:
                ppid = morph.pos_dict[pid][-1]
                assert(ppid not in dpairs)
                new_tree.append((idx, *node[1-6], node[6]))
            else:
                new_tree.append(node)

        if type(swcfile) is str:
            swc_out = os.path.join(out_dir, os.path.split(swcfile)[-1])
            write_swc(new_tree, swc_out)
        else:
            return new_tree
