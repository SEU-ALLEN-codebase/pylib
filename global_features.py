##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-07
#Description:               
##########################################################
import os, glob
import sys
import time
import subprocess
import pandas as pd
import numpy as np


__FEAT_NAMES22__ = [
    'Nodes', 'SomaSurface', 'Stems', 'Bifurcations',
    'Branches', 'Tips', 'OverallWidth', 'OverallHeight', 'OverallDepth',
    'AverageDiameter', 'Length', 'Surface', 'Volume',
    'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder',
    'AverageContraction', 'AverageFragmentation',
    'AverageParent-daughterRatio', 'AverageBifurcationAngleLocal',
    'AverageBifurcationAngleRemote', 'HausdorffDimension'
]

FEAT_NAME_DICT = {
    'N_node': 'Nodes',
    'Soma_surface': 'SomaSurface',
    'N_stem': 'Stems',
    'Number of Bifurcatons': 'Bifurcations',
    'Number of Branches': 'Branches',
    'Number of Tips': 'Tips',
    'Overall Width': 'OverallWidth',
    'Overall Height': 'OverallHeight',
    'Overall Depth': 'OverallDepth',
    'Average Diameter': 'AverageDiameter',
    'Total Length': 'Length',
    'Total Surface': 'Surface',
    'Total Volume': 'Volume',
    'Max Euclidean Distance': 'MaxEuclideanDistance',
    'Max Path Distance': 'MaxPathDistance',
    'Max Branch Order': 'MaxBranchOrder',
    'Average Contraction': 'AverageContraction',
    'Average Fragmentation': 'AverageFragmentation',
    'Average Parent-daughter Ratio': 'AverageParent-daughterRatio',
    'Average Bifurcation Angle Local': 'AverageBifurcationAngleLocal',
    'Average Bifurcation Angle Remote': 'AverageBifurcationAngleRemote',
    'Hausdorff Dimension': 'HausdorffDimension'
}

def calc_global_features(swc_file, vaa3d='/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x'):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x global_neuron_feature -f compute_feature -i "{swc_file}"'
    p = subprocess.check_output(cmd_str, shell=True)
        
    output = p.decode().splitlines()[37:-2]
    info_dict = {}
    for s in output:
        it1, it2 = s.split(':')
        it1 = it1.strip()
        it2 = it2.strip()
        info_dict[it1] = float(it2)

    # extract the target result
    #print(info_dict)
    features = []
    features.append(int(info_dict['N_node']))
    features.append(info_dict['Soma_surface'])
    features.append(int(info_dict['N_stem']))
    features.append(int(info_dict['Number of Bifurcatons']))
    features.append(int(info_dict['Number of Branches']))
    features.append(int(info_dict['Number of Tips']))
    features.append(info_dict['Overall Width'])
    features.append(info_dict['Overall Height'])
    features.append(info_dict['Overall Depth'])
    features.append(info_dict['Average Diameter'])
    features.append(info_dict['Total Length'])
    features.append(info_dict['Total Surface'])
    features.append(info_dict['Total Volume'])
    features.append(info_dict['Max Euclidean Distance'])
    features.append(info_dict['Max Path Distance'])
    features.append(info_dict['Max Branch Order'])
    features.append(info_dict['Average Contraction'])
    features.append(info_dict['Average Fragmentation'])
    features.append(info_dict['Average Parent-daughter Ratio'])
    features.append(info_dict['Average Bifurcation Angle Local'])
    features.append(info_dict['Average Bifurcation Angle Remote'])
    features.append(info_dict['Hausdorff Dimension'])

    return features


def calc_global_features_from_folder(swc_dir, outfile=None, robust=True):

    features_all = []
    iswc = 0
    t0 = time.time()
    for swcfile in glob.glob(os.path.join(swc_dir, '*swc')):
        print(swcfile)
        prefix = os.path.splitext(os.path.split(swcfile)[-1])[0]
        #prefix = os.path.split(swcfile)[-1]
        if robust:
            try:
                features = calc_global_features(swcfile)
            except ValueError:
                continue
        else:
            features = calc_global_features(swcfile)
        features_all.append([prefix, *features])

        iswc += 1
        if iswc % 10 ==  0:
            print(f'--> {iswc} in {time.time() - t0:.2f} s')

    df = pd.DataFrame(features_all, columns=['',  *__FEAT_NAMES22__])
    if outfile is not None:
        df.to_csv(outfile, float_format='%g', index=False)
    return df


