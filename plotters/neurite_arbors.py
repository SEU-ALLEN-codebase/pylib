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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from swc_handler import parse_swc, write_swc, get_specific_neurite, NEURITE_TYPES
from morph_topo import morphology


class NeuriteArbors:
    def __init__(self, swcfile, soma_params=None):
        tree = parse_swc(swcfile)
        self.morph = morphology.Morphology(tree)
        self.morph.get_critical_points()

        self.soma_params = soma_params
        if soma_params is not None:
            self.soma_xyz = self.morph.tree[self.morph.index_soma][2:5]


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

       
    def plot_morph_mip(self, type_id, xxyy=None, mip='z', color='r', figname='temp.png', 
                       out_dir='.', show_name=False, linewidth=2, bkg_transparent=False):
        paths = self.get_paths_of_specific_neurite(type_id, mip=mip)
        
        if bkg_transparent:
            fig = plt.figure(figsize=(8,8), facecolor='none', edgecolor='none')
            ax = plt.gca()
            ax.set_facecolor('none')
        else:
            fig = plt.figure(figsize=(8,8))


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
        
        # soma visualization
        if self.soma_params is not None:
            if mip == 'z':
                soma_xy = self.soma_xyz[:2]
            elif mip == 'x':
                soma_xy = self.soma_xyz[1:]
            elif mip == 'y':
                soma_xy = (self.soma_xyz[0], self.soma_xyz[2])
            plt.scatter(soma_xy[0], soma_xy[1], s=self.soma_params['size'], 
                        color=self.soma_params['color'], alpha=self.soma_params['alpha'],
                        zorder=100)

        # title
        if show_name:
            plt.title(figname)

        if bkg_transparent:
            plt.savefig(os.path.join(out_dir, f'{figname}.png'), dpi=300, transparent=True,
                        bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(os.path.join(out_dir, f'{figname}.png'), dpi=300)
        plt.close()


    def get_morph_mip(self, type_id, img_shape=None, xxyy=None, mip='z', color='r', 
                      linewidth=2, bkg_transparent=False):
        """
        绘制神经元形态的投影图，确保与原始图像尺寸一致
        
        参数:
            img_shape: 原始图像的形状(h,w)，用于确保尺寸一致
            xxyy: 如果不为None，应提供(xmin, xmax, ymin, ymax)坐标范围
            return_fig: 如果为True，返回图像数组；否则保存到文件
        """
        paths = self.get_paths_of_specific_neurite(type_id, mip=mip)
        
        # 计算坐标范围
        try:
            all_paths = np.vstack(paths)
            if xxyy is None:
                xxyy = (all_paths.min(axis=0)[0], all_paths.max(axis=0)[0], 
                        all_paths.min(axis=0)[1], all_paths.max(axis=0)[1])
        except ValueError:
            xxyy = (0, 1, 0, 1)  # 默认范围，如果没有路径
        
        # 设置图形尺寸与原始图像保持相同比例
        if img_shape is not None:
            h, w = img_shape
            figsize = (8, 8*(h/w)) if w > h else (8*(w/h), 8)
            dpi = max(w, h) / 8  # 根据尺寸自动计算DPI
        else:
            figsize = (8, 8)
            dpi = 100
        
        # 创建图形
        fig = plt.figure(figsize=figsize, facecolor='none' if bkg_transparent else 'white')
        ax = fig.add_axes([0, 0, 1, 1], facecolor='none' if bkg_transparent else 'white')
        
        # 绘制路径
        for path in paths:
            ax.plot(path[:,0], path[:,1], color=color, lw=linewidth)
        
        # 设置相同的坐标范围
        ax.set_xlim(xxyy[0], xxyy[1])
        ax.set_ylim(xxyy[2], xxyy[3])
        ax.invert_yaxis()  # 通常图像坐标系y轴是反的
        
        # 隐藏坐标轴
        ax.set_axis_off()
        
        # 绘制胞体
        if self.soma_params is not None:
            if mip == 'z':
                soma_xy = self.soma_xyz[:2]
            elif mip == 'x':
                soma_xy = self.soma_xyz[1:]
            elif mip == 'y':
                soma_xy = (self.soma_xyz[0], self.soma_xyz[2])
            ax.scatter(soma_xy[0], soma_xy[1], s=self.soma_params['size'],
                      color=self.soma_params['color'], alpha=self.soma_params['alpha'],
                      zorder=100)
        
        # 将图形转换为numpy数组
        canvas = FigureCanvas(fig)
        canvas.draw()
        morph_img = np.array(canvas.renderer.buffer_rgba())#[..., :3]  # 去掉alpha通道
        
        # 调整尺寸与原始图像完全一致
        if img_shape is not None:
            from PIL import Image
            morph_img = np.array(Image.fromarray(morph_img).resize((w, h), Image.LANCZOS))
        
        plt.close(fig)
        return morph_img
        

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
    


