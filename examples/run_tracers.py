#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : run_tracers.py
#   Author       : Yufeng Liu
#   Date         : 2021-07-28
#   Description  : 
#
#================================================================

import os, glob
import subprocess
import numpy as np
import timeout_decorator

from multiprocessing.pool import Pool

class BaseTracer(object):
    DEFAULT_TIMEOUT = 3600

    def __init__(self, vaa3d_path=None, timeout=None):
        self.vaa3d_path = vaa3d_path
        if timeout is None:
            self.timeout = self.DEFAULT_TIMEOUT
        else:
            self.timeout = timeout

    def __call__(self, infile, outfile):
        pass

    #@timeout_decorator.timeout(DEFAULT_TIMEOUT)
    def run(self, cmd_str):
        try:
            out = subprocess.check_output(cmd_str, timeout=self.timeout, shell=True)
        except subprocess.TimeoutExpired:
            print(f'Time expired error for cmd: cmd_str')
            out = ''
        finally:
            out = ''
        return out


class APP1(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(APP1, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x vn2 -f app1 -i {infile} -p NULL 0 40 1; mv {infile}*_app1.swc {folder}'
        out = super().run(cmd_str)
        return out

class APP2(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(APP2, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x vn2 -f app2 -i {infile} -p NULL 0 10 1 1 0 0 5 0 0 0; mv {infile}*_app2.swc {folder}'
        out = super().run(cmd_str)
        return out

class APP2_NEW1(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(APP2_NEW1, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        outfile = f'{infile}_app2new1.swc'
        cmd_str = f'{self.vaa3d_path} -x vn2 -f app2 -i {infile} -o {outfile} -p NULL 0 AUTO 1 1 1 1 5 0 0 0; mv {outfile} {folder}'
        out = super().run(cmd_str)
        return out

class APP2_NEW2(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(APP2_NEW2, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        outfile = f'{infile}_app2new2.swc'
        cmd_str = f'{self.vaa3d_path} -x vn2 -f app2 -i {infile} -o {outfile} -p NULL 0 AUTO 1 1 0 1 5 0 0 0; mv {outfile} {folder}'
        out = super().run(cmd_str)
        return out

class APP2_NEW3(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(APP2_NEW3, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        outfile = f'{infile}_app2new3.swc'
        cmd_str = f'{self.vaa3d_path} -x vn2 -f app2 -i {infile} -o {outfile} -p NULL 0 10 1 1 1 1 5 0 0 0; mv {outfile} {folder}'
        out = super().run(cmd_str)
        return out

class MOST(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(MOST, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x MOST -f MOST_trace -i {infile} -p 1 40; mv {infile}*_MOST.swc {folder}'
        out = super().run(cmd_str)
        return out

class NEUTUBE(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(NEUTUBE, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x neuTube -f neutube_trace -i {infile} -p 1 1; mv {infile}*_neutube.swc {folder}'
        out = super().run(cmd_str)
        return out

class SNAKE(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(SNAKE, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x snake -f snake_trace -i {infile} -p 1; mv {infile}*_snake.swc {folder}'
        out = super().run(cmd_str)
        return out

class SimpleTracing1(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(SimpleTracing1, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x SimpleTracing -f tracing -i {infile} -o {infile}_simple.swc; mv {infile}*_simple.swc {folder}'
        out = super().run(cmd_str)
        return out

class SimpleTracing2(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(SimpleTracing2, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x SimpleTracing -f ray_shooting -i {infile} -o {infile}_Rayshooting.swc; mv {infile}*_Rayshooting.swc {folder}'
        out = super().run(cmd_str)
        return out

class SimpleTracing3(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(SimpleTracing3, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x SimpleTracing -f dfs -i {infile} -o {infile}_Rollerball.swc; mv {infile}*_Rollerball.swc {folder}'
        out = super().run(cmd_str)
        return out

class TreMap(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(TreMap, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x TReMap -f trace_mip -i {infile} -p 0 1 10 0 1 0 5; mv {infile}*_TreMap.swc {folder}'
        out = super().run(cmd_str)
        return out

class MST(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(MST, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x MST_tracing -f trace_mst -i {infile} -p 1 5; mv {infile}*_MST_Tracing.swc {folder}'
        out = super().run(cmd_str)
        return out

class NeuroGPSTree(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(NeuroGPSTree, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x NeuroGPSTree -f tracing_func -i {infile} -p 1 1 1 10; mv {infile}*_NeuroGPSTree.swc {folder}'
        out = super().run(cmd_str)
        return out

class NeuroGPSTree2(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(NeuroGPSTree2, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x NeuroGPSTree -f tracing_func -i {infile} -p 0.5 0.5 1 15 10 150; mv {infile}*_NeuroGPSTree.swc {folder}'
        out = super().run(cmd_str)
        return out

class FMST(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(FMST, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x fastmarching_spanningtree -f tracing_func -i {infile}; mv {infile}*_fastmarching_spanningtree.swc {folder}'
        out = super().run(cmd_str)
        return out

class MeanShift(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(MeanShift, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x BJUT_meanshift -f meanshift -i {infile}; rm {infile}*init_meanshift.swc; mv {infile}*_meanshift.swc {folder}'
        out = super().run(cmd_str)
        return out

class CWlab11(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(CWlab11, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x CWlab_method1_version1 -f tracing_func -i {infile} -p 1; mv {infile}*_Cwlab_ver1.swc {folder}'
        out = super().run(cmd_str)
        return out

class LCM_boost(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(LCM_boost, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x LCM_boost -f LCM_boost -i {infile} -o {infile}_LCMboost.swc; mv {infile}*_LCMboost.swc {folder}'
        out = super().run(cmd_str)
        return out

class LCM_boost_2(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(LCM_boost_2, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        outfile = f'{infile}_LCMboost_2.swc'
        cmd_str = f'{self.vaa3d_path} -x LCM_boost -f LCM_boost_2 -i {infile} -o {outfile}; mv {outfile} {folder}'
        out = super().run(cmd_str)
        return out

class LCM_boost_3(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(LCM_boost_3, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        outfile = f'{infile}_LCMboost_3.swc'
        cmd_str = f'{self.vaa3d_path} -x LCM_boost -f LCM_boost_3 -i {infile} -o {outfile}; mv {outfile} {folder}'
        out = super().run(cmd_str)
        return out

class NeuroStalker(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(NeuroStalker, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x NeuroStalker -f tracing_func -i {infile} -p 1 1 1 5 5 30; mv {infile}*_NeuroStalker.swc {folder}'
        out = super().run(cmd_str)
        return out

class nctuTW(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(nctuTW, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x nctuTW -f tracing_func -i {infile} -p NULL; mv {infile}*_nctuTW.swc {folder}'
        out = super().run(cmd_str)
        return out

class tips_GD(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(tips_GD, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x tips_GD -f tracing_func -i {infile}; mv {infile}*_nctuTW_GD.swc {folder}'
        out = super().run(cmd_str)
        return out

class SimpleAxisAnalyzer(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(SimpleAxisAnalyzer, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x SimpleAxisAnalyzer -f medial_axis_analysis -i {infile}; mv {infile}*_axis_analyzer.swc {folder}'
        out = super().run(cmd_str)
        return out

class NeuronChaser(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(NeuronChaser, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x NeuronChaser -f nc_func -i {infile} -p 1 10 0.6 15 60 30 5 1 0; mv {infile}*_NeuronChaser.swc {folder}'
        out = super().run(cmd_str)
        return out

class NeuronChaser2(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(NeuronChaser2, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x NeuronChaser -f nc_func -i {infile} -p 1 10 0.7 20 60 10 5 1 0; mv {infile}*_NeuronChaser.swc {folder}'
        out = super().run(cmd_str)
        return out

class smartTracing(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(smartTracing, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x smartTrace -f smartTrace -i {infile}; mv {infile}*_smartTracing.swc {folder}'
        out = super().run(cmd_str)
        return out

class neutu_autotrace(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(neutu_autotrace, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x neutu_autotrace -f tracing -i {infile}; mv {infile}*_neutu_autotrace.swc {folder}'
        out = super().run(cmd_str)
        return out

class Advantra(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(Advantra, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x Advantra -f advantra_func -i {infile} -p 10 0.3 0.6 15 60 30 5 1; mv {infile}*_Advantra.swc {folder}'
        out = super().run(cmd_str)
        return out

class Advantra2(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(Advantra2, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x Advantra -f advantra_func -i {infile} -p 10 0.5 0.7 20 60 30 5 1; mv {infile}*_Advantra.swc {folder}'
        out = super().run(cmd_str)
        return out

class RegMST(BaseTracer):
    def __init__(self, vaa3d_path=None, p0=21, p1=200):
        super(RegMST, self).__init__(vaa3d_path)
        self.p0 = p0
        self.p1 = p1

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        vaa3d_dir = os.path.split(self.vaa3d_path)[0]
        # there are files lacking here!
        cmd_str = f'{self.vaa3d_path} -x RegMST -f tracing_func -i {infile} -p {vaa3d_dir}/filter_banks/oof_fb_3d_scale_1_2_3_5_size_13_sep_cpd_rank_49.txt  {vaa3d_dir}/filter_banks/oof_fb_3d_scale_1_2_3_5_size_13_weigths_cpd_rank_49.txt {vaa3d_dir}/filter_banks/proto_filter_AC_lap_633_822_sep_cpd_rank_49.txt {vaa3d_dir}/filter_banks/proto_filter_AC_lap_633_822_weigths_cpd_rank_49.txt 1 2 {vaa3d_dir}/trained_models/model_S/Regressor_ac_0.cfg {vaa3d_dir}/trained_models/model_S/Regressor_ac_1.cfg {self.p0} {self.p1}; mv {infile}*_MST_Tracing_Ws_{self.p0}_th_{self.p1}.swc {folder}'
        out = super().run(cmd_str)
        return out

class EnsembleNeuronTracer(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(EnsembleNeuronTracer, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x EnsembleNeuronTracerBasic -f tracing_func -i {infile}; mv {infile}*_EnsembleNeuronTracerBasic.swc {folder}'
        out = super().run(cmd_str)
        return out

class EnsembleNeuronTracerV2n(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(EnsembleNeuronTracerV2n, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x EnsembleNeuronTracerV2n -f tracing_func -i {infile}; mv {infile}*_EnsembleNeuronTracerV2n.swc {folder}'
        out = super().run(cmd_str)
        return out

class EnsembleNeuronTracerV2s(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(EnsembleNeuronTracerV2s, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x EnsembleNeuronTracerV2s -f tracing_func -i {infile}; mv {infile}*_EnsembleNeuronTracerV2s.swc {folder}'
        out = super().run(cmd_str)
        return out

class threeDTraceSWC(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(threeDTraceSWC, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x aVaaTrace3D -f func1 -i {infile} -p 20 2 2.5; mv {infile}*_pyzh.swc {folder}'
        out = super().run(cmd_str)
        return out

class threeDTraceSWC2(BaseTracer):
    def __init__(self, vaa3d_path=None):
        super(threeDTraceSWC2, self).__init__(vaa3d_path)

    def __call__(self, infile, outfile):
        folder = os.path.split(outfile)[0]
        cmd_str = f'{self.vaa3d_path} -x aVaaTrace3D -f func1 -i {infile} -p 50 5 2.5; mv {infile}*_pyzh.swc {folder}'
        out = super().run(cmd_str)
        return out


class TracingRunner(object):
    def __init__(self, vaa3d_path=None, tracers=None):
        self.vaa3d_path = vaa3d_path
        self.tracers = tracers
        self.tracers_dict = self._init_tracers(tracers, vaa3d_path)

    @staticmethod
    def _init_tracers(tracers, vaa3d_path):
        print(f'Initializing {len(tracers)} tracers: {tracers}')
        tracers_dict = {}
        gvs = globals()
        for tracer in tracers:
            cls = gvs[tracer](vaa3d_path=vaa3d_path)
            tracers_dict[tracer] = cls
        return tracers_dict

    def wrapper_func_in_tracer(self, tracer, imgdir, outdir, file_ext, distinguish_str='[0-9]'):
        for imgfile in glob.glob(os.path.join(imgdir, f'*{distinguish_str}.{file_ext}')):
            imgname = os.path.split(imgfile)[-1]
            prefix = os.path.splitext(imgname)[0]
            outfile = os.path.join(outdir, f'{prefix}.swc')
            if len(glob.glob(os.path.join(outdir, f'{prefix}*swc'))) > 0:
                continue
            
            out = self.tracers_dict[tracer](imgfile, outfile)
        #return out

    def run_in_tracer(self, imgdir, outdir, nprocessors, with_subdir, file_ext, distinguish_str):
        # initialize task list
        args_list = []
        for tracer in self.tracers:
            # create folders if not exists
            trace_dir = os.path.join(outdir, tracer)
            if not os.path.exists(trace_dir):
                os.mkdir(trace_dir)

            if not with_subdir:
                args_list.append((tracer, imgdir, trace_dir, file_ext, distinguish_str))
            else:
                for cur_path in glob.glob(os.path.join(imgdir, '*')):
                    if not os.path.isdir(cur_path):
                        continue
                    folder_name = os.path.split(cur_path)[-1]
                    out_path = os.path.join(trace_dir, folder_name)
                    if not os.path.exists(out_path):
                        os.mkdir(out_path)
                    args_list.append((tracer, cur_path, out_path, file_ext, distinguish_str))

        print(f'Number of files to process: {len(args_list)}')
            
        pt = Pool(nprocessors)
        pt.starmap(self.wrapper_func_in_tracer, args_list)
        pt.close()
        pt.join()

    def wrapper_func_in_tracer_file(self, tracer, infile, outfile):
        cls = self.tracers_dict[tracer]
        return cls(infile, outfile)

    def run_in_tracer_file(self, imgdir, outdir, nprocessors, with_subdir, file_ext, distinguish_str):
        # initialize task list
        args_list = []
        for tracer in self.tracers:
            # create folders if not exists
            trace_dir = os.path.join(outdir, tracer)
            if not os.path.exists(trace_dir):
                os.mkdir(trace_dir)
            if not with_subdir:
                for imgfile in glob.glob(os.path.join(imgdir, f'*{distinguish_str}.{file_ext}')):
                    imgname = os.path.split(imgfile)[-1]
                    prefix = os.path.splitext(imgname)[0]
                    outfile = os.path.join(trace_dir, f'{prefix}.swc')
                    if len(glob.glob(os.path.join(trace_dir, f'{prefix}*swc'))) > 0:
                        continue
                    args_list.append((tracer, imgfile, outfile))
            else:
                for cur_path in glob.glob(os.path.join(imgdir, '*')):
                    if not os.path.isdir(cur_path):
                        continue
                    folder_name = os.path.split(cur_path)[-1]
                    out_path = os.path.join(trace_dir, folder_name)
                    if not os.path.exists(out_path):
                        os.mkdir(out_path)
                    for imgfile in glob.glob(os.path.join(cur_path, f'*{distinguish_str}.{file_ext}')):
                        imgname = os.path.split(imgfile)[-1]
                        prefix = os.path.splitext(imgname)[0]
                        outfile = os.path.join(out_path, f'{prefix}.swc')
                        if len(glob.glob(os.path.join(out_path, f'{prefix}*swc'))) > 0:
                            continue
                        args_list.append((tracer, imgfile, outfile))
                    
                    

        np.random.shuffle(args_list)

        print(f'Number of files to process: {len(args_list)}')
            
        pt = Pool(nprocessors)
        pt.starmap(self.wrapper_func_in_tracer_file, args_list)
        pt.close()
        pt.join()
        
if __name__ == '__main__':
    data_name = 'images'
    imgdir = f'/media/lyf/storage/seu_mouse/misc/SWC2IMG_gold163/{data_name}'
    outdir = f'/media/lyf/storage/seu_mouse/misc/SWC2IMG_gold163/traced_{data_name}'
    
    vaa3d_path = '/home/lyf/Softwares/installation/Vaa3D_v3.447/Vaa3D_CentOS_64bit_v3.447/start_vaa3d.sh'
    file_ext = 'tif'
    with_subdir = True
    distinguish_str = 'swc2img'

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    #tracers = ['APP1', 'APP2', 'MOST', 'NEUTUBE', 'SNAKE', 'SimpleTracing1', 
    #           'SimpleTracing2', 'SimpleTracing3', 'TreMap', 'MST', 
    #           'NeuroGPSTree', 'FMST', 'MeanShift', 'CWlab11', 'LCM_boost', 'NeuroStalker', 
    #           'nctuTW', 'tips_GD', 'SimpleAxisAnalyzer', 'NeuronChaser', 'smartTracing',
    #           'neutu_autotrace', 'Advantra', 'RegMST', 'EnsembleNeuronTracer', 
    #           'EnsembleNeuronTracerV2n', 'EnsembleNeuronTracerV2s', 'threeDTraceSWC']
    #tracers = ['APP1', 'APP2', 'MOST', 'NEUTUBE', 'SNAKE', 'SimpleTracing1', 
    #           'SimpleTracing2', 'SimpleTracing3', 'TreMap', 'MST', 
    #           'NeuroGPSTree', 'MeanShift', 'CWlab11', 'LCM_boost', 'NeuroStalker', 
    #           'SimpleAxisAnalyzer', 'NeuronChaser', 'smartTracing',
    #           'neutu_autotrace', 'Advantra', 'RegMST', 'EnsembleNeuronTracer', 
    #           'EnsembleNeuronTracerV2n', 'EnsembleNeuronTracerV2s']

    #tracers = ['APP2_NEW1', 'NeuroGPSTree2', 'NeuronChaser2',
    #           'Advantra2', 'APP2_NEW2', 'APP2_NEW3', 
    #          ]
    tracers = ['APP2_NEW1', 'Advantra2', 'NeuronChaser2']
    # FMST is very memory-intensive!
    # tips_GD, threeDTraceSWC, EnsembleNeuronTracer are problematic
    nprocessors = 6

    tr = TracingRunner(vaa3d_path, tracers)
    tr.run_in_tracer_file(imgdir, outdir, nprocessors, with_subdir=with_subdir, file_ext=file_ext, distinguish_str=distinguish_str)

