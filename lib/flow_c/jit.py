# CXX=g++-4.9 CC=gcc-4.9 python jit.py
from torch.utils.cpp_extension import load
import os
cur_dir = os.path.abspath(os.path.dirname(__file__))
gpu_flow = load(
    'gpu_flow',
    ['gpu_flow.cpp', 'gpu_flow_kernel.cu'],
    build_directory=cur_dir,
    verbose=True,
    extra_cuda_cflags=[
        '-arch=sm_52',  # sm_35, sm_61
        '--ptxas-options=-v',
        '-c',
        '--compiler-options',
        "'-fPIC'"
    ])
help(gpu_flow)
