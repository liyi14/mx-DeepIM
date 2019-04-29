#CXX=g++-4.9 CC=gcc-4.9 python jit.py
rm -rf gpu_flow.cpython*
#CUDAHOME=/usr/local/cuda-9.1 python setup_linux.py build_ext --inplace
python setup_linux.py build_ext --inplace