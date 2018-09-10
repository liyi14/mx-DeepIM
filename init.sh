#!/bin/bash

mkdir -p ./data
mkdir -p ./output
mkdir -p ./external/mxnet
mkdir -p ./model/pretrained_model


cd lib/flow_c
python setup_linux.py build_ext --inplace
## in python3, if problems with setup_linux.py,
## install pytorch first to use jit compile by uncomment the folowing lines
#pwd
#sh build.sh
cd ../..
