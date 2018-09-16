### DeepIM: Deep Iterative Matching for 6D Pose Estimation    
Yi Li, Gu Wang, Xiangyang Ji, Yu Xiang and Dieter Fox.  
In ECCV, 2018.    
[arXiv](https://arxiv.org/abs/1804.00175), [project page](https://rse-lab.cs.washington.edu/projects/deepim/)
    
The major contributors of this repository include [Yi Li](https://github.com/liyi14) and [Gu Wang](https://github.com/wangg12).

### Citing DeepIM
If you find DeepIM useful in your research, please consider citing:
```
@inproceedings{li2017deepim,
title     = {DeepIM: Deep Iterative Matching for 6D Pose Estimation},
author    = {Yi Li and Gu Wang and Xiangyang Ji and Yu Xiang and Dieter Fox},
booktitle = {European Conference on Computer Vision (ECCV)},
year      = {2018}
}
```
## Overall Framework
<p align="center">
<img src='assets/intro.png' width='600'>
<p>

### Network Structure
<p align="center">
<img src='assets/net_structure.png' width='600'>
<p>

### Zoom In Operation
<p align="center">
<img src='assets/zoom_in.png' width='600'>
</p>


## Main Results

### LINEMOD
<p align="center">
<img src='assets/LM6d_table.png' width='600'>
</p>

### Occlusion LINEMOD
<p align="center">
<img src='assets/LM6d_Occ_results.png' width='800'>
</p>
<p align="center">
<img src='assets/LM6d_Occ_results_pictures.png' width='600'>
</p>

### Unseen Objects from ModelNet
<p align="center">
<img src='assets/Unseen.png' width='600'>
</p>

The red and green lines
represent the edges of 3D model projected from the initial poses and our refined poses
respectively.

## Requirements: Software

1. MXNet from [the official repository](https://github.com/dmlc/mxnet). 
We tested our code on [MXNet@(commit fc9e70b)](https://github.com/apache/incubator-mxnet/commit/fc9e70b).
   
    Or use the prebuilt version following [the installation guide.](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=GPU).   
    ```
        nvcc --version
        pip install mxnet-cu90 # (change to your cuda version)
    ```
   
2. Python 2.7. We recommend using Anaconda. (python 3.x should also be OK.) 
3. GLFW for OpenGL: 
`sudo apt-get install libglfw3-dev libglfw3` (on Ubuntu 16.04)   
4. Python packages might missing: 
	```
	conda install scipy
	pip install Cython
	pip install opencv-python
	pip install easydict
	pip install pyyaml
	```
   glumpy:
   ```
   pip install pyopengl packaging appdirs pyopengl triangle cython glfw
   # clone the lastest glumpy (there is a bug in the pip version)
   git clone https://github.com/glumpy/glumpy.git
   cd glumpy
   pip install .
   ``` 	

5. (Optional) Use tensorboard to visualize loss: 

    Install [mxboard](https://github.com/awslabs/mxboard) 
    following [https://github.com/awslabs/mxboard#installation](https://github.com/awslabs/mxboard#installation).
    
    ``pip install mxboard`` 



## Requirements: Hardware

Any NVIDIA GPUs with at least 4GB memory should be OK.

## Installation

1. Clone the DeepIM repository, and we'll call the directory that you cloned mx-DeepPose as ${DeepIM_ROOT}.
```
git clone https://github.com/liyi14/mx-DeepIM.git
```


2. Install MXNet:

    2.0 Ignore this step if you have installed the prebuilt version.

	2.1 Clone MXNet and checkout to [MXNet@(commit fc9e70b)](https://github.com/dmlc/mxnet/tree/fc9e70b) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout fc9e70b
	git submodule update
	```
	or use the latest master directly (code is tested under mxnet 1.2.0).
	
	2.2 Install the MXNet Python binding by
	
	***Note: If you will actively switch between different versions of MXNet, please follow 2.3***
	```
	cd python
	sudo python setup.py install
	```
	2.3 For advanced users, 
	you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)`,        
	and modify `MXNET_VERSION` in `./experiments/deepim/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`.    
	Thus you can switch among different versions of MXNet quickly.

3. Initialize DeepIM:

    3.1 In the root directory of DeepIM, run `sh init.sh` to initialize the DeepIM project.   
    (**Note:** For python3, need to install pytorch first to jit compile flow_c module.)

    3.2 Link (i.e. `ln -sf`) the root folder of data to `./data`.
    
     

## Preparation for Training & Testing


1. Prepare datasets, see `./toolkit/` and [prepare_data.md](./prepare_data.md) for details.
   
   **Note:** `real` corresponds to `observed` in the paper.
   
    The datasets should be put in folder:
	```
	./data/
	```

2. Please download FlowNet model manually from [Google Drive](https://drive.google.com/open?id=1wU1QkG5h5Yyg0G-l9SjoGOIzUgLT4ntR)
 or [Baidu NetDisk](https://pan.baidu.com/s/1qjLw8_gNzcywbWXa2NK9ig) (password: shga), 
and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/flownet-0000.params
	```

## Usage

1. All of our experiment settings (GPU, dataset, etc.) are kept in yaml config files at folder `./experiments/deepim/cfgs`

2. To perform experiments, run the python scripts with the corresponding config file as input. For example, to train and test DeepIM models with pre-trained FlowNet, use the following command
    ```
    python experiments/deepim/deepim_train_test.py --cfg experiments/deepim/deepim/your_cfg.yaml
    ```
    A cache folder would be created automatically to save the model and the log under `output/deepim/`.
3. Please find more details in config files and in our code.

## Misc.

Code has been tested under:

- Ubuntu 14.04/16.04 with 4 GTX 1080Ti GPUs or a single GTX 1070 GPU



