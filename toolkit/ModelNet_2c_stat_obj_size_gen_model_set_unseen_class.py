# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import numpy as np
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
from lib.pair_matching.RT_transform import *
from lib.utils.mkdir_if_missing import mkdir_if_missing
import random
from lib.render_glumpy.render_py_light_modelnet_multi import Render_Py_Light_ModelNet_Multi

random.seed(2333)
np.random.seed(1234)

classes = ['airplane', 'bed', 'bench',  'car', 'chair', 'toilet', 'sink',
            'stool', 'piano',
           # 'laptop',

             'range_hood',  #'stairs',
             'tv_stand',
            'bookshelf',
             'mantel',

           'door', 'glass_box',
           'guitar',
           'wardrobe', # test
           # 'plant',
           'xbox',

           'bathtub', 'table', 'monitor', 'sofa', 'night_stand']
train_classes = ['airplane', 'bed', 'bench',  'car', 'chair', 'toilet', 'sink',
            'stool', 'piano']
test_classes = ['range_hood',  #'stairs',
             'tv_stand',
            'bookshelf',
             'mantel',
               'door', 'glass_box',
               'guitar',
               'wardrobe', # test
               # 'plant',
               'xbox',

           'bathtub', 'table', 'monitor', 'sofa', 'night_stand']
print(classes)


# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]) # LM
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000


########################
modelnet_root = '/data/wanggu/Downloads/modelnet' # NB: change to your dir
modelnet40_root = os.path.join(modelnet_root, 'ModelNet40')
image_set_dir = os.path.join(modelnet_root, 'modelnet_render_v1/image_set')

model_set_dir = os.path.join(modelnet_root, 'model_set')
mkdir_if_missing(model_set_dir)


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        size_in_MB = file_info.st_size / (1024. * 1024.)
        return size_in_MB
        # return convert_bytes(file_info.st_size)


# sel_classes = train_classes
sel_classes = test_classes
num_models = 30

def stat_obj_size():
    train_cls_models = {}
    test_cls_models = {}
    for cls_i, cls_name in enumerate(sel_classes):
        # if not cls_name in ['door', 'glass_box', 'wardrobe', 'plant', 'xbox']: # 'car'
        #     continue
        print(cls_name)
        class_dir = os.path.join(modelnet40_root, cls_name)


        train_model_path_dict = {}
        test_model_path_dict = {}
        for set in ['train', 'test']:
            if set == 'test':
                model_list = [fn.replace('my_val_', '').replace('.txt', '')
                              for fn in os.listdir(os.path.join(image_set_dir)) if 'my_val_{}'.format(cls_name) in fn]


            if set == 'test':
                n_model_list = ['{}/{}/{}'.format(cls_name, set, model_name) for model_name in model_list]
                if len(n_model_list) > 50:
                    test_cls_models[cls_name] = n_model_list[:50]
                else:
                    test_cls_models[cls_name] = n_model_list


        with open(os.path.join(model_set_dir, 'unseen_test_{}.txt'.format(cls_name)), 'w') as f:
            for model_name in test_cls_models[cls_name]:
                f.write(model_name + '\n')


        def check_render():
            texture_path = os.path.join(modelnet_root, 'gray_texture.png')
            # init render machines
            brightness_ratios = [0.7]  ###################
            render_machine = Render_Py_Light_ModelNet_Multi(train_model_path_list, texture_path, K, width, height, ZNEAR, ZFAR,
                                                      brightness_ratios)

            pose = np.zeros((3, 4))
            rot_q = np.random.normal(0, 1, 4)
            rot_q = rot_q / np.linalg.norm(rot_q)
            pose[:3, :3] = quat2mat(rot_q)
            pose[:3, 3] = np.array([0, 0, 0.5])

            idx = 2  # random.randint(0, 100)

            # generate random light_position
            if idx % 6 == 0:
                light_position = [1, 0, 1]
            elif idx % 6 == 1:
                light_position = [1, 1, 1]
            elif idx % 6 == 2:
                light_position = [0, 1, 1]
            elif idx % 6 == 3:
                light_position = [-1, 1, 1]
            elif idx % 6 == 4:
                light_position = [-1, 0, 1]
            elif idx % 6 == 5:
                light_position = [0, 0, 1]
            else:
                raise Exception("???")
            # print("light_position a: {}".format(light_position))
            light_position = np.array(light_position) * 0.5
            # inverse yz
            light_position[0] += pose[0, 3]
            light_position[1] -= pose[1, 3]
            light_position[2] -= pose[2, 3]
            # print("light_position b: {}".format(light_position))

            colors = np.array([1, 1, 1])  # white light
            intensity = np.random.uniform(0.9, 1.1, size=(3,))
            colors_randk = random.randint(0, colors.shape[0] - 1)
            light_intensity = colors[colors_randk] * intensity
            # print('light intensity: ', light_intensity)

            # randomly choose a render machine
            rm_randk = random.randint(0, len(brightness_ratios) - 1)
            # print('brightness ratio:', brightness_ratios[rm_randk])
            # get render result
            rgb_gl, depth_gl = render_machine.render(0,
                                                     mat2quat(pose[:3, :3]), pose[:, -1],
                                                     light_position,
                                                     light_intensity,
                                                     brightness_k=rm_randk)
            rgb_gl = rgb_gl.astype('uint8')



def load_points_from_obj():
    from glumpy import app, gl, gloo, glm, data, log
    model_path = os.path.join(cur_dir, '../data/ModelNet/ModelNet40/airplane/train/airplane_0005.obj')
    vertices, indices = data.objload("{}"
                                     .format(model_path), rescale=True)
    vertices['position'] = vertices['position'] / 10.

    points = np.array(vertices['position'])
    print(type(points))
    print(points.shape, points.min(0), points.max(0), points.max(0) - points.min(0))


if __name__ == "__main__":
    stat_obj_size()
    # load_points_from_obj()
























