# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import numpy as np
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
import random

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
print(classes)


# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]) # LM
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000


########################
modelnet_root = '/data/wanggu/Downloads/modelnet'   # NB: change to your dir
modelnet40_root = os.path.join(modelnet_root, 'ModelNet40')

sel_classes = classes

def convert_obj():
    for cls_i, cls_name in enumerate(sel_classes):
        if not cls_name  in ['bathtub', 'table', 'monitor', 'sofa', 'night_stand']: # 'car'
            continue
        print(cls_name)
        class_dir = os.path.join(modelnet40_root, cls_name)

        for set in ['train', 'test']:
            model_list = [fn for fn in os.listdir(os.path.join(class_dir, set)) if '.off' in fn]
            model_list.sort()
            model_folder = os.path.join(class_dir, set)
            for model_name in model_list:
                print(set, model_name)
                model_prefix = model_name.split('.')[0]
                if os.path.exists(os.path.join(model_folder, '{}.obj'.format(model_prefix))):
                    continue

                # convert obj model
                cmd = 'cd {0}; meshlabserver -i {0}/{1}.off -o {1}.obj -om vc vn wt -s {2}/modelnet_set_texture.mlx; cd -'.format(
                            model_folder,
                            model_prefix,
                            modelnet_root)
                print(cmd)
                os.system(cmd)

                if not os.path.exists(os.path.join(model_folder, '{}.obj'.format(model_prefix))):
                    cmd = 'cd {0}; meshlabserver -i {0}/{1}.off -o {1}.obj -om vc vn wt -s {2}/modelnet_set_texture_flat.mlx; cd -'.format(
                        model_folder,
                        model_prefix,
                        modelnet_root)
                    print(cmd)
                    os.system(cmd)
                    if not os.path.exists(os.path.join(model_folder, '{}.obj'.format(model_prefix))):
                        print('************************{}/{} Failed to convert obj ***************************'.format(set, model_name))

                # if not os.path.exists(os.path.join(model_folder, '{}.png'.format(model_name))):
                #     print('************************{} Failed to set texture ***************************'.format(model_name))


if __name__ == "__main__":
    # meshlabserver is not stable, may need to run multiple times
    convert_obj()























