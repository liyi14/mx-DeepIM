# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang
# --------------------------------------------------------
'''
For more precise evaluation, use the diameters in models_info.txt
'''
from __future__ import print_function, division
import os, sys
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))

version = 'v1'
model_root = os.path.join(cur_dir, '../../data/LINEMOD_6D/LM6d_converted/models/LM6d_render_{}/models'.format(version))
print ("target path: {}".format(model_root))


idx2class = {1: 'ape',
            2: 'benchvise',
            3: 'bowl',
            4: 'camera',
            5: 'can',
            6: 'cat',
            7: 'cup',
            8: 'driller',
            9: 'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            13: 'iron',
            14: 'lamp',
            15: 'phone'
}
classes = idx2class.values()
classes = sorted(classes)

def class2idx(class_name, idx2class=idx2class):
    for k,v in idx2class.items():
        if v == class_name:
            return k

def load_object_points():
    points = {}

    for cls_idx, cls_name in idx2class.items():
        point_file = os.path.join(model_root, cls_name, 'points.xyz')
        assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
        points[cls_name] = np.loadtxt(point_file)

    return points

def write_extents():
    points_dict = load_object_points()
    extents = np.zeros((len(classes), 3))
    for i, cls_name in enumerate(classes):
        extents[i, :] = 2 * np.max(np.abs(points_dict[cls_name]), 0)
    # print(extents)
    extent_file = os.path.join(model_root, 'extents.txt')
    # with open(extent_file, 'w') as f:
    np.savetxt(extent_file, extents, fmt="%.6f", delimiter=' ')

if __name__ == "__main__":
    write_extents()
    print("{} finished".format(__file__))
