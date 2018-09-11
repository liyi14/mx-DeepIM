# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division

import sys, os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '..'))
import numpy as np

class_list = ['{:02d}'.format(i) for i in range(1, 31)]
TLESS_root = os.path.join(cur_dir, '../data/TLESS')
src_model_root = os.path.join(TLESS_root, 't-less_v2/models_reconst')


width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0


def read_rot(path):
    rot = np.loadtxt(path, skiprows=1)
    print('rot: ', rot)
    return rot


def read_trans(path):
    trans = np.loadtxt(path, skiprows=1)
    print('trans: ', trans/100.)
    return trans/100.

def read_transform(path):
    transform = np.loadtxt(path, skiprows=1)[:, 1].reshape((3, 4))
    return transform

def read_points_from_mesh(mesh_path):
    '''
    ply
    :param mesh_path: 
    :return: 
    '''
    with open(mesh_path, 'r') as f:
        i = 0
        points = []
        for line in f:
            i += 1
            if i <= 15:
                continue
            line_list = line.strip('\r\n').split()
            if len(line_list) < 9:
                break
            xyz = [float(m)/1000. for m in line_list[:3]]
            points.append(xyz)
        points_np = np.array(points)
    return points_np

def scale_ply(mesh_path, res_mesh_path, transform=None):
    '''
    ply: mm to m
    :param mesh_path: 
    :return: 
    '''
    f_res = open(res_mesh_path, 'w')
    with open(mesh_path, 'r') as f:
        i = 0
        # points = []
        for line in f:
            line = line.strip('\r\n')
            i += 1
            if i <= 15:
                res_line = line + '\n'
            line_list = line.split()

            if len(line_list) < 9:
                res_line = '{}\n'.format(line)

            if len(line_list) == 9:
                xyz = [float(m)/1000. for m in line_list[:3]]
                if transform is not None:
                    R = transform[:3, :3]
                    T = transform[:3, 3]
                    xyz = np.array(xyz)
                    xyz_new = R.dot(xyz.reshape((3, 1))) + T.reshape((3, 1))
                    xyz = xyz_new.reshape((3,))
                for i in range(3):
                    line_list[i] = '{}'.format(xyz[i])
                res_line = ' '.join(line_list) + '\n'

            # print(res_line)
            f_res.write(res_line)


def scale_ply_main():
    for cls_idx, cls_name in enumerate(class_list):
        print(cls_idx, cls_name)
        # if cls_name != '01':
        #     continue
        mesh_path = os.path.join(src_model_root, 'obj_{}.ply'.format(cls_name))

        if not os.path.exists(mesh_path):
            print("{} not exists!".format(mesh_path))

        res_mesh_path = mesh_path.replace('.ply', '_scaled.ply')
        scale_ply(mesh_path, res_mesh_path)


def check_model_points():
    '''
    R * oldmesh + T = mesh
    :return: 
    '''
    mesh_path = os.path.join(src_model_root, 'obj_01.ply')
    points_mesh = read_points_from_mesh(mesh_path)

    print('points_mesh: ', points_mesh.shape)
    print(points_mesh[:10, :])


# check_model_points()

# =================================


if __name__ == "__main__":
    scale_ply_main()

