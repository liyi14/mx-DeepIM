# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division

import os
import sys
cur_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(cur_dir, '../'))
import numpy as np
import numpy.linalg as LA
from glumpy import app, gl, gloo, glm, data, log
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
log.setLevel(logging.WARNING)  # ERROR, WARNING, DEBUG, INFO
import cv2
from lib.pair_matching.RT_transform import *
from lib.render_glumpy.render_py import Render_Py
from lib.utils.projection import se3_inverse, se3_mul
# from lib.py_pcl import pcl_icp

width = 640
height = 480
zNear = 0.25
zFar = 6.0
DEPTH_FACTOR = 1000
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]) # LM

def load_object_points(point_path):
    assert os.path.exists(point_path), 'Path does not exist: {}'.format(point_path)
    points = np.loadtxt(point_path)
    return points

def points_to_2D(points, R, T, K):
    """
    :param points: (N, 3) 
    :param R: (3, 3)
    :param T: (3, )
    :param K: (3, 3)
    :return: 
    """
    points_in_world = np.matmul(R, points.T) + T.reshape((3, 1)) # (3, N)
    points_in_camera = np.matmul(K, points_in_world) # (3, N)
    N = points_in_world.shape[1]
    points_2D = np.zeros((2, N))
    points_2D[0, :] = points_in_camera[0, :] / points_in_world[2, :]
    points_2D[1, :] = points_in_camera[1, :] / points_in_world[2, :]
    z = points_in_world[2, :]
    return points_2D, z

def angle_axis_to_quat(angle, rot_axis):
    angle = angle % (2 * np.pi)
    # print(angle)
    q = np.zeros(4)
    q[0] = np.cos(0.5 * angle)
    q[1:] = np.sin(0.5 * angle) * rot_axis
    if q[0] < 0:
        q *= -1
    # print('norm of q: ', LA.norm(q))
    q = q / LA.norm(q)
    # print('norm of q: ', LA.norm(q))
    return q


if __name__ == "__main__":
    cls_name = 'bowl'
    model_folder = './data/LINEMOD_6D/models/{}'.format(cls_name)
    print('init render machine...')
    render_machine = Render_Py(model_folder, K, width, height, zNear, zFar)

    def rotate(angle, rot_axis, pose_gt, p_center=np.array([0,0,0])):
        rot_sym_q = angle_axis_to_quat(angle, rot_axis)
        rot_sym_m = quat2mat(rot_sym_q)

        # print(rot_sym_m)
        rot_res = R_transform(pose_gt[:3, :3], rot_sym_m, rot_coord='model')
        rot_res_q = mat2quat(rot_res)

        rgb_gl, depth_gl = render_machine.render(rot_res_q, pose_gt[:, 3] + p_center)
        rgb_gl = rgb_gl.astype('uint8')
        return rgb_gl, depth_gl

    pose_gt = np.loadtxt('./data/LM6d_render_v1/data/render_real/bowl/000001-pose.txt',
                         skiprows=1)

    im_c = cv2.imread('./data/LM6d_render_v1/data/render_real/bowl/000001-color.png')
    depth_c = cv2.imread('./data/LM6d_render_v1/data/render_real/bowl/000001-depth.png',
                         cv2.IMREAD_UNCHANGED) / float(DEPTH_FACTOR)

    im_render_gt, _ = render_machine.render(mat2quat(pose_gt[:3, :3]), pose_gt[:, 3])

    # symmetry axis
    angle = np.pi
    p_center = np.array([0, 0, 0])
    rot_axis = np.array([0, 0, -1])

    im_rot, depth_rot = rotate(angle, rot_axis, pose_gt,  p_center=p_center)
    #
    im_rot_1, depth_rot_1 = rotate(angle=np.pi * 0.5, rot_axis=rot_axis, pose_gt=pose_gt, p_center=p_center)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.subplot(2, 3, 1)
    plt.imshow(im_c[:,:,[2, 1, 0]])
    plt.title('im_render_real')

    plt.subplot(2, 3, 2)
    plt.imshow(im_rot[:,:,[2, 1, 0]])
    plt.title('rotated image')

    plt.subplot(2, 3, 3)
    im_diff = np.abs(im_c - im_rot)
    plt.imshow(im_diff[:,:,[2, 1, 0]])
    plt.title('im diff')

    plt.subplot(2,3,4)
    plt.imshow(im_rot_1[:, :, [2,1,0]])
    plt.title('rotated image 1')

    plt.subplot(2,3,5)
    im_diff_gt = np.abs(im_c - im_render_gt)
    plt.imshow(im_diff_gt[:,:,[2,1,0]])
    plt.title('diff of im_render_gt')

    plt.show()















