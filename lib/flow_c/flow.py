# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function
import sys
import os
import os.path as osp
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))

import numpy as np

from .gpu_flow import gpu_flow


def gpu_flow_wrapper(device_id):

    def _flow(depth_src, depth_tgt, KT, Kinv):
        return gpu_flow(depth_src, depth_tgt, KT, Kinv, device_id)

    return _flow


if __name__ == '__main__':
    import cv2
    from time import time
    import matplotlib.pyplot as plt
    from lib.pair_matching import RT_transform

    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    DEPTH_FACTOR = 1000.0
    flow_thresh = 3E-3
    batch_size = 8
    height = 480
    width = 640
    wh_rep = False
    device = 'cuda:0'
    src_img_idx = ['{:06}'.format(x*100 + 1) for x in range(batch_size)]
    tgt_img_idx = ['{:06}'.format(x*100 + 31) for x in range(batch_size)]
    class_name = 'driller'
    cls_idx = 8
    data_dir = osp.join(cur_dir, '../../data/LINEMOD_6D/LM6d_converted/LM6d_refine/data/')
    model_dir = osp.join(cur_dir, '../../data/LINEMOD_6D/models/{}'.format(class_name))
    pose_path = osp.join(data_dir, 'gt_observed/%s/{}-pose.txt' % (class_name))
    color_path = osp.join(data_dir, 'gt_observed/%s/{}-color.png' % (class_name))
    depth_path = osp.join(data_dir, 'gt_observed/%s/{}-depth.png' % (class_name))

    # prepare input data
    v_depth_src = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_depth_tgt = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_pose_src = np.array([np.loadtxt(pose_path.format(x), skiprows=1) for x in src_img_idx])
    v_pose_tgt = np.array([np.loadtxt(pose_path.format(x), skiprows=1) for x in tgt_img_idx])
    KT_array = np.zeros((batch_size, 3, 4))

    for i in range(batch_size):
        depth_src_path = depth_path.format(src_img_idx[i])
        depth_tgt_path = depth_path.format(tgt_img_idx[i])
        assert osp.exists(depth_src_path), 'no {}'.format(depth_src_path)
        assert osp.exists(depth_tgt_path), 'no {}'.format(depth_tgt_path)
        v_depth_src[i, 0, :, :] = cv2.imread(depth_src_path, cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR
        v_depth_tgt[i, 0, :, :] = cv2.imread(depth_tgt_path, cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR
        se3_m = np.zeros([3, 4])
        se3_rotm, se3_t = RT_transform.calc_se3(v_pose_src[i], v_pose_tgt[i])
        se3_m[:, :3] = se3_rotm
        se3_m[:, 3] = se3_t
        KT_array[i] = np.dot(K, se3_m)

    Kinv = np.linalg.inv(np.matrix(K))
    gpu_flow_machine = gpu_flow_wrapper(0)

    for i in range(10):
        tic = time()
        flow_all, flow_weights_all = \
            gpu_flow_machine(v_depth_src.astype(np.float32),
                             v_depth_tgt.astype(np.float32),
                             KT_array.astype(np.float32),
                             np.array(Kinv).astype(np.float32))
        print('{} s'.format(time() - tic))
        print(flow_all.shape, np.unique(flow_all))
        print(flow_weights_all.shape, np.unique(flow_weights_all))

        for j in range(batch_size):
            img_src_path = color_path.format(src_img_idx[j])
            assert osp.exists(img_src_path), 'no {}'.format(img_src_path)
            img_tgt_path = color_path.format(tgt_img_idx[j])
            assert osp.exists(img_tgt_path)
            img_src = cv2.imread(img_src_path, cv2.IMREAD_COLOR)
            img_src = img_src[:, :, [2, 1, 0]]
            img_tgt = cv2.imread(img_tgt_path, cv2.IMREAD_COLOR)
            img_tgt = img_tgt[:, :, [2, 1, 0]]

            flow_img_tgt = np.zeros((height, width, 3), np.uint8)
            flow_img_src = np.zeros((height, width, 3), np.uint8)

            print('flow_all(unique): \n', np.unique(flow_all))
            flow = np.squeeze(flow_all[j, :, :, :].transpose((1, 2, 0)))
            flow_weights = np.squeeze(flow_weights_all[j, 0, :, :])

            for h in range(height):
                for w in range(width):
                    if flow_weights[h, w]:
                        cur_flow = flow[h, w, :]
                        flow_img_src = cv2.line(flow_img_src, (np.round(w).astype(int), np.round(h).astype(int)),
                                                (np.round(w).astype(int), np.round(h).astype(int)),
                                                (255, h * 255 / height, w * 255 / width), 5)
                        flow_img_tgt = cv2.line(
                            flow_img_tgt, (np.round(w + cur_flow[1]).astype(int), np.round(h + cur_flow[0]).astype(int)),
                            (np.round(w + cur_flow[1]).astype(int), np.round(h + cur_flow[0]).astype(int)),
                            (255, h * 255 / height, w * 255 / width), 5)

            depth_src = cv2.imread(depth_path.format(src_img_idx[j]), cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR

            fig = plt.figure()
            fig.add_subplot(2, 3, 1)
            plt.imshow(img_src)
            plt.axis('off')
            plt.title('img_src')

            fig.add_subplot(2, 3, 2)
            plt.imshow(img_tgt)
            plt.axis('off')
            plt.title('img_tgt')

            fig.add_subplot(2, 3, 3)
            plt.imshow(depth_src)
            plt.axis('off')
            plt.title('depth_src')

            fig.add_subplot(2, 3, 4)
            plt.imshow(flow_img_src)
            plt.title('flow_img_src')
            plt.axis('off')

            fig.add_subplot(2, 3, 5)
            plt.imshow(flow_img_tgt)
            plt.title('flow_img_tgt')
            plt.axis('off')

            fig.add_subplot(2, 3, 6)
            plt.imshow(flow_img_src - flow_img_tgt)
            plt.title('flow_img_src - flow_img_tgt')
            plt.axis('off')
            plt.show()
