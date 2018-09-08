# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division, absolute_import
import os, sys
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))
import numpy as np

from lib.flow_c.gpu_flow import gpu_flow

def gpu_flow_wrapper(device_id):
    def _flow(depth_src, depth_tgt, KT, Kinv):
        return gpu_flow(depth_src, depth_tgt, KT, Kinv, device_id)
    return _flow

if __name__ == '__main__':
    import cv2
    import numpy as np
    from time import time
    from lib.pair_matching import RT_transform

    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    DEPTH_FACTOR = 10000.0
    flow_thresh = 3E-3
    batch_size = 8
    height = 480
    width = 640
    wh_rep = False
    device = 'cuda:0'
    src_img_idx = ['{:06}'.format(x * 100 + 1) for x in range(batch_size)]
    tgt_img_idx = ['{:06}'.format(x * 100 + 31) for x in range(batch_size)]
    class_name = '035_power_drill'  # '002_master_chef_can'
    model_dir = os.path.join(cur_dir, '../../data/LOV/models/{}'.format(class_name))
    pose_path = os.path.join(cur_dir, '../../data/render_v5/data/render_real/%s/0006/{}-pose.txt' % (class_name))
    color_path = os.path.join(cur_dir, '../../data/render_v5/data/render_real/%s/0006/{}-color.png' % (class_name))
    depth_path = os.path.join(cur_dir, '../../data/render_v5/data/render_real/%s/0006/{}-depth.png' % (class_name))

    # prepare input data
    v_depth_imgn = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_depth_real = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_pose_src = np.array([np.loadtxt(pose_path.format(x), skiprows=1) for x in src_img_idx])
    v_pose_tgt = np.array([np.loadtxt(pose_path.format(x), skiprows=1) for x in tgt_img_idx])
    KT_array = np.zeros((batch_size, 3, 4))

    for i in range(batch_size):
        v_depth_imgn[i, 0, :, :] = cv2.imread(depth_path.format(src_img_idx[i]),
                                              cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR
        v_depth_real[i, 0, :, :] = cv2.imread(depth_path.format(tgt_img_idx[i]),
                                              cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR
        se3_m = np.zeros([3, 4])
        se3_rotm, se3_t = RT_transform.calc_se3(v_pose_src[i], v_pose_tgt[i])
        se3_m[:, :3] = se3_rotm
        se3_m[:, 3] = se3_t
        KT_array[i] = np.dot(K, se3_m)

    Kinv = np.linalg.inv(np.matrix(K))
    gpu_flow_machine = gpu_flow_wrapper(0)

    for i in range(10):
        t = time()
        flow, valid = \
            gpu_flow_machine(v_depth_imgn.astype(np.float32),
                             v_depth_real.astype(np.float32),
                             KT_array.astype(np.float32),
                             np.array(Kinv).astype(np.float32))
        print(time() - t)
        print(flow.shape, np.unique(flow))
        print(valid.shape, np.unique(valid))
