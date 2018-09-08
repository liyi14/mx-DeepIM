# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import cv2

def stat_depth(pairdb, config):
    '''
    stat max and min value of depth in a pairdb
    :param pairdb:
    :param config:
    :return:
    '''
    all_depth = [x['depth_rendered'] for x in pairdb]
    all_depth = list(set(all_depth))
    max_val = -1
    min_val = 9999999999
    for idx, f in enumerate(all_depth):
        depth_rendered = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32)
        x = np.max(depth_rendered)
        y = np.min(depth_rendered)
        max_val = x if x>max_val else max_val
        min_val = y if y<min_val else min_val
    print('max of depth value is {}, min of depth value is {}'.format(max_val, min_val))

    return max_val, min_val
