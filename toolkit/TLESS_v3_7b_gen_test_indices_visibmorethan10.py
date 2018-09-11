# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
remove test instance that the visible surface is lower than 10%
'''
from __future__ import print_function, division

import sys, os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '..'))
import numpy as np
import yaml

# =================== global settings ======================
video_list = ['{:02d}'.format(i) for i in range(1, 21)]
sel_videos = ['02']

class_list = ['{:02d}'.format(i) for i in range(1, 31)]
sel_classes = ['05', '06']


TLESS_root = os.path.join(cur_dir, '../data/TLESS')



width = 640
height = 480


K_0 = np.array([[1075.65091572, 0, 320.], [0, 1073.90347929, 240.], [0, 0, 1]]) # Primesense

# in test set, K is different for each image
ZNEAR = 0.25
ZFAR = 6.0

DEPTH_FACTOR = 10000


test_data_root = os.path.join(TLESS_root, 'TLESS_render_v3/data/real')
image_set_dir = os.path.join(TLESS_root, 'TLESS_render_v3/image_set')

def load_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt_dict = yaml.load(f)
    return gt_dict


def main():
    test_indices_dict = {video_name:{cls_name: [] for cls_name in sel_classes} for video_name in sel_videos}
    for video_name in sel_videos: # 02
        print('video name: {}'.format(video_name))
        gt_path = os.path.join(TLESS_root, 't-less_v2/test_primesense/{}/gt.yml'.format(video_name))
        gt_dict = load_gt(gt_path)

        test_stat_path = os.path.join(TLESS_root, 'test_primesense_gt_stats/{}_delta=15.yml'.format(video_name))
        with open(test_stat_path, 'r') as f:
            test_stat = yaml.load(f)
        for img_id in gt_dict.keys():
            for idx, instance in enumerate(gt_dict[img_id]):
                obj_id = instance['obj_id']
                cls_name = '{:02d}'.format(obj_id)
                if cls_name in sel_classes:
                    visib_fract = test_stat[img_id][idx]['visib_fract']
                    if visib_fract > 0.1:
                        test_indices_dict[video_name][cls_name].append(img_id)

    for cls_idx, cls_name in enumerate(class_list):
        if not cls_name in sel_classes:
            continue
        print(cls_idx, cls_name)
        for video_name in sel_videos: # 02
            print('video name: {}'.format(video_name))
            test_set_file = os.path.join(image_set_dir, 'my_val_v1_video{}_{}.txt'.format(video_name, cls_name))
            new_test_set_file = os.path.join(image_set_dir, 'my_val_v1_ND10_video{}_{}.txt'.format(video_name, cls_name))
            test_indices = []
            with open(test_set_file, 'r') as f:
                lines = [line.strip('\r\n') for line in f.readlines()]

            for line in lines:
                real_idx, rendered_idx = line.split()
                img_id = int(real_idx.split('/')[-1].split('_')[0])
                if img_id in test_indices_dict[video_name][cls_name]:
                    test_indices.append(line)

            with open(new_test_set_file, 'w') as f:
                for line in test_indices:
                    f.write(line + '\n')



if __name__ == "__main__":
    main()