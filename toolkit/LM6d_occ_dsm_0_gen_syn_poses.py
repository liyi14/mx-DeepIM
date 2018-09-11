# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
'''
use YCB trans to make multiple objects on the same plane as much as possible
'''
from __future__ import print_function, division
import numpy as np
import scipy.io as sio
import os, sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))
from lib.pair_matching.RT_transform import *
from lib.utils.mkdir_if_missing import mkdir_if_missing
from six.moves import cPickle
import copy
from tqdm import tqdm
import random
random.seed(1234)
np.random.seed(2345)


data_dir = os.path.join(cur_path, '..', 'data', 'render_v5', 'data', 'real')

render_real_dir = os.path.join(cur_path, '..', 'data', 'render_v5', 'data', 'render_real')
# ape(ape, test)

# classes = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
# sel_classes = classes

K_lm = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
K_lov = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
# version = 'v1'
# image_set = 'all'


pose_dict = {} #  {real_prefix: {class: euler angles, translations}}
euler_stat = {}
trans_stat = {}

real_indices = []
videos = ["{:04d}".format(i) for i in range(48)]
for video in videos:
    filelist = [fn for fn in os.listdir(os.path.join(data_dir, video)) if 'color' in fn]
    for idx, fn in enumerate(filelist):
        if idx % 4 == 0:
            real_indices.append("{}/{}".format(video, fn.split('-')[0]))


#print(real_indices)
print(len(real_indices))

LOV_classes = ['__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box',
                       '005_tomato_soup_can', '006_mustard_bottle', \
                       '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                       '019_pitcher_base', \
                       '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                       '037_scissors', '040_large_marker', \
                       '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']


idx2class = {1: 'ape',
            # 2: 'benchviseblue',
            # 4: 'camera',
            5: 'can',
            6: 'cat',
            8: 'driller',
            9: 'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            # 13: 'iron',
            # 14: 'lamp',
            # 15: 'phone'
}


def class2idx(class_name, idx2class=idx2class):
    for k,v in idx2class.items():
        if v == class_name:
            return k

classes = idx2class.values()
classes.sort()

def get_poses_from_meta(meta_path): 
    meta_data = sio.loadmat(meta_path)
    cls_idxs = meta_data['cls_indexes']
    poses = []
    for class_idx in cls_idxs:
        inner_id = np.where(np.squeeze(meta_data['cls_indexes']) == class_idx)
    	if len(meta_data['poses'].shape) == 2:
            pose = meta_data['poses']
        else:
            pose = np.squeeze(meta_data['poses'][:, :, inner_id])
        poses.append(pose)
    return poses


def stat_lov():
    res_dir = os.path.join(cur_path, '..', 'data', 'LINEMOD_6D', 'pose_stat_v2')
    mkdir_if_missing(res_dir)
    if os.path.exists(os.path.join(res_dir, 'trans_from_LOV.pkl')):
        trans_dict = cPickle.load(open(os.path.join(res_dir, 'trans_from_LOV.pkl'), 'rb'))
    else:
        pose_dict = {}
        trans_list = []
        trans_lm_list = []
        trans_dict = {}
        for j, real_idx in enumerate(tqdm(real_indices)):
            meta_path = os.path.join(data_dir, real_idx + '-meta.mat')
            poses = get_poses_from_meta(meta_path)
            tmp_pose = np.zeros((len(poses), 6), dtype='float32')
            tmp_trans = np.zeros((len(poses), 3), dtype='float32')
            for i, pose in enumerate(poses):
                rot_euler = mat2euler(pose[:3, :3])
                trans = pose[:3, 3]
                tmp_pose[i, :3] = rot_euler
                tmp_pose[i, 3:] = trans
                trans_list.append(trans)

                trans_lm = np.dot(np.dot(np.linalg.inv(K_lm), K_lov), trans.reshape((3, 1)))
                trans_lm = trans_lm.reshape((3,))
                trans_lm_list.append(trans_lm)
                tmp_trans[i, :] = trans_lm
            pose_dict["{:06d}".format(j)] = tmp_pose
            trans_dict["{:06d}".format(j)] = tmp_trans

        trans_array = np.array(trans_list)
        trans_mean = np.mean(trans_array, 0)
        trans_std = np.std(trans_array, 0)
        print('trans, ', 'mean: ', trans_mean, 'std: ', trans_std)

        trans_lm_array = np.array(trans_lm_list)
        trans_lm_mean = np.mean(trans_lm_array, 0)
        trans_lm_std = np.std(trans_lm_array, 0)
        print('trans lm, ', 'mean: ', trans_lm_mean, 'std: ', trans_lm_std)

        print(len(pose_dict))


        # cPickle.dump(pose_dict, open(os.path.join(res_dir, 'LOV_pose_dict.pkl'), 'wb'), 2)
        # {prefix: array(num_posex7)}, num_pose is uncertain

        cPickle.dump(trans_dict, open(os.path.join(res_dir, 'trans_from_LOV.pkl'), 'wb'), 2)

    return trans_dict

## stat olm pose ------------------------------------------------------------

def angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))  # -> cosine of the angle
    rad = np.arccos(np.clip(c, -1, 1))
    deg = rad / np.pi * 180
    return deg

def stat_lm6d():
    real_set_dir = os.path.join(cur_path, '..', 'data', 'LM6d_render_v1', 'image_set/real')
    render_real_dir = os.path.join(cur_path, '../data/LM6d_render_v1/data/render_real')
    # real_set_dir = os.path.join(cur_path, '..', 'data', 'LM6d_occ_ds_multi', 'image_set/real')
    # render_real_dir = os.path.join(cur_path, '../data/LM6d_occ_ds_multi/data/render_real')

    pz = np.array([0, 0, 1])
    new_points = {}
    pose_dict = {}
    trans_stat = {}
    quat_stat = {}
    for cls_idx, cls_name in enumerate(classes):
        if cls_name != 'eggbox':
            continue
        new_points[cls_name] = {'pz': []}
        train_idx_file = os.path.join(real_set_dir, "{}_train.txt".format(cls_name))
        # train_idx_file = os.path.join(real_set_dir, "NDtrain_real_{}.txt".format(cls_name))
        with open(train_idx_file, 'r') as f:
            real_indices = [line.strip() for line in f.readlines()]

        num_real = len(real_indices)
        pose_dict[cls_name] = np.zeros((num_real, 7))  # quat, translation
        trans_stat[cls_name] = {}
        quat_stat[cls_name] = {}
        for real_i, real_idx in enumerate(tqdm(real_indices)):
            prefix = real_idx.split('/')[1]
            pose_path = os.path.join(render_real_dir, cls_name, '{}-pose.txt'.format(prefix))
            assert os.path.exists(pose_path), 'path {} not exists'.format(pose_path)
            pose = np.loadtxt(pose_path, skiprows=1)
            rot = pose[:3, :3]
            # print(rot)
            quat = np.squeeze(mat2quat(rot))
            src_trans = pose[:3, 3]
            pose_dict[cls_name][real_i, :4] = quat
            pose_dict[cls_name][real_i, 4:] = src_trans

            new_pz = np.dot(rot, pz.reshape((-1, 1))).reshape((3,))
            new_points[cls_name]['pz'].append(new_pz)
        new_points[cls_name]['pz'] = np.array(new_points[cls_name]['pz'])
        new_points[cls_name]['pz_mean'] = np.mean(new_points[cls_name]['pz'], 0)
        new_points[cls_name]['pz_std'] = np.std(new_points[cls_name]['pz'], 0)

        trans_mean = np.mean(pose_dict[cls_name][:, 4:], 0)
        trans_std = np.std(pose_dict[cls_name][:, 4:], 0)
        trans_max = np.max(pose_dict[cls_name][:, 4:], 0)
        print('trans mean: {}'.format(trans_mean))
        print('trans std: {}'.format(trans_std))
        print('trans max: {}'.format(trans_max))

        trans_stat[cls_name]['trans_mean'] = trans_mean
        trans_stat[cls_name]['trans_std'] = trans_std

        quat_mean = np.mean(pose_dict[cls_name][:, :4], 0)
        quat_std = np.std(pose_dict[cls_name][:, :4], 0)
        quat_stat[cls_name]['quat_mean'] = quat_mean
        quat_stat[cls_name]['quat_std'] = quat_std

        print('new z: ', 'mean: ', new_points[cls_name]['pz_mean'], 'std: ', new_points[cls_name]['pz_std'])

        new_points[cls_name]['angle'] = [] # angle between mean vector and points
        pz_mean = new_points[cls_name]['pz_mean']
        for p_i in range(new_points[cls_name]['pz'].shape[0]):
            deg = angle(pz_mean, new_points[cls_name]['pz'][p_i, :])
            new_points[cls_name]['angle'].append(deg)
        new_points[cls_name]['angle'] = np.array(new_points[cls_name]['angle'])

        print('angle mean: ', np.mean(new_points[cls_name]['angle']),
              'angle std: ', np.std(new_points[cls_name]['angle']),
              'angle max: ', np.max(new_points[cls_name]['angle']))
        new_points[cls_name]['angle_max'] = np.max(new_points[cls_name]['angle']) ###############
        print()


        def vis_points():
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            sel_p = 'pz'
            ax = plt.figure().add_subplot(111, projection='3d')
            ax.scatter(new_points[cls_name][sel_p][:,0], new_points[cls_name][sel_p][:,1], new_points[cls_name][sel_p][:,2],
                       c='r', marker='^')
            ax.scatter(0,0,0,c='b',marker='o')
            ax.scatter(0, 0, 1, c='b', marker='o')
            ax.scatter(0, 1, 0, c='b', marker='o')
            ax.scatter(1, 0, 0, c='b', marker='o')
            ax.quiver(0,0,0, 0,0,1)
            pz_mean = new_points[cls_name]['pz_mean']
            ax.quiver(0, 0, 0, pz_mean[0], pz_mean[1], pz_mean[2])

            ax.scatter(pz_mean[0], pz_mean[1], pz_mean[2], c='b', marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            plt.title(cls_name+'-'+sel_p)
            plt.show()
        vis_points()
    return pose_dict, quat_stat, trans_stat, new_points


def stat_lm6d_occ_test():
    real_set_dir = os.path.join(cur_path, '..', 'data', 'LM6d_render_v1', 'image_set/real')
    render_real_dir = os.path.join(cur_path, '../data/LM6d_render_v1/data/render_real/occ_test')

    pz = np.array([0, 0, 1])
    new_points = {}
    pose_dict = {}
    trans_stat = {}
    quat_stat = {}
    for cls_idx, cls_name in enumerate(classes):
        if cls_name != 'eggbox':
            continue
        new_points[cls_name] = {'pz': []}
        test_idx_file = os.path.join(real_set_dir, "occLM_val_real_{}.txt".format(cls_name))
        with open(test_idx_file, 'r') as f:
            real_indices = [line.strip() for line in f.readlines()]

        num_real = len(real_indices)
        pose_dict[cls_name] = np.zeros((num_real, 7))  # quat, translation
        trans_stat[cls_name] = {}
        quat_stat[cls_name] = {}
        for real_i, real_idx in enumerate(tqdm(real_indices)):
            prefix = real_idx.split('/')[1]
            pose_path = os.path.join(render_real_dir, cls_name, '{}-pose.txt'.format(prefix))
            assert os.path.exists(pose_path), 'path {} not exists'.format(pose_path)
            pose = np.loadtxt(pose_path, skiprows=1)
            rot = pose[:3, :3]
            # print(rot)
            quat = np.squeeze(mat2quat(rot))
            src_trans = pose[:3, 3]
            pose_dict[cls_name][real_i, :4] = quat
            pose_dict[cls_name][real_i, 4:] = src_trans

            new_pz = np.dot(rot, pz.reshape((-1, 1))).reshape((3,))
            new_points[cls_name]['pz'].append(new_pz)
        new_points[cls_name]['pz'] = np.array(new_points[cls_name]['pz'])
        new_points[cls_name]['pz_mean'] = np.mean(new_points[cls_name]['pz'], 0)
        new_points[cls_name]['pz_std'] = np.std(new_points[cls_name]['pz'], 0)

        trans_mean = np.mean(pose_dict[cls_name][:, 4:], 0)
        trans_std = np.std(pose_dict[cls_name][:, 4:], 0)
        trans_max = np.max(pose_dict[cls_name][:, 4:], 0)
        print('trans mean: {}'.format(trans_mean))
        print('trans std: {}'.format(trans_std))
        print('trans max: {}'.format(trans_max))

        trans_stat[cls_name]['trans_mean'] = trans_mean
        trans_stat[cls_name]['trans_std'] = trans_std

        quat_mean = np.mean(pose_dict[cls_name][:, :4], 0)
        quat_std = np.std(pose_dict[cls_name][:, :4], 0)
        quat_stat[cls_name]['quat_mean'] = quat_mean
        quat_stat[cls_name]['quat_std'] = quat_std

        print('new z: ', 'mean: ', new_points[cls_name]['pz_mean'], 'std: ', new_points[cls_name]['pz_std'])

        new_points[cls_name]['angle'] = [] # angle between mean vector and points
        pz_mean = new_points[cls_name]['pz_mean']
        for p_i in range(new_points[cls_name]['pz'].shape[0]):
            deg = angle(pz_mean, new_points[cls_name]['pz'][p_i, :])
            new_points[cls_name]['angle'].append(deg)
        new_points[cls_name]['angle'] = np.array(new_points[cls_name]['angle'])

        print('angle mean: ', np.mean(new_points[cls_name]['angle']),
              'angle std: ', np.std(new_points[cls_name]['angle']),
              'angle max: ', np.max(new_points[cls_name]['angle']))
        new_points[cls_name]['angle_max'] = np.max(new_points[cls_name]['angle']) ###############
        print()


        def vis_points():
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            sel_p = 'pz'
            ax = plt.figure().add_subplot(111, projection='3d')
            ax.scatter(new_points[cls_name][sel_p][:,0], new_points[cls_name][sel_p][:,1], new_points[cls_name][sel_p][:,2],
                       c='r', marker='^')
            ax.scatter(0,0,0,c='b',marker='o')
            ax.scatter(0, 0, 1, c='b', marker='o')
            ax.scatter(0, 1, 0, c='b', marker='o')
            ax.scatter(1, 0, 0, c='b', marker='o')
            ax.quiver(0,0,0, 0,0,1)
            pz_mean = new_points[cls_name]['pz_mean']
            ax.quiver(0, 0, 0, pz_mean[0], pz_mean[1], pz_mean[2])

            ax.scatter(pz_mean[0], pz_mean[1], pz_mean[2], c='b', marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            plt.title('occ_test-'+cls_name+'-'+sel_p)
            plt.show()
        vis_points()
    return pose_dict, quat_stat, trans_stat, new_points



def gen_poses():
    NUM_IMAGES = 20000
    pz = np.array([0,0,1])
    pose_dict, quat_stat, trans_stat, new_points = stat_lm6d()
    trans_lm_dict = stat_lov()

    real_prefix_list = ['{:06d}'.format(i + 1) for i in range(NUM_IMAGES)]
    sel_classes = copy.deepcopy(classes)
    num_class = len(sel_classes)
    syn_pose_dict = {} # prefix: {} for prefix in real_prefix_list}  # store poses

    syn_pose_dir = os.path.join(cur_path, '..', 'data', 'LM6d_occ_render_v1', 'syn_poses_multi')
    mkdir_if_missing(syn_pose_dir)

    for i in tqdm(range(NUM_IMAGES)):
        real_prefix = real_prefix_list[i]


        # randomly choose a set of transes
        rand_k = random.randint(0, len(trans_lm_dict.keys()) - 1)
        sel_transes = trans_lm_dict[trans_lm_dict.keys()[rand_k]]
        num_pose = sel_transes.shape[0]
        if num_pose < 3:
            continue

        syn_pose_dict[real_prefix] = {}
        random.shuffle(sel_classes)
        gen_classes = sel_classes[:num_pose]
        for cls_i, cls_name in enumerate(gen_classes):
            # if cls_name != 'driller':
            #     continue
            src_quat_mean = quat_stat[cls_name]['quat_mean']
            src_quat_std = quat_stat[cls_name]['quat_std']
            src_trans_mean = trans_stat[cls_name]['trans_mean']
            src_trans_std = trans_stat[cls_name]['trans_std']
            deg_max = new_points[cls_name]['angle_max'] + 10

            gen_this_pose = True
            # generate trans ------------------------------------------------
            tgt_trans = sel_transes[cls_i].copy()
            # print('tgt_trans: ', tgt_trans)
            tgt_trans += np.random.normal(0, 0.05, 1)

            # r_dist, t_dist = calc_rt_dist_q(tgt_quat, src_quat, tgt_trans, src_trans)
            transform = np.matmul(K_lm, tgt_trans.reshape(3, 1))
            center_x = float(transform[0] / transform[2])
            center_y = float(transform[1] / transform[2])
            count = 0
            while (not (0.1 < tgt_trans[2] < 1.2)
                   or not (48 < center_x < (640 - 48) and 48 < center_y < (480 - 48))):
                # randomly generate a pose
                tgt_trans = sel_transes[cls_i].copy()
                tgt_trans += np.random.normal(0, 0.05, 1)

                transform = np.matmul(K_lm, tgt_trans.reshape(3, 1))
                center_x = float(transform[0] / transform[2])
                center_y = float(transform[1] / transform[2])
                count += 1
                if count % 500 == 0:
                    print(real_prefix, cls_name, count,
                          "48 < center_x < (640-48): {}, 48 < center_y < (480-48): {}".format(
                             48 < center_x < (640 - 48), 48 < center_y < (480 - 48)))
                    print("\tcenter_x:{}, center_y:{}, tgt_trans: {}".format(center_x, center_y, tgt_trans))
                if count == 5000:
                    gen_this_pose = False
                    break

            # randomly generate a quat -------------------------------------------------
            tgt_quat = np.random.normal(0, 1, 4)
            tgt_quat = tgt_quat / np.linalg.norm(tgt_quat)
            if tgt_quat[0] < 0:
                tgt_quat *= -1

            tgt_rot_m = quat2mat(tgt_quat)
            new_pz = np.dot(tgt_rot_m, pz.reshape((-1, 1))).reshape((3,))
            pz_mean = new_points[cls_name]['pz_mean']
            deg = angle(new_pz, pz_mean)
            count = 0
            while (deg > deg_max):
                tgt_quat = np.random.normal(0, 1, 4)
                tgt_quat = tgt_quat / np.linalg.norm(tgt_quat)
                if tgt_quat[0] < 0:
                    tgt_quat *= -1

                tgt_rot_m = quat2mat(tgt_quat)
                new_pz = np.dot(tgt_rot_m, pz.reshape((-1, 1))).reshape((3,))
                pz_mean = new_points[cls_name]['pz_mean']
                deg = angle(new_pz, pz_mean)
                count += 1
                if count % 100 == 0:
                    print(real_prefix, cls_name, count, "deg < deg_max={}: {}".format(
                              deg_max, deg <= deg_max))
                    print(
                        "\tdeg:{}".format(deg))
                if count == 5000:
                    gen_this_pose = False
                    break
            # ---------------------------------------------------------------------------------
            if gen_this_pose:
                tgt_pose_q = np.zeros((7,), dtype='float32')
                tgt_pose_q[:4] = tgt_quat
                tgt_pose_q[4:] = tgt_trans
                syn_pose_dict[real_prefix][cls_name] = tgt_pose_q


    i = 0
    for k,v in syn_pose_dict.items():
        if len(v.keys()) >= 2:
            i += 1
    print('{} indices are successfully generated.'.format(i))

    # write pose
    poses_file = os.path.join(syn_pose_dir, 'LM6d_occ_dsm_all_syn_pose.pkl')
    with open(poses_file, 'wb') as f:
        cPickle.dump(syn_pose_dict, f, 2)


if __name__ == "__main__":
    # trans_lm_array = stat_lov()
    # pose_dict, quat_stat, trans_stat, new_points = stat_lm6d()
    # pose_dict, quat_stat, trans_stat, new_points = stat_lm6d_occ_test()
    gen_poses()
