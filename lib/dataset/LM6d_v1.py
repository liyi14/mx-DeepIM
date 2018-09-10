# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division, absolute_import
import six
from six.moves import cPickle, xrange
import cv2
import os
import numpy as np
from .imdb import IMDB
from lib.utils.projection import se3_inverse, se3_mul
from lib.utils.print_and_log import print_and_log
from lib.utils.pose_error import *
from lib.pair_matching.RT_transform import calc_rt_dist_m

class LM6D_v1(IMDB):
    def __init__(self, cfg, image_set, root_path, devkit_path, class_name, result_path=None, mask_size=-1, binary_thresh=None, mask_syn_name=''):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        #image_set = image_set[len(year) + 1 : len(image_set)]
        if len(mask_syn_name)>0:
            super(LM6D_v1, self).__init__('LM6d_render_v1_'+mask_syn_name, image_set, root_path, devkit_path, result_path)  # set self.name
        else:
            super(LM6D_v1, self).__init__('LM6D_render_v1', image_set, root_path, devkit_path, result_path)  # set self.name
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.render_real_data_path = os.path.join(devkit_path, 'data', 'render_real')
        self.real_data_path = os.path.join(devkit_path, 'data', 'real')
        self._linemod_path = self._get_default_path()

        if image_set.startswith('yu_val_v02'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_Yu_v02')
        elif image_set.startswith('yu_val'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_Yu_v01')
        elif image_set.startswith('train_v1') or image_set.startswith('my_val_v1') or image_set.startswith(
                'my_minival_v1'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v1')
        elif image_set.startswith('train_v2') or image_set.startswith('my_val_v2') or image_set.startswith(
                'my_minival_v2'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v2')
        elif image_set.startswith('train_v3') or image_set.startswith('my_val_v3') or image_set.startswith(
                'my_minival_v3'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v3')
        elif image_set.startswith('train_v4') or image_set.startswith('my_val_v4') or image_set.startswith(
                'my_minival_v4'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v4')
        elif image_set.startswith('train_v5') or image_set.startswith('my_val_v5') or image_set.startswith(
                'my_minival_v5'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v5')
        elif image_set.startswith('train_v6') or image_set.startswith('my_val_v6') or image_set.startswith(
                'my_minival_v6'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v6')
        elif image_set.startswith('train_v7') or image_set.startswith('my_val_v7') or image_set.startswith(
                'my_minival_v7'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v7')
        elif image_set.startswith('train_v8') or image_set.startswith('my_val_v8') or image_set.startswith(
                'my_minival_v8'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v8')
        elif image_set.startswith('train_v9') or image_set.startswith('my_val_v9') or image_set.startswith(
                'my_minival_v9'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered_v9')
        elif image_set.startswith('train') or image_set.startswith('my_val') or image_set.startswith('my_minival'):
            self.rendered_data_path = os.path.join(devkit_path, 'data', 'rendered')
        else:
            raise Exception("unknown prefix of "+image_set)

        print('LM6d_v1 rendered_data_path: {}'.format(self.rendered_data_path))


        self.mask_syn_path = ''
        if image_set.startswith('train'):
            self.phase = 'train'
            if len(mask_syn_name) > 0:
                self.mask_syn_path = os.path.join(devkit_path, 'data', mask_syn_name)
        elif image_set.startswith('my_val') or image_set.startswith('yu_val') or image_set.startswith('my_minival'):
            self.phase = 'val'
        else:
            raise Exception("unknown prefix of "+image_set)

        if image_set.startswith('yu_val_v02'):
            if os.path.exists(os.path.join(devkit_path, 'data', 'mask_Yu_v02')):
                self.mask_est_path = os.path.join(devkit_path, 'data', 'mask_Yu_v02')
            else:
                self.mask_est_path = '' # os.path.join(devkit_path, 'data', 'mask_Yu_v01')
        elif image_set.startswith('yu_val'):
            if os.path.exists(os.path.join(devkit_path, 'data', 'mask_Yu_v01')):
                self.mask_est_path = os.path.join(devkit_path, 'data', 'mask_Yu_v01')
            else:
                self.mask_est_path = '' # os.path.join(devkit_path, 'data', 'mask_Yu_v01')
        else:
            self.mask_est_path = ''

        if image_set.startswith('yu_val_v02'):
            if os.path.exists(os.path.join(devkit_path, 'data', 'mask_Yu_v02')):
                self.mask_est_path = os.path.join(devkit_path, 'data', 'mask_Yu_v02')
            else:
                self.mask_est_path = '' # os.path.join(devkit_path, 'data', 'mask_Yu_v01')
        else:
            self.mask_est_path = ''

        self.idx2class = {
                     1: 'ape',
                     2: 'benchviseblue',
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
        self.classes = self.idx2class.values()
        self.classes = sorted(self.classes)

        self.cur_class = class_name
        self.num_classes = len(self.idx2class.keys())
        self.image_set_index = self.load_image_set_index()
        self.num_pairs = len(self.image_set_index)
        print('num_pairs', self.num_pairs)
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

        self._points = self._load_object_points()
        self._diameters = self._load_object_diameter()

    def _get_default_path(self):
        """
        Return the default path where LOV is expected to be installed.
        """
        ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
        return os.path.join(ROOT_DIR, 'data', 'LINEMOD_6D')


    def _load_object_points(self):

        points = {}

        for cls_idx, cls_name in self.idx2class.items():
            point_file = os.path.join(self._linemod_path, 'models', cls_name, 'points.xyz')
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[cls_name] = np.loadtxt(point_file)

        return points

    def _load_object_diameter(self):
        diameters = {}
        models_info_file = os.path.join(self._linemod_path, 'models', 'models_info.txt')
        assert os.path.exists(models_info_file), 'Path does not exist: {}'.format(models_info_file)
        with open(models_info_file, 'r') as f:
            for line in f:
                line_list = line.strip().split()
                cls_idx = int(line_list[0])
                if not cls_idx in self.idx2class.keys():
                    continue
                diameter = float(line_list[2])
                cls_name = self.idx2class[cls_idx]
                diameters[cls_name] = diameter / 1000.

        return diameters

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.devkit_path, 'image_set', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip().split(' ') for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index, type, check=True, cls_name=''):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        if type == 'real':
            image_file = os.path.join(self.real_data_path, index + '-color.png')
        elif type == 'render_real':
            image_file = os.path.join(self.render_real_data_path, cls_name, index.split('/')[1] + '-color.png')
        elif type == 'rendered':
            image_file = os.path.join(self.rendered_data_path, index + '-color.png')
        if check:
            assert os.path.exists(image_file), 'type: {}, Path does not exist: {}'.format(type, image_file)
        return image_file

    def depth_path_from_index(self, index, type, check=True, cls_name=''):
        """
        given image index, find out the full path of depth map
        :param index: index of a specific image
        :return: full path of depth image
        """
        if type == 'real':
            depth_file = os.path.join(self.real_data_path, index + '-depth.png')
        elif type == 'render_real':
            depth_file = os.path.join(self.render_real_data_path, cls_name, index.split('/')[1] + '-depth.png')
        elif type == 'rendered':
            depth_file = os.path.join(self.rendered_data_path, index + '-depth.png')
        if check:
            assert os.path.exists(depth_file), 'type:{}, Path does not exist: {}'.format(type, depth_file)
        return depth_file

    def segmentation_path_from_index(self, index, type, check=True):
        """
        given image index, find out the full path of segmentation class
        :param index: index of a specific image
        :return: full path of segmentation class
        """
        if type == 'real_gt':
            seg_class_file = os.path.join(self.real_data_path, index + '-label.png')
        elif type == 'rendered':
            seg_class_file = os.path.join(self.rendered_data_path, index + '-label.png')
        elif type == 'syn':
            seg_class_file = os.path.join(self.mask_syn_path, index + '-label.png')
        elif type == 'real_est':
            seg_class_file = os.path.join(self.mask_est_path, index.split('/')[0], self.cur_class, index.split('/')[1] + '-label.png')
        if check:
            assert os.path.exists(seg_class_file), 'Path does not exist: {}'.format(seg_class_file)
        return seg_class_file

    def pose_from_index(self, index, type, cls_name=''):
        """
        given image index, find out the full path of segmentation class
        :param index: index of a specific image
        :return: full path of segmentation class
        """
        if type == 'real' or type == 'render_real':
            pose_file = os.path.join(self.render_real_data_path, cls_name, index.split('/')[1] + '-pose.txt')
        elif type == 'rendered':
            pose_file = os.path.join(self.rendered_data_path, index + '-pose.txt')
        assert os.path.exists(pose_file), 'type:{}, Path does not exist: {}'.format(type, pose_file)
        return np.loadtxt(pose_file, skiprows=1)

    def gt_pairdb(self):
        """
        return ground truth match pair dataset
        :return: imdb[pair_index]['image_real', 'image_rendered', 'height', 'width',
                                  'pose_real', 'pose_est', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_pairdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                if six.PY3:
                    pairdb = cPickle.load(fid, encoding='latin1')
                else:
                    pairdb = cPickle.load(fid)
            print('{} gt pairdb loaded from {}'.format(self.name, cache_file))
            return pairdb
        gt_pairdb = [self.load_render_annotation(pair_index) for pair_index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_pairdb, fid, 2)
        print('wrote gt pairdb to {}'.format(cache_file))

        return gt_pairdb

    def class2idx(self, class_name):
        for k, v in self.idx2class.items():
            if v == class_name:
                return k

    def load_render_annotation(self, pair_index):
        """

        :param self:
        :param pair_index:
        :return:
        """
        # from lib.utils.tictoc import tic,toc
        pair_rec = dict()
        cls_name = self.cur_class
        pair_rec['gt_class'] = cls_name

        pair_rec['image_real'] = self.image_path_from_index(pair_index[0], 'real')
        pair_rec['image_render_real'] = self.image_path_from_index(pair_index[0], 'render_real', cls_name=cls_name)
        pair_rec['image_rendered'] = self.image_path_from_index(pair_index[1], 'rendered')
        size_real = cv2.imread(pair_rec['image_real']).shape
        size_rendered = cv2.imread(pair_rec['image_rendered']).shape
        assert size_real==size_rendered
        pair_rec['height'] = size_real[0]
        pair_rec['width'] = size_real[1]
        pair_rec['depth_real'] = self.depth_path_from_index(pair_index[0], 'real')
        pair_rec['depth_render_real'] = self.depth_path_from_index(pair_index[0], 'render_real', cls_name=cls_name)
        pair_rec['depth_rendered'] = self.depth_path_from_index(pair_index[1], 'rendered')

        pair_rec['pose_real'] = self.pose_from_index(pair_index[0], 'real', cls_name=cls_name)
        pair_rec['pose_est'] = self.pose_from_index(pair_index[1], 'rendered')

        pair_rec['mask_real_gt'] = self.segmentation_path_from_index(pair_index[0], 'real_gt')
        if self.mask_est_path != '':
            pair_rec['mask_real_est'] = self.segmentation_path_from_index(pair_index[0], 'real_est')
        if self.phase == 'train' and self.mask_syn_path != '':
            pair_rec['mask_syn'] = self.segmentation_path_from_index(pair_index[0], 'syn')
        pair_rec['mask_idx'] = self.class2idx(pair_rec['gt_class'])

        pair_rec['pair_flipped'] = False
        pair_rec['img_flipped'] = False
        pair_rec['data_syn'] = False
        return pair_rec

    def evaluate_flow(self, flow_pred, flow_gt, flow_type):
        assert len(flow_pred)==len(flow_gt), 'flow_pred and flow_gt length not equal'
        sum_EPE = 0
        num_inst = 0
        for i in range(len(flow_gt)):
            cur_flow_pred = flow_pred[i]
            cur_flow_gt = flow_gt[i][flow_type][0]
            cur_flow_gt_vis = flow_gt[i][flow_type][1]
            x_diff = (cur_flow_gt[:,:,0]-cur_flow_pred[:,:,0])[cur_flow_gt_vis!=0]
            y_diff = (cur_flow_gt[:,:,1]-cur_flow_pred[:,:,1])[cur_flow_gt_vis!=0]
            diff = np.sqrt(np.square(x_diff)+np.square(y_diff))
            sum_EPE += diff.sum()
            num_inst += np.sum(cur_flow_gt_vis!=0)
        return float(sum_EPE)/num_inst

    def evaluate_pose(self, config, all_poses_est, all_poses_gt, logger):
        # evaluate and display
        print_and_log('evaluating pose', logger)
        rot_thresh_list = np.arange(1, 11, 1)
        trans_thresh_list = np.arange(0.01, 0.11, 0.01)
        num_metric = len(rot_thresh_list)
        num_iter = config.TEST.test_iter
        rot_acc = np.zeros((self.num_classes, num_iter, num_metric))
        trans_acc = np.zeros((self.num_classes, num_iter, num_metric))
        space_acc = np.zeros((self.num_classes, num_iter, num_metric))

        num_valid_class = 0
        for cls_idx, cls_name in enumerate(self.classes):
            if not (all_poses_est[cls_idx][0] and all_poses_gt[cls_idx][0]):
                continue
            num_valid_class += 1
            for iter_i in range(num_iter):
                curr_poses_gt = all_poses_gt[cls_idx][0]
                num = len(curr_poses_gt)
                curr_poses_est = all_poses_est[cls_idx][iter_i]

                cur_rot_rst = np.zeros((num, 1))
                cur_trans_rst = np.zeros((num, 1))

                for j in range(num):
                    r_dist_est, t_dist_est = calc_rt_dist_m(curr_poses_est[j], curr_poses_gt[j])
                    if cls_name == 'eggbox' and r_dist_est > 90:
                        RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                        curr_pose_est_sym = se3_mul(curr_poses_est[j], RT_z)
                        r_dist_est, t_dist_est = calc_rt_dist_m(curr_pose_est_sym, curr_poses_gt[j])
                    cur_rot_rst[j,0] = r_dist_est
                    cur_trans_rst[j,0] = t_dist_est

                for thresh_idx in range(num_metric):
                    rot_acc[cls_idx, iter_i, thresh_idx] = np.mean(cur_rot_rst < rot_thresh_list[thresh_idx])
                    trans_acc[cls_idx, iter_i, thresh_idx] = np.mean(cur_trans_rst < trans_thresh_list[thresh_idx])
                    space_acc[cls_idx, iter_i, thresh_idx] = np.mean(np.logical_and(cur_rot_rst < rot_thresh_list[thresh_idx],
                                                                                 cur_trans_rst < trans_thresh_list[thresh_idx]))

            show_list = [1, 4, 9]
            print_and_log("------------ {} -----------".format(cls_name), logger)
            print_and_log("{:>24}: {:>7}, {:>7}, {:>7}"
                          .format("[rot_thresh, trans_thresh", "RotAcc", "TraAcc", "SpcAcc"),
                          logger)
            for iter_i in range(num_iter):
                print_and_log("** iter {} **".format(iter_i+1), logger)
                print_and_log("{:<16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                              .format('average_accuracy',
                                      '[{:>2}, {:>4}]'.format(-1,-1),
                                      np.mean(rot_acc[cls_idx, iter_i, :])*100,
                                      np.mean(trans_acc[cls_idx, iter_i, :])*100,
                                      np.mean(space_acc[cls_idx, iter_i, :])*100),
                              logger)
                for i, show_idx in enumerate(show_list):
                    print_and_log("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                                  .format('average_accuracy',
                                          '[{:>2}, {:>4}]'.format(rot_thresh_list[show_idx],
                                                           trans_thresh_list[show_idx]),
                                          rot_acc[cls_idx, iter_i, show_idx]*100,
                                          trans_acc[cls_idx, iter_i, show_idx]*100,
                                          space_acc[cls_idx, iter_i, show_idx]*100),
                                  logger)
        print(' ')
        # overall performance
        for iter_i in range(num_iter):
            show_list = [1, 4, 9]
            print_and_log("---------- performance over {} classes -----------".format(num_valid_class), logger)
            print_and_log("** iter {} **".format(iter_i+1), logger)
            print_and_log("{:>24}: {:>7}, {:>7}, {:>7}"
                          .format("[rot_thresh, trans_thresh", "RotAcc", "TraAcc", "SpcAcc"),
                          logger)
            print_and_log("{:<16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                          .format('average_accuracy',
                                  '[{:>2}, {:>4}]'.format(-1,-1),
                                  np.sum(rot_acc[:, iter_i, :]) / (num_valid_class*num_metric) * 100,
                                  np.sum(trans_acc[:, iter_i, :]) / (num_valid_class*num_metric) * 100,
                                  np.sum(space_acc[:, iter_i, :]) / (num_valid_class*num_metric) * 100),
                          logger)
            for i, show_idx in enumerate(show_list):
                print_and_log("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                              .format('average_accuracy',
                                      '[{:>2}, {:>4}]'.format(rot_thresh_list[show_idx],
                                                       trans_thresh_list[show_idx]),
                                      np.sum(rot_acc[:, iter_i, show_idx]) / num_valid_class * 100,
                                      np.sum(trans_acc[:, iter_i, show_idx]) / num_valid_class * 100,
                                      np.sum(space_acc[:, iter_i, show_idx]) / num_valid_class * 100),
                              logger)
            print(' ')

    def evaluate_pose_add(self, config, all_poses_est, all_poses_gt, output_dir, logger):
        '''

        :param config: 
        :param all_poses_est: 
        :param all_poses_gt: 
        :param output_dir:
        :param logger: 
        :return: 
        '''
        print_and_log('evaluating pose add', logger)
        eval_method = 'add'
        num_iter = config.TEST.test_iter

        count_all = np.zeros((self.num_classes,), dtype=np.float32)
        count_correct = {k: np.zeros((self.num_classes, num_iter), dtype=np.float32) for k in ['0.02', '0.05', '0.10']}

        threshold_002 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_005 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_010 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32),
                                 (self.num_classes, num_iter, 1))  # (num_class, num_iter, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((self.num_classes, num_iter, num_thresh), dtype=np.float32)

        for i, cls_name in enumerate(self.classes):
            threshold_002[i, :] = 0.02 * self._diameters[cls_name]
            threshold_005[i, :] = 0.05 * self._diameters[cls_name]
            threshold_010[i, :] = 0.10 * self._diameters[cls_name]
            threshold_mean[i, :, :] *= self._diameters[cls_name]

        num_valid_class = 0
        for cls_idx, cls_name in enumerate(self.classes):
            if not (all_poses_est[cls_idx][0] and all_poses_gt[cls_idx][0]):
                continue
            num_valid_class += 1
            for iter_i in range(num_iter):
                curr_poses_gt = all_poses_gt[cls_idx][0]
                num = len(curr_poses_gt)
                curr_poses_est = all_poses_est[cls_idx][iter_i]

                for j in xrange(num):
                    if iter_i == 0:
                        count_all[cls_idx] += 1
                    RT = curr_poses_est[j]  # est pose
                    pose_gt = curr_poses_gt[j]  # gt pose
                    if  cls_name == 'eggbox' or cls_name == 'glue' or cls_name == 'bowl' or cls_name == 'cup':
                        eval_method = 'adi'
                        error = adi(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                    self._points[cls_name])
                    else:
                        error = add(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                    self._points[cls_name])

                    if error < threshold_002[cls_idx, iter_i]:
                        count_correct['0.02'][cls_idx, iter_i] += 1
                    if error < threshold_005[cls_idx, iter_i]:
                        count_correct['0.05'][cls_idx, iter_i] += 1
                    if error < threshold_010[cls_idx, iter_i]:
                        count_correct['0.10'][cls_idx, iter_i] += 1
                    for thresh_i in xrange(num_thresh):
                        if error < threshold_mean[cls_idx, iter_i, thresh_i]:
                            count_correct['mean'][cls_idx, iter_i, thresh_i] += 1

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plot_data = {}

        for cls_idx, cls_name in enumerate(self.classes):
            if count_all[cls_idx] == 0:
                continue
            plot_data[cls_name] = []
            for iter_i in range(num_iter):
                print_and_log("** {}, iter {} **".format(cls_name, iter_i + 1), logger)
                from scipy.integrate import simps
                area = simps(count_correct['mean'][cls_idx, iter_i] / float(count_all[cls_idx]),
                             dx = dx) / 0.1

                fig = plt.figure()
                x_s = np.arange(0, 0.1, dx).astype(np.float32)
                y_s = count_correct['mean'][cls_idx, iter_i] / float(count_all[cls_idx])
                plot_data[cls_name].append((x_s, y_s))
                plt.plot(x_s, y_s, '-')
                plt.xlim(0, 0.1)
                plt.ylim(0, 1)
                plt.xlabel("Average distance threshold in meter (symmetry)")
                plt.ylabel("accuracy")
                plt.savefig(os.path.join(output_dir, 'acc_thres_{}_iter{}.png'.format(cls_name, iter_i + 1)),
                            dpi=fig.dpi)

                print_and_log('threshold=[0.0, 0.10], area: {:.2f}'.format(area * 100), logger)
                print_and_log('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['0.02'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['0.02'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['0.05'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['0.05'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['0.10'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['0.10'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log(" ", logger)

        with open(os.path.join(output_dir, '{}_xys.pkl'.format(eval_method)), 'wb') as f:
            cPickle.dump(plot_data, f, protocol=2)

        print_and_log("="*30, logger)

        print(' ')
        # overall performance of add
        # iter sum of 'mean'
        count_correct_mean_all = np.sum(count_correct['mean'], 0)
        for iter_i in range(num_iter):
            print_and_log("---------- add performance over {} classes -----------".format(num_valid_class), logger)
            print_and_log("** iter {} **".format(iter_i + 1), logger)
            area = simps(count_correct_mean_all[iter_i] / float(np.sum(count_all[:])),
                                      dx=dx) / 0.1

            print_and_log('threshold=[0.0, 0.10], area: {:.2f}'.format(area * 100), logger)
            print_and_log('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                np.sum(count_correct['0.02'][:, iter_i]),
                np.sum(count_all[:]),
                100 * float(np.sum(count_correct['0.02'][:, iter_i])) / float(np.sum(count_all[:]))), logger)
            print_and_log('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                np.sum(count_correct['0.05'][:, iter_i]),
                np.sum(count_all[:]),
                100 * float(np.sum(count_correct['0.05'][:, iter_i])) / float(np.sum(count_all[:]))), logger)
            print_and_log('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                np.sum(count_correct['0.10'][:, iter_i]),
                np.sum(count_all[:]),
                100 * float(np.sum(count_correct['0.10'][:, iter_i])) / float(np.sum(count_all[:]))), logger)
            print(' ')

        print_and_log("=" * 30, logger)


    def evaluate_pose_arp_2d(self, config, all_poses_est, all_poses_gt, output_dir, logger):
        '''
        evaluate average re-projection 2d error
        :param config: 
        :param all_poses_est: 
        :param all_poses_gt: 
        :param output_dir:
        :param logger: 
        :return: 
        '''
        print_and_log('evaluating pose average re-projection 2d error', logger)
        num_iter = config.TEST.test_iter
        K = config.dataset.INTRINSIC_MATRIX

        count_all = np.zeros((self.num_classes,), dtype=np.float32)
        count_correct = {k: np.zeros((self.num_classes, num_iter), dtype=np.float32) for k in ['2', '5', '10', '20']}

        threshold_2 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_5 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_10 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_20 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        dx = 0.1
        threshold_mean = np.tile(np.arange(0, 50, dx).astype(np.float32),
                                 (self.num_classes, num_iter, 1))  # (num_class, num_iter, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((self.num_classes, num_iter, num_thresh), dtype=np.float32)

        for i in xrange(self.num_classes):
            threshold_2[i, :] = 2
            threshold_5[i, :] = 5
            threshold_10[i, :] = 10
            threshold_20[i, :] = 20


        num_valid_class = 0
        for cls_idx, cls_name in enumerate(self.classes):
            if not (all_poses_est[cls_idx][0] and all_poses_gt[cls_idx][0]):
                continue
            num_valid_class += 1
            for iter_i in range(num_iter):
                curr_poses_gt = all_poses_gt[cls_idx][0]
                num = len(curr_poses_gt)
                curr_poses_est = all_poses_est[cls_idx][iter_i]

                for j in xrange(num):
                    if iter_i == 0:
                        count_all[cls_idx] += 1
                    RT = curr_poses_est[j]  # est pose
                    pose_gt = curr_poses_gt[j]  # gt pose

                    error_rotation = re(RT[:3, :3], pose_gt[:3, :3])
                    if cls_name == 'eggbox' and error_rotation > 90:
                        RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                        RT_sym = se3_mul(RT, RT_z)
                        error = arp_2d(RT_sym[:3, :3], RT_sym[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                       self._points[cls_name], K)
                    else:
                        error = arp_2d(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                       self._points[cls_name], K)

                    if error < threshold_2[cls_idx, iter_i]:
                        count_correct['2'][cls_idx, iter_i] += 1
                    if error < threshold_5[cls_idx, iter_i]:
                        count_correct['5'][cls_idx, iter_i] += 1
                    if error < threshold_10[cls_idx, iter_i]:
                        count_correct['10'][cls_idx, iter_i] += 1
                    if error < threshold_20[cls_idx, iter_i]:
                        count_correct['20'][cls_idx, iter_i] += 1
                    for thresh_i in xrange(num_thresh):
                        if error < threshold_mean[cls_idx, iter_i, thresh_i]:
                            count_correct['mean'][cls_idx, iter_i, thresh_i] += 1
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # store plot data
        plot_data = {}

        for cls_idx, cls_name in enumerate(self.classes):
            if count_all[cls_idx] == 0:
                continue
            plot_data[cls_name] = []
            for iter_i in range(num_iter):
                print_and_log("** {}, iter {} **".format(cls_name, iter_i + 1), logger)
                from scipy.integrate import simps
                area = simps(count_correct['mean'][cls_idx, iter_i] / float(count_all[cls_idx]),
                             dx=dx) / (50.0)

                fig = plt.figure()
                x_s = np.arange(0, 50, dx).astype(np.float32)
                y_s = 100 * count_correct['mean'][cls_idx, iter_i] / float(count_all[cls_idx])
                plot_data[cls_name].append((x_s, y_s))
                plt.plot(x_s, y_s, '-')
                plt.xlim(0, 50)
                plt.ylim(0, 100)
                plt.grid(True)
                plt.xlabel("px")
                plt.ylabel("correctly estimated poses in %")
                plt.savefig(os.path.join(output_dir, 'arp_2d_{}_iter{}.png'.format(cls_name, iter_i + 1)),
                            dpi=fig.dpi)

                print_and_log('threshold=[0, 50], area: {:.2f}'.format(area * 100), logger)
                print_and_log('threshold=2, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['2'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['2'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=5, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['5'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['5'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['10'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['10'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=20, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['20'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['20'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log(" ", logger)

        with open(os.path.join(output_dir, 'arp_2d_xys.pkl'), 'wb') as f:
            cPickle.dump(plot_data, f, protocol=2)
        print_and_log("=" * 30, logger)

        print(' ')
        # overall performance of arp 2d
        count_correct_mean_all = np.sum(count_correct['mean'], 0)
        for iter_i in range(num_iter):
            print_and_log("---------- arp 2d performance over {} classes -----------".format(num_valid_class), logger)
            print_and_log("** iter {} **".format(iter_i + 1), logger)
            area = simps(count_correct_mean_all[iter_i] / float(np.sum(count_all[:])),
                         dx=dx) / (50.0)

            print_and_log('threshold=[0, 50], area: {:.2f}'.format(area * 100), logger)
            print_and_log('threshold=2, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                np.sum(count_correct['2'][:, iter_i]),
                np.sum(count_all[:]),
                100 * float(np.sum(count_correct['2'][:, iter_i])) / float(np.sum(count_all[:]))), logger)
            print_and_log('threshold=5, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                np.sum(count_correct['5'][:, iter_i]),
                np.sum(count_all[:]),
                100 * float(np.sum(count_correct['5'][:, iter_i])) / float(np.sum(count_all[:]))), logger)
            print_and_log('threshold=10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                np.sum(count_correct['10'][:, iter_i]),
                np.sum(count_all[:]),
                100 * float(np.sum(count_correct['10'][:, iter_i])) / float(np.sum(count_all[:]))), logger)
            print_and_log('threshold=20, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                np.sum(count_correct['20'][:, iter_i]),
                np.sum(count_all[:]),
                100 * float(np.sum(count_correct['20'][:, iter_i])) / float(np.sum(count_all[:]))), logger)
            print_and_log(" ", logger)

        print_and_log("=" * 30, logger)


################################

    def evaluate_pose_dict(self, config, all_poses_est, all_poses_gt, logger):
        # evaluate and display
        print_and_log('evaluating pose', logger)
        rot_thresh_list = np.arange(1, 11, 1)
        trans_thresh_list = np.arange(0.01, 0.11, 0.01)
        num_metric = len(rot_thresh_list)
        num_iter = config.TEST.test_iter
        rot_acc = np.zeros((self.num_classes, num_iter, num_metric))
        trans_acc = np.zeros((self.num_classes, num_iter, num_metric))
        space_acc = np.zeros((self.num_classes, num_iter, num_metric))

        num_valid_class = 0
        for cls_idx, cls_name in enumerate(self.classes):
            cls_idx_p = cls_idx + 1 # cls_idx in poses starts from '__background__'

            if not (all_poses_est[cls_idx_p][0]): # and all_poses_gt[cls_idx][0]):
                continue
            num_valid_class += 1
            curr_poses_gt = all_poses_gt[cls_idx_p]
            num = len(curr_poses_gt)
            for iter_i in range(num_iter):
                curr_poses_est = [all_poses_est[cls_idx_p][iter_i][j] for j in range(num)]

                cur_rot_rst = np.zeros((num, 1))
                cur_trans_rst = np.zeros((num, 1))

                for j in range(num):
                    r_dist_est, t_dist_est = calc_rt_dist_m(curr_poses_est[j], curr_poses_gt[j])
                    if cls_name == 'eggbox' and r_dist_est > 90:
                        RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                        curr_pose_est_sym = se3_mul(curr_poses_est[j], RT_z)
                        r_dist_est, t_dist_est = calc_rt_dist_m(curr_pose_est_sym, curr_poses_gt[j])
                    cur_rot_rst[j,0] = r_dist_est
                    cur_trans_rst[j,0] = t_dist_est

                for thresh_idx in range(num_metric):
                    rot_acc[cls_idx, iter_i, thresh_idx] = np.mean(cur_rot_rst < rot_thresh_list[thresh_idx])
                    trans_acc[cls_idx, iter_i, thresh_idx] = np.mean(cur_trans_rst < trans_thresh_list[thresh_idx])
                    space_acc[cls_idx, iter_i, thresh_idx] = np.mean(np.logical_and(cur_rot_rst < rot_thresh_list[thresh_idx],
                                                                                 cur_trans_rst < trans_thresh_list[thresh_idx]))

            show_list = [1, 4, 9]
            print_and_log("------------ {} -----------".format(cls_name), logger)
            print_and_log("{:>24}: {:>7}, {:>7}, {:>7}"
                          .format("[rot_thresh, trans_thresh", "RotAcc", "TraAcc", "SpcAcc"),
                          logger)
            for iter_i in range(num_iter):
                print_and_log("** iter {} **".format(iter_i+1), logger)
                print_and_log("{:<16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                              .format('average_accuracy',
                                      '[{:>2}, {:>4}]'.format(-1,-1),
                                      np.mean(rot_acc[cls_idx, iter_i, :])*100,
                                      np.mean(trans_acc[cls_idx, iter_i, :])*100,
                                      np.mean(space_acc[cls_idx, iter_i, :])*100),
                              logger)
                for i, show_idx in enumerate(show_list):
                    print_and_log("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                                  .format('average_accuracy',
                                          '[{:>2}, {:>4}]'.format(rot_thresh_list[show_idx],
                                                           trans_thresh_list[show_idx]),
                                          rot_acc[cls_idx, iter_i, show_idx]*100,
                                          trans_acc[cls_idx, iter_i, show_idx]*100,
                                          space_acc[cls_idx, iter_i, show_idx]*100),
                                  logger)
        print(' ')
        # overall performance
        for iter_i in range(num_iter):
            show_list = [1, 4, 9]
            print_and_log("---------- performance over {} classes -----------".format(num_valid_class), logger)
            print_and_log("** iter {} **".format(iter_i+1), logger)
            print_and_log("{:>24}: {:>7}, {:>7}, {:>7}"
                          .format("[rot_thresh, trans_thresh", "RotAcc", "TraAcc", "SpcAcc"),
                          logger)
            print_and_log("{:<16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                          .format('average_accuracy',
                                  '[{:>2}, {:>4}]'.format(-1,-1),
                                  np.sum(rot_acc[:, iter_i, :]) / (num_valid_class*num_metric) * 100,
                                  np.sum(trans_acc[:, iter_i, :]) / (num_valid_class*num_metric) * 100,
                                  np.sum(space_acc[:, iter_i, :]) / (num_valid_class*num_metric) * 100),
                          logger)
            for i, show_idx in enumerate(show_list):
                print_and_log("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                              .format('average_accuracy',
                                      '[{:>2}, {:>4}]'.format(rot_thresh_list[show_idx],
                                                       trans_thresh_list[show_idx]),
                                      np.sum(rot_acc[:, iter_i, show_idx]) / num_valid_class * 100,
                                      np.sum(trans_acc[:, iter_i, show_idx]) / num_valid_class * 100,
                                      np.sum(space_acc[:, iter_i, show_idx]) / num_valid_class * 100),
                              logger)
            print(' ')

    def evaluate_pose_add_dict(self, config, all_poses_est, all_poses_gt, output_dir, logger):
        '''

        :param config:
        :param all_poses_est:
        :param all_poses_gt:
        :param output_dir:
        :param logger:
        :return:
        '''
        print_and_log('evaluating pose add', logger)
        eval_method = 'add'
        num_iter = config.TEST.test_iter

        count_all = np.zeros((self.num_classes,), dtype=np.float32)
        count_correct = {k: np.zeros((self.num_classes, num_iter), dtype=np.float32) for k in ['0.02', '0.05', '0.10']}

        threshold_002 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_005 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_010 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32),
                                 (self.num_classes, num_iter, 1))  # (num_class, num_iter, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((self.num_classes, num_iter, num_thresh), dtype=np.float32)

        for i, cls_name in enumerate(self.classes):
            threshold_002[i, :] = 0.02 * self._diameters[cls_name]
            threshold_005[i, :] = 0.05 * self._diameters[cls_name]
            threshold_010[i, :] = 0.10 * self._diameters[cls_name]
            threshold_mean[i, :, :] *= self._diameters[cls_name]

        num_valid_class = 0
        for cls_idx, cls_name in enumerate(self.classes):
            cls_idx_p = cls_idx + 1
            if not (all_poses_est[cls_idx_p][0]): # and all_poses_gt[cls_idx][0]):
                continue
            num_valid_class += 1
            curr_poses_gt = all_poses_gt[cls_idx_p]
            num = len(curr_poses_gt)
            for iter_i in range(num_iter):
                curr_poses_est = [all_poses_est[cls_idx_p][iter_i][j] for j in range(num)]

                for j in xrange(num):
                    if iter_i == 0:
                        count_all[cls_idx] += 1
                    RT = curr_poses_est[j]  # est pose
                    pose_gt = curr_poses_gt[j]  # gt pose
                    if  cls_name == 'eggbox' or cls_name == 'glue' or cls_name == 'bowl' or cls_name == 'cup':
                        eval_method = 'adi'
                        error = adi(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                    self._points[cls_name])
                    else:
                        error = add(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                    self._points[cls_name])

                    if error < threshold_002[cls_idx, iter_i]:
                        count_correct['0.02'][cls_idx, iter_i] += 1
                    if error < threshold_005[cls_idx, iter_i]:
                        count_correct['0.05'][cls_idx, iter_i] += 1
                    if error < threshold_010[cls_idx, iter_i]:
                        count_correct['0.10'][cls_idx, iter_i] += 1
                    for thresh_i in xrange(num_thresh):
                        if error < threshold_mean[cls_idx, iter_i, thresh_i]:
                            count_correct['mean'][cls_idx, iter_i, thresh_i] += 1

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plot_data = {}

        for cls_idx, cls_name in enumerate(self.classes):
            if count_all[cls_idx] == 0:
                continue
            plot_data[cls_name] = []
            for iter_i in range(num_iter):
                print_and_log("** {}, iter {} **".format(cls_name, iter_i + 1), logger)
                from scipy.integrate import simps
                area = simps(count_correct['mean'][cls_idx, iter_i] / float(count_all[cls_idx]),
                             dx=dx) / 0.1

                fig = plt.figure()
                x_s = np.arange(0, 0.1, dx).astype(np.float32)
                y_s = count_correct['mean'][cls_idx, iter_i] / float(count_all[cls_idx])
                plot_data[cls_name].append((x_s, y_s))
                plt.plot(x_s, y_s, '-')
                plt.xlim(0, 0.1)
                plt.ylim(0, 1)
                plt.xlabel("Average distance threshold in meter (symmetry)")
                plt.ylabel("accuracy")
                plt.savefig(os.path.join(output_dir, 'acc_thres_{}_iter{}.png'.format(cls_name, iter_i + 1)),
                            dpi=fig.dpi)

                print_and_log('threshold=[0.0, 0.10], area: {:.2f}'.format(area * 100), logger)
                print_and_log('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['0.02'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['0.02'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['0.05'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['0.05'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['0.10'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['0.10'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log(" ", logger)

        with open(os.path.join(output_dir, '{}_xys.pkl'.format(eval_method)), 'wb') as f:
            cPickle.dump(plot_data, f, protocol=2)

        print_and_log("="*30, logger)




    def evaluate_pose_arp_2d_dict(self, config, all_poses_est, all_poses_gt, output_dir, logger):
        '''
        evaluate average re-projection 2d error
        :param config:
        :param all_poses_est:
        :param all_poses_gt:
        :param output_dir:
        :param logger:
        :return:
        '''
        print_and_log('evaluating pose average re-projection 2d error', logger)
        num_iter = config.TEST.test_iter
        K = config.dataset.INTRINSIC_MATRIX

        count_all = np.zeros((self.num_classes,), dtype=np.float32)
        count_correct = {k: np.zeros((self.num_classes, num_iter), dtype=np.float32) for k in ['2', '5', '10', '20']}

        threshold_2 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_5 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_10 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        threshold_20 = np.zeros((self.num_classes, num_iter), dtype=np.float32)
        dx = 0.1
        threshold_mean = np.tile(np.arange(0, 50, dx).astype(np.float32),
                                 (self.num_classes, num_iter, 1))  # (num_class, num_iter, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((self.num_classes, num_iter, num_thresh), dtype=np.float32)

        for i in xrange(self.num_classes):
            threshold_2[i, :] = 2
            threshold_5[i, :] = 5
            threshold_10[i, :] = 10
            threshold_20[i, :] = 20


        num_valid_class = 0
        for cls_idx, cls_name in enumerate(self.classes):
            cls_idx_p = cls_idx + 1
            if not (all_poses_est[cls_idx_p][0]): # and all_poses_gt[cls_idx][0]):
                continue
            num_valid_class += 1
            curr_poses_gt = all_poses_gt[cls_idx_p]
            num = len(curr_poses_gt)
            for iter_i in range(num_iter):
                curr_poses_est = [all_poses_est[cls_idx_p][iter_i][j] for j in range(num)]

                for j in xrange(num):
                    if iter_i == 0:
                        count_all[cls_idx] += 1
                    RT = curr_poses_est[j]  # est pose
                    pose_gt = curr_poses_gt[j]  # gt pose

                    error_rotation = re(RT[:3, :3], pose_gt[:3, :3])
                    if cls_name == 'eggbox' and error_rotation > 90:
                        RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                        RT_sym = se3_mul(RT, RT_z)
                        error = arp_2d(RT_sym[:3, :3], RT_sym[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                       self._points[cls_name], K)
                    else:
                        error = arp_2d(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                       self._points[cls_name], K)

                    if error < threshold_2[cls_idx, iter_i]:
                        count_correct['2'][cls_idx, iter_i] += 1
                    if error < threshold_5[cls_idx, iter_i]:
                        count_correct['5'][cls_idx, iter_i] += 1
                    if error < threshold_10[cls_idx, iter_i]:
                        count_correct['10'][cls_idx, iter_i] += 1
                    if error < threshold_20[cls_idx, iter_i]:
                        count_correct['20'][cls_idx, iter_i] += 1
                    for thresh_i in xrange(num_thresh):
                        if error < threshold_mean[cls_idx, iter_i, thresh_i]:
                            count_correct['mean'][cls_idx, iter_i, thresh_i] += 1
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # store plot data
        plot_data = {}

        for cls_idx, cls_name in enumerate(self.classes):
            if count_all[cls_idx] == 0:
                continue
            plot_data[cls_name] = []
            for iter_i in range(num_iter):
                print_and_log("** {}, iter {} **".format(cls_name, iter_i + 1), logger)
                from scipy.integrate import simps
                area = simps(count_correct['mean'][cls_idx, iter_i] / float(count_all[cls_idx]),
                             dx=dx) / (50.0)

                fig = plt.figure()
                x_s = np.arange(0, 50, dx).astype(np.float32)
                y_s = 100 * count_correct['mean'][cls_idx, iter_i] / float(count_all[cls_idx])
                plot_data[cls_name].append((x_s, y_s))
                plt.plot(x_s, y_s, '-')
                plt.xlim(0, 50)
                plt.ylim(0, 100)
                plt.grid(True)
                plt.xlabel("px")
                plt.ylabel("correctly estimated poses in %")
                plt.savefig(os.path.join(output_dir, 'arp_2d_{}_iter{}.png'.format(cls_name, iter_i + 1)),
                            dpi=fig.dpi)

                print_and_log('threshold=[0, 50], area: {:.2f}'.format(area * 100), logger)
                print_and_log('threshold=2, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['2'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['2'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=5, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['5'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['5'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['10'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['10'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log('threshold=20, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['20'][cls_idx, iter_i],
                    count_all[cls_idx],
                    100 * float(count_correct['20'][cls_idx, iter_i]) / float(count_all[cls_idx])), logger)
                print_and_log(" ", logger)

        with open(os.path.join(output_dir, 'arp_2d_xys.pkl'), 'wb') as f:
            cPickle.dump(plot_data, f, protocol=2)
        print_and_log("=" * 30, logger)
