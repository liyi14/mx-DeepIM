# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division, absolute_import
import numpy as np
from lib.render_glumpy.render_py_multi import Render_Py
import cv2
import os
import time
import mxnet as mx
import six
from six.moves import cPickle

from .module import MutableModule
from lib.utils.PrefetchingIter import PrefetchingIter

from lib.pair_matching.flow import calc_flow
from lib.utils.show_flows import sintel_compute_color

from lib.pair_matching.RT_transform import *
from lib.pair_matching.data_pair import update_data_batch

from lib.utils.print_and_log import print_and_log
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.utils.image import resize

class Predictor(object):
    def __init__(self, config, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes,
                                  CUDA9=config.CUDA9)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]

def pred_eval(config, predictor, test_data, imdb_test, vis=False, ignore_cache=None, logger=None, pairdb=None):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb_test: image database
    :param vis: controls visualization
    :param ignore_cache: ignore the saved cache file
    :param logger: the logger instance
    :return:
    """
    print(imdb_test.result_path)
    print('test iter size: ', config.TEST.test_iter)
    pose_err_file = os.path.join(imdb_test.result_path, imdb_test.name + '_pose_iter{}.pkl'.format(config.TEST.test_iter))
    if os.path.exists(pose_err_file) and not ignore_cache and not vis:
        with open(pose_err_file, 'rb') as fid:
            if six.PY3:
                [all_rot_err, all_trans_err, all_poses_est, all_poses_gt] = cPickle.load(fid, encoding='latin1')
            else:
                [all_rot_err, all_trans_err, all_poses_est, all_poses_gt] = cPickle.load(fid)
        imdb_test.evaluate_pose(config, all_poses_est, all_poses_gt, logger)
        pose_add_plots_dir = os.path.join(imdb_test.result_path, 'add_plots')
        mkdir_if_missing(pose_add_plots_dir)
        imdb_test.evaluate_pose_add(config, all_poses_est, all_poses_gt, output_dir=pose_add_plots_dir, logger=logger)
        pose_arp2d_plots_dir = os.path.join(imdb_test.result_path, 'arp_2d_plots')
        mkdir_if_missing(pose_arp2d_plots_dir)
        imdb_test.evaluate_pose_arp_2d(config, all_poses_est, all_poses_gt, output_dir=pose_arp2d_plots_dir,
                                       logger=logger)
        return

    assert vis or not test_data.shuffle
    assert config.TEST.BATCH_PAIRS == 1
    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    num_pairs = len(pairdb)
    height = 480
    width = 640

    data_time, net_time, post_time = 0.0, 0.0, 0.0

    sum_EPE_all = 0.0
    num_inst_all = 0.0
    sum_EPE_viz = 0.0
    num_inst_viz = 0.0
    sum_EPE_vizbg = 0.0
    num_inst_vizbg = 0.0
    sum_PoseErr = [np.zeros((len(imdb_test.classes)+1, 2)) for batch_idx in range(config.TEST.test_iter)]

    all_rot_err = [[[]for j in range(config.TEST.test_iter)] for batch_idx in range(len(imdb_test.classes))] # num_cls x test_iter
    all_trans_err = [[[]for j in range(config.TEST.test_iter)] for batch_idx in range(len(imdb_test.classes))]

    all_poses_est = [[[] for j in range(config.TEST.test_iter)] for batch_idx in range(len(imdb_test.classes))]
    all_poses_gt = [[[] for j in range(config.TEST.test_iter)] for batch_idx in range(len(imdb_test.classes))]

    num_inst = np.zeros(len(imdb_test.classes)+1)

    K = config.dataset.INTRINSIC_MATRIX
    if (config.TEST.test_iter > 1 or config.TEST.VISUALIZE) and True:
        print("************* start setup render_glumpy environment... ******************")
        if config.dataset.dataset.startswith('ModelNet'):
            from lib.render_glumpy.render_py_light_modelnet_multi import Render_Py_Light_ModelNet_Multi
            modelnet_root = config.modelnet_root
            texture_path = os.path.join(modelnet_root, 'gray_texture.png')

            model_path_list = [os.path.join(config.dataset.model_dir, '{}.obj'.format(model_name))
                                    for model_name in config.dataset.class_name]
            render_machine = Render_Py_Light_ModelNet_Multi(model_path_list,
                                                            texture_path, K, width, height,
                                                            config.dataset.ZNEAR,
                                                            config.dataset.ZFAR,
                                                            brightness_ratios=[0.7])
        else:
            render_machine = Render_Py(config.dataset.model_dir, config.dataset.class_name, K, width, height,
                                   config.dataset.ZNEAR, config.dataset.ZFAR)

        def render(render_machine, pose, cls_idx, K=None):
            if config.dataset.dataset.startswith('ModelNet'):
                idx = 2
                # generate random light_position
                if idx % 6 == 0:
                    light_position = [1, 0, 1]
                elif idx % 6 == 1:
                    light_position = [1, 1, 1]
                elif idx % 6 == 2:
                    light_position = [0, 1, 1]
                elif idx % 6 == 3:
                    light_position = [-1, 1, 1]
                elif idx % 6 == 4:
                    light_position = [-1, 0, 1]
                elif idx % 6 == 5:
                    light_position = [0, 0, 1]
                else:
                    raise Exception("???")
                light_position = np.array(light_position) * 0.5
                # inverse yz
                light_position[0] += pose[0, 3]
                light_position[1] -= pose[1, 3]
                light_position[2] -= pose[2, 3]

                colors = np.array([1, 1, 1])  # white light
                intensity = np.random.uniform(0.9, 1.1, size=(3,))
                colors_randk = 0
                light_intensity = colors[colors_randk] * intensity

                # randomly choose a render machine
                rm_randk = 0  # random.randint(0, len(brightness_ratios) - 1)
                rgb_gl, depth_gl = render_machine.render(
                    cls_idx,
                    pose[:3, :3],
                    pose[:3, 3],
                    light_position,
                    light_intensity,
                    brightness_k=rm_randk,
                    r_type='mat')
                rgb_gl = rgb_gl.astype('uint8')
            else:
                rgb_gl, depth_gl = render_machine.render(
                    cls_idx,
                    pose[:3, :3], pose[:, 3], r_type='mat', K=K)
                rgb_gl = rgb_gl.astype('uint8')
            return rgb_gl, depth_gl
        print("***************setup render_glumpy environment succeed ******************")

    if config.TEST.PRECOMPUTED_ICP:
        print('precomputed_ICP')
        config.TEST.test_iter = 1
        all_rot_err = [[[]for j in range(1)] for batch_idx in range(len(imdb_test.classes))]
        all_trans_err = [[[]for j in range(1)] for batch_idx in range(len(imdb_test.classes))]

        all_poses_est = [[[] for j in range(1)] for batch_idx in range(len(imdb_test.classes))]
        all_poses_gt = [[[] for j in range(1)] for batch_idx in range(len(imdb_test.classes))]

        xy_trans_err = [[[]for j in range(1)] for batch_idx in range(len(imdb_test.classes))]
        z_trans_err = [[[]for j in range(1)] for batch_idx in range(len(imdb_test.classes))]
        for idx in range(len(pairdb)):
            pose_path = pairdb[idx]['depth_rendered'][:-10]+'-pose_icp.txt'
            pose_real_est = np.loadtxt(pose_path, skiprows=1)
            pose_real = pairdb[idx]['pose_real']
            r_dist_est, t_dist_est = calc_rt_dist_m(pose_real_est, pose_real)
            xy_dist = np.linalg.norm(pose_real_est[:2, -1]-pose_real[:2, -1])
            z_dist = np.linalg.norm(pose_real_est[-1, -1]-pose_real[-1, -1])
            print("{}: r_dist_est: {}, t_dist_est: {}, xy_dist: {}, z_dist: {}".format(
                    idx, r_dist_est, t_dist_est, xy_dist, z_dist))
            class_id = imdb_test.classes.index(pairdb[idx]['gt_class'])
            # store poses estimation and gt
            all_poses_est[class_id][0].append(pose_real_est)
            all_poses_gt[class_id][0].append(pairdb[idx]['pose_real'])
            all_rot_err[class_id][0].append(r_dist_est)
            all_trans_err[class_id][0].append(t_dist_est)
            xy_trans_err[class_id][0].append(xy_dist)
            z_trans_err[class_id][0].append(z_dist)
        all_rot_err = np.array(all_rot_err)
        all_trans_err = np.array(all_trans_err)
        print("rot = {} +/- {}".format(np.mean(all_rot_err[class_id][0]), np.std(all_rot_err[class_id][0])))
        print("trans = {} +/- {}".format(np.mean(all_trans_err[class_id][0]), np.std(all_trans_err[class_id][0])))
        num_list = all_trans_err[class_id][0]
        print("xyz: {:.2f} +/- {:.2f}".format(np.mean(num_list)*100, np.std(num_list)*100))
        num_list = xy_trans_err[class_id][0]
        print("xy: {:.2f} +/- {:.2f}".format(np.mean(num_list)*100, np.std(num_list)*100))
        num_list = z_trans_err[class_id][0]
        print("z: {:.2f} +/- {:.2f}".format(np.mean(num_list)*100, np.std(num_list)*100))

        imdb_test.evaluate_pose(config, all_poses_est, all_poses_gt, logger)
        pose_add_plots_dir = os.path.join(imdb_test.result_path, 'add_plots_precomputed_ICP')
        mkdir_if_missing(pose_add_plots_dir)
        imdb_test.evaluate_pose_add(config, all_poses_est, all_poses_gt, output_dir = pose_add_plots_dir, logger=logger)
        pose_arp2d_plots_dir = os.path.join(imdb_test.result_path, 'arp_2d_plots_precomputed_ICP')
        mkdir_if_missing(pose_arp2d_plots_dir)
        imdb_test.evaluate_pose_arp_2d(config, all_poses_est, all_poses_gt, output_dir=pose_arp2d_plots_dir, logger=logger)
        return

    if config.TEST.BEFORE_ICP:
        print('before_ICP')
        config.TEST.test_iter = 1
        all_rot_err = [[[]for j in range(1)] for batch_idx in range(len(imdb_test.classes))]
        all_trans_err = [[[]for j in range(1)] for batch_idx in range(len(imdb_test.classes))]

        all_poses_est = [[[] for j in range(1)] for batch_idx in range(len(imdb_test.classes))]
        all_poses_gt = [[[] for j in range(1)] for batch_idx in range(len(imdb_test.classes))]

        xy_trans_err = [[[]for j in range(1)] for batch_idx in range(len(imdb_test.classes))]
        z_trans_err = [[[]for j in range(1)] for batch_idx in range(len(imdb_test.classes))]
        for idx in range(len(pairdb)):
            pose_path = pairdb[idx]['depth_rendered'][:-10]+'-pose.txt'
            pose_real_est = np.loadtxt(pose_path, skiprows=1)
            pose_real = pairdb[idx]['pose_real']
            r_dist_est, t_dist_est = calc_rt_dist_m(pose_real_est, pose_real)
            xy_dist = np.linalg.norm(pose_real_est[:2, -1]-pose_real[:2, -1])
            z_dist = np.linalg.norm(pose_real_est[-1, -1]-pose_real[-1, -1])
            class_id = imdb_test.classes.index(pairdb[idx]['gt_class'])
            # store poses estimation and gt
            all_poses_est[class_id][0].append(pose_real_est)
            all_poses_gt[class_id][0].append(pairdb[idx]['pose_real'])
            all_rot_err[class_id][0].append(r_dist_est)
            all_trans_err[class_id][0].append(t_dist_est)
            xy_trans_err[class_id][0].append(xy_dist)
            z_trans_err[class_id][0].append(z_dist)

        all_trans_err = np.array(all_trans_err)
        imdb_test.evaluate_pose(config, all_poses_est, all_poses_gt, logger)
        pose_add_plots_dir = os.path.join(imdb_test.result_path, 'add_plots_before_ICP')
        mkdir_if_missing(pose_add_plots_dir)
        imdb_test.evaluate_pose_add(config, all_poses_est, all_poses_gt, output_dir=pose_add_plots_dir, logger=logger)
        pose_arp2d_plots_dir = os.path.join(imdb_test.result_path, 'arp_2d_plots_before_ICP')
        mkdir_if_missing(pose_arp2d_plots_dir)
        imdb_test.evaluate_pose_arp_2d(config, all_poses_est, all_poses_gt, output_dir=pose_arp2d_plots_dir,
                                       logger=logger)
        return

    # ------------------------------------------------------------------------------
    t_start = time.time()
    t = time.time()
    for idx, data_batch in enumerate(test_data):
        if np.sum(pairdb[idx]['pose_est']) == -12: # NO POINT VALID IN INIT POSE
            print(idx)
            class_id = imdb_test.classes.index(pairdb[idx]['gt_class'])
            for pose_iter_idx in range(config.TEST.test_iter):
                all_poses_est[class_id][pose_iter_idx].append(pairdb[idx]['pose_est'])
                all_poses_gt[class_id][pose_iter_idx].append(pairdb[idx]['pose_real'])

                r_dist = 1000
                t_dist = 1000
                all_rot_err[class_id][pose_iter_idx].append(r_dist)
                all_trans_err[class_id][pose_iter_idx].append(t_dist)
                sum_PoseErr[pose_iter_idx][class_id, :] += np.array([r_dist, t_dist])
                sum_PoseErr[pose_iter_idx][-1, :] += np.array([r_dist, t_dist])
                # post process
            if idx % 50 == 0:
                print_and_log('testing {}/{} data {:.4f}s net {:.4f}s calc_gt {:.4f}s'.format((idx + 1), num_pairs,
                                                                  data_time / (idx + 1) * test_data.batch_size,
                                                                  net_time / (idx + 1) * test_data.batch_size,
                                                                  post_time / (idx + 1) * test_data.batch_size),
                              logger)
            print("NO POINT_VALID IN rendered")
            continue
        data_time += time.time() - t

        t = time.time()

        pose_est = pairdb[idx]['pose_est']
        if np.sum(pose_est) == -12:
            print(idx)
            class_id = imdb_test.classes.index(pairdb[idx]['gt_class'])
            num_inst[class_id] += 1
            num_inst[-1] += 1
            for pose_iter_idx in range(config.TEST.test_iter):
                all_poses_est[class_id][pose_iter_idx].append(pose_est)
                all_poses_gt[class_id][pose_iter_idx].append(pairdb[idx]['pose_real'])

            # post process
            if idx % 50 == 0:
                print_and_log('testing {}/{} data {:.4f}s net {:.4f}s calc_gt {:.4f}s'.format((idx + 1), num_pairs,
                                                      data_time / (idx + 1) * test_data.batch_size,
                                                      net_time / (idx + 1) * test_data.batch_size,
                                                      post_time / (idx + 1) * test_data.batch_size),
                              logger)

            t = time.time()
            continue

        output_all = predictor.predict(data_batch)
        net_time += time.time() - t

        t = time.time()
        rst_iter = []
        for output in output_all:
            cur_rst = {}
            cur_rst['se3'] = np.squeeze(output['se3_output'].asnumpy()).astype('float32')

            if not config.TEST.FAST_TEST and config.network.PRED_FLOW:
                cur_rst['flow'] = np.squeeze(
                    output['flow_est_crop_output'].asnumpy().transpose((2,3,1,0))).astype('float16')
            else:
                cur_rst['flow'] = None
            if config.network.PRED_MASK and config.TEST.UPDATE_MASK not in ['init', 'box_rendered']:
                mask_pred = np.squeeze(output['mask_real_est_pred_output'].asnumpy()).astype('float32')
                cur_rst['mask_pred'] = mask_pred

            rst_iter.append(cur_rst)

        post_time += time.time() - t
        sample_ratio = 1 # 0.01
        for batch_idx in range(0, test_data.batch_size):
            # if config.TEST.VISUALIZE and not (r_dist>15 and t_dist>0.05):
            #     continue # 3388, 5326
            # calculate the flow error --------------------------------------------
            t = time.time()
            if config.network.PRED_FLOW and not config.TEST.FAST_TEST:
                # evaluate optical flow
                flow_gt = par_generate_gt(config, pairdb[idx])
                if config.network.PRED_FLOW:
                    all_diff = calc_EPE_one_pair(rst_iter[batch_idx], flow_gt, 'flow')
                sum_EPE_all += all_diff['epe_all']
                num_inst_all += all_diff['num_all']
                sum_EPE_viz += all_diff['epe_viz']
                num_inst_viz += all_diff['num_viz']
                sum_EPE_vizbg += all_diff['epe_vizbg']
                num_inst_vizbg += all_diff['num_vizbg']

            # calculate the se3 error ---------------------------------------------
            # evaluate se3 estimation
            pose_est = pairdb[idx]['pose_est']
            class_id = imdb_test.classes.index(pairdb[idx]['gt_class'])
            num_inst[class_id] += 1
            num_inst[-1] += 1
            post_time += time.time() - t

            # iterative refine se3 estimation --------------------------------------------------
            for pose_iter_idx in range(config.TEST.test_iter):
                t = time.time()
                pose_real_est = RT_transform(pose_est, rst_iter[0]['se3'][:-3], rst_iter[0]['se3'][-3:],
                                             config.dataset.trans_means, config.dataset.trans_stds,
                                             config.network.ROT_COORD)

                # calculate error
                r_dist, t_dist = calc_rt_dist_m(pose_real_est, pairdb[idx]['pose_real'])

                # store poses estimation and gt
                all_poses_est[class_id][pose_iter_idx].append(pose_real_est)
                all_poses_gt[class_id][pose_iter_idx].append(pairdb[idx]['pose_real'])

                all_rot_err[class_id][pose_iter_idx].append(r_dist)
                all_trans_err[class_id][pose_iter_idx].append(t_dist)
                sum_PoseErr[pose_iter_idx][class_id, :] += np.array([r_dist, t_dist])
                sum_PoseErr[pose_iter_idx][-1, :] += np.array([r_dist, t_dist])
                if config.TEST.VISUALIZE:
                    print("idx {}, iter {}: rError: {}, tError: {}".format(idx+batch_idx, pose_iter_idx+1, r_dist, t_dist))

                post_time += time.time() - t

                # # if more than one iteration
                if pose_iter_idx < (config.TEST.test_iter-1) or config.TEST.VISUALIZE:
                    t = time.time()
                    # get refined image
                    K_path = pairdb[idx]['image_real'][:-10] + '-K.txt'
                    if os.path.exists(K_path):
                        K = np.loadtxt(K_path)
                    image_refined, depth_refined = render(render_machine, pose_real_est,
                                                          config.dataset.class_name.index(pairdb[idx]['gt_class']), K=K)
                    image_refined = image_refined[:, :, :3]

                    # update minibatch
                    update_package = [{'image_rendered': image_refined,
                                       'src_pose': pose_real_est}]
                    if config.network.INPUT_DEPTH:
                        update_package[0]['depth_rendered'] = depth_refined
                    if config.network.INPUT_MASK:
                        mask_rendered_refined = np.zeros(depth_refined.shape)
                        mask_rendered_refined[depth_refined > 0.2] = 1
                        update_package[0]['mask_rendered'] = mask_rendered_refined
                        if config.network.PRED_MASK:
                            # init, box_rendered, mask_rendered, box_real, mask_real
                            if config.TEST.UPDATE_MASK == 'box_rendered':
                                input_names = [blob_name[0] for blob_name in data_batch.provide_data[0]]
                                update_package[0]['mask_real_est'] = np.squeeze(
                                    data_batch.data[0][input_names.index('mask_rendered')].asnumpy()[batch_idx])
                            elif config.TEST.UPDATE_MASK == 'box_real_est' or config.TEST.UPDATE_MASK == 'mask_real_est':
                                mask_pred = rst_iter[0]['mask_pred']
                                update_package[0]['mask_real_est'] = mask_pred
                            elif config.TEST.UPDATE_MASK == 'init':
                                pass
                            else:
                                raise Exception('Unknown UPDATE_MASK type: {}'.format(config.network.UPDATE_MASK))

                    pose_est = pose_real_est
                    data_batch = update_data_batch(config, data_batch, update_package)

                    data_time += time.time() - t

                    # forward and get rst
                    if pose_iter_idx < config.TEST.test_iter-1:
                        t = time.time()
                        # visualize_data_batch(config, data_batch, idx, pose_iter_idx)
                        output_all = predictor.predict(data_batch)
                        net_time += time.time() -t

                        t = time.time()
                        rst_iter = []
                        for output in output_all:
                            cur_rst = {}
                            if config.network.REGRESSOR_NUM == 1:
                                # rot_param = 3 if config.network.ROT_TYPE == "EULER" else 4
                                # rot = np.squeeze(output['se3_output'].asnumpy()).astype('float32')[:rot_param]
                                # trans = np.squeeze(output['se3_i2r_output'].asnumpy()).astype('float32')[-3:]
                                #
                                # cur_rst['se3'] = np.concatenate((rot, trans))
                                cur_rst['se3'] = np.squeeze(output['se3_output'].asnumpy()).astype('float32')


                            if not config.TEST.FAST_TEST and config.network.PRED_FLOW:
                                cur_rst['flow'] = np.squeeze(
                                    output['flow_est_crop_output'].asnumpy().transpose((2,3,1,0))).astype('float16')
                            else:
                                cur_rst['flow'] = None

                            if config.network.PRED_MASK and config.TEST.UPDATE_MASK not in ['init', 'box_rendered']:
                                mask_pred = np.squeeze(output['mask_real_est_pred_output'].asnumpy()).astype('float32')
                                cur_rst['mask_pred'] = mask_pred

                            rst_iter.append(cur_rst)
                            post_time += time.time() - t

        # post process
        if idx % 50 == 0:
            print_and_log('testing {}/{} data {:.4f}s net {:.4f}s calc_gt {:.4f}s'.format((idx+1), num_pairs,
                                                                               data_time / (idx+1) * test_data.batch_size,
                                                                               net_time / (idx+1) * test_data.batch_size,
                                                                               post_time / (idx+1) * test_data.batch_size),
                          logger)

        t = time.time()

    all_rot_err = np.array(all_rot_err)
    all_trans_err = np.array(all_trans_err)

    # save inference results
    if not config.TEST.VISUALIZE:
        with open(pose_err_file, 'wb') as f:
            print("saving result cache to {}".format(pose_err_file),)
            cPickle.dump([all_rot_err, all_trans_err, all_poses_est, all_poses_gt], f, protocol=2)
            print("done")

    if config.network.PRED_FLOW:
        print_and_log('evaluate flow:', logger)
        print_and_log('EPE all: {}'.format(sum_EPE_all / max(num_inst_all, 1.0)), logger)
        print_and_log('EPE ignore unvisible: {}'.format(sum_EPE_vizbg / max(num_inst_vizbg, 1.0)), logger)
        print_and_log('EPE visible: {}'.format(sum_EPE_viz / max(num_inst_viz, 1.0)), logger)

    print_and_log('evaluate pose:', logger)
    imdb_test.evaluate_pose(config, all_poses_est, all_poses_gt, logger)
    # evaluate pose add
    pose_add_plots_dir = os.path.join(imdb_test.result_path, 'add_plots')
    mkdir_if_missing(pose_add_plots_dir)
    imdb_test.evaluate_pose_add(config, all_poses_est, all_poses_gt, output_dir=pose_add_plots_dir, logger=logger)
    pose_arp2d_plots_dir = os.path.join(imdb_test.result_path, 'arp_2d_plots')
    mkdir_if_missing(pose_arp2d_plots_dir)
    imdb_test.evaluate_pose_arp_2d(config, all_poses_est, all_poses_gt, output_dir=pose_arp2d_plots_dir, logger=logger)

    print_and_log('using {} seconds in total'.format(time.time()-t_start), logger)

def visualize_data_batch(cfg, data_batch, idx, iter_idx):
    import matplotlib.pyplot as plt
    input_names = [blob_name[0] for blob_name in data_batch.provide_data[0]]
    blob_name = data_batch.provide_data[0][0][0]
    image_real = np.squeeze(data_batch.data[0][input_names.index('image_real')].asnumpy().transpose([2,3,1,0]))
    image_real[:, :, 0] += 103.939
    image_real[:, :, 1] += 116.779
    image_real[:, :, 2] += 123.68
    image_real = image_real[:, :, [2,1,0]]
    cv2.imwrite('iter/{:06}_{:02}_ir.jpg'.format(idx, iter_idx), image_real.astype('uint8'))
    image_rendered = np.squeeze(data_batch.data[0][input_names.index('image_rendered')].asnumpy().transpose([2,3,1,0]))
    image_rendered[:, :, 0] += 103.939
    image_rendered[:, :, 1] += 116.779
    image_rendered[:, :, 2] += 123.68
    image_rendered = image_rendered[:, :, [2,1,0]]
    cv2.imwrite('iter/{:06}_{:02}_ii.jpg'.format(idx, iter_idx), image_rendered.astype('uint8'))
    mask_real = np.squeeze(data_batch.data[0][input_names.index('mask_real_est')].asnumpy().transpose([2,3,1,0]))
    cv2.imwrite('iter/{:06}_{:02}_mr.jpg'.format(idx, iter_idx), (mask_real*255).astype('uint8'))
    mask_rendered = np.squeeze(data_batch.data[0][input_names.index('mask_rendered')].asnumpy().transpose([2,3,1,0]))
    cv2.imwrite('iter/{:06}_{:02}_mi.jpg'.format(idx, iter_idx), (mask_rendered*255).astype('uint8'))

def visualize_se3_flow(cfg, image_real, mesh_real, image_rendered, mesh_rendered, image_refined, mesh_est, pose_iter_idx):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax_im_real = fig.add_subplot(2,4,1)
    ax_im_real.set_title('image_rendered')
    plt.axis('off')
    plt.imshow(image_rendered)
    ax_im_rendered = fig.add_subplot(2,4,2)
    ax_im_rendered.set_title('mesh_rendered')
    plt.axis('off')
    plt.imshow(mesh_rendered)
    ax_d_real = fig.add_subplot(2,4,3)
    ax_d_real.set_title('image_refined_iter{}'.format(pose_iter_idx))
    plt.axis('off')
    plt.imshow(image_refined)
    ax_d_rendered = fig.add_subplot(2,4,4)
    if config.network.FLOW_I2R:
        ax_d_rendered.set_title('mesh_real_est_iter{}'.format(pose_iter_idx))
    elif config.network.FLOW_R2I:
        ax_d_rendered.set_title('mesh_rendered_est_iter{}'.format(pose_iter_idx))
    plt.axis('off')
    plt.imshow(mesh_est)
    ax_d_rendered = fig.add_subplot(2,4,5)
    ax_d_rendered.set_title('image_diff')
    plt.axis('off')
    plt.imshow(np.abs(image_real-image_refined))
    ax_d_rendered = fig.add_subplot(2,4,6)
    ax_d_rendered.set_title('mesh_diff')
    plt.axis('off')
    if config.network.FLOW_I2R:
        mask_real = (mesh_real[:, :, 0] != 0).astype(np.float32)
        mask_real_est = (mesh_est[:, :, 0] != 0).astype(np.float32)
        mesh_diff = np.dstack((mask_real_est, mask_real, mask_real_est))
    elif config.network.FLOW_R2I:
        mask_rendered = (mesh_rendered[:, :, 0] != 0).astype(np.float32)
        mask_rendered_est = (mesh_est[:, :, 0] != 0).astype(np.float32)
        mesh_diff = np.dstack((mask_rendered_est, mask_rendered, mask_rendered_est))
    plt.imshow(mesh_diff)
    ax_d_rendered = fig.add_subplot(2,4,7)
    ax_d_rendered.set_title('image_real')
    plt.axis('off')
    plt.imshow(image_real)
    ax_d_rendered = fig.add_subplot(2,4,8)
    ax_d_rendered.set_title('mesh_real')
    plt.axis('off')
    plt.imshow(mesh_real)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

def visualize_se3(cfg, image_real, depth_real, image_rendered, depth_rendered, image_refined, depth_refined, pose_iter_idx):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax_im_real = fig.add_subplot(2,4,1)
    ax_im_real.set_title('image_rendered')
    plt.axis('off')
    plt.imshow(image_rendered)
    ax_d_real = fig.add_subplot(2,4,3)
    ax_d_real.set_title('image_refined_iter{}'.format(pose_iter_idx))
    plt.axis('off')
    plt.imshow(image_refined)
    ax_d_rendered = fig.add_subplot(2,4,5)
    ax_d_rendered.set_title('image_diff')
    plt.axis('off')
    plt.imshow(np.abs(image_real-image_refined))
    ax_d_rendered = fig.add_subplot(2,4,7)
    ax_d_rendered.set_title('image_real')
    plt.axis('off')
    plt.imshow(image_real)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

def visualize_se3_cv2(cfg, image_real, image_refined, save_path, legend):
    import matplotlib.pyplot as plt
    show_img = image_real - image_refined
    show_img = show_img[:, :, [2,1,0]]
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(show_img)
    fig.gca().text(10, 50, legend, color='white')
    plt.savefig(save_path, aspect='normal')

def visualize_mask(cfg, zoom_image_real, image_real, zoom_mask_prob, mask_pred, mask_real,
                   image_refined, save_path, legend, title=None, invert_color=False):
    import matplotlib.pyplot as plt

    zoom_mask_pred_bin = np.zeros(zoom_mask_prob.shape)
    zoom_mask_pred_bin[zoom_mask_prob >= 0.5] = 1.0

    fig = plt.figure(frameon=False, figsize=(8, 6), dpi=100)
    fig.add_subplot(2,3,1)
    plt.imshow(zoom_image_real[:, :, [2, 1, 0]])
    plt.title('zoom_image_real')

    fig.add_subplot(2,3,2)
    plt.imshow(zoom_mask_pred_bin)
    plt.title('zoom_mask_pred')

    fig.add_subplot(2,3,3)
    plt.imshow(image_real)
    plt.title('image_real')

    fig.add_subplot(2,3,4)
    plt.imshow(mask_pred)
    plt.title('mask_pred')

    fig.add_subplot(2, 3, 5)
    plt.imshow(mask_real)
    plt.title('mask_real')

    fig.gca().text(10, 25, title, color='green', bbox=dict(facecolor='white', alpha=0.8))
    fig.gca().text(10, 70, legend, color='red', bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(save_path, aspect='normal')

def visualize_flow(cfg, pair_rec, flow_i2r_est, image_src, depth_src, pose_src,
                   image_tgt, depth_tgt, pose_tgt, visible_est=None):
    import matplotlib.pyplot as plt
    K = cfg.dataset.INTRINSIC_MATRIX

    flow, visible, _ = calc_flow(depth_src, pose_src, pose_tgt, K, depth_tgt, standard_rep=config.network.STANDARD_FLOW_REP)
    visible = visible
    def kernel_calc_epe(flow_pred, flow_gt, visible):
        h_diff = (flow_gt[:, :, 0] - flow_pred[:, :, 0])[visible != 0]
        w_diff = (flow_gt[:, :, 1] - flow_pred[:, :, 1])[visible != 0]
        diff = np.sqrt(np.square(h_diff) + np.square(w_diff))
        print(flow_gt.shape, diff.shape, flow_pred.shape, visible.shape, np.sum(visible), np.max(depth_tgt), np.max(depth_src))
        return diff.sum()/np.logical_and(flow_gt[:,:,0]!=0, flow_gt[:,:,1]!=0).sum()

    r_dist, t_dist = calc_rt_dist_m(pose_src, pose_tgt)
    print('epe: {}, rotation dist: {}, translation dist: {}'.format(kernel_calc_epe(flow_i2r_est, flow, visible), r_dist, t_dist))

    fig = plt.figure()
    ax_im_src = fig.add_subplot(3,4,1)
    ax_im_src.set_title('image_src')
    plt.axis('off')
    plt.imshow(image_src)

    ax_im_tgt = fig.add_subplot(3,4,2)
    ax_im_tgt.set_title('image_tgt')
    plt.axis('off')
    plt.imshow(image_tgt)

    ax_d_src = fig.add_subplot(3,4,3)
    ax_d_src.set_title('depth_src')
    plt.axis('off')
    plt.imshow(depth_src)

    ax_d_tgt = fig.add_subplot(3,4,4)
    ax_d_tgt.set_title('depth_tgt')
    plt.axis('off')
    plt.imshow(depth_tgt)

    height = depth_tgt.shape[0]
    width = depth_tgt.shape[1]
    mesh_tgt = np.zeros((height, width, 3), np.uint8)
    mesh_src = np.zeros((height, width, 3), np.uint8)
    mesh_tgt_est = np.zeros((height, width, 3), np.uint8)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x*20))
    for h in range(height):
        for w in range(width):
            if visible[h, w]:
                cur_flow = flow[h,w,:].flatten()
                if not config.network.STANDARD_FLOW_REP:
                    cur_flow = cur_flow[[1,0]]
                cur_flow_est = flow_i2r_est[h,w,:]
                if not config.network.STANDARD_FLOW_REP:
                    cur_flow_est = cur_flow_est[[1,0]]
                point_color = [sigmoid(float(h)/height-0.5)*200+50, sigmoid(0.5-float(w)/width)*200+50,
                               sigmoid(float(h)/height+float(w)/width-1)*200+50]

                mesh_src = cv2.circle(mesh_src, (np.round(w).astype(int), np.round(h).astype(int)),
                                       1, point_color)
                mesh_tgt = cv2.circle(mesh_tgt, (np.round(w + cur_flow[0]).astype(int), np.round(h + cur_flow[1]).astype(int)),
                                   1, point_color)
                mesh_tgt_est = cv2.circle(mesh_tgt_est, (np.round(w + cur_flow_est[0]).astype(int),
                                                         np.round(h + cur_flow_est[1]).astype(int)),
                                          1, point_color)
    weight = np.dstack((visible, visible))


    ax_m_src = fig.add_subplot(3,4,5)
    ax_m_src.set_title('mesh_src')
    plt.axis('off')
    plt.imshow(mesh_src)

    ax_m_tgt = fig.add_subplot(3,4,6)
    ax_m_tgt.set_title('mesh_tgt_gt')
    plt.axis('off')
    plt.imshow(mesh_tgt)

    ax_m_tgt_est = fig.add_subplot(3,4,7)
    ax_m_tgt_est.set_title('mesh_tgt_est')
    plt.axis('off')
    sintel_vis = sintel_compute_color(np.vstack((flow[:,:,[1,0]], flow_i2r_est[:,:,[1,0]]*weight)))
    flow_i2r_vis = sintel_vis[:sintel_vis.shape[0]/2, :, :]
    flow_i2r_est_vis = sintel_vis[sintel_vis.shape[0]/2:, :, :]
    plt.imshow(mesh_tgt_est)

    ax_m_diff = fig.add_subplot(3,4,8)
    ax_m_diff.set_title('mesh_diff')
    plt.axis('off')
    mask_real = (mesh_tgt[:, :, 0] != 0).astype(np.float32)
    mask_real_est = (mesh_tgt_est[:, :, 0] != 0).astype(np.float32)
    mesh_diff = np.dstack((mask_real_est, mask_real, mask_real_est))
    plt.imshow(mesh_diff)

    ax_flow_i2r = fig.add_subplot(3,4,9)
    ax_flow_i2r.set_title('flow_i2r')
    plt.imshow(flow_i2r_vis)
    plt.axis('off')

    ax_flow_i2r_est = fig.add_subplot(3,4,10)
    ax_flow_i2r_est.set_title('flow_i2r_est')
    plt.imshow(flow_i2r_est_vis)
    plt.axis('off')

    if not (visible_est is None):
        ax_visible_i2r_diff_est = fig.add_subplot(3, 4, 11)
        ax_visible_i2r_diff_est.set_title('visible_est_diff')
        plt.imshow(np.dstack((visible_est, visible, visible_est)).astype(np.float32))
        plt.axis('off')

        ax_visible_i2r_est = fig.add_subplot(3, 4, 12)
        ax_visible_i2r_est.set_title('visible_est')
        plt.imshow(np.dstack((visible_est, visible_est, visible_est)).astype(np.float32))
        plt.axis('off')

    ax_rendered_list = {'ax_image_rendered': ax_im_src,
                    'ax_depth_rendered': ax_d_src,
                    'ax_mesh_rendered': ax_m_src,
                    'ax_flow_i2r': ax_flow_i2r,
                    'ax_flow_i2r_est': ax_flow_i2r_est}
    ax_real_list = {'ax_image_real': ax_im_tgt,
                    'ax_depth_real': ax_d_tgt,
                    'ax_mesh_real': ax_m_tgt,
                    'ax_mesh_real_est': ax_m_tgt_est}
    data = {'flow_i2r': flow,
            'visible': visible,
            'flow_i2r_est': flow_i2r_est}
    cid = fig.canvas.mpl_connect('button_press_event', get_onclick(ax_real_list, ax_rendered_list, data, fig))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    # plt.show()

    return mesh_src, mesh_tgt, mesh_tgt_est

def par_generate_gt(config, pair_rec, flow_depth_rendered=None):
    from lib.pair_matching.flow import calc_flow
    target_size, max_size = config.SCALES[0][0], config.SCALES[0][1]

    if flow_depth_rendered is None:
        flow_depth_rendered = cv2.imread(pair_rec['depth_rendered'],
                                     cv2.IMREAD_UNCHANGED).astype(np.float32)
        flow_depth_rendered /= config.dataset.DEPTH_FACTOR

        flow_depth_rendered, _ = resize(flow_depth_rendered, target_size, max_size)

    if 'depth_render_real' in pair_rec:
        flow_depth_real = cv2.imread(pair_rec['depth_render_real'],
                                 cv2.IMREAD_UNCHANGED).astype(np.float32)
        flow_depth_real /= config.dataset.DEPTH_FACTOR
    else:
        print('not using render_real depth in par_generate_gt')
        flow_depth_real = cv2.imread(pair_rec['depth_real'],
                                 cv2.IMREAD_UNCHANGED).astype(np.float32)
        flow_depth_real /= config.dataset.DEPTH_FACTOR

    flow_depth_real, _ = resize(flow_depth_real, target_size, max_size)

    if 'mask_real_gt' or 'mask_real_est' in pair_rec:
        mask_real_path = pair_rec['mask_real_gt']
        assert os.path.exists(mask_real_path), '%s does not exist'.format(pair_rec['mask_real_gt'])
        mask_real = cv2.imread(mask_real_path, cv2.IMREAD_UNCHANGED)
        mask_real, _ = resize(mask_real, target_size, max_size)
        flow_depth_real[mask_real != pair_rec['mask_idx']] = 0

    if config.network.FLOW_I2R:
        if config.FLOW_CLASS_AGNOSTIC:
            flow_i2r, visible, _ = calc_flow(flow_depth_rendered, pair_rec['pose_est'], pair_rec['pose_real'],
                                             config.dataset.INTRINSIC_MATRIX, flow_depth_real,
                                             standard_rep=config.network.STANDARD_FLOW_REP)
            flow_i2r_list = [flow_i2r, visible, np.logical_and(visible == 0, flow_depth_rendered == 0)]
        else:
            raise Exception('NOT_IMPLEMENTED')
    else:
        flow_i2r_list = None

    if config.network.FLOW_R2I:
        if config.FLOW_CLASS_AGNOSTIC:
            flow_r2i, visible, _ = calc_flow(flow_depth_real, pair_rec['pose_real'], pair_rec['pose_est'],
                                             config.dataset.INTRINSIC_MATRIX, flow_depth_rendered,
                                             standard_rep=config.network.STANDARD_FLOW_REP)
            flow_r2i_list = [flow_r2i, visible, np.logical_and(visible == 0, flow_depth_real == 0)]
        else:
            raise Exception('NOT_IMPLEMENTED')
    else:
        flow_r2i_list = None
    return {'flow_i2r': flow_i2r_list, 'flow_r2i': flow_r2i_list}

def calc_EPE_one_pair(flow_pred_list, flow_gt, flow_type):
    cur_flow_pred = flow_pred_list[flow_type]
    cur_flow_gt = flow_gt[flow_type][0]
    visible = flow_gt[flow_type][1]
    bg = flow_gt[flow_type][2]

    x_diff = cur_flow_gt[:, :, 0] - cur_flow_pred[:, :, 0]
    y_diff = cur_flow_gt[:, :, 1] - cur_flow_pred[:, :, 1]
    point_diff = np.sqrt(np.square(x_diff) + np.square(y_diff))
    all_diff = {'epe_all': point_diff.sum(),
                'num_all': point_diff.size,
                'epe_viz': point_diff[visible==1].sum(),
                'num_viz': visible.sum(),
                'epe_vizbg': point_diff[np.logical_or(visible, bg)].sum(),
                'num_vizbg': np.logical_or(visible, bg).sum()}
    return all_diff

def get_onclick(ax_real_list, ax_rendered_list, data, fig):
    def onclick(event):
        point_rendered = np.round([event.xdata, event.ydata]).astype(int)
        print('point_rendered: {}'.format(point_rendered))
        color_list = ['b', 'c', 'y', 'm', 'r']
        color = color_list[np.random.randint(len(color_list))]
        flow_i2r = data['flow_i2r']
        flow_i2r_est = data['flow_i2r_est']
        for sub_ax_rendered in ax_rendered_list:
            ax_rendered_list[sub_ax_rendered].scatter(point_rendered[0], point_rendered[1], c=color, s=7, marker='.')
        cur_flow_gt = flow_i2r[point_rendered[1], point_rendered[0], :].flatten()
        if not config.network.STANDARD_FLOW_REP:
            cur_flow_gt = cur_flow_gt[[1, 0]]
        point_real = [point_rendered[0]+cur_flow_gt[0],
                      point_rendered[1]+cur_flow_gt[1]]
        print('point_real: {}'.format(point_real))

        for sub_ax_real in ax_real_list:
            ax_real_list[sub_ax_real].scatter(point_real[0], point_real[1], c=color, s=7, marker='.')
        cur_flow_est = flow_i2r_est[point_rendered[1], point_rendered[0], :].flatten()
        if not config.network.STANDARD_FLOW_REP:
            cur_flow_est = cur_flow_est[[1, 0]]
        point_real_est = [point_rendered[0] + cur_flow_est[0],
                          point_rendered[1] + cur_flow_est[1]]
        print('point_real_est: {}'.format(point_real_est))
        for sub_ax_real in ax_real_list:
            ax_real_list[sub_ax_real].scatter(point_real_est[0], point_real_est[1], c=color, s=7, marker='x')
        fig.canvas.draw()
    return onclick
