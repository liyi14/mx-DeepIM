# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
from six.moves import xrange
import numpy as np
from lib.utils.image import *
import lib.utils.image as image
from lib.pair_matching.RT_transform import calc_RT_delta, mat2euler, mat2quat
from lib.utils.tictoc import tic, toc


def get_data_pair_test_batch(pairdb, config):
    """
    return a dict of train batch
    :param segdb: ['image', 'flipped']
    :param config: the config setting
    :return: data, label, im_info
    """
    im_observed, im_rendered, scale_ind_list = get_pair_image(
        pairdb, config, 'test')
    if config.network.INPUT_DEPTH:
        depth_observed, depth_rendered = get_pair_depth(
            pairdb, config, scale_ind_list, 'test')
    if config.network.INPUT_MASK:
        mask_observed, _, mask_rendered = get_pair_mask(
            pairdb, config, scale_ind_list, 'test')

    im_info = [
        np.array([pairdb[i]['height'], pairdb[i]['width']], dtype=np.float32)
        for i in xrange(len(pairdb))
    ]

    num_pair = len(pairdb)
    for i in range(num_pair):
        class_index_tensor = [[] for i in xrange(num_pair)]
        class_index_tensor[i] = np.array(
            config.dataset.class_name.index(pairdb[i]['gt_class'])).reshape(1)
        class_index_array = my_tensor_vstack(class_index_tensor)

    data = []
    for i in xrange(len(pairdb)):
        cur_batch = {
            'image_observed': im_observed[i],
            'image_rendered': im_rendered[i],
            'src_pose': np.array(pairdb[i]['pose_rendered']).reshape((1, 3,
                                                                      4)),
            'class_index': class_index_array
        }

        if config.network.INPUT_DEPTH:
            cur_batch['depth_observed'] = depth_observed[i]
            cur_batch['depth_rendered'] = depth_rendered[i]

        if config.network.INPUT_MASK:
            cur_batch['mask_observed'] = mask_observed[i]
            cur_batch['mask_rendered'] = mask_rendered[i]

        data.append(cur_batch)
    label = {}

    return data, label, im_info


def update_data_batch(config, data_batch, update_package):
    import mxnet.ndarray as nd
    for ctx_idx, data in enumerate(data_batch.data):
        package = update_package[ctx_idx]
        for blob_idx, blob in enumerate(data):
            blob_name = data_batch.provide_data[ctx_idx][blob_idx][0]
            blob_shape = data_batch.provide_data[ctx_idx][blob_idx][1]
            if blob_name not in package:
                continue
            update_data = package[blob_name]
            if blob_name.startswith('image'):
                target_size = min(blob_shape[2:])
                max_size = max(blob_shape[2:])
                image_update, _ = image.resize(update_data, target_size,
                                               max_size)
                image_update = image.transform(image_update,
                                               config.network.PIXEL_MEANS)
                image_update = nd.array(image_update)
                data_batch.data[ctx_idx][blob_idx] = image_update
            elif blob_name.startswith('src_pose'):
                src_pose_update = nd.array(update_data[np.newaxis, :, :])
                data_batch.data[ctx_idx][blob_idx] = src_pose_update
            elif blob_name.startswith('depth'):
                target_size = min(blob_shape[2:])
                max_size = max(blob_shape[2:])
                depth_update, _ = image.resize(update_data, target_size,
                                               max_size)
                depth_update = nd.array(
                    depth_update[np.newaxis, np.newaxis, :, :])
                data_batch.data[ctx_idx][blob_idx] = depth_update
            elif blob_name.startswith('mask_observed'):
                if config.TEST.UPDATE_MASK == 'box_rendered':
                    update_data = np.copy(package['mask_rendered'])
                    mask_observed = np.zeros(update_data.shape)
                    x_max = np.max(update_data, 0)
                    y_max = np.max(update_data, 1)
                    nz_x = np.nonzero(x_max)[0]
                    nz_y = np.nonzero(y_max)[0]
                    x_start = np.min(nz_x)
                    x_end = np.max(nz_x)
                    y_start = np.min(nz_y)
                    y_end = np.max(nz_y)
                    mask_observed[y_start:y_end, x_start:
                                  x_end] = 1.  # rectangle
                elif config.TEST.UPDATE_MASK == 'box_observed':
                    mask_observed = np.zeros(update_data.shape)
                    x_max = np.max(update_data, 0)
                    y_max = np.max(update_data, 1)
                    nz_x = np.nonzero(x_max)[0]
                    nz_y = np.nonzero(y_max)[0]
                    if len(nz_x) == 0 or len(nz_y) == 0:
                        raise Exception("no point valid in mask_observed")

                    x_start = np.min(nz_x)
                    x_end = np.max(nz_x)
                    y_start = np.min(nz_y)
                    y_end = np.max(nz_y)
                    mask_observed[y_start:y_end, x_start:
                                  x_end] = 1.  # rectangle
                else:
                    mask_observed = update_data
                mask_update = nd.array(
                    mask_observed[np.newaxis, np.newaxis, :, :])
                data_batch.data[ctx_idx][blob_idx] = mask_update
            elif blob_name.startswith('mask_rendered'):
                mask_update = nd.array(
                    update_data[np.newaxis, np.newaxis, :, :])
                data_batch.data[ctx_idx][blob_idx] = mask_update
            else:
                raise Exception("NOT_IMPLEMENTED")
    return data_batch


point_clound_dict = []


def get_data_pair_train_batch(pairdb, config):
    """
    return a dict of train batch
    :param pairdb: ['image_observed', 'image_rendered', 'height', 'width',
                    'depth_observed', 'depth_rendered', 'pose_observed', 'pose_rendered',
                    'flipped']
    :param config: ['INTRINSIC_MATRIX', 'DEPTH_FACTOR', ...]
    :return: data: ['image_observed', 'image_rendered', 'depth_observed', 'depth_rendered']
             label: ['flow', 'flow_weights'(RT)]
    """
    num_pair = len(pairdb)
    random_k = np.random.randint(18)
    im_observed, im_rendered, scale_ind_list = get_pair_image(
        pairdb, config, phase='train', random_k=random_k)
    depth_gt_observed = get_gt_observed_depth(
        pairdb, config, scale_ind_list, random_k=random_k)
    if config.network.INPUT_DEPTH:
        depth_observed, depth_rendered = get_pair_depth(
            pairdb, config, scale_ind_list, phase='train', random_k=random_k)
    if config.network.PRED_MASK or config.network.INPUT_MASK:
        mask_observed, mask_gt_observed, mask_rendered = get_pair_mask(
            pairdb, config, scale_ind_list, phase='train', random_k=random_k)
    im_observed_array = my_tensor_vstack(im_observed)
    im_rendered_array = my_tensor_vstack(im_rendered)

    depth_gt_observed_array = my_tensor_vstack(depth_gt_observed)
    if config.network.INPUT_DEPTH:
        depth_observed_array = my_tensor_vstack(depth_observed)
        depth_rendered_array = my_tensor_vstack(depth_rendered)
    if config.network.INPUT_MASK or config.network.PRED_MASK:
        mask_observed_array = my_tensor_vstack(mask_observed)
        mask_gt_observed_array = my_tensor_vstack(mask_gt_observed)
        mask_rendered_array = my_tensor_vstack(mask_rendered)

    if config.network.PRED_FLOW:
        flow_i2r, flow_i2r_weights, X_rendered_tensor, X_observed_tensor \
            = get_pair_flow(pairdb, config, scale_ind_list, phase='train', random_k=random_k)
        flow_i2r_array = my_tensor_vstack(flow_i2r)
        flow_i2r_weights_array = my_tensor_vstack(flow_i2r_weights)

    if config.train_iter.SE3_PM_LOSS:
        X_obj, X_obj_weights = get_point_cloud_model(config, pairdb)
        X_obj_array = my_tensor_vstack(X_obj)
        X_obj_weights_array = my_tensor_vstack(X_obj_weights)

    # SE3
    rot_i2r_tensor = [[] for i in range(num_pair)]
    trans_i2r_tensor = [[] for i in range(num_pair)]
    src_pose_tensor = [[] for i in range(num_pair)]
    tgt_pose_tensor = [[] for i in range(num_pair)]
    tgt_points_tensor = [[] for i in range(num_pair)]
    for i in range(num_pair):
        rot_i2r, trans_i2r = calc_RT_delta(
            pairdb[i]['pose_rendered'], pairdb[i]['pose_observed'],
            config.dataset.trans_means, config.dataset.trans_stds,
            config.network.ROT_COORD, config.network.ROT_TYPE)

        rot_i2r_tensor[i] = np.array(rot_i2r).reshape((1, -1))
        trans_i2r_tensor[i] = np.array(trans_i2r).reshape((1, -1))
        src_pose_tensor[i] = np.array(pairdb[i]['pose_rendered']).reshape(
            (1, 3, 4))
        tgt_pose_tensor[i] = np.array(pairdb[i]['pose_observed']).reshape(
            (1, 3, 4))
        if config.train_iter.SE3_PM_LOSS:
            tgt_points = get_point_cloud_observed(
                config,
                X_obj_array[i],
                pose_observed=np.array(pairdb[i]['pose_observed']))
            tgt_points_tensor[i] = np.expand_dims(
                tgt_points, axis=0)  # (1, 3, 3000)

    rot_i2r_array = my_tensor_vstack(rot_i2r_tensor)
    trans_i2r_array = my_tensor_vstack(trans_i2r_tensor)
    src_pose_array = my_tensor_vstack(src_pose_tensor)  # before refinment
    tgt_pose_array = my_tensor_vstack(tgt_pose_tensor)  # refinement target
    tgt_points_array = my_tensor_vstack(tgt_points_tensor)

    for i in range(num_pair):
        class_index_tensor = [[] for i in range(num_pair)]
        class_index_tensor[i] = np.array(
            config.dataset.class_name.index(
                pairdb[i]['gt_class'])).reshape(num_pair)
        class_index_array = my_tensor_vstack(class_index_tensor)
    data = {
        'image_observed': im_observed_array,
        'image_rendered': im_rendered_array,
        'depth_gt_observed': depth_gt_observed_array,
        'class_index': class_index_array,
        'src_pose': src_pose_array,
        'tgt_pose': tgt_pose_array
    }

    if config.network.INPUT_DEPTH:
        data['depth_observed'] = depth_observed_array
        data['depth_rendered'] = depth_rendered_array

    if config.network.INPUT_MASK:
        data['mask_observed'] = mask_observed_array
        data['mask_rendered'] = mask_rendered_array

    label = {'rot': rot_i2r_array, 'trans': trans_i2r_array}

    if config.network.PRED_MASK:
        label['mask_gt_observed'] = mask_gt_observed_array

    if config.network.PRED_FLOW:
        label['flow'] = flow_i2r_array
        label['flow_weights'] = flow_i2r_weights_array

    if config.train_iter.SE3_PM_LOSS:
        label['point_cloud_model'] = X_obj_array
        label['point_cloud_weights'] = X_obj_weights_array
        label['point_cloud_observed'] = tgt_points_array

    return {'data': data, 'label': label}
