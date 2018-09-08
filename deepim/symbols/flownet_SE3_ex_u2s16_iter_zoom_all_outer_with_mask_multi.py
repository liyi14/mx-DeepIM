# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import mxnet as mx
import numpy as np
from lib.utils.symbol import Symbol
from operator_py.transform3d import *
from operator_py.flow_updater import *

from operator_py.zoom_image import *
from operator_py.zoom_flow import *
from operator_py.zoom_trans import *
from operator_py.zoom_depth import *
from operator_py.zoom_mask import *
from operator_py.zoom_image_with_factor import *
from operator_py.zoom_mask_with_factor import *
from operator_py.group_picker import *


class flownet_SE3_ex_u2s16_iter_zoom_all_outer_with_mask_multi(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.workspace = 4096
        self.units = (2, 3, 3, 3, 3)
        self.filter_list = [64, 128, 256, 512]

    def get_flownet(self, data_iter, big_cfg, small_cfg, share_dict, iter_idx):
        if 'depth_real' in data_iter.keys():
            if big_cfg.network.INPUT_MASK and big_cfg.network.PRED_MASK:
                data = mx.symbol.Concat(data_iter['image_real'] / 255.0, data_iter['image_rendered'] / 255.0,
                                        data_iter['depth_real'] / 255.0, data_iter['depth_rendered'] / 255.0,
                                        data_iter['mask_real_est'],
                                        data_iter['mask_rendered'],
                                        dim=1)
            else:
                data = mx.symbol.Concat(data_iter['image_real']/255.0, data_iter['image_rendered']/255.0,
                                    data_iter['depth_real']/255.0, data_iter['depth_rendered']/255.0, dim=1)
        else:
            if big_cfg.network.INPUT_MASK and big_cfg.network.PRED_MASK:
                data = mx.symbol.Concat(data_iter['image_real'] / 255.0, data_iter['image_rendered'] / 255.0,
                                        data_iter['mask_real_est'],
                                        data_iter['mask_rendered'],
                                        dim=1)
            else:
                data = mx.symbol.Concat(data_iter['image_real'] / 255.0, data_iter['image_rendered'] / 255.0, dim=1)
        flow_conv1 = mx.symbol.Convolution(name='flow_conv1_iter{}'.format(iter_idx), data=data, num_filter=64, pad=(3, 3),
                                           kernel=(7, 7), stride=(2, 2), no_bias=False,
                                           weight=share_dict['flow_conv1_weight'], bias=share_dict['flow_conv1_bias'])
        ReLU1 = mx.symbol.LeakyReLU(name='ReLU1_iter{}'.format(iter_idx), data=flow_conv1, act_type='leaky', slope=0.1)
        # scale 4
        conv2 = mx.symbol.Convolution(name='conv2_iter{}'.format(iter_idx), data=ReLU1, num_filter=128, pad=(2, 2),
                                      kernel=(5, 5), stride=(2, 2), no_bias=False, weight=share_dict['conv2_weight'],
                                      bias=share_dict['conv2_bias'])
        ReLU2 = mx.symbol.LeakyReLU(name='ReLU2_iter{}'.format(iter_idx), data=conv2, act_type='leaky', slope=0.1)
        # scale 8
        conv3 = mx.symbol.Convolution(name='conv3_iter{}'.format(iter_idx), data=ReLU2, num_filter=256, pad=(2, 2),
                                      kernel=(5, 5), stride=(2, 2), no_bias=False, weight=share_dict['conv3_weight'],
                                      bias=share_dict['conv3_bias'])
        ReLU3 = mx.symbol.LeakyReLU(name='ReLU3_iter{}'.format(iter_idx), data=conv3, act_type='leaky', slope=0.1)
        conv3_1 = mx.symbol.Convolution(name='conv3_1_iter{}'.format(iter_idx), data=ReLU3, num_filter=256, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1), no_bias=False,
                                        weight=share_dict['conv3_1_weight'], bias=share_dict['conv3_1_bias'])
        ReLU4 = mx.symbol.LeakyReLU(name='ReLU4_iter{}'.format(iter_idx), data=conv3_1, act_type='leaky', slope=0.1)
        # scale 16
        conv4 = mx.symbol.Convolution(name='conv4_iter{}'.format(iter_idx), data=ReLU4, num_filter=512, pad=(1, 1),
                                      kernel=(3, 3), stride=(2, 2), no_bias=False, weight=share_dict['conv4_weight'],
                                      bias=share_dict['conv4_bias'])
        ReLU5 = mx.symbol.LeakyReLU(name='ReLU5_iter{}'.format(iter_idx), data=conv4, act_type='leaky', slope=0.1)
        conv4_1 = mx.symbol.Convolution(name='conv4_1_iter{}'.format(iter_idx), data=ReLU5, num_filter=512, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1), no_bias=False,
                                        weight=share_dict['conv4_1_weight'], bias=share_dict['conv4_1_bias'])
        ReLU6 = mx.symbol.LeakyReLU(name='ReLU6_iter{}'.format(iter_idx), data=conv4_1, act_type='leaky', slope=0.1)
        # scale 32
        conv5 = mx.symbol.Convolution(name='conv5_iter{}'.format(iter_idx), data=ReLU6, num_filter=512, pad=(1, 1),
                                      kernel=(3, 3), stride=(2, 2), no_bias=False, weight=share_dict['conv5_weight'],
                                      bias=share_dict['conv5_bias'])
        ReLU7 = mx.symbol.LeakyReLU(name='ReLU7_iter{}'.format(iter_idx), data=conv5, act_type='leaky', slope=0.1)
        conv5_1 = mx.symbol.Convolution(name='conv5_1_iter{}'.format(iter_idx), data=ReLU7, num_filter=512, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1), no_bias=False,
                                        weight=share_dict['conv5_1_weight'], bias=share_dict['conv5_1_bias'])
        ReLU8 = mx.symbol.LeakyReLU(name='ReLU8_iter{}'.format(iter_idx), data=conv5_1, act_type='leaky', slope=0.1)
        # scale 64
        conv6 = mx.symbol.Convolution(name='conv6_iter{}'.format(iter_idx), data=ReLU8, num_filter=1024, pad=(1, 1),
                                      kernel=(3, 3), stride=(2, 2), no_bias=False, weight=share_dict['conv6_weight'],
                                      bias=share_dict['conv6_bias'])
        ReLU9 = mx.symbol.LeakyReLU(name='ReLU9_iter{}'.format(iter_idx), data=conv6, act_type='leaky', slope=0.1)
        conv6_1 = mx.symbol.Convolution(name='conv6_1_iter{}'.format(iter_idx), data=ReLU9, num_filter=1024, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1), no_bias=False,
                                        weight=share_dict['conv6_1_weight'], bias=share_dict['conv6_1_bias'])
        ReLU10 = mx.symbol.LeakyReLU(name='ReLU10_iter{}'.format(iter_idx), data=conv6_1, act_type='leaky', slope=0.1)

        # se3
        flatten_0 = mx.symbol.Flatten(name='flatten_0_iter{}'.format(iter_idx), data=ReLU10)

        fc6 = mx.symbol.FullyConnected(name='fc6_iter{}'.format(iter_idx), data=flatten_0, num_hidden=256, no_bias=False,
                                       weight=share_dict['fc6_weight'], bias=share_dict['fc6_bias'])
        relu_fc6 = mx.symbol.LeakyReLU(name='ReLU11_iter{}'.format(iter_idx), data=fc6, act_type='leaky', slope=0.1)

        fc7 = mx.symbol.FullyConnected(name='fc7_iter{}'.format(iter_idx), data=relu_fc6, num_hidden=256, no_bias=False,
                                       weight=share_dict['fc7_weight'], bias=share_dict['fc7_bias'])
        relu_fc7 = mx.symbol.LeakyReLU(name='ReLU12_iter{}'.format(iter_idx), data=fc7, act_type='leaky', slope=0.1)

        feat_list = [relu_fc7, None]

        # flow branch
        if big_cfg.network.PRED_FLOW or big_cfg.network.PRED_MASK:
            # scale 64
            Convolution1 = mx.symbol.Convolution(name='Convolution1_iter{}'.format(iter_idx), data=ReLU10, num_filter=2,
                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False,
                 weight=share_dict['Convolution1_weight'], bias=share_dict['Convolution1_bias']) #(B,2,8,10)
            # scale 32
            deconv5 = mx.symbol.Deconvolution(name='deconv5_iter{}'.format(iter_idx), data=ReLU10, num_filter=512,
                pad=(0, 0), kernel=(4, 4), stride=(2, 2), no_bias=False,
                weight=share_dict['deconv5_weight'], bias=share_dict['deconv5_bias']) # (B,512, 18, 22)
            crop_deconv5 = mx.symbol.Crop(name='crop_deconv5_iter{}'.format(iter_idx), *[deconv5, ReLU8], offset=(1, 1)) #(B,512,15,20)
            ReLU11 = mx.symbol.LeakyReLU(name='ReLU11_iter{}'.format(iter_idx), data=crop_deconv5, act_type='leaky',
                                         slope=0.1)
            upsample_flow6to5 = mx.symbol.Deconvolution(name='upsample_flow6to5_iter{}'.format(iter_idx),
                                                        data=Convolution1, num_filter=2, pad=(0, 0), kernel=(4, 4),
                                                        stride=(2, 2), no_bias=False,
                                                        weight=share_dict['upsample_flow6to5_weight'],
                                                        bias=share_dict['upsample_flow6to5_bias'])
            crop_upsampled_flow6_to_5 = mx.symbol.Crop(name='crop_upsampled_flow6_to_5', *[upsample_flow6to5, ReLU8],
                                                       offset=(1, 1)) #(B,2,15,20)
            Concat2 = mx.symbol.Concat(name='Concat2_iter{}'.format(iter_idx), *[ReLU8, ReLU11, crop_upsampled_flow6_to_5]) #(B, 1026,15,20)
            Convolution2 = mx.symbol.Convolution(name='Convolution2_iter{}'.format(iter_idx), data=Concat2, num_filter=2,
                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False,
                weight=share_dict['Convolution2_weight'], bias=share_dict['Convolution2_bias']) #(B,2,15,20)
            # scale 16
            deconv4 = mx.symbol.Deconvolution(name='deconv4_iter{}'.format(iter_idx), data=Concat2, num_filter=256,
                                        pad=(0, 0), kernel=(4, 4), stride=(2, 2), no_bias=False,
                                        weight=share_dict['deconv4_weight'], bias=share_dict['deconv4_bias']) # (B,256,32,42)
            crop_deconv4 = mx.symbol.Crop(name='crop_deconv4_iter{}'.format(iter_idx), *[deconv4, ReLU6], offset=(1, 1))
            ReLU12 = mx.symbol.LeakyReLU(name='ReLU12_iter{}'.format(iter_idx), data=crop_deconv4, act_type='leaky',
                                         slope=0.1)
            upsample_flow5to4 = mx.symbol.Deconvolution(name='upsample_flow5to4_iter{}'.format(iter_idx),
                data=Convolution2, num_filter=2, pad=(0, 0), kernel=(4, 4),
                stride=(2, 2), no_bias=False,
                weight=share_dict['upsample_flow5to4_weight'], bias=share_dict['upsample_flow5to4_bias'])
            crop_upsampled_flow5_to_4 = mx.symbol.Crop(name='crop_upsampled_flow5_to_4_iter{}'.format(iter_idx),
                *[upsample_flow5to4, ReLU6], offset=(1, 1)) #(B,2,32,42)
            Concat3 = mx.symbol.Concat(name='Concat3_iter{}'.format(iter_idx), *[ReLU6, ReLU12, crop_upsampled_flow5_to_4]) #(B,770,30,40)
            feat_list[1] = Concat3

        return feat_list


    def get_share_dict(self, cfg):
        share_dict = {
            'flow_conv1_weight': mx.sym.Variable('flow_conv1_weight'),
            'flow_conv1_bias': mx.sym.Variable('flow_conv1_bias'),
            'conv2_weight': mx.sym.Variable('conv2_weight'),
            'conv2_bias': mx.sym.Variable('conv2_bias'),
            'conv3_weight': mx.sym.Variable('conv3_weight'),
            'conv3_bias': mx.sym.Variable('conv3_bias'),
            'conv3_1_weight': mx.sym.Variable('conv3_1_weight'),
            'conv3_1_bias': mx.sym.Variable('conv3_1_bias'),
            'conv4_weight': mx.sym.Variable('conv4_weight'),
            'conv4_bias': mx.sym.Variable('conv4_bias'),
            'conv4_1_weight': mx.sym.Variable('conv4_1_weight'),
            'conv4_1_bias': mx.sym.Variable('conv4_1_bias'),
            'conv5_weight': mx.sym.Variable('conv5_weight'),
            'conv5_bias': mx.sym.Variable('conv5_bias'),
            'conv5_1_weight': mx.sym.Variable('conv5_1_weight'),
            'conv5_1_bias': mx.sym.Variable('conv5_1_bias'),
            'conv6_weight': mx.sym.Variable('conv6_weight'),
            'conv6_bias': mx.sym.Variable('conv6_bias'),
            'conv6_1_weight': mx.sym.Variable('conv6_1_weight'),
            'conv6_1_bias': mx.sym.Variable('conv6_1_bias'),
            'fc6_weight': mx.sym.Variable('fc6_weight'),
            'fc6_bias': mx.sym.Variable('fc6_bias'),
            'fc7_weight': mx.sym.Variable('fc7_weight'),
            'fc7_bias': mx.sym.Variable('fc7_bias'),
            'deconv5_weight': mx.sym.Variable('deconv5_weight'),
            'deconv5_bias': mx.sym.Variable('deconv5_bias'),
            'deconv4_weight': mx.sym.Variable('deconv4_weight'),
            'deconv4_bias': mx.sym.Variable('deconv4_bias'),
            'upsample_flow5to4_bias': mx.sym.Variable('upsample_flow5to4_bias'),
            'upsample_flow5to4_weight': mx.sym.Variable('upsample_flow5to4_weight'),
            'upsample_flow6to5_bias': mx.sym.Variable('upsample_flow6to5_bias'),
            'upsample_flow6to5_weight': mx.sym.Variable('upsample_flow6to5_weight'),
            'Convolution1_bias': mx.sym.Variable('Convolution1_bias'),
            'Convolution1_weight': mx.sym.Variable('Convolution1_weight'),
            'Convolution2_bias': mx.sym.Variable('Convolution2_bias'),
            'Convolution2_weight': mx.sym.Variable('Convolution2_weight'),
            'Convolution3_bias': mx.sym.Variable('Convolution3_bias'),
            'Convolution3_weight': mx.sym.Variable('Convolution3_weight'),
            'upsampling_weight': mx.sym.Variable('upsampling_weight')
        }
        if cfg.network.PRED_MASK:
            share_dict['mask_conv3_bias'] = mx.sym.Variable('mask_conv3_bias')
            share_dict['mask_conv3_weight'] = mx.sym.Variable('mask_conv3_weight')
            share_dict['mask_upsampling_weight'] = mx.sym.Variable('mask_upsampling_weight')
        if cfg.network.REGRESSOR_NUM == 1:
            share_dict['rot_bias'] = mx.sym.Variable('rot_bias')
            share_dict['rot_weight'] = mx.sym.Variable('rot_weight')
            share_dict['trans_bias'] = mx.sym.Variable('trans_bias')
            share_dict['trans_weight'] = mx.sym.Variable('trans_weight')
        else:
            for regressor_idx in range(cfg.network.REGRESSOR_NUM):
                share_dict['rot_class{}_bias'.format(regressor_idx)] = mx.sym.Variable(
                    'rot_iter{}_bias'.format(regressor_idx))
                share_dict['rot_iter{}_weight'.format(regressor_idx)] = mx.sym.Variable(
                    'rot_iter{}_weight'.format(regressor_idx))
                share_dict['trans_iter{}_bias'.format(regressor_idx)] = mx.sym.Variable(
                    'trans_iter{}_bias'.format(regressor_idx))
                share_dict['trans_iter{}_weight'.format(regressor_idx)] = mx.sym.Variable(
                    'trans_iter{}_weight'.format(regressor_idx))
        return share_dict

    def get_loss(self, big_cfg, small_cfg, conv_feat, deconv_feat, share_dict, labels, group_list, regressor_idx):
        pred = {}

        if big_cfg.network.PRED_FLOW:
            # get flow prediction
            flow_est = mx.symbol.Convolution(name='Convolution3', data=deconv_feat,
                                                 num_filter=2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False,
                                                 weight=share_dict['Convolution3_weight'], bias=share_dict['Convolution3_bias'])
            flow_est_resize = mx.symbol.Deconvolution(data=flow_est, num_filter=2, kernel=(32, 32),
                                                          stride=(16, 16), num_group=2, no_bias=True, name='upsampleing',
                                                          attr={'lr_mult': '0.0'}, workspace=self.workspace,
                                                          weight=share_dict['upsampling_weight'])

            # flow loss
            flow_est_crop = mx.symbol.Crop(*[flow_est_resize, labels['image_real']], offset=(8, 8),
                                               name='flow_est_crop')
            flow_loss_ = labels['flow_weights'] * \
                             mx.sym.square(data=(flow_est_crop - labels['flow'] / big_cfg.dataset.NORMALIZE_FLOW),
                                           name='flow_loss')

            flow_loss = mx.sym.MakeLoss(name='flow_loss', data=flow_loss_,
                                        grad_scale=small_cfg.LW_FLOW / (480 * 640))
            group_list.append(mx.sym.BlockGrad(flow_est_crop * big_cfg.dataset.NORMALIZE_FLOW))
            group_list.append(flow_loss)

        # get se3 prediction
        if big_cfg.network.REGRESSOR_NUM == 1:
            rot_est = mx.symbol.FullyConnected(name='rot', data=conv_feat, num_hidden=4,
                                               no_bias=False, weight=share_dict['rot_weight'],
                                               bias=share_dict['rot_bias'])
            zoom_trans_est = mx.symbol.FullyConnected(name='trans', data=conv_feat, num_hidden=3,
                                                      no_bias=False, weight=share_dict['trans_weight'],
                                                      bias=share_dict['trans_bias'])
        else:
            rot_est = mx.symbol.FullyConnected(name='rot', data=conv_feat, num_hidden=4, no_bias=False,
                                               weight=share_dict['rot_iter{}_weight'.format(regressor_idx)],
                                               bias=share_dict['rot_iter{}_bias'.format(regressor_idx)])
            zoom_trans_est = mx.symbol.FullyConnected(name='trans', data=conv_feat,
                                                      num_hidden=3,
                                                      no_bias=False, weight=share_dict[
                    'trans_iter{}_weight'.format(regressor_idx)],
                                                      bias=share_dict['trans_iter{}_bias'.format(regressor_idx)])

        rot_est_norm = mx.sym.L2Normalization(data=rot_est, name='normalize_quat')
        trans_est = mx.sym.Custom(zoom_factor=labels['zoom_factor'],
                                      trans_delta=zoom_trans_est,
                                      name='invZoomTrans', op_type='ZoomTrans',
                                      b_inv_zoom=True, b_zoom_grad=False)

        pred['rot_est_norm'] = mx.sym.BlockGrad(rot_est_norm)
        pred['trans_est'] = mx.sym.BlockGrad(trans_est)

        rot_gt = labels['rot_gt']
        trans_gt = labels['trans_gt']
        zoom_trans_gt = labels['zoom_trans_gt']
        group_list.append(mx.symbol.BlockGrad(rot_est_norm))
        group_list.append(mx.symbol.BlockGrad(rot_gt))
        group_list.append(mx.symbol.BlockGrad(trans_est))
        group_list.append(mx.symbol.BlockGrad(trans_gt))
        # se3 norm loss
        if small_cfg.SE3_DIST_LOSS:
            # rotation
            rot_est_norm_dot = mx.sym.expand_dims(rot_est_norm, axis=1, name='expand_rot_est')
            rot_gt_dot = mx.sym.expand_dims(rot_gt, axis=1, name='expand_rot_gt')
            rot_loss_ = 1 - mx.sym.square(mx.sym.batch_dot(rot_gt_dot, rot_est_norm_dot, transpose_b=True),
                                          name='rot_loss')

            rot_loss = mx.sym.MakeLoss(name='rot_loss', data=rot_loss_,
                                       grad_scale=small_cfg.LW_ROT)
            group_list.append(rot_loss)

            # translation
            if small_cfg.TRANS_LOSS_TYPE == 'L2':
                trans_loss_ = mx.sym.square(data=(zoom_trans_est - zoom_trans_gt), name='trans_loss')
            elif small_cfg.TRANS_LOSS_TYPE == 'smooth_L1':
                trans_loss_ = mx.sym.smooth_l1(data=(zoom_trans_est - zoom_trans_gt),
                                               name='trans_loss',
                                               scalar=small_cfg.TRANS_SMOOTH_L1_SCALAR)
            elif small_cfg.TRANS_LOSS_TYPE == 'L1':
                trans_loss_ = mx.sym.abs(data=(zoom_trans_est - zoom_trans_gt), name='trans_loss')
            else:
                raise Exception('Does not support small_cfg.TRANS_LOSS_TYPE: {}'.format(small_cfg.TRANS_LOSS_TYPE))

            trans_loss = mx.sym.MakeLoss(name='trans_loss', data=trans_loss_,
                                         grad_scale=small_cfg.LW_TRANS)
            group_list.append(trans_loss)

        # se3 point matching loss
        if small_cfg.SE3_PM_LOSS:
            pose_src = labels['src_pose']
            point_cloud_weights = labels['point_cloud_weights']
            point_cloud_model = labels['point_cloud_model']
            point_cloud_real = labels['point_cloud_real']

            point_cloud_real_est = mx.sym.Custom(point_cloud=point_cloud_model, rotation=rot_est_norm,
                                                 translation=trans_est,
                                                 pose_src=pose_src,
                                                 name='Transform3D_real',
                                                 op_type='Transform3D', T_means=big_cfg.dataset.trans_means.flatten(),
                                                 T_stds=big_cfg.dataset.trans_stds.flatten(),
                                                 rot_coord=big_cfg.network.ROT_COORD)

            # calculate the point matching loss
            norm_term = big_cfg.dataset.NORMALIZE_3D_POINT
            if small_cfg.SE3_PM_LOSS_TYPE == 'L1':
                point_matching_loss_all = mx.sym.abs(data=(point_cloud_real_est - point_cloud_real) / norm_term,
                                                   name='point_matching_loss_all_')
                point_matching_loss_valid = point_cloud_weights * point_matching_loss_all

            elif small_cfg.SE3_PM_LOSS_TYPE == 'L2':
                point_matching_loss_all = mx.sym.square(data=(point_cloud_real_est - point_cloud_real) / norm_term,
                                                      name='point_matching_loss_all_')
                point_matching_loss_valid = mx.sym.broadcast_mul(point_cloud_weights, point_matching_loss_all)
            elif small_cfg.SE3_PM_LOSS_TYPE == 'smooth_L1':
                point_matching_loss_all = mx.sym.smooth_l1(data=(point_cloud_real_est - point_cloud_real) / norm_term,
                                                         name='point_matching_loss_all_',
                                                         scalar=small_cfg.SE3_PM_SL1_SCALAR)
                point_matching_loss_valid = point_cloud_weights * point_matching_loss_all
            else:
                raise Exception('Unknown Point Matching Loss Type: {}'.format(small_cfg.SE3_PM_LOSS_TYPE))

            point_matching_norm = small_cfg.NUM_3D_SAMPLE
            point_matching_loss = mx.sym.MakeLoss(name='point_matching_loss', data=point_matching_loss_valid,
                                                  grad_scale=small_cfg.LW_PM / point_matching_norm)
            group_list.append(point_matching_loss)

        # mask
        if big_cfg.network.PRED_MASK:
            # get mask prediction
            mask_pred = mx.symbol.Convolution(name='mask_conv3', data=deconv_feat,
                                              num_filter=1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False,
                                              weight=share_dict['mask_conv3_weight'],
                                              bias=share_dict['mask_conv3_bias'])
            mask_pred_resize = mx.symbol.Deconvolution(data=mask_pred, num_filter=1, kernel=(32, 32),
                                                       stride=(16, 16), no_bias=True,
                                                       name='mask_upsampling',
                                                       attr={'lr_mult': '0.0'}, workspace=self.workspace,
                                                       weight=share_dict['mask_upsampling_weight'])

            # mask loss
            mask_pred_resize_crop = mx.symbol.Crop(*[mask_pred_resize, labels['image_real']], offset=(8, 8),
                                                   name='mask_pred_resize_crop')

            mask_prob = mx.sym.LogisticRegressionOutput(name='mask_prob',
                                                        data=mask_pred_resize_crop, label=labels['mask_real_gt'],
                                                        grad_scale=small_cfg.LW_MASK)

            mask_pred_bin = mx.sym.round(data=mask_prob, name='mask_pred_bin')
            unzoomed_mask_pred = mx.sym.Custom(zoom_factor=labels['zoom_factor'],
                                               mask=mask_pred_bin,
                                               name='invZoomMask', op_type='ZoomMaskWithFactor',
                                               height=480, width=640, b_inv_zoom=True)
            group_list.append(mask_prob) # zoomed
            group_list.append(mx.sym.BlockGrad(labels['mask_real_gt'],
                                               name='zoom_mask_real_gt'))
            group_list.append(mx.sym.BlockGrad(unzoomed_mask_pred, name='unzoomed_mask_pred'))

            pred['mask_real_est'] = mx.sym.BlockGrad(mask_pred_bin)

        return pred

    def get_train_symbol(self, cfg):
        """
        get symbol for training
        :param cfg: config['NUM_CLASSES']
        :return: the symbol for training
        """
        # prepare for forward
        share_dict = self.get_share_dict(cfg)
        small_cfg = cfg['train_iter']

        data_iter = {} # for get_conv
        labels_dict = {}

        class_index = mx.sym.Variable('class_index')
        labels_dict['class_index'] = class_index
        # poses
        src_pose = mx.symbol.Variable(name="src_pose")
        tgt_pose = mx.symbol.Variable(name="tgt_pose")
        labels_dict['src_pose'] = src_pose
        labels_dict['tgt_pose'] = tgt_pose

        # images and masks
        image_real = mx.symbol.Variable(name="image_real")
        image_rendered = mx.symbol.Variable(name="image_rendered")
        group_list = [mx.sym.BlockGrad(image_real), mx.sym.BlockGrad(image_rendered)]

        if cfg.network.INPUT_MASK or cfg.network.PRED_MASK:
            mask_vis_dict = {}
            mask_real_est = mx.symbol.Variable(name="mask_real_est")
            mask_real_gt = mx.symbol.Variable(name='mask_real_gt')
            mask_rendered = mx.symbol.Variable(name='mask_rendered')
            # get zoom factor using mask real est and mask rendered
            zoom_mask_real_est, zoom_mask_real_gt, zoom_mask_rendered, zoom_factor \
                = mx.sym.Custom(mask_real_est=mask_real_est, mask_real_gt=mask_real_gt,
                                mask_rendered=mask_rendered, src_pose=labels_dict['src_pose'],
                                K=cfg.dataset.INTRINSIC_MATRIX.flatten(),
                                name='ZoomMask', op_type='ZoomMask',
                                height=480, width=640)
            data_iter['mask_real_est'] = zoom_mask_real_est
            data_iter['mask_rendered'] = zoom_mask_rendered
            labels_dict['mask_real_gt'] = zoom_mask_real_gt

            mask_vis_dict['mask_real_est'] = zoom_mask_real_est
            mask_vis_dict['mask_rendered'] = zoom_mask_rendered
            mask_vis_dict['mask_real_gt'] = zoom_mask_real_gt
            # zoom image
            zoom_image_real, zoom_image_rendered \
                = mx.sym.Custom(zoom_factor=zoom_factor,
                                image_real=image_real, image_rendered=image_rendered,
                                name='ZoomImageWithFactor', op_type='ZoomImageWithFactor',
                                height=480, width=640, pixel_means=cfg.network.PIXEL_MEANS.flatten())

        else:
            # get zoom factor using image
            zoom_image_real, zoom_image_rendered, zoom_factor \
                = mx.sym.Custom(image_real=image_real, image_rendered=image_rendered,
                                src_pose=labels_dict['src_pose'],
                                K=cfg.dataset.INTRINSIC_MATRIX.flatten(),
                                name='ZoomImage', op_type='ZoomImage',
                                height=480, width=640, pixel_means=cfg.network.PIXEL_MEANS.flatten())
        data_iter['image_real'] = zoom_image_real
        data_iter['image_rendered'] = zoom_image_rendered
        labels_dict['image_real'] = image_real # for get_loss
        labels_dict['zoom_factor'] = zoom_factor

        if cfg.network.INPUT_MASK:
            mask_vis_dict['image_real'] = mx.sym.BlockGrad(zoom_image_real, name='zoom_image_real')
            mask_vis_dict['image_rendered'] = mx.sym.BlockGrad(zoom_image_rendered, name='zoom_image_rendered')

        depth_render_real = mx.symbol.BlockGrad(mx.symbol.Variable(name='depth_render_real'))

        # se3
        rot_gt = mx.symbol.Variable(name='rot')
        trans_gt = mx.symbol.Variable(name='trans')
        labels_dict['rot_gt'] = rot_gt
        zoom_trans_gt = mx.sym.Custom(zoom_factor=zoom_factor, trans_delta=trans_gt,
                                   name='ZoomTrans', op_type='ZoomTrans',
                                   b_inv_zoom=False)
        labels_dict['trans_gt'] = trans_gt
        labels_dict['zoom_trans_gt'] = zoom_trans_gt

        # depth
        if cfg.network.INPUT_DEPTH:
            depth_real = mx.symbol.Variable(name="depth_real")
            depth_rendered = mx.symbol.Variable(name="depth_rendered")
            zoom_depth_real, zoom_depth_rendered \
                = mx.sym.Custom(zoom_factor=zoom_factor, depth_real=depth_real, depth_rendered=depth_rendered,
                                name='ZoomDepth', op_type='ZoomDepth',
                                height=480, width=640)
            data_iter['depth_real'] = zoom_depth_real
            data_iter['depth_rendered'] = zoom_depth_rendered

        # flow
        if cfg.network.PRED_FLOW:
            flow = mx.symbol.Variable(name='flow')
            flow_weights = mx.symbol.Variable(name='flow_weights')
            zoom_flow, zoom_flow_weights \
                = mx.sym.Custom(flow=flow, flow_weights=flow_weights,
                                name='ZoomFlow', op_type='ZoomFlow', height=480, width=640,
                                zoom_factor=zoom_factor, b_inv_zoom=False)
            labels_dict['flow'] = zoom_flow
            labels_dict['flow_weights'] = zoom_flow_weights

        # point matching loss
        if small_cfg.SE3_PM_LOSS:
            point_cloud_model = mx.symbol.Variable(name='point_cloud_model')
            point_cloud_weights = mx.symbol.Variable(name='point_cloud_weights')
            point_cloud_real = mx.symbol.Variable(name='point_cloud_real')
            labels_dict['point_cloud_model'] = point_cloud_model
            labels_dict['point_cloud_weights'] = point_cloud_weights
            labels_dict['point_cloud_real'] = point_cloud_real


        # forward iter 0
        conv_feat, deconv_feat = self.get_flownet(data_iter, cfg, small_cfg, share_dict, 0)
        pred_iter = self.get_loss(cfg, small_cfg, conv_feat, deconv_feat, share_dict, labels_dict,
                                  group_list, regressor_idx=0)

        if cfg.network.TRAIN_ITER:
            group_list.append(mx.sym.BlockGrad(src_pose))
            group_list.append(mx.sym.BlockGrad(tgt_pose))

        group_list.append(depth_render_real)
        if cfg.network.INPUT_MASK:
            group_list.append(mx.sym.BlockGrad(mask_real_est))
            group_list.append(mx.sym.BlockGrad(mask_real_gt))
            group_list.append(mx.sym.BlockGrad(mask_rendered))

        if cfg.network.INPUT_MASK and cfg.network.PRED_MASK and cfg.TRAIN.VISUALIZE: # for visualize
            # for i in range(1, cfg.network.TRAIN_ITER_SIZE+1):
            group_list.append(mask_vis_dict['mask_real_est'])       # zoomed
            group_list.append(mask_vis_dict['mask_real_gt']) # zoomed
            group_list.append(mask_vis_dict['mask_rendered'])    # zoomed
            group_list.append(mask_vis_dict['image_real'])   # zoomed
            group_list.append(mask_vis_dict['image_rendered'])   # zoomed

        # zoom_image_real, zoom_image_rendered = mx.sym.Custom(image_real=image_real, image_rendered=image_rendered,
        #                                                    name='SimpleZoom_vis', op_type='SimpleZoom',
        #                                                    height=480, width=640, pixel_means=cfg.network.PIXEL_MEANS.flatten())
        group_list.append(labels_dict['class_index'])
        if cfg.TRAIN.VISUALIZE:
            group_list.append(mx.sym.BlockGrad(zoom_mask_real_gt)) # zoomed
            group_list.append(mx.sym.BlockGrad(zoom_mask_real_est)) # zoomed
            group_list.append(mx.sym.BlockGrad(zoom_mask_rendered))
        group_list.append(data_iter['image_real']) # zoomed
        group_list.append(data_iter['image_rendered'])
        group = mx.symbol.Group(group_list)
        self.sym = group

        return group

    def get_test_symbol_share(self, cfg):
        """
        get symbol for testing
        :param cfg: config['NUM_CLASSES']
        :return: the symbol for testing
        """
        group_list = []
        share_dict = self.get_share_dict(cfg)

        data_iter = {}
        # images and masks
        image_real = mx.symbol.Variable(name="image_real")
        image_rendered = mx.symbol.Variable(name="image_rendered")
        src_pose = mx.symbol.Variable(name="src_pose")
        class_index = mx.symbol.Variable(name="class_index")

        if cfg.network.INPUT_MASK:
            mask_real_est = mx.symbol.Variable(name="mask_real_est")
            mask_real_gt = mask_real_est # test phase has no gt
            mask_rendered = mx.symbol.Variable(name='mask_rendered')
            # get zoom factor using mask real est and mask rendered
            zoom_mask_real_est, _, zoom_mask_rendered, zoom_factor \
                = mx.sym.Custom(mask_real_est=mask_real_est, mask_real_gt=mask_real_gt,
                                mask_rendered=mask_rendered, src_pose=src_pose,
                                K=cfg.dataset.INTRINSIC_MATRIX.flatten(),
                                name='ZoomMask', op_type='ZoomMask',
                                height=480, width=640)
            data_iter['mask_real_est'] = zoom_mask_real_est
            data_iter['mask_rendered'] = zoom_mask_rendered
            # zoom image
            zoom_image_real, zoom_image_rendered \
                = mx.sym.Custom(zoom_factor=zoom_factor,
                                image_real=image_real, image_rendered=image_rendered,
                                name='ZoomImageWithFactor', op_type='ZoomImageWithFactor',
                                height=480, width=640, pixel_means=cfg.network.PIXEL_MEANS.flatten())
        else:
            # zoom image
            zoom_image_real, zoom_image_rendered, zoom_factor \
                = mx.sym.Custom(image_real=image_real, image_rendered=image_rendered,
                                src_pose=src_pose,
                                K=cfg.dataset.INTRINSIC_MATRIX.flatten(),
                                name='ZoomImage', op_type='ZoomImage',
                                height=480, width=640, pixel_means=cfg.network.PIXEL_MEANS.flatten())
        data_iter['image_real'] = zoom_image_real
        data_iter['image_rendered'] = zoom_image_rendered

        # depth
        if cfg.network.INPUT_DEPTH:
            depth_real = mx.symbol.Variable(name="depth_real")
            depth_rendered = mx.symbol.Variable(name="depth_rendered")
            zoom_depth_real, zoom_depth_rendered \
                = mx.sym.Custom(zoom_factor=zoom_factor, depth_real=depth_real, depth_rendered=depth_rendered,
                                name='ZoomDepth', op_type='ZoomDepth',
                                height=480, width=640)
            data_iter['depth_real'] = zoom_depth_real
            data_iter['depth_rendered'] = zoom_depth_rendered

        conv_feat, deconv_feat = self.get_flownet(data_iter, cfg, cfg.train_iter, share_dict, 0)
        mask_score = None
        if cfg.network.PRED_MASK and (cfg.TEST.UPDATE_MASK not in ['init', 'box_rendered'] or not cfg.TEST.FAST_TEST):

            # get mask prediction
            mask_pred = mx.symbol.Convolution(name='mask_conv3', data=deconv_feat,
                                              num_filter=1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
            mask_pred_resize = mx.symbol.Deconvolution(data=mask_pred, num_filter=1, kernel=(32, 32),
                                                       stride=(16, 16), no_bias=True,
                                                       name='mask_upsampling',
                                                       attr={'lr_mult': '0.0'}, workspace=self.workspace)

            mask_pred_resize_crop = mx.symbol.Crop(*[mask_pred_resize, image_real], offset=(8, 8),
                                                   name='mask_pred_resize_crop')
            zoom_mask_real_est_pred = mx.sym.Activation(data=mask_pred_resize_crop, act_type='sigmoid',
                                                      name="zoom_mask_real_est_prob_iter")
            zoom_mask_real_est_pred_bin = mx.sym.round(zoom_mask_real_est_pred, name='zoom_mask_real_est_pred_bin')

            mask_real_est_pred = mx.sym.Custom(zoom_factor=zoom_factor, mask=zoom_mask_real_est_pred_bin,
                           name='invZoomMask', op_type='ZoomMaskWithFactor',
                           height=480, width=640, b_inv_zoom=True)
            group_list.append(zoom_mask_real_est_pred)
            group_list.append(mx.sym.BlockGrad(zoom_mask_real_est_pred_bin, name='zoom_mask_pred_bin'))
            group_list.append(mx.sym.BlockGrad(zoom_mask_real_est, name='zoom_mask_real_est'))
            group_list.append(mx.sym.BlockGrad(mask_real_est_pred, name='mask_real_est_pred'))
            group_list.append(mx.sym.BlockGrad(zoom_image_real, name='zoom_image_real'))
            group_list.append(mx.sym.BlockGrad(zoom_image_rendered, name='zoom_image_rendered'))

        # flow
        if cfg.network.PRED_FLOW:
            flow_est = mx.symbol.Convolution(name='Convolution3', data=deconv_feat, num_filter=2,
                                             pad=(1, 1), kernel=(3, 3),
                                             stride=(1, 1), no_bias=False)
            flow_est_resize = mx.symbol.Deconvolution(data=flow_est, num_filter=2, kernel=(32, 32),
                                                      stride=(16, 16),
                                                      num_group=2, no_bias=True, name='upsampling',
                                                      attr={'lr_mult': '0.0'},
                                                      workspace=self.workspace) * cfg.dataset.NORMALIZE_FLOW
            zoom_flow_est_crop = mx.symbol.Crop(*[flow_est_resize, image_real], offset=(8, 8), name='zoom_flow_est_crop')
            flow_est = mx.sym.Custom(name='invZoomFlow', op_type='ZoomFlow', height=480, width=640,
                                     flow=zoom_flow_est_crop, zoom_factor=zoom_factor,
                                     b_inv_zoom=True)
            flow_est_crop = mx.symbol.Crop(*[flow_est, image_real], offset=(0, 0), name='flow_est_crop')
            group_list.append(flow_est_crop)

        #se3
        rot_param = 3 if cfg.network.ROT_TYPE == "EULER" else 4
        rot_est = mx.symbol.FullyConnected(name='rot', data=conv_feat,
                                           num_hidden=rot_param, no_bias=False)
        zoom_trans_est_ = mx.symbol.FullyConnected(name='trans', data=conv_feat, num_hidden=3,
                                                   no_bias=False)
        trans_est = mx.sym.Custom(zoom_factor=zoom_factor, trans_delta=zoom_trans_est_,
                                  name='invZoomTrans', op_type='ZoomTrans',
                                  b_inv_zoom=True)
        se3_est = mx.symbol.Concat(rot_est, trans_est, dim=1, name='se3')
        group_list.append(se3_est)

        group_list.append(mx.sym.BlockGrad(class_index))
        group_list.append(mx.sym.BlockGrad(zoom_image_real))
        group_list.append(mx.sym.BlockGrad(zoom_image_rendered))

        group = mx.symbol.Group(group_list)
        self.sym = group
        return group

    def get_test_symbol_sep(self, cfg):
        """
        get symbol for testing
        :param cfg: config['NUM_CLASSES']
        :return: the symbol for testing
        """
        group_list = []
        share_dict = self.get_share_dict(cfg)

        data_iter = {}
        # images and masks
        image_real = mx.symbol.Variable(name="image_real")
        image_rendered = mx.symbol.Variable(name="image_rendered")
        src_pose = mx.symbol.Variable(name="src_pose")
        class_index = mx.symbol.Variable(name="class_index")

        if cfg.network.WITH_MASK:
            mask_real_est = mx.symbol.Variable(name="mask_real_est")
            mask_real_gt = mask_real_est ######## test phase has no gt
            mask_rendered = mx.symbol.Variable(name='mask_rendered')
            # get zoom factor using mask real est and mask rendered
            zoom_mask_real_est, zoom_mask_real_gt, zoom_mask_rendered, zoom_factor \
                = mx.sym.Custom(mask_real_est=mask_real_est, mask_real_gt=mask_real_gt,
                                mask_rendered=mask_rendered, src_pose=src_pose,
                                K=cfg.dataset.INTRINSIC_MATRIX.flatten(),
                                name='ZoomMask', op_type='ZoomMask',
                                height=480, width=640)
            data_iter['mask_real_est'] = zoom_mask_real_est
            data_iter['mask_rendered'] = zoom_mask_rendered
            # zoom image
            zoom_image_real, zoom_image_rendered \
                = mx.sym.Custom(zoom_factor=zoom_factor,
                                image_real=image_real, image_rendered=image_rendered,
                                name='ZoomImageWithFactor', op_type='ZoomImageWithFactor',
                                height=480, width=640, pixel_means=cfg.network.PIXEL_MEANS.flatten())
        else:
            # zoom image
            zoom_image_real, zoom_image_rendered, zoom_factor\
                = mx.sym.Custom(image_real=image_real, image_rendered=image_rendered,
                                src_pose=src_pose,
                                K=cfg.dataset.INTRINSIC_MATRIX.flatten(),
                                name='ZoomImage', op_type='ZoomImage',
                                height=480, width=640, pixel_means=cfg.network.PIXEL_MEANS.flatten())
        data_iter['image_real'] = zoom_image_real
        data_iter['image_rendered'] = zoom_image_rendered

        # depth
        if cfg.network.WITH_DEPTH:
            depth_real = mx.symbol.Variable(name="depth_real")
            depth_rendered = mx.symbol.Variable(name="depth_rendered")
            zoom_depth_real, zoom_depth_rendered \
                = mx.sym.Custom(zoom_factor=zoom_factor, depth_real=depth_real, depth_rendered=depth_rendered,
                                name='ZoomDepth', op_type='ZoomDepth',
                                height=480, width=640)
            data_iter['depth_real'] = zoom_depth_real
            data_iter['depth_rendered'] = zoom_depth_rendered

        conv_feat, deconv_feat = self.get_flownet(data_iter, cfg, cfg.train_iter, share_dict, 0)
        mask_score = None
        if cfg.network.PRED_MASK and (cfg.TEST.UPDATE_MASK not in ['init', 'box_rendered'] or not cfg.TEST.FAST_TEST):

            # get mask prediction
            mask_pred = mx.symbol.Convolution(name='mask_conv3', data=deconv_feat,
                                              num_filter=1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
            mask_pred_resize = mx.symbol.Deconvolution(data=mask_pred, num_filter=1, kernel=(32, 32),
                                                       stride=(16, 16), no_bias=True,
                                                       name='mask_upsampling',
                                                       attr={'lr_mult': '0.0'}, workspace=self.workspace)

            mask_pred_resize_crop = mx.symbol.Crop(*[mask_pred_resize, image_real], offset=(8, 8),
                                                   name='mask_pred_resize_crop')
            zoom_mask_real_est_pred = mx.sym.Activation(data=mask_pred_resize_crop, act_type='sigmoid',
                                                      name="zoom_mask_real_est_prob_iter")
            zoom_mask_real_est_pred_bin = mx.sym.round(zoom_mask_real_est_pred, name='zoom_mask_real_est_pred_bin')

            mask_real_est_pred = mx.sym.Custom(zoom_factor=zoom_factor, mask=zoom_mask_real_est_pred_bin,
                           name='invZoomMask', op_type='ZoomMaskWithFactor',
                           height=480, width=640, b_inv_zoom=True)
            group_list.append(zoom_mask_real_est_pred)
            group_list.append(mx.sym.BlockGrad(zoom_mask_real_est_pred_bin, name='zoom_mask_pred_bin'))
            group_list.append(mx.sym.BlockGrad(zoom_mask_real_est, name='zoom_mask_real_est'))
            group_list.append(mx.sym.BlockGrad(mask_real_est_pred, name='mask_real_est_pred'))
            group_list.append(mx.sym.BlockGrad(zoom_image_real, name='zoom_image_real'))
            group_list.append(mx.sym.BlockGrad(zoom_image_rendered, name='zoom_image_rendered'))

        # flow
        if cfg.train_iter.flow:
            flow_est = mx.symbol.Convolution(name='Convolution3', data=deconv_feat, num_filter=2,
                                                 pad=(1, 1), kernel=(3, 3),
                                                 stride=(1, 1), no_bias=False)
            flow_est_resize = mx.symbol.Deconvolution(data=flow_est, num_filter=2, kernel=(32, 32),
                                                          stride=(16, 16),
                                                          num_group=2, no_bias=True, name='upsampling',
                                                          attr={'lr_mult': '0.0'},
                                                          workspace=self.workspace) * cfg.dataset.NORMALIZE_FLOW
            zoom_flow_est_crop = mx.symbol.Crop(*[flow_est_resize, image_real], offset=(8, 8), name='zoom_flow_est_crop')
            flow_est = mx.sym.Custom(name='invZoomFlow', op_type='ZoomFlow', height=480, width=640,
                                              flow=zoom_flow_est_crop, zoom_factor=zoom_factor,
                                              b_inv_zoom=True)
            flow_est_crop = mx.symbol.Crop(*[flow_est, image_real], offset=(0, 0), name='flow_est_crop')
            group_list.append(flow_est_crop)

        # se3
        rot_param = 3 if cfg.network.SE3_TYPE == "EULER" else 4
        rot_est_ = mx.symbol.FullyConnected(name='rot_', data=conv_feat,
                                            num_hidden=rot_param*len(cfg.dataset.class_name), no_bias=False)
        rot_est = mx.sym.Custom(input_data=rot_est_, group_idx=class_index,
                                    name='pick_rot', op_type='GroupPicker', group_num=len(cfg.dataset.class_name))

        zoom_trans_est_ = mx.symbol.FullyConnected(name='trans', data=conv_feat,
                                                  num_hidden=3*len(cfg.dataset.class_name), no_bias=False)
        zoom_trans_est = mx.sym.Custom(input_data=zoom_trans_est_, group_idx=class_index,
                                           name='pick_trans', op_type='GroupPicker',
                                           group_num=len(cfg.dataset.class_name))
        trans_est = mx.sym.Custom(zoom_factor=zoom_factor, trans_delta=zoom_trans_est,
                                  name='ZoomTrans', op_type='ZoomTrans',
                                  b_inv_zoom=True)
        se3_est = mx.symbol.Concat(rot_est, trans_est, dim=1, name='se3')
        group_list.append(se3_est)

        group = mx.symbol.Group(group_list)
        self.sym = group
        return group

    def get_symbol(self, cfg, is_train=True):
        """
        return a generated symbol, it also need to be assigned to self.sym
        """

        if is_train:
            self.sym = self.get_train_symbol(cfg)
        else:
            if cfg.network.REGRESSOR_NUM == 1:
                self.sym = self.get_test_symbol_share(cfg)
            else:
                raise Exception("NOT REVIEWED")
                self.sym = self.get_test_symbol_sep(cfg)

        return self.sym

    def init_weights(self, cfg, arg_params, aux_params):
        if cfg.network.pretrained == 'xavier':
            init_xavier = mx.init.Xavier()
            for param_name in arg_params:
                if param_name.endwith('bias'):
                    arg_params[param_name] = mx.nd.zeros(shape=self.arg_shape_dict[param_name])
                elif param_name.endwith('weight'):
                    init_xavier._init_weight(param_name, arg_params[param_name])
        else:
            num_extra_in_channel = 0
            if cfg.network.INPUT_DEPTH:
                num_extra_in_channel += 2 # corresponding to the real and rendered images respectively
            if cfg.network.INPUT_MASK:
                num_extra_in_channel += 2
            if num_extra_in_channel > 0:
                flow_conv1_rgb_weight = arg_params['flow_conv1_weight']
                rgb_kernel_shape = arg_params['flow_conv1_weight'].shape
                d_kernel_shape = list(rgb_kernel_shape)
                d_kernel_shape[1] = num_extra_in_channel
                flow_conv1_d_weight = mx.ndarray.zeros(d_kernel_shape)
                flow_conv1_rgbd_weight = mx.ndarray.concat(flow_conv1_rgb_weight, flow_conv1_d_weight, dim=1)
                arg_params['flow_conv1_weight'] = flow_conv1_rgbd_weight

            print("in init_weights, arg_shape_dict: ", self.arg_shape_dict)
            # print("in init_weights, arg_params: ", arg_params)

            # init from flownet
            if cfg.network.init_from_flownet:
                print("### init from flownet ###")
                arg_params['fc6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc6_bias'])
                arg_params['fc6_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['fc6_weight'])
                arg_params['fc7_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc7_bias'])
                arg_params['fc7_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['fc7_weight'])

                rot_bias_init = mx.nd.zeros(shape=self.arg_shape_dict['rot_bias'])
                arg_params['rot_bias'] = rot_bias_init

                if cfg.network.ROT_TYPE == 'EULER':
                    arg_params['rot_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['rot_weight'])
                elif cfg.network.ROT_TYPE == 'QUAT':
                    # make the initial rotation estimate closer to 0 degree
                    rot_weight = np.random.rand(self.arg_shape_dict['rot_weight'][0],
                                                    self.arg_shape_dict['rot_weight'][1])*0.01
                    rot_weight[0, :] = np.random.rand(self.arg_shape_dict['rot_weight'][1])+0.01
                    arg_params['rot_weight'] = mx.nd.array(rot_weight)
                arg_params['trans_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['trans_bias'])
                arg_params['trans_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['trans_weight'])

                init_xavier = mx.init.Xavier()
                init_xavier._init_weight('fc6_weight', arg_params['fc6_weight'])
                init_xavier._init_weight('fc7_weight', arg_params['fc7_weight'])

                if cfg.network.PRED_FLOW:
                    init = mx.init.Initializer()
                    arg_params['upsampling_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['upsampling_weight'])
                    init._init_bilinear('upsample_weight', arg_params['upsampling_weight'])

                if cfg.network.PRED_MASK:
                    arg_params['mask_conv3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['mask_conv3_bias'])
                    arg_params['mask_conv3_weight'] = mx.random.normal(0, 0.01,
                                                                       shape=self.arg_shape_dict['mask_conv3_weight'])
                    init = mx.init.Initializer()
                    arg_params['mask_upsampling_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['mask_upsampling_weight'])
                    init._init_bilinear('mask_upsampling_weight', arg_params['mask_upsampling_weight'])

                print('arg_shape_dict: ', self.arg_shape_dict.keys())

            # init from pre-trained
            for k in self.arg_shape_dict:
                if (not k in arg_params) and (k.endswith('weight') or k.endswith('bias')):
                    # weights not found in arg_params
                    if not cfg.network.REGRESSOR_NUM == 1:
                        print("copy weight to {}".format(k))
                        # if not share regressor, then init them
                        if k.startswith('rot_iter') and k.endswith('weight'):
                            arg_params[k] = arg_params['rot_weight'].copy()
                        elif k.startswith('rot_iter') and k.endswith('bias'):
                            arg_params[k] = arg_params['rot_bias'].copy()
                        elif k.startswith('trans_iter') and k.endswith('weight'):
                            arg_params[k] = arg_params['trans_weight'].copy()
                        elif k.startswith('trans_iter') and k.endswith('bias'):
                            arg_params[k] = arg_params['trans_bias'].copy()
                        else:
                            print("What's this? ", k)
                    else:
                        assert k in arg_params, k
            print("all weights assigned")


