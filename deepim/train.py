# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import _init_paths

from lib.pair_matching import RT_transform
import time
from six.moves import xrange
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train deepim network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    parser.add_argument('--gpus', help='specify the gpu to be use', required=True, type=str)
    parser.add_argument('--temp', help='turn on visualization', action='store_true')
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))


import shutil
import numpy as np
import mxnet as mx

from symbols import *
from core import callback, metric
from core.loader import TrainDataLoader
from core.module import MutableModule
from lib.utils.load_data import load_gt_pairdb, merge_pairdb
from lib.utils.load_model import load_param
from lib.utils.PrefetchingIter import PrefetchingIter
from lib.utils.create_logger import create_logger
from lib.utils.lr_scheduler import WarmupMultiFactorScheduler
from lib.utils.print_and_log import print_and_log

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    if not os.path.basename(args.cfg).split('.')[0].endswith('temp'):
        new_args_name = os.path.basename(args.cfg).split('.')[0]+'_{}gpus.yaml'.format(len(ctx))
    else:
        new_args_name = args.cfg
    if args.vis:
        config.TRAIN.VISUALIZE = True
    logger, final_output_path = create_logger(config.output_path, new_args_name, config.dataset.image_set, args.temp)
    prefix = os.path.join(final_output_path, prefix)
    logger.info('called with args {}'.format(args))

    print(config.train_iter.SE3_PM_LOSS)
    if config.train_iter.SE3_PM_LOSS:
        print("SE3_PM_LOSS == True")
    else:
        print("SE3_PM_LOSS == False")

    if not config.network.STANDARD_FLOW_REP:
        print_and_log("[h, w] representation for flow is dep", logger)

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    datasets = [dset for dset in config.dataset.dataset.split('+')]
    print("config.dataset.class_name: {}".format(config.dataset.class_name))
    print("image_sets: {}".format(image_sets))
    if datasets[0].startswith('ModelNet'):
        pairdbs = [load_gt_pairdb(config, datasets[i], image_sets[i] + class_name.split('/')[-1], config.dataset.root_path,
                                  config.dataset.dataset_path,
                                  class_name=class_name,
                                  result_path=final_output_path)
                   for class_name in config.dataset.class_name
                   for i in range(len(image_sets))]
    else:
        pairdbs = [load_gt_pairdb(config, datasets[i], image_sets[i]+class_name, config.dataset.root_path,
                              config.dataset.dataset_path,
                              class_name=class_name,
                              result_path=final_output_path)
               for class_name in config.dataset.class_name
               for i in range(len(image_sets))]
    pairdb = merge_pairdb(pairdbs)

    if not args.temp:
        src_file = os.path.join(curr_path, 'symbols', config.symbol + '.py')
        dst_file = os.path.join(final_output_path, '{}_{}.py'.format(config.symbol, time.strftime('%Y-%m-%d-%H-%M')))
        os.popen('cp {} {}'.format(src_file, dst_file))

    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=True)

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_PAIRS * batch_size

    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load training data
    train_data = TrainDataLoader(sym, pairdb, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE, ctx=ctx)

    train_data.get_batch_parallel()
    max_scale = [max([v[0] for v in config.SCALES]), max(v[1] for v in config.SCALES)]
    max_data_shape = [('image_observed', (config.TRAIN.BATCH_PAIRS, 3, max_scale[0], max_scale[1])),
                      ('image_rendered', (config.TRAIN.BATCH_PAIRS, 3, max_scale[0], max_scale[1])),
                      ('depth_gt_observed', (config.TRAIN.BATCH_PAIRS, 1, max_scale[0], max_scale[1])),
                      ('src_pose', (config.TRAIN.BATCH_PAIRS, 3, 4)),
                      ('tgt_pose', (config.TRAIN.BATCH_PAIRS, 3, 4))]
    if config.network.INPUT_DEPTH:
        max_data_shape.append(('depth_observed', (config.TRAIN.BATCH_PAIRS, 1, max_scale[0], max_scale[1])))
        max_data_shape.append(('depth_rendered', (config.TRAIN.BATCH_PAIRS, 1, max_scale[0], max_scale[1])))
    if config.network.INPUT_MASK:
        max_data_shape.append(('mask_observed', (config.TRAIN.BATCH_PAIRS, 1, max_scale[0], max_scale[1])))
        max_data_shape.append(('mask_rendered', (config.TRAIN.BATCH_PAIRS, 1, max_scale[0], max_scale[1])))

    rot_param = 3 if config.network.ROT_TYPE=="EULER" else 4
    max_label_shape = [('rot', (config.TRAIN.BATCH_PAIRS, rot_param)),
                       ('trans', (config.TRAIN.BATCH_PAIRS, 3))]
    if config.network.PRED_FLOW:
        max_label_shape.append(('flow', (config.TRAIN.BATCH_PAIRS, 2, max_scale[0], max_scale[1])))
        max_label_shape.append(('flow_weights', (config.TRAIN.BATCH_PAIRS, 2, max_scale[0], max_scale[1])))
    if config.train_iter.SE3_PM_LOSS:
        max_label_shape.append(('point_cloud_model', (config.TRAIN.BATCH_PAIRS, 3, config.train_iter.NUM_3D_SAMPLE)))
        max_label_shape.append(('point_cloud_weights', (config.TRAIN.BATCH_PAIRS, 3, config.train_iter.NUM_3D_SAMPLE)))
        max_label_shape.append(('point_cloud_observed', (config.TRAIN.BATCH_PAIRS, 3, config.train_iter.NUM_3D_SAMPLE)))
    if config.network.PRED_MASK:
        max_label_shape.append(('mask_gt_observed', (config.TRAIN.BATCH_PAIRS, 1, max_scale[0], max_scale[1])))

    # max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape, max_label_shape)
    print_and_log('providing maximum shape, {}, {}'.format(max_data_shape, max_label_shape), logger)

    """
    # infer max shape
    max_label_shape = [('label', (config.TRAIN.BATCH_IMAGES, 1, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape, max_label_shape)
    print('providing maximum shape', max_data_shape, max_label_shape)
    """
    # infer shape
    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    print_and_log('\ndata_shape_dict: {}\n'.format(data_shape_dict), logger)
    sym_instance.infer_shape(data_shape_dict)

    print('************(wg): infering shape **************')
    internals = sym.get_internals()
    _, out_shapes, _ = internals.infer_shape(**data_shape_dict)
    print(sym.list_outputs())
    shape_dict = dict(zip(internals.list_outputs(), out_shapes))
    pprint.pprint(shape_dict)

    # load and initialize params
    if config.TRAIN.RESUME:
        print('continue training from ', begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    elif pretrained=='xavier':
        print('xavier')
        # arg_params = {}
        # aux_params = {}
        # sym_instance.init_weights(config, arg_params, aux_params)
    else:
        print(pretrained)
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        print('arg_params: ', arg_params.keys())
        print('aux_params: ', aux_params.keys())
        if not config.network.skip_initialize:
            sym_instance.init_weights(config, arg_params, aux_params)

    # check parameter shapes
    if pretrained != 'xavier':
        sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    label_names = [k[0] for k in train_data.provide_label_single]

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in xrange(batch_size)],
                        max_label_shapes=[max_label_shape for _ in xrange(batch_size)],
                        fixed_param_prefix=fixed_param_prefix,
                        CUDA9=config.CUDA9, config=config)


    # decide training params
    # metrics
    eval_metrics = mx.metric.CompositeEvalMetric()

    metric_list = []
    iter_idx = 0
    if config.network.PRED_FLOW:
        metric_list.append(metric.Flow_L2LossMetric(config, iter_idx))
        metric_list.append(metric.Flow_CurLossMetric(config, iter_idx))
    if config.train_iter.SE3_DIST_LOSS:
        metric_list.append(metric.Rot_L2LossMetric(config, iter_idx))
        metric_list.append(metric.Trans_L2LossMetric(config, iter_idx))
    if config.train_iter.SE3_PM_LOSS:
        metric_list.append(metric.PointMatchingLossMetric(config, iter_idx))
    if config.network.PRED_MASK:
        metric_list.append(metric.MaskLossMetric(config, iter_idx))

    # Visualize Training Batches
    if config.TRAIN.VISUALIZE:
        metric_list.append(metric.SimpleVisualize(config))
        # metric_list.append(metric.MaskVisualize(config, save_dir = final_output_path))
        metric_list.append(metric.MinibatchVisualize(config)) # flow visualization

    for child_metric in metric_list:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    epoch_end_callback = mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True)
    # decide learning rate
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(pairdb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)

    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr,
                                              config.TRAIN.warmup_step)


    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    # train
    if config.TRAIN.optimizer == 'adam':
        optimizer_params = {'learning_rate': lr}
        if pretrained == 'xavier':
            init = mx.init.Mixed(['rot_weight|trans_weight', '.*'],
                                 [mx.init.Zero(),
                                  mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)])
            mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
                    batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
                    optimizer='adam', optimizer_params=optimizer_params,
                    begin_epoch=begin_epoch, num_epoch=end_epoch,
                    prefix=prefix, initializer=init, force_init=True)
        else:
            mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
                    batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
                    optimizer='adam',
                    arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch,
                    prefix=prefix)
    elif config.TRAIN.optimizer == 'sgd':
        # optimizer
        optimizer_params = {'momentum': config.TRAIN.momentum,
                            'wd': config.TRAIN.wd,
                            'learning_rate': lr,
                            'lr_scheduler': lr_scheduler,
                            'rescale_grad': 1.0,
                            'clip_gradient': None}
        if pretrained == 'xavier':
            init = mx.init.Mixed(['rot_weight|trans_weight', '.*'],
                                 [mx.init.Zero(),
                                  mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)])
            mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
                    batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
                    optimizer='sgd', optimizer_params=optimizer_params,
                    begin_epoch=begin_epoch, num_epoch=end_epoch,
                    prefix=prefix, initializer=init, force_init=True)
        else:
            mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
                    batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
                    optimizer='sgd', optimizer_params=optimizer_params,
                    arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch,
                    prefix=prefix)


def main():
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    if len(ctx) != config.NUM_GPUS:
        print("********** WARNING: length of context doesn't match num_gpus set in config, {} vs. {} ************".format(len(ctx), config.NUM_GPUS))
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)


if __name__ == '__main__':
    main()
