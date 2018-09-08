# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import _init_paths

import argparse
import os
import sys
import time
import logging
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Test a DeepIM Network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # testing
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--vis_video', help='turn on video visualization', action='store_true')
    parser.add_argument('--vis_video_zoom', help='turn on zoom video visualization', action='store_true')
    parser.add_argument('--iter_test', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--gpus', help='specify the gpu to be use', required=True, type=str)
    parser.add_argument('--temp', help='turn on visualization', action='store_true')
    parser.add_argument('--alt', help='switch between rotation and translation', action='store_true')
    parser.add_argument('--refine', help='turn on visualization', action='store_true')
    parser.add_argument('--skip_flow', help='whether skip flow during test', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

if args.vis_video:
    config.TEST.VIS_VIDEO = True

if args.vis_video_zoom:
    config.TEST.VIS_VIDEO_ZOOM = True

import pprint
import mxnet as mx

from symbols import *
from lib.dataset import *
from core.loader import TestDataLoader
from core.tester import Predictor, pred_eval
from lib.utils.load_data import load_gt_pairdb, merge_pairdb
from lib.utils.load_model import load_param
from lib.utils.create_logger import create_logger

def test_deepim():
    config.TRAIN.MASK_SYN = False

    if args.vis or args.vis_video or args.vis_video_zoom:
        config.TEST.VISUALIZE = True
        config.TEST.FAST_TEST = False
    if args.iter_test:
        config.TEST.test_iter = 5
    if args.refine:
        config.TEST.test_iter = 1
    if args.skip_flow:
        config.network.FLOW_I2R = False
        config.network.FLOW_R2I = False
        config.train_iter0.FLOW_I2R = False
        config.train_iter0.FLOW_R2I = False
        config.train_iter1.FLOW_I2R = False
        config.train_iter1.FLOW_R2I = False
        config.train_iter2.FLOW_I2R = False
        config.train_iter2.FLOW_R2I = False
        config.train_iter3.FLOW_I2R = False
        config.train_iter3.FLOW_R2I = False
    epoch = config.TEST.test_epoch
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    if len(ctx) != config.NUM_GPUS:
        print("********** WARNING: length of context doesn't match num_gpus set in config, {} vs. {} **********".\
            format(len(ctx), config.NUM_GPUS))

    image_set = config.dataset.test_image_set
    root_path = config.dataset.root_path
    dataset = config.dataset.dataset.split('+')[0]
    dataset_path = config.dataset.dataset_path

    if not os.path.basename(args.cfg).split('.')[0].endswith('temp'):
        new_args_name = os.path.basename(args.cfg).split('.')[0]+'_{}gpus.yaml'.format(config.NUM_GPUS)
    else:
        new_args_name = args.cfg
    if config.TEST.VISUALIZE or args.temp:
        logger, final_output_path = create_logger(config.output_path, new_args_name, image_set, True)
    else:
        logger, final_output_path = create_logger(config.output_path, new_args_name, image_set)
    prefix = os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix)

    pprint.pprint(config)
    logger.info('testing config:{}\n'.format(pprint.pformat(config)))

    # load symbol and testing data
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    if config.dataset.dataset.startswith('ModelNet'):
        imdb_test = eval(dataset)(config, image_set+config.dataset.class_name[0].split('/')[-1],
                                  root_path, dataset_path,
                              class_name=config.dataset.class_name[0],
                              result_path=final_output_path)
        print(imdb_test)
        pairdbs = [load_gt_pairdb(config, dataset, image_set + class_name.split('/')[-1], config.dataset.root_path,
                                  config.dataset.dataset_path,
                                  class_name=class_name,
                                  result_path=final_output_path)
                   for class_name in config.dataset.class_name]
        pairdb = merge_pairdb(pairdbs)
    else:
        imdb_test = eval(dataset)(config, image_set + config.dataset.class_name[0], root_path, dataset_path,
                                  class_name=config.dataset.class_name[0],
                                  result_path=final_output_path)
        print(imdb_test)
        pairdbs = [load_gt_pairdb(config, dataset, image_set+class_name, config.dataset.root_path,
                              config.dataset.dataset_path,
                              class_name=class_name, result_path=final_output_path)
               for class_name in config.dataset.class_name]
        pairdb = merge_pairdb(pairdbs)

    # get test data iter
    test_data = TestDataLoader(pairdb, config=config, batch_size=len(ctx))

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    # load model and check parameters
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]

    # create predictor
    predictor = Predictor(config, sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval(config, predictor, test_data, imdb_test, vis=args.vis, ignore_cache=args.ignore_cache, logger=logger, pairdb=pairdb)
    print(args.cfg, config.TEST.test_epoch)

def main():
    print(args)
    test_deepim()


if __name__ == '__main__':
    main()
