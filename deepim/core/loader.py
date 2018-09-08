# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import mxnet as mx
import random
from six.moves import xrange
import math

from mxnet.executor_manager import _split_input_slice
from lib.utils.image import tensor_vstack, my_tensor_vstack
from lib.pair_matching.data_pair import get_data_pair_train_batch, get_data_pair_test_batch
from PIL import Image
from multiprocessing import Pool

class TestDataLoader(mx.io.DataIter):
    def __init__(self, pairdb, config, batch_size=1, shuffle=False):
        super(TestDataLoader, self).__init__()

        # save parameters as properties
        self.pairdb = pairdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.config = config

        # infer properties from roidb
        self.size = len(self.pairdb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['image_real', 'image_rendered', 'src_pose', 'class_index']
        if config.network.INPUT_DEPTH:
            self.data_name.append('depth_real')
            self.data_name.append('depth_rendered')
        if config.network.INPUT_MASK:
            self.data_name.append('mask_real_est')
            self.data_name.append('mask_rendered')


        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [None for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        pairdb = [self.pairdb[self.index[i]] for i in range(cur_from, cur_to)]

        data, label, im_info = get_data_pair_test_batch(pairdb, self.config)

        self.data = [[mx.nd.array(data[i][name]) for name in self.data_name] for i in xrange(len(data))]
        self.im_info = im_info

class TrainDataLoader(mx.io.DataIter):
    def __init__(self, sym, pairdb, config, batch_size=1, shuffle=False, ctx=None, work_load_list=None):
        """
        This Iter will provide seg data to Deeplab network
        :param sym: to infer shape
        :param pairdb: must be preprocessed
        :param config: config file
        :param batch_size: must divide BATCH_SIZE(128)
        :param crop_height: the height of cropped image
        :param crop_width: the width of cropped image
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :return: DataLoader
        """
        super(TrainDataLoader, self).__init__()

        # save parameters as properties
        self.sym = sym
        self.pairdb = pairdb
        self.config = config
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.ctx = ctx

        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        # infer properties from pairdb
        self.size = len(pairdb)
        self.index = np.arange(self.size)

        # decide data and label names
        self.data_name = ['image_real', 'image_rendered', 'depth_render_real', 'class_index', 'src_pose', 'tgt_pose']

        if config.network.INPUT_DEPTH:
            self.data_name.append('depth_real')
            self.data_name.append('depth_rendered')

        if config.network.INPUT_MASK:
            self.data_name.append('mask_real_est')
            self.data_name.append('mask_rendered')

        self.label_name = ['rot', 'trans']

        if config.network.PRED_MASK:
            self.label_name.append('mask_real_gt')

        if config.network.PRED_FLOW:
            self.label_name.append('flow')
            self.label_name.append('flow_weights')

        if config.train_iter.SE3_PM_LOSS:
            self.label_name.append('point_cloud_model')
            self.label_name.append('point_cloud_weights')
            self.label_name.append('point_cloud_real')

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # init multi-process pool
        self.pool = Pool(processes=len(ctx))
        # self.pool = Pool(processes=self.batch_size)

        # get first batch to fill in provide_data and provide_label
        random.seed(6)
        np.random.seed(3)
        self.rseed = np.random.randint(999999, size=[99999])
        np.random.seed(self.rseed[0])
        np.delete(self.rseed, 0)
        self.reset()

        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.label))]

    @property
    def provide_data_single(self):
        print('data_single: ', [(k, v.shape) for k, v in zip(self.data_name, self.data[0])])
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        print('label_single: ', [(k, v.shape) for k, v in zip(self.label_name, self.label[0])])
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if np.size(self.rseed) == 0:
                self.rseed = np.random.random([9999])
            np.random.seed(self.rseed[0])
            np.delete(self.rseed, 0)
            np.random.shuffle(self.index)
            print(self.index[1:10])

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []

        max_shapes = dict(max_data_shape + max_label_shape)
        _, label_shape, _ = self.sym.infer_shape(**max_shapes)
        label_shape = [(self.label_name[0], label_shape)]
        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        pairdb = [self.pairdb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        multiprocess_results = []
        for idx, islice in enumerate(slices):
            """
            ipairdb = [pairdb[i] for i in range(islice.start, islice.stop)]
            multiprocess_results.append(self.pool.apply_async(
                get_data_pair_train_batch, (ipairdb, self.config)))
            """
            for i in range(islice.start, islice.stop):
                multiprocess_results.append(self.pool.apply_async(
                    get_data_pair_train_batch, ([pairdb[i]], self.config)))

        if False:
            temp = get_data_pair_train_batch([pairdb[islice.start]], self.config) # for debug
            print('**'*20)
            print(pairdb[0]['image_real'])
            print('data:')
            for k in temp['data'].keys():
                print("\t{}, {}".format(k, temp['data'][k].shape))
            print('label:')
            for k in temp['label'].keys():
                print("\t{}, {}".format(k, temp['label'][k].shape))
            print(temp['label']['rot'])
            print(temp['label']['trans'])
            from lib.pair_matching.RT_transform import calc_rt_dist_m
            r_dist, t_dist = calc_rt_dist_m(temp['data']['src_pose'][0], temp['data']['tgt_pose'][0])
            print("{}: R_dist: {}, T_dist: {}".format(self.cur, r_dist, t_dist))
            print('**'*20)
            image_real = (temp['data']['image_real'][0].transpose([1,2,0])+128).astype(np.uint8)
            print(np.max(image_real))
            print(np.min(image_real))
            image_rendered = (temp['data']['image_rendered'][0].transpose([1,2,0])+128).astype(np.uint8)
            mask_real_gt = np.squeeze(temp['label']['mask_real_gt'])
            mask_real_est = np.squeeze(temp['data']['mask_real_est'])
            mask_rendered = np.squeeze(temp['data']['mask_rendered'])
            if 'flow' in temp['label']:
                print('in loader, flow: ', temp['label']['flow'].shape, np.unique(temp['label']['flow']))
                print('in loader, flow weights: ',
                      temp['label']['flow_weights'].shape, np.unique(temp['label']['flow_weights']))
            import matplotlib.pyplot as plt
            plt.subplot(2,3,1)
            plt.imshow(mask_real_est)
            plt.subplot(2,3,2)
            plt.imshow(mask_real_gt)
            plt.subplot(2,3,3)
            plt.imshow(mask_rendered)
            plt.subplot(2,3,4)
            plt.imshow(image_real)
            plt.subplot(2,3,5)
            plt.imshow(image_rendered)
            plt.show()
            # plt.savefig('image_real_rendered_{}'.format(self.cur), aspect='normal')

        rst = [multiprocess_result.get() for multiprocess_result in multiprocess_results]

        batch_per_gpu = int(self.batch_size / len(ctx))
        data_list = []
        label_list = []
        for i in range(len(ctx)):
            sample_data_list = [_['data'] for _ in rst]
            sample_label_list = [_['label'] for _ in rst]
            batch_data = {}
            batch_label = {}
            for key in sample_data_list[0]:
                batch_data[key] = \
                    my_tensor_vstack([sample_data_list[j][key] for j in range(i*batch_per_gpu, (i+1)*batch_per_gpu)])
            for key in sample_label_list[0]:
                batch_label[key] = \
                    my_tensor_vstack([sample_label_list[j][key] for j in range(i*batch_per_gpu, (i+1)*batch_per_gpu)])
            data_list.append(batch_data)
            label_list.append(batch_label)

        """
        data_list = [_['data'] for _ in rst]
        label_list = [_['label'] for _ in rst]
        """
        self.data = [[mx.nd.array(data_on_i[key]) for key in self.data_name] for data_on_i in data_list]
        self.label = [[mx.nd.array(label_on_i[key]) for key in self.label_name] for label_on_i in label_list]

