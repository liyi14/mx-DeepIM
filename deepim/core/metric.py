# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division, absolute_import
import os, sys
import mxnet as mx
import numpy as np
import time

def get_flow_names_iter(cfg):
    label = []
    if cfg.network.PRED_MASK:
        label.append('mask_real_gt')
    label.append('rot')
    label.append('trans')
    if cfg.network.PRED_FLOW:
        label.append('flow')
        label.append('flow_weight')
    if cfg.train_iter.SE3_PM_LOSS:
        label.append('point_cloud_model')
        label.append('point_cloud_weights')
        label.append('point_cloud_real')

    pred = ['image_real', 'image_rendered']
    iter_idx = 0
    if cfg.network.PRED_FLOW:
        pred.append('flow_est_crop')
        pred.append('flow_loss')
    pred.append('rot_est')
    pred.append('rot_gt')
    pred.append('trans_est') # zoomed
    pred.append('trans_gt') # zoomed
    if cfg['train_iter']['SE3_DIST_LOSS']:
        pred.append('rot_loss')
        pred.append('trans_loss')
    if cfg['train_iter']['SE3_PM_LOSS']:
        pred.append('point_matching_loss')
    if cfg['network']['INPUT_MASK'] and cfg['network']['PRED_MASK']:
        pred.append('mask_prob') # zoomed
        pred.append('mask_gt') # zoomed
        pred.append('mask_pred')  # unzoomed

    #debug
    pred.append('debug_term')
    return pred, label

class Flow_L2LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg, iter_idx=-1):
        super(Flow_L2LossMetric, self).__init__('Flow_L2Loss')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.iter_idx = iter_idx
        self.show_interval = cfg.default.frequent

    def update(self, labels, preds):
        name1 = 'flow_loss' if self.iter_idx==-1 else 'flow_loss'
        flow_loss = preds[self.pred.index(name1)].asnumpy()
        name2 = 'flow_weight' if self.iter_idx == 0 else 'flow_weight'
        self.sum_metric += np.sum(flow_loss)
        self.num_inst += 480*640

class Flow_CurLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg, iter_idx=-1):
        super(Flow_CurLossMetric, self).__init__('Flow_CurLoss')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.iter_idx = iter_idx
        self.show_interval = cfg.default.frequent

    def update(self, labels, preds):
        name1 = 'flow_loss' if self.iter_idx==-1 else 'flow_loss'
        flow_loss = preds[self.pred.index(name1)].asnumpy()
        self.sum_metric = np.sum(flow_loss)
        self.num_inst = 480*640

class Rot_L2LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg, iter_idx=-1):
        super(Rot_L2LossMetric, self).__init__('Rot_L2Loss')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.iter_idx = iter_idx
        self.show_interval = cfg.default.frequent

    def update(self, labels, preds):
        name1 = 'rot_loss' if self.iter_idx==-1 else 'rot_loss'
        rot_loss = preds[self.pred.index(name1)].asnumpy()
        self.sum_metric += np.sum(rot_loss)
        self.num_inst += 1

class Trans_L2LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg, iter_idx=-1):
        super(Trans_L2LossMetric, self).__init__('Trans_L2Loss')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.iter_idx = iter_idx
        self.show_interval = cfg.default.frequent

    def update(self, labels, preds):
        name1 = 'trans_loss' if self.iter_idx==-1 else 'trans_loss'
        trans_loss = preds[self.pred.index(name1)].asnumpy()
        self.sum_metric += np.sum(trans_loss)
        self.num_inst += 1

class PointMatchingLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg, iter_idx=-1):
        super(PointMatchingLossMetric, self).__init__('PointMatchingLoss')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.iter_idx = iter_idx
        self.show_interval = cfg.default.frequent
        self.sample_per_iter = cfg['train_iter']['NUM_3D_SAMPLE']

    def update(self, labels, preds):
        name1 = 'point_matching_loss' if self.iter_idx==-1 else 'point_matching_loss'
        point_matching_loss = preds[self.pred.index(name1)].asnumpy()
        self.sum_metric += np.sum(point_matching_loss)
        self.num_inst += self.sample_per_iter

class MaskLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg, iter_idx=-1):
        super(MaskLossMetric, self).__init__('MaskLoss')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.iter_idx = iter_idx
        self.show_interval = cfg.default.frequent

    def update(self, labels, preds):
        name1 = 'mask_prob' if self.iter_idx == -1 else 'mask_prob'
        name2 = 'mask_gt' if self.iter_idx == -1 else 'mask_gt'
        mask_prob = preds[self.pred.index(name1)].asnumpy()
        mask_gt = preds[self.pred.index(name2)].asnumpy()
        mask_loss = - (mask_gt * np.log(mask_prob + 1e-19) + (1 - mask_gt) * np.log(1 - mask_prob + 1e-19))
        self.sum_metric += np.sum(mask_loss)
        self.num_inst += 480*640


class SimpleVisualize(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(SimpleVisualize, self).__init__('SimpleVisualize')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.show_interval = 1
        self.train_iter_size = cfg.network.TRAIN_ITER_SIZE
        self.cfg = cfg

    def update(self, labels, preds):
        import matplotlib.pyplot as plt
        import cv2
        num_imgs = preds[self.pred.index('image_real')].shape[0]
        sel_img_idx = np.random.randint(0, num_imgs, 1)
        image_real = preds[self.pred.index('image_real')].asnumpy()*0.9+128
        image_real = np.squeeze(image_real[sel_img_idx, :, :, :]).transpose(1, 2, 0)/255
        image_rendered = preds[self.pred.index('image_rendered')].asnumpy()+128
        image_rendered = np.squeeze(image_rendered[sel_img_idx, :, :, :]).transpose(1, 2, 0)/255
        zoom_mask_real_gt = np.squeeze(preds[-5].asnumpy()[sel_img_idx, 0, :, :])
        zoom_mask_real_est = np.squeeze(preds[-4].asnumpy()[sel_img_idx, 0, :, :])
        zoom_mask_rendered = np.squeeze(preds[-3].asnumpy()[sel_img_idx, 0, :, :])
        zoom_image_real = preds[-2].asnumpy()+128
        zoom_image_real = np.squeeze(zoom_image_real[sel_img_idx, :, :, :]).transpose(1, 2, 0)/255
        zoom_image_real = np.maximum(zoom_image_real, 0.0)
        zoom_image_real = np.minimum(zoom_image_real, 1.0)
        zoom_image_rendered = preds[-1].asnumpy()+128
        zoom_image_rendered = np.squeeze(zoom_image_rendered[sel_img_idx, :, :, :]).transpose(1, 2, 0)/255
        zoom_image_rendered = np.maximum(zoom_image_rendered, 0.0)
        zoom_image_rendered = np.minimum(zoom_image_rendered, 1.0)

        for iter_idx in range(1):
            print('rot_est_iter_{}: {}, \n' \
                  'rot_gt_iter_{} : {}'\
                .format(iter_idx, preds[self.pred.index('rot_est')].asnumpy()[sel_img_idx],
                        iter_idx, preds[self.pred.index('rot_gt')].asnumpy()[sel_img_idx]))
            print('trans_est_iter_{}: {}, \n' \
                  'trans_gt_iter_{} : {}'\
                .format(iter_idx, preds[self.pred.index('trans_est')].asnumpy()[sel_img_idx],
                        iter_idx, preds[self.pred.index('trans_gt')].asnumpy()[sel_img_idx]))
        if self.cfg['train_iter']['SE3_DIST_LOSS']:
            rot_loss_iter0 = preds[self.pred.index('rot_loss')].asnumpy()
            trans_loss_iter0 = preds[self.pred.index('trans_loss')].asnumpy()
            print('rot_loss_iter0: ', rot_loss_iter0[sel_img_idx])
            print('trans_loss_iter0: ', trans_loss_iter0[sel_img_idx])

        fig = plt.figure()
        fig.add_subplot(3, 3, 1)
        plt.axis('off')
        plt.title('image_real')
        plt.imshow(image_real)
        fig.add_subplot(3, 3, 2)
        plt.axis('off')
        plt.title('image_rendered')
        plt.imshow(image_rendered)
        fig.add_subplot(3, 3, 3)
        plt.axis('off')
        plt.title('image_real-image_rendered')
        plt.imshow(np.abs(image_real-image_rendered[:, :, [2,1,0]]))
        fig.add_subplot(3, 3, 4)
        plt.axis('off')
        plt.title('zoom_image_real')
        plt.imshow(zoom_image_real)
        fig.add_subplot(3, 3, 5)
        plt.axis('off')
        plt.title('zoom_image_rendered')
        plt.imshow(zoom_image_rendered)
        fig.add_subplot(3, 3, 6)
        plt.axis('off')
        plt.title('zoom_image_real-zoom_image_rendered')
        plt.imshow(np.abs(zoom_image_real-zoom_image_rendered))

        fig.add_subplot(3, 3, 7)
        plt.axis('off')
        plt.title('zoom_mask_real_input')
        plt.imshow(zoom_mask_real_est)
        fig.add_subplot(3, 3, 8)
        plt.axis('off')
        plt.title('zoom_mask_rendered_input')
        plt.imshow(zoom_mask_rendered)

        fig.add_subplot(3, 3, 9)
        plt.axis('off')
        plt.title('zoom_mask_real_gt')
        plt.imshow(zoom_mask_real_gt)
        plt.show()

class MaskVisualize(mx.metric.EvalMetric):
    def __init__(self, cfg, save_dir=None):
        super(MaskVisualize, self).__init__('MaskVisualize')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.show_interval = cfg.default.frequent #1
        self.train_iter_size = cfg.network.TRAIN_ITER_SIZE
        self.cfg = cfg
        self.save_dir = os.path.join(save_dir, 'yu_vis_train')
        self.max_to_keep = 100
        self.idx = 1

    def smooth_l1(self, diff):
        diff = np.abs(diff)
        rst = diff-0.5
        rst[diff<1] = np.square(diff[diff<1])/2
        return rst

    def l2(selfs, diff):
        return np.square(diff)

    def save_fig(self, im, save_path):
        import matplotlib
        # matplotlib.use('Agg') # when X11 is not available
        import matplotlib.pyplot as plt
        fig = plt.figure(frameon=False, figsize=(8, 6), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(im)
        plt.savefig(save_path, aspect='normal')

    def update(self, labels, preds):
        from lib.utils.mkdir_if_missing import mkdir_if_missing
        num_imgs = preds[self.pred.index('image_real')].shape[0]
        sel_img_idx = np.random.randint(0, num_imgs, 1)
        image_real = preds[self.pred.index('image_real')].asnumpy()*0.9 + 128
        image_real = np.squeeze(image_real[sel_img_idx, :, :, :]).transpose(1, 2, 0) / 255
        image_real = np.maximum(image_real, 0.0)
        image_real = np.minimum(image_real, 1.0)

        image_rendered = preds[self.pred.index('image_rendered')].asnumpy() + 128
        image_rendered = np.squeeze(image_rendered[sel_img_idx, :, :, :]).transpose(1, 2, 0) / 255

        zoom_image_real = preds[-2].asnumpy() + 128
        zoom_image_real = np.squeeze(zoom_image_real[sel_img_idx, :, :, :]).transpose(1, 2, 0) / 255
        zoom_image_real = np.maximum(zoom_image_real, 0.0)
        zoom_image_real = np.minimum(zoom_image_real, 1.0)
        zoom_image_rendered = preds[-1].asnumpy() + 128
        zoom_image_rendered = np.squeeze(zoom_image_rendered[sel_img_idx, :, :, :]).transpose(1, 2, 0) / 255
        zoom_image_rendered = np.maximum(zoom_image_rendered, 0.0)
        zoom_image_rendered = np.minimum(zoom_image_rendered, 1.0)

        if self.cfg.network.WITH_MASK and self.cfg.network.PRED_MASK:
            # input
            zoom_mask_real_gt = np.squeeze(preds[-5].asnumpy()[sel_img_idx, 0, :, :])
            zoom_mask_real_est = np.squeeze(preds[-4].asnumpy()[sel_img_idx, 0, :, :])
            zoom_mask_rendered = np.squeeze(preds[-3].asnumpy()[sel_img_idx, 0, :, :])

            # output
            zoom_mask_prob = np.squeeze(preds[self.pred.index('mask_prob_iter0')].asnumpy()[sel_img_idx, 0, :, :])
            zoom_mask_pred_bin = np.round(zoom_mask_prob)


        if self.cfg.network.PRED_FLOW: # flow
            import cv2
            flow_est = preds[self.pred.index('flow_est_crop')].asnumpy()
            print('flow_est:', flow_est.shape)
            flow_est = np.squeeze(flow_est[sel_img_idx, :, :, :]).transpose(1, 2, 0)
            flow_loss = preds[self.pred.index('flow_loss')].asnumpy()

            flow = labels[self.label.index('flow')].asnumpy()
            print('flow: ', flow.shape)
            flow = np.squeeze(flow[sel_img_idx, :, :, :]).transpose(1, 2, 0)
            flow_weights = labels[self.label.index('flow_weight')].asnumpy()
            flow_weights = np.squeeze(flow_weights[sel_img_idx, :, :, :]).transpose([1, 2, 0])
            visible = np.squeeze(flow_weights[:, :, 0]) != 0
            print('image_rendered: ', image_rendered.shape, image_rendered.min(), image_rendered.max())

            height = image_real.shape[0]
            width = image_rendered.shape[1]
            mesh_real = np.zeros((height, width, 3), np.uint8)
            mesh_rendered = np.zeros((height, width, 3), np.uint8)
            mesh_real_est = np.zeros((height, width, 3), np.uint8)
            for h in range(0, height, 3):
                for w in range(0, width, 3):
                    if visible[h, w]:
                        cur_flow = flow[h, w, :]
                        cur_flow_est = flow_est[h, w, :]
                        mesh_rendered = cv2.circle(mesh_rendered, (np.round(w).astype(int), np.round(h).astype(int)), 1,
                                               (h * 255 / height, 255 - w * 255 / width, w * 255 / width), 5)

                        mesh_real = cv2.circle(mesh_real,
                                               (np.round(w + cur_flow[1]).astype(int),
                                                np.round(h + cur_flow[0]).astype(int)), 1,
                                               (h * 255 / height, 255 - w * 255 / width, w * 255 / width), 5)

                        point = np.round([w + cur_flow_est[1], h + cur_flow_est[0]]).astype(int)
                        point[0] = min(max(point[0], 0), width)
                        point[1] = min(max(point[1], 0), height)
                        mesh_real_est = cv2.circle(mesh_real_est, (point[0], point[1]), 1,
                                                   (h * 255 / height, 255 - w * 255 / width, 127), 5)

            print('est_loss: {}'.format(np.sum(flow_weights * self.l2(flow - flow_est))))
            print('act_loss: {}'.format(np.sum(flow_loss)))

        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        mkdir_if_missing(self.save_dir)

        self.save_fig(image_real, '{}/{}_image_real.png'.format(self.save_dir, time_str))

        self.save_fig(image_rendered, '{}/{}_image_rendered.png'.format(self.save_dir, time_str))

        self.save_fig(zoom_image_real, '{}/{}_zoom_image_real.png'.format(self.save_dir, time_str))


        self.save_fig(zoom_image_rendered, '{}/{}_zoom_image_rendered.png'.format(self.save_dir, time_str))

        if self.cfg.network.PRED_MASK:
            self.save_fig(zoom_mask_real_est, '{}/{}_zoom_mask_real_est.png'.format(self.save_dir, time_str))

            self.save_fig(zoom_mask_real_gt, '{}/{}_zoom_mask_real_gt.png'.format(self.save_dir, time_str))

            self.save_fig(zoom_mask_rendered, '{}/{}_zoom_mask_rendered.png'.format(self.save_dir, time_str))

            self.save_fig(zoom_mask_pred_bin, '{}/{}_zoom_mask_pred_bin.png'.format(self.save_dir, time_str))

        if self.cfg.network.PRED_FLOW:
            self.save_fig(mesh_real, '{}/{}_mesh_real.png'.format(self.save_dir, time_str))

            self.save_fig(mesh_rendered, '{}/{}_mesh_rendered.png'.format(self.save_dir, time_str))

            self.save_fig(mesh_real_est, '{}/{}_mesh_real_est.png'.format(self.save_dir, time_str))
        print('=====================')

class MinibatchVisualize(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(MinibatchVisualize, self).__init__('MinibatchVisualize')
        self.pred, self.label = get_flow_names_iter(cfg)
        self.show_interval = 1 # cfg.default.frequent

    def smooth_l1(self, diff):
        diff = np.abs(diff)
        rst = diff - 0.5
        rst[diff<1] = np.square(diff[diff < 1]) / 2
        return rst

    def l2(selfs, diff):
        return np.square(diff)

    def update(self, labels, preds):

        if self.num_inst % self.show_interval*25 != 0:
            self.sum_metric = 666
            self.num_inst += 1
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        num_imgs = preds[self.pred.index('image_real')].shape[0]
        sel_img_idx = np.random.randint(0, num_imgs, 1)
        image_real = preds[self.pred.index('image_real')].asnumpy()+128
        image_real = np.squeeze(image_real[sel_img_idx, :, :, :]).transpose(1, 2, 0)/255
        image_rendered = preds[self.pred.index('image_rendered')].asnumpy() + 128
        image_rendered = np.squeeze(image_rendered[sel_img_idx, :, :, :]).transpose(1, 2, 0)/255
        flow_est = preds[self.pred.index('flow_est_crop')].asnumpy()
        print('flow_est:', flow_est.shape, np.unique(flow_est))
        flow_est = np.squeeze(flow_est[sel_img_idx, :, :, :]).transpose(1, 2, 0)

        flow_loss = preds[self.pred.index('flow_loss')].asnumpy()

        flow = labels[self.label.index('flow')].asnumpy()
        print('flow: ', flow.shape, np.unique(flow))
        flow = np.squeeze(flow[sel_img_idx, :, :, :]).transpose(1, 2, 0)
        flow_weights = labels[self.label.index('flow_weight')].asnumpy()
        flow_weights = np.squeeze(flow_weights[sel_img_idx, :, :, :]).transpose([1,2,0])
        visible = np.squeeze(flow_weights[:, :, 0]) != 0
        print('flow weights: ', visible.shape, np.unique(visible))
        print('image_rendered: ', image_rendered.shape, image_rendered.min(), image_rendered.max())

        fig = plt.figure()
        font_size = 5
        plt.axis('off')
        fig.add_subplot(2, 3, 1)
        plt.imshow(image_real)
        plt.title('image_real', fontsize=font_size)
        fig.add_subplot(2, 3, 2)
        plt.imshow(image_rendered)
        plt.title('image_rendered', fontsize=font_size)

        height = image_real.shape[0]
        width = image_rendered.shape[1]
        mesh_real = np.zeros((height, width, 3), np.uint8)
        mesh_rendered = np.zeros((height, width, 3), np.uint8)
        mesh_real_est = np.zeros((height, width, 3), np.uint8)
        for h in range(0, height, 3):
            for w in range(0, width, 3):
                if visible[h, w]:
                    cur_flow = flow[h, w, :]
                    cur_flow_est = flow_est[h, w, :]
                    mesh_rendered = cv2.circle(mesh_rendered, (np.round(w).astype(int), np.round(h).astype(int)), 1,
                                         (h * 255 / height, 255 - w * 255 / width, w * 255 / width), 5)

                    mesh_real = cv2.circle(mesh_real,
                                         (np.round(w + cur_flow[1]).astype(int), np.round(h + cur_flow[0]).astype(int)), 1,
                                         (h * 255 / height, 255 - w * 255 / width, w * 255 / width), 5)

                    point = np.round([w+cur_flow_est[1], h+cur_flow_est[0]]).astype(int)
                    point[0] = min(max(point[0], 0), width)
                    point[1] = min(max(point[1], 0), height)
                    mesh_real_est = cv2.circle(mesh_real_est, (point[0], point[1]), 1,
                                             (h*255/height, 255-w*255/width, 127), 5)

        print('est_loss: {}'.format(np.sum(flow_weights * self.l2(flow - flow_est))))
        print('act_loss: {}'.format(np.sum(flow_loss)))
        fig.add_subplot(2, 3, 4)
        plt.imshow(mesh_real)
        plt.title('mesh_real', fontsize=font_size)

        fig.add_subplot(2, 3, 5)
        plt.imshow(mesh_rendered)
        plt.title('mesh_rendered', fontsize=font_size)

        fig.add_subplot(2, 3, 6)
        plt.imshow(mesh_real_est)
        plt.title('mesh_real_est', fontsize=font_size)

        plt.show()

        self.sum_metric = 666
        self.num_inst += 1
