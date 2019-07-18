# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
"""
flow updater
"""
from __future__ import print_function, division
import sys
import os

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../external/mxnet/mxnet_v00_origin"))

import cv2
import mxnet as mx
import numpy as np
from lib.pair_matching import RT_transform


class flowUpdaterOperator(mx.operator.CustomOp):
    def __init__(self, K, thresh, batch_size, height, width, wh_rep=False):
        super(flowUpdaterOperator, self).__init__()
        Kinv = np.linalg.inv(np.matrix(K))
        self.batch_K = np.tile(K.reshape((1, 3, 3)), (batch_size, 1, 1))
        self.thresh = thresh
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.spatial_dim = self.height * self.width
        self.wh_rep = wh_rep
        self.base_coord = None
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(1, width * height, 3)

        R = np.array(Kinv * x2d.transpose())  # Kinv is np.matrix type
        self.R = R
        self.batch_R = np.tile(R.reshape((1, 3, height * width)), (self.batch_size, 1, 1))

    def forward(self, is_train, req, in_data, out_data, aux):
        batch_size = in_data[0].shape[0]
        ctx = in_data[0].context

        depth_src = in_data[0]
        depth_tgt = in_data[1]
        pose_src = in_data[2].asnumpy()
        pose_tgt = in_data[3].asnumpy()

        if self.base_coord is None:
            self.batch_K = mx.nd.array(self.batch_K, ctx=ctx)
            self.R = mx.nd.array(self.R, ctx=ctx)
            self.batch_R = mx.nd.array(self.batch_R, ctx=ctx)

            w_ori, h_ori = np.meshgrid(
                np.linspace(0, self.width - 1, self.width), np.linspace(0, self.height - 1, self.height)
            )
            self.w_ori = np.tile(w_ori.reshape((1, 1, -1)), (self.batch_size, 1, 1))
            self.h_ori = np.tile(h_ori.reshape((1, 1, -1)), (self.batch_size, 1, 1))

        batch_se3_m = np.zeros((self.batch_size, 3, 4))
        for i in range(batch_size):
            se3_rotm, se3_t = RT_transform.calc_se3(pose_src[i], pose_tgt[i])
            batch_se3_m[i, :, :3] = se3_rotm
            batch_se3_m[i, :, 3] = se3_t
        batch_se3 = mx.nd.array(batch_se3_m, ctx=ctx)

        batch_ones = mx.nd.ones((batch_size, 1, self.width * self.height), ctx=ctx)
        X = mx.nd.broadcast_mul(depth_src.reshape((self.batch_size, 1, self.width * self.height)), self.batch_R)
        transform = mx.nd.batch_dot(self.batch_K, batch_se3)
        temp = mx.nd.concat(X, batch_ones, dim=1)
        Xp = mx.nd.batch_dot(transform, temp)
        w_proj, h_proj, z_proj = mx.nd.split(Xp, axis=1, num_outputs=3)
        z_proj = z_proj + 1e-15
        pw = mx.nd.minimum(mx.nd.maximum(mx.nd.round(w_proj / z_proj), 0), self.width - 1).asnumpy().astype(np.int)
        ph = mx.nd.minimum(mx.nd.maximum(mx.nd.round(h_proj / z_proj), 0), self.height - 1).asnumpy().astype(np.int)
        pz = z_proj.asnumpy()
        valid_in_src = (depth_src.reshape((self.batch_size, 1, self.height * self.width))).asnumpy() > 1e-10
        depth_tgt = depth_tgt.asnumpy()
        depth_mapping = np.array([depth_tgt[i, 0, ph[i], pw[i]] for i in range(self.batch_size)])
        visible_in_tgt = np.abs(depth_mapping - pz) < self.thresh
        valid_points = np.logical_and(valid_in_src, visible_in_tgt)

        w_diff = pw - self.w_ori
        w_diff[valid_points == 0] = 0
        w_diff.reshape((self.batch_size, 1, self.height * self.width))
        h_diff = ph - self.h_ori
        h_diff[valid_points == 0] = 0
        h_diff.reshape((self.batch_size, 1, self.height * self.width))
        if self.wh_rep:
            flow = np.concatenate([w_diff, h_diff], axis=1).reshape((self.batch_size, 2, self.height, self.width))
        else:
            flow = np.concatenate([h_diff, w_diff], axis=1).reshape((self.batch_size, 2, self.height, self.width))
        # print(flow)
        flow_nd = mx.nd.array(flow, ctx=ctx)
        valid_points = valid_points.reshape((self.batch_size, 1, self.height, self.width))
        flow_weights_nd = mx.nd.array(valid_points, ctx=ctx)
        flow_weights_nd = mx.nd.tile(flow_weights_nd, (1, 2, 1, 1))

        self.assign(out_data[0], req[0], flow_nd)
        self.assign(out_data[1], req[1], flow_weights_nd)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for ind, _ in enumerate(out_grad):
            self.assign(in_grad[ind], req[ind], 0)


@mx.operator.register("FlowUpdater")
class flowUpdaterProp(mx.operator.CustomOpProp):
    def __init__(self, K, thresh, batch_size, height, width, wh_rep="False"):
        super(flowUpdaterProp, self).__init__(True)
        self.K = np.fromstring(K[1:-1], dtype=np.float32, sep=" ").reshape((3, 3))
        self.thresh = float(thresh)
        self.batch_size = int(batch_size)
        self.height = int(height)
        self.width = int(width)
        self.wh_rep = wh_rep.lower() == "true"

    def list_arguments(self):
        return ["depth_src", "depth_tgt", "pose_src", "pose_tgt"]

    def list_outputs(self):
        output_list = ["flow", "flow_weights"]
        return output_list

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        height = in_shape[0][2]
        width = in_shape[0][3]

        depth_src_shape = (batch_size, 1, height, width)
        depth_tgt_shape = depth_src_shape
        pose_src_shape = in_shape[2]
        pose_tgt_shape = in_shape[3]
        input_shape = [depth_src_shape, depth_tgt_shape, pose_src_shape, pose_tgt_shape]

        flow_shape = (batch_size, 2, height, width)
        flow_weights_shape = flow_shape
        output_shape = [flow_shape, flow_weights_shape]

        return input_shape, output_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        input_type = [dtype, dtype, dtype, dtype]
        output_type = [dtype, dtype]
        return input_type, output_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return flowUpdaterOperator(self.K, self.thresh, self.batch_size, self.height, self.width, self.wh_rep)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # configs
    thresh = 1e-3
    step = 1e-4
    ctx = mx.gpu(0)
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    DEPTH_FACTOR = 10000.0
    flow_thresh = 3e-3
    batch_size = 8
    height = 480
    width = 640
    wh_rep = False
    src_img_idx = ["{:06}".format(x * 100 + 1) for x in range(batch_size)]
    tgt_img_idx = ["{:06}".format(x * 100 + 31) for x in range(batch_size)]
    class_name = "035_power_drill"  # '002_master_chef_can'
    model_dir = os.path.join(cur_dir, "../../data/LOV/models/{}".format(class_name))
    pose_path = os.path.join(cur_dir, "../../data/render_v5/data/render_real/%s/0006/{}-pose.txt" % (class_name))
    color_path = os.path.join(cur_dir, "../../data/render_v5/data/render_real/%s/0006/{}-color.png" % (class_name))
    depth_path = os.path.join(cur_dir, "../../data/render_v5/data/render_real/%s/0006/{}-depth.png" % (class_name))

    # initialize layer
    depth_rendered = mx.sym.Variable("depth_rendered")
    depth_real = mx.sym.Variable("depth_real")
    pose_src = mx.sym.Variable("pose_src")
    pose_tgt = mx.sym.Variable("pose_tgt")
    proj2d = mx.sym.Custom(
        depth_src=depth_rendered,
        depth_tgt=depth_real,
        pose_src=pose_src,
        pose_tgt=pose_tgt,
        K=K.flatten(),
        name="updater",
        op_type="FlowUpdater",
        batch_size=batch_size,
        height=height,
        width=width,
        thresh=flow_thresh,
        wh_rep=wh_rep,
    )

    # prepare input data
    v_depth_rendered = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_depth_real = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_pose_src = np.array([np.loadtxt(pose_path.format(x), skiprows=1) for x in src_img_idx])
    v_pose_tgt = np.array([np.loadtxt(pose_path.format(x), skiprows=1) for x in tgt_img_idx])
    for i in range(batch_size):
        v_depth_rendered[i, 0, :, :] = (
            cv2.imread(depth_path.format(src_img_idx[i]), cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR
        )
        v_depth_real[i, 0, :, :] = cv2.imread(depth_path.format(tgt_img_idx[i]), cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR

    # bind
    exe1 = proj2d.simple_bind(
        ctx=ctx,
        depth_rendered=v_depth_rendered.shape,
        depth_real=v_depth_real.shape,
        pose_src=v_pose_src.shape,
        pose_tgt=v_pose_tgt.shape,
    )

    # forward
    for i in range(5):
        exe1.arg_dict["depth_rendered"][:] = mx.nd.array(v_depth_rendered, ctx=ctx)
        exe1.arg_dict["depth_real"][:] = mx.nd.array(v_depth_real, ctx=ctx)
        exe1.arg_dict["pose_src"][:] = mx.nd.array(v_pose_src, ctx=ctx)
        exe1.arg_dict["pose_tgt"][:] = mx.nd.array(v_pose_tgt, ctx=ctx)
        t = time.time()
        exe1.forward(is_train=True)
        flow_all = exe1.outputs[0].asnumpy()
        flow_weights_all = exe1.outputs[1].asnumpy()
        print("using {:.2} seconds".format(time.time() - t))

        for j in range(batch_size):
            img_rendered = cv2.imread(color_path.format(src_img_idx[j]), cv2.IMREAD_COLOR)
            img_rendered = img_rendered[:, :, [2, 1, 0]]
            img_real = cv2.imread(color_path.format(tgt_img_idx[j]), cv2.IMREAD_COLOR)
            img_real = img_real[:, :, [2, 1, 0]]

            img_tgt = np.zeros((height, width, 3), np.uint8)
            img_src = np.zeros((height, width, 3), np.uint8)

            print("flow_all(unique): \n", np.unique(flow_all))
            flow = np.squeeze(flow_all[j, :, :, :].transpose((1, 2, 0)))
            flow_weights = np.squeeze(flow_weights_all[j, 0, :, :])
            for h in range(height):
                for w in range(width):
                    if flow_weights[h, w]:
                        cur_flow = flow[h, w, :]
                        img_src = cv2.line(
                            img_src,
                            (np.round(w).astype(int), np.round(h).astype(int)),
                            (np.round(w).astype(int), np.round(h).astype(int)),
                            (255, h * 255 / height, w * 255 / width),
                            5,
                        )
                        img_tgt = cv2.line(
                            img_tgt,
                            (np.round(w + cur_flow[1]).astype(int), np.round(h + cur_flow[0]).astype(int)),
                            (np.round(w + cur_flow[1]).astype(int), np.round(h + cur_flow[0]).astype(int)),
                            (255, h * 255 / height, w * 255 / width),
                            5,
                        )

            depth_rendered = cv2.imread(depth_path.format(src_img_idx[j]), cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR

            fig = plt.figure()
            fig.add_subplot(2, 3, 1)
            plt.imshow(img_rendered)
            plt.title("img_rendered")

            fig.add_subplot(2, 3, 2)
            plt.imshow(img_real)
            plt.title("img_real")

            fig.add_subplot(2, 3, 3)
            plt.imshow(depth_rendered)
            plt.title("depth_rendered")

            fig.add_subplot(2, 3, 4)
            plt.imshow(img_src)
            plt.title("flow_img_src")
            plt.axis("off")

            fig.add_subplot(2, 3, 5)
            plt.imshow(img_tgt)
            plt.title("flow_img_tgt")
            plt.axis("off")

            fig.add_subplot(2, 3, 6)
            plt.imshow(img_src - img_tgt)
            plt.title("flow_img_src - flow_img_tgt")
            plt.axis("off")
            plt.show()
