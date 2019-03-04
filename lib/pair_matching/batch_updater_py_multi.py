# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import division, print_function
import os
import cv2
import mxnet as mx
import numpy as np
from lib.pair_matching import RT_transform
from lib.flow_c.flow import gpu_flow_wrapper
import time
from lib.render_glumpy.render_py_multi import Render_Py

cur_dir = os.path.abspath(os.path.dirname(__file__))


class batchUpdaterPyMulti:
    def __init__(self, big_cfg, height, width):
        self.big_cfg = big_cfg
        self.model_dir = big_cfg.dataset.model_dir
        self.rot_coord = big_cfg.network.ROT_COORD
        self.pixel_means = big_cfg.network.PIXEL_MEANS[[2, 1, 0]]
        self.pixel_means = self.pixel_means.reshape([3, 1, 1]).astype(np.float32)
        self.K = big_cfg.dataset.INTRINSIC_MATRIX
        self.T_means = big_cfg.dataset.trans_means
        self.T_stds = big_cfg.dataset.trans_stds
        self.height = height
        self.width = width
        self.zNear = big_cfg.dataset.ZNEAR
        self.zFar = big_cfg.dataset.ZFAR

        self.render_machine = None
        if big_cfg.dataset.dataset.startswith("ModelNet"):
            self.modelnet_root = big_cfg.modelnet_root
            self.texture_path = os.path.join(self.modelnet_root, "gray_texture.png")
            from lib.render_glumpy.render_py_light_modelnet_multi import (
                Render_Py_Light_ModelNet_Multi,
            )

            self.model_path_list = [
                os.path.join(self.model_dir, "{}.obj".format(model_name))
                for model_name in big_cfg.dataset.class_name
            ]
            self.render_machine = Render_Py_Light_ModelNet_Multi(
                self.model_path_list,
                self.texture_path,
                self.K,
                self.width,
                self.height,
                self.zNear,
                self.zFar,
                brightness_ratios=[0.7],
            )
        else:
            self.render_machine = Render_Py(
                self.model_dir,
                big_cfg.dataset.class_name,
                self.K,
                self.width,
                self.height,
                self.zNear,
                self.zFar,
            )

        self.reinit = True

        self.batch_size = big_cfg.TRAIN.BATCH_PAIRS  # will update according to data
        self.Kinv = np.linalg.inv(np.matrix(self.K))
        print("build render_machine: ", self.render_machine)

    def get_names(self, big_cfg):
        """

        :param small_cfg:
        :return:
        """
        pred = ["image_observed", "image_rendered"]
        # pred = []
        if big_cfg.network.PRED_FLOW:
            pred.append("flow_est_crop")
            pred.append("flow_loss")
        pred.append("rot_est")
        pred.append("rot_gt")
        pred.append("trans_est")
        pred.append("trans_gt")
        if big_cfg.train_iter.SE3_DIST_LOSS:
            pred.append("rot_loss")
            pred.append("trans_loss")
        if big_cfg.train_iter.SE3_PM_LOSS:
            pred.append("point_matching_loss")
        if self.big_cfg["network"]["PRED_MASK"]:
            pred.append("zoom_mask_prob")
            pred.append("zoom_mask_gt_observed")
            pred.append("mask_pred")  # unzoomed

        return pred

    def forward(self, data_batch, preds, big_cfg):
        """
        :param data:
            image_observed
            image_rendered
            depth_gt_observed
            - depth_observed
            - depth_rendered
            - mask_real
            src_pose
            tgt_pose
        :param label:
            rot_i2r
            trans_i2r
            - flow_i2r
            - flow_i2r_weights
            - point_cloud_model
            - point_cloud_weights
            - point_cloud_real
        :param preds:
            image_observed
            image_rendered
            - flow_i2r_est
            - flow_i2r_loss
            rot_i2r
            trans_i2r
            - rot_i2r_loss
            - trans_i2r_loss
            - point_matching_loss
        :return updated_batch:
        """
        data_array = data_batch.data
        label_array = data_batch.label
        num_ctx = len(data_array)
        pred_names = self.get_names(big_cfg)
        init_time = 0
        render_time = 0
        image_time = 0
        flow_time = 0
        update_time = 0
        mask_time = 0
        io_time = 0
        data_names = [x[0] for x in data_batch.provide_data[0]]
        label_names = [x[0] for x in data_batch.provide_label[0]]
        src_pose_all = [
            data_array[ctx_i][data_names.index("src_pose")].asnumpy()
            for ctx_i in range(num_ctx)
        ]
        tgt_pose_all = [
            data_array[ctx_i][data_names.index("tgt_pose")].asnumpy()
            for ctx_i in range(num_ctx)
        ]
        class_index_all = [
            data_array[ctx_i][data_names.index("class_index")].asnumpy()
            for ctx_i in range(num_ctx)
        ]
        t = time.time()
        # print("pred lens: {}".format(len(preds)))
        # for i in preds:
        #     print(i[0].shape)
        rot_est_all = [
            preds[pred_names.index("rot_est")][ctx_i].asnumpy()
            for ctx_i in range(num_ctx)
        ]
        trans_est_all = [
            preds[pred_names.index("trans_est")][ctx_i].asnumpy()
            for ctx_i in range(num_ctx)
        ]
        init_time += time.time() - t

        if self.big_cfg.network.PRED_FLOW:
            depth_gt_observed_all = [
                data_array[ctx_i][data_names.index("depth_gt_observed")].asnumpy()
                for ctx_i in range(num_ctx)
            ]

        for ctx_i in range(num_ctx):
            batch_size = data_array[ctx_i][0].shape[0]
            assert batch_size == self.batch_size, "{} vs. {}".format(
                batch_size, self.batch_size
            )
            cur_ctx = data_array[ctx_i][0].context
            t = time.time()
            src_pose = src_pose_all[
                ctx_i
            ]  # data_array[ctx_i][data_names.index('src_pose')].asnumpy()
            tgt_pose = tgt_pose_all[
                ctx_i
            ]  # data_array[ctx_i][data_names.index('tgt_pose')].asnumpy()
            if self.big_cfg.network.PRED_FLOW:
                depth_gt_observed = depth_gt_observed_all[
                    ctx_i
                ]  # data_array[ctx_i][data_names.index('depth_gt_observed')] # ndarray

            class_index = class_index_all[
                ctx_i
            ]  # data_array[ctx_i][data_names.index('class_index')].asnumpy()
            rot_est = rot_est_all[
                ctx_i
            ]  # preds[pred_names.index('rot_est')][ctx_i].asnumpy()
            trans_est = trans_est_all[
                ctx_i
            ]  # preds[pred_names.index('trans_est')][ctx_i].asnumpy()
            init_time += time.time() - t

            refined_image_array = np.zeros((batch_size, 3, self.height, self.width))
            refined_depth_array = np.zeros((batch_size, 1, self.height, self.width))
            rot_res_array = np.zeros((batch_size, 4))
            trans_res_array = np.zeros((batch_size, 3))
            refined_pose_array = np.zeros((batch_size, 3, 4))
            KT_array = np.zeros((batch_size, 3, 4))
            for batch_idx in range(batch_size):
                pre_pose = np.squeeze(src_pose[batch_idx])
                r_delta = np.squeeze(rot_est[batch_idx])
                t_delta = np.squeeze(trans_est[batch_idx])

                refined_pose = RT_transform.RT_transform(
                    pre_pose,
                    r_delta,
                    t_delta,
                    self.T_means,
                    self.T_stds,
                    rot_coord=self.rot_coord,
                )
                t = time.time()
                if not self.big_cfg.dataset.dataset.startswith("ModelNet"):
                    refined_image, refined_depth = self.render_machine.render(
                        class_index[batch_idx].astype("int"),
                        refined_pose[:3, :3],
                        refined_pose[:3, 3],
                        r_type="mat",
                    )
                else:
                    idx = 2  # random.randint(0, 100)

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
                    # print("light_position a: {}".format(light_position))
                    light_position = np.array(light_position) * 0.5
                    # inverse yz
                    light_position[0] += refined_pose[0, 3]
                    light_position[1] -= refined_pose[1, 3]
                    light_position[2] -= refined_pose[2, 3]
                    # print("light_position b: {}".format(light_position))

                    colors = np.array([1, 1, 1])  # white light
                    intensity = np.random.uniform(0.9, 1.1, size=(3,))
                    colors_randk = 0  # random.randint(0, colors.shape[0] - 1)
                    light_intensity = colors[colors_randk] * intensity
                    # print('light intensity: ', light_intensity)

                    # randomly choose a render machine
                    rm_randk = 0  # random.randint(0, len(brightness_ratios) - 1)
                    refined_image, refined_depth = self.render_machine.render(
                        class_index[batch_idx].astype("int"),
                        refined_pose[:3, :3],
                        refined_pose[:3, 3],
                        light_position,
                        light_intensity,
                        brightness_k=rm_randk,
                        r_type="mat",
                    )
                render_time += time.time() - t

                # process refined_image
                t = time.time()
                refined_image = (
                    refined_image[:, :, [2, 1, 0]]
                    .transpose([2, 0, 1])
                    .astype(np.float32)
                )
                refined_image -= self.pixel_means
                image_time += time.time() - t

                # get se3_res
                rot_res, trans_res = RT_transform.calc_RT_delta(
                    refined_pose,
                    np.squeeze(tgt_pose[batch_idx]),
                    self.T_means,
                    self.T_stds,
                    rot_coord=self.rot_coord,
                    rot_type="QUAT",
                )
                # print('{}, {}: {}, {}'.format(ctx_i, batch_idx, r_delta, rot_res))

                refined_pose_array[batch_idx] = refined_pose
                refined_image_array[batch_idx] = refined_image
                refined_depth_array[batch_idx] = refined_depth.reshape(
                    (1, self.height, self.width)
                )
                rot_res_array[batch_idx] = rot_res
                trans_res_array[batch_idx] = trans_res

                se3_m = np.zeros([3, 4])
                se3_rotm, se3_t = RT_transform.calc_se3(
                    refined_pose, np.squeeze(tgt_pose[batch_idx])
                )
                se3_m[:, :3] = se3_rotm
                se3_m[:, 3] = se3_t
                KT_array[batch_idx] = np.dot(self.K, se3_m)

            if self.big_cfg.network.PRED_MASK:
                t = time.time()
                refined_mask_rendered_array = np.zeros(refined_depth_array.shape)
                refined_mask_rendered_array[
                    refined_depth_array > 0.2
                ] = 1  # if the mask_rendered input is depth
                mask_time += time.time() - t

            update_package = {
                "image_rendered": refined_image_array,
                "depth_rendered": refined_depth_array,
                "src_pose": refined_pose_array,
                "rot": rot_res_array,
                "trans": trans_res_array,
            }
            if self.big_cfg.network.PRED_FLOW:
                t = time.time()
                gpu_flow_machine = gpu_flow_wrapper(cur_ctx.device_id)
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.subplot(1,2,1)
                # plt.imshow(refined_depth_array[0,0])
                # plt.subplot(1,2,2)
                # plt.imshow(depth_gt_observed[0,0])
                # plt.show()

                refined_flow, refined_flow_valid = gpu_flow_machine(
                    refined_depth_array.astype(np.float32),
                    depth_gt_observed.astype(np.float32),
                    KT_array.astype(np.float32),
                    np.array(self.Kinv).astype(np.float32),
                )
                # problem with py3
                # print('updater, flow: ', refined_flow.shape, np.unique(refined_flow))
                # print('updater, flow weights: ', refined_flow_valid.shape, np.unique(refined_flow_valid))
                # print('KT: ', KT_array[0])
                # print('Kinv: ', self.Kinv)
                flow_time += time.time() - t
                refined_flow_weights = np.tile(refined_flow_valid, [1, 2, 1, 1])
                update_package["flow"] = refined_flow
                update_package["flow_weights"] = refined_flow_weights
            if self.big_cfg.network.INPUT_MASK:
                update_package["mask_rendered"] = refined_mask_rendered_array

            t = time.time()
            data_array[ctx_i] = self.update_data_batch(
                data_array[ctx_i], data_names, update_package
            )
            label_array[ctx_i] = self.update_data_batch(
                label_array[ctx_i], label_names, update_package
            )
            update_time += time.time() - t

        t = time.time()
        new_data_batch = mx.io.DataBatch(
            data=data_array,
            label=label_array,
            pad=data_batch.pad,
            index=data_batch.index,
            provide_data=data_batch.provide_data,
            provide_label=data_batch.provide_label,
        )
        io_time += time.time() - t
        # print("---------------------------------")
        # print("init_time: {:.3f} sec".format(init_time))
        # print("render_time: {:.3f} sec".format(render_time))
        # print("image_time: {:.3f} sec".format(image_time))
        # print("flow_time: {:.3f} sec".format(flow_time))
        # print("mask_time: {:.3f} sec".format(mask_time))
        # print("update_time: {:.3f} sec".format(update_time))
        # print("io_time: {:.3f} sec".format(io_time))
        # print("all_time: {:.3f} sec".format(time.time() - t_all))
        # print("---------------------------------")
        return new_data_batch

    def update_data_batch(self, data, data_names, update_package):
        import mxnet.ndarray as nd

        for blob_idx, blob_name in enumerate(data_names):
            if blob_name not in update_package:
                continue
            # print('blob_idx: {}, blob_name: {} -- {}'.format(blob_idx, blob_name, np.max(update_package[blob_name])))
            data[blob_idx] = nd.array(update_package[blob_name])
        return data


if __name__ == "__main__":
    # configs
    thresh = 1e-3
    step = 1e-4
    ctx = mx.gpu(0)
    # K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    K = np.array(
        [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
    )
    pixel_means = np.array([128, 127, 126])[:, np.newaxis, np.newaxis]
    T_means = np.array([0.0, 0.0, 0.0])
    T_stds = np.array([1.0, 1.0, 1.0])
    batch_size = 4
    num_3d_sample = 3000
    height = 480
    width = 640
    zNear = 0.1
    zFar = 6.0
    rot_coord = "MODEL"
    src_img_idx = ["{:06}".format(x * 100 + 1) for x in range(batch_size)]
    tgt_img_idx = ["{:06}".format(x * 100 + 11) for x in range(batch_size)]
    class_name = ["driller"]  # '002_master_chef_can'
    model_dir = os.path.join(cur_dir, "../../data/LINEMOD_Dataset/models/")
    pose_path = os.path.join(
        cur_dir,
        "../../data/render_v5/data/render_real/%s/0006/{}-pose.txt" % (class_name),
    )
    color_path = os.path.join(
        cur_dir,
        "../../data/render_v5/data/render_real/%s/0006/{}-color.png" % (class_name),
    )
    depth_path = os.path.join(
        cur_dir,
        "../../data/render_v5/data/render_real/%s/0006/{}-depth.png" % (class_name),
    )

    from easydict import EasyDict as edict

    big_cfg = edict()
    big_cfg.network = edict()
    big_cfg.network.ROT_COORD = "CAMERA"
    big_cfg.network.IMAGE_STRIDE = 0
    big_cfg.network.PIXEL_MEANS = np.array([128, 127, 126])
    big_cfg.dataset = edict()
    big_cfg.dataset.INTRINSIC_MATRIX = K
    big_cfg.dataset.trans_means = T_means
    big_cfg.dataset.trans_stds = T_stds
    big_cfg.dataset.ZNEAR = 0.25
    big_cfg.dataset.ZFAR = 6.0
    big_cfg.dataset.model_dir = model_dir
    big_cfg.dataset.class_name = class_name
    big_cfg.network.STANDARD_FLOW_REP = False
    big_cfg.network.WITH_MASK = True
    big_cfg.network.PRED_MASK = True
    big_cfg.network.PRED_FLOW = True

    last_small_cfg = edict()
    last_small_cfg["SE3_DIST_LOSS"] = False
    last_small_cfg["SE3_PM_LOSS"] = True

    big_cfg.TRAIN = edict()
    big_cfg.TRAIN.BATCH_PAIRS = batch_size
    big_cfg.train_iter = last_small_cfg

    bu_py = batchUpdaterPyMulti(big_cfg, height, width)

    from six.moves import cPickle
    import six

    with open("data_batch2.pkl", "rb") as fid:
        if six.PY3:
            old_data_batch = cPickle.load(fid, encoding="latin1")
        else:
            old_data_batch = cPickle.load(fid)
    with open("preds2.pkl", "rb") as fid:
        if six.PY3:
            preds = cPickle.load(fid, encoding="latin1")
        else:
            preds = cPickle.load(fid)
    new_data_batch = bu_py.forward(old_data_batch, preds, last_small_cfg)
    new_data_batch = bu_py.forward(old_data_batch, preds, last_small_cfg)

    t = time.time()
    new_data_batch = bu_py.forward(old_data_batch, preds, last_small_cfg)
    print("using {:.2} seconds".format(time.time() - t))

    with open("data_batch2.pkl", "rb") as fid:
        if six.PY3:
            old_data_batch = cPickle.load(fid, encoding="latin1")
        else:
            old_data_batch = cPickle.load(fid)
    # forward
    for i in range(2):
        import matplotlib.pyplot as plt

        data_names = [x[0] for x in old_data_batch.provide_data[i]]
        print("data_names:", data_names)
        label_names = [x[0] for x in old_data_batch.provide_label[i]]
        print("label_names:", label_names)
        old_data = old_data_batch.data[i]
        new_data = new_data_batch.data[i]
        old_label = old_data_batch.label[i]
        new_label = new_data_batch.label[i]
        old_image_rendered = old_data[data_names.index("image_rendered")].asnumpy()
        new_image_rendered = new_data[data_names.index("image_rendered")].asnumpy()

        old_image_observed = old_data[data_names.index("image_observed")].asnumpy()
        new_image_observed = new_data[data_names.index("image_observed")].asnumpy()

        if big_cfg.network.PRED_MASK:
            old_mask_rendered = old_data[data_names.index("mask_rendered")].asnumpy()
            new_mask_rendered = new_data[data_names.index("mask_rendered")].asnumpy()
            old_mask_observed = old_data[data_names.index("mask_observed")].asnumpy()
            new_mask_observed = new_data[data_names.index("mask_observed")].asnumpy()

        if last_small_cfg["FLOW"]:
            old_flow = old_label[label_names.index("flow")].asnumpy()
            old_flow_weights = old_label[label_names.index("flow_weights")].asnumpy()
            new_flow = new_label[label_names.index("flow")].asnumpy()
            new_flow_weights = new_label[label_names.index("flow_weights")].asnumpy()

        for j in range(batch_size):
            #
            def get_im(image, j, pixel_means=pixel_means):
                im = np.squeeze(image[j]).astype(np.float32)
                print("im:", np.max(im), np.min(im))
                im += pixel_means[[2, 1, 0], :, :]
                im[im < 0] = 0
                im[im > 255] = 255
                im = im.astype(np.uint8)
                im = im.transpose([1, 2, 0])
                return im

            old_im_i = get_im(old_image_rendered, j)
            new_im_i = get_im(new_image_rendered, j)

            old_im_r = get_im(old_image_observed, j)
            new_im_r = get_im(new_image_observed, j)

            new_rot_i2r = new_label[label_names.index("rot_i2r")].asnumpy()[j]
            new_trans_i2r = new_label[label_names.index("trans_i2r")].asnumpy()[j]
            src_pose = new_data[data_names.index("src_pose")].asnumpy()[j]
            tgt_pose = new_data[data_names.index("tgt_pose")].asnumpy()[j]
            refined_pose = RT_transform.RT_transform(
                src_pose,
                new_rot_i2r,
                new_trans_i2r,
                T_means,
                T_stds,
                rot_coord="CAMERA",
            )
            print(
                "src_pose: \n{}\nrefined_pose: \n{}\n, tgt_pose: \n{}\n, delta: \n{}\n".format(
                    src_pose, refined_pose, tgt_pose, refined_pose - tgt_pose
                )
            )

            def vis_flow(
                flow_i2r, flow_i2r_weights, sel_img_idx, image_observed, image_rendered
            ):
                flow_i2r = np.squeeze(flow_i2r[sel_img_idx, :, :, :]).transpose(1, 2, 0)
                flow_i2r_weights = np.squeeze(
                    flow_i2r_weights[sel_img_idx, :, :, :]
                ).transpose([1, 2, 0])
                visible = np.squeeze(flow_i2r_weights[:, :, 0]) != 0
                print(
                    "image_rendered: ",
                    image_rendered.shape,
                    image_rendered.min(),
                    image_rendered.max(),
                )

                fig = plt.figure()
                font_size = 5
                plt.axis("off")
                fig.add_subplot(2, 3, 1)
                plt.imshow(image_observed)
                plt.title("image_observed", fontsize=font_size)
                fig.add_subplot(2, 3, 2)
                plt.imshow(image_rendered)
                plt.title("image_rendered", fontsize=font_size)

                height = image_observed.shape[0]
                width = image_rendered.shape[1]
                mesh_observed = np.zeros((height, width, 3), np.uint8)
                mesh_rendered = np.zeros((height, width, 3), np.uint8)
                for h in range(0, height, 3):
                    for w in range(0, width, 3):
                        if visible[h, w]:
                            cur_flow = flow_i2r[h, w, :]
                            mesh_rendered = cv2.circle(
                                mesh_rendered,
                                (np.round(w).astype(int), np.round(h).astype(int)),
                                1,
                                (
                                    h * 255 / height,
                                    255 - w * 255 / width,
                                    w * 255 / width,
                                ),
                                5,
                            )

                            mesh_observed = cv2.circle(
                                mesh_observed,
                                (
                                    np.round(w + cur_flow[1]).astype(int),
                                    np.round(h + cur_flow[0]).astype(int),
                                ),
                                1,
                                (
                                    h * 255 / height,
                                    255 - w * 255 / width,
                                    w * 255 / width,
                                ),
                                5,
                            )

                fig.add_subplot(2, 3, 4)
                plt.imshow(mesh_observed)
                plt.title("mesh_observed", fontsize=font_size)

                fig.add_subplot(2, 3, 5)
                plt.imshow(mesh_rendered)
                plt.title("mesh_rendered", fontsize=font_size)

                plt.show()

            if big_cfg.network.PRED_FLOW:
                # print('old flow')
                # vis_flow(old_flow, old_flow_weights, j, old_im_r, old_im_i)
                print("new flow")
                vis_flow(new_flow, new_flow_weights, j, new_im_r, new_im_i)

            if big_cfg.network.PRED_MASK:

                def get_mask(mask, j):
                    m = np.squeeze(mask[j])
                    return m

                old_mask_i = get_mask(old_mask_rendered, j)
                old_mask_r_est = get_mask(old_mask_observed, j)
                new_mask_i = get_mask(new_mask_rendered, j)
                new_mask_r_est = get_mask(new_mask_observed, j)

                fig = plt.figure()
                font_size = 5
                plt.axis("off")
                fig.add_subplot(2, 2, 1)
                plt.imshow(old_mask_i)
                plt.title("old_mask_i", fontsize=font_size)
                fig.add_subplot(2, 2, 2)
                plt.imshow(old_mask_r_est)
                plt.title("old_mask_r_est", fontsize=font_size)

                fig.add_subplot(2, 2, 3)
                plt.imshow(new_mask_i)
                plt.title("new_mask_i", fontsize=font_size)
                fig.add_subplot(2, 2, 4)
                plt.imshow(new_mask_r_est)
                plt.title("new_mask_r_est", fontsize=font_size)
                plt.show()

            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(old_im_i)
            plt.title("old_im_i")
            fig.add_subplot(1, 2, 2)
            plt.imshow(new_im_i)
            plt.title("new_im_i")
            plt.show()
