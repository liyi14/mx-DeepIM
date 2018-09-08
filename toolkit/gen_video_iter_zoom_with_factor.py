# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang
# --------------------------------------------------------
'''
python toolkit/gen_video_iter_zoom_with_factor.py --exp_dir /data/wanggu/PoseEst/mx-DeepPose/output/deepim_v3/deepim_v3_flownet_LM_SIXD_OCC_v1_se3_ex_u2s16_multi_all_iter_v01_zoom_RFMx4_camera2_ds_light_8epoch_4gpus/yu_val_mini_/video_iter
'''
from __future__ import division, print_function
import os
import sys
import numpy as np
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, '..'))
import cv2
sys.path.insert(1, os.path.join(cur_path, '../../external/mxnet/mxnet_v00_0303'))
import mxnet as mx
sys.path.insert(1, os.path.join(cur_path, '../deepim'))
from operator_py.zoom_image_with_factor import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from lib.utils.mkdir_if_missing import mkdir_if_missing
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate video iter with zoom factor')
    parser.add_argument('--exp_dir', required=True, help='exp_dir for generating video iter, e.g.: '
        'output/deepim_v3/deepim_v3_flownet_LM_SIXD_OCC_v1_se3_ex_u2s16_multi_all_iter_v01_zoom_RFMx4_camera2_ds_light_8epoch_4gpus/yu_val_mini_/video_iter')
    args = parser.parse_args()
    return args


def my_cmp(x, y):
    x_idx = int(x.split('/')[-1].split('.')[0].split('_')[-1])
    y_idx = int(y.split('/')[-1].split('.')[0].split('_')[-1])
    return x_idx - y_idx

idx2class = {
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


def class2idx(class_name):
    for k, v in idx2class.items():
        if v == class_name:
            return k

def load_object_diameter():
    diameters = {}
    models_info_file = os.path.join(cur_path, '../data/LINEMOD_6D/models/models_info.txt')
    assert os.path.exists(models_info_file), 'Path does not exist: {}'.format(models_info_file)
    with open(models_info_file, 'r') as f:
        for line in f:
            line_list = line.strip().split()
            cls_idx = int(line_list[0])
            if not cls_idx in idx2class.keys():
                continue
            diameter = float(line_list[2])
            cls_name = idx2class[cls_idx]
            diameters[cls_name] = diameter / 1000.

    return diameters

diameters = load_object_diameter()

def read_info(pose_info_path):
    with open(pose_info_path, 'r') as f:
        infos = [line.strip('\r\n') for line in f.readlines()]
        if 'initial' in infos[2]:
            title = 'Initial Pose'
        elif 'iter1' in infos[2]:
            title = 'Iter_1 Pose'
        elif 'iter2' in infos[2]:
            title = 'Iter_2 Pose'
        elif 'iter3' in infos[2]:
            title = 'Iter_3 Pose'
        elif 'iter4' in infos[2]:
            title = 'Iter_4 Pose'
        elif 'gt' in infos[2]:
            title = 'Ground Truth Pose'
        else:
            pass
        # infos[2] = infos[2][infos[2].find('r_dist'):]
        cls_name = infos[1].split('_')[1]
        diameter = diameters[cls_name]
        # ADD/ADI metric
        add = float(infos[3].split(':')[1])
        add_name = infos[3].split(':')[0]
        add_rel = add / diameter * 100
        add_info = "{}: {:.2f}/{:.2f}={:.2f}%".format(add_name, add*1000, diameter*1000, add_rel)
        # color_info = "gray: gt, red: initial, green: refined"
        legend = add_info #"{}\n{}".format(add_info, color_info)

        zoom_factor_str = infos[-1]
        zoom_factor = [float(e) for e in zoom_factor_str.split()]
        zoom_factor = np.array(zoom_factor)
        # legend = '\n'.join(infos[1:-1])

    return (title, legend, zoom_factor)

def main():

    args = parse_args()
    exp_dir = args.exp_dir
    process_images = True
    print('exp_dir: ', exp_dir)

    ctx = mx.gpu(0)
    pixel_means = np.array([0, 0, 0]) 
    batch_size = 1
    height = 600
    width = 800

    # initialize layer
    image_real_sym = mx.sym.Variable('image_real')
    image_rendered_sym = mx.sym.Variable('image_rendered')
    zoom_factor_sym = mx.sym.Variable('zoom_factor')
    zoom_op = mx.sym.Custom(zoom_factor=zoom_factor_sym, image_real=image_real_sym, image_rendered=image_rendered_sym,
                           pixel_means=pixel_means.flatten(),
                           name='updater', op_type='ZoomImageWithFactor',
                           height=height, width=width, high_light_center=False)


    pose_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if 'pose' in d]
    pose_dirs = sorted(pose_dirs)


    pose_path_list = []
    for pose_dir in pose_dirs:
        files = [os.path.join(pose_dir, fn) for fn in os.listdir(pose_dir) if '.png' in fn]
        files = sorted(files)
        for i in range(len(files)):
            if i == 0 or i == len(files) - 1:
                for j in range(5):
                    pose_path_list.append(files[i])
            else:
                pose_path_list.append(files[i])


    save_dir = os.path.join(exp_dir, '../zoom_video_iter/')
    mkdir_if_missing(save_dir)

    # zoom in
    def get_zoom_iamges(image_path, save_dir, is_initial=False, is_last=False, use_first_zoom_factor=True):
        legend_loc = 50
        t = 1
        cmap = {1: [1.0, 0.0, 0.0, t], 2: [0.75, 0.75, 0.75, t], 3: [0.0, 1.0, 0.0, t]}
        labels = {1: 'Initial', 2: 'GT', 3: 'Refined'}
        patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in range(1, 4)]
        if use_first_zoom_factor:
            init_info_path = image_path.replace(os.path.basename(image_path)[os.path.basename(image_path).find('iter'):],
                                           'iter_00_info.txt')
            _, _, zoom_factor = read_info(init_info_path)

            info_path = image_path.replace('.png', '_info.txt')
            title, legend, _ = read_info(info_path)
        else:
            info_path = image_path.replace('.png', '_info.txt')
            title, legend, zoom_factor = read_info(info_path)

        zoom_factor = zoom_factor[None, :]
        # print(zoom_factor)
        image_real = cv2.imread(image_path, cv2.IMREAD_COLOR).transpose([2, 0, 1])[None, :, :, :]
        image_rendered = image_real.copy()
        exe1 = zoom_op.simple_bind(ctx=ctx, zoom_factor=zoom_factor.shape, image_real=image_real.shape,
                                   image_rendered=image_rendered.shape)

        def simple_forward(exe1, zoom_factor, image_real, image_rendered, ctx=ctx, is_train=False):
            print('zoom factor: ', zoom_factor)
            exe1.arg_dict['zoom_factor'][:] = mx.nd.array(zoom_factor, ctx=ctx)
            exe1.arg_dict['image_real'][:] = mx.nd.array(image_real, ctx=ctx)
            exe1.arg_dict['image_rendered'][:] = mx.nd.array(image_rendered, ctx=ctx)
            exe1.forward(is_train=is_train)

        if is_initial:
            # original
            fig = plt.figure(frameon=False, figsize=(8, 6), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            # print(image_real[0].shape)
            ax.imshow(image_real[0].transpose((1,2,0))[:, :, [2, 1, 0]])
            fig.gca().text(10, 25, title, color='green', bbox=dict(facecolor='white', alpha=0.8))
            fig.gca().text(10, legend_loc, legend, color='red', bbox=dict(facecolor='white', alpha=0.8))
            plt.legend(handles=patches, loc=4, borderaxespad=0.)
            # plt.show()
            save_d = os.path.join(save_dir, os.path.dirname(image_path).split('/')[-1])
            mkdir_if_missing(save_d)
            save_path = os.path.join(save_d, os.path.basename(image_path).replace('.png', '_0.png'))
            plt.savefig(save_path, aspect='normal')
            plt.close()


            #################### (1/3)
            wx, wy, tx, ty = zoom_factor[0]
            delta = (1 - wx) / 3
            zoom_factor_1 = np.zeros((1, 4))
            zoom_factor_1[0, 0] = 1 - delta
            zoom_factor_1[0, 1] = 1 - delta
            zoom_factor_1[0, 2] = tx / 3
            zoom_factor_1[0, 3] = ty / 3

            simple_forward(exe1, zoom_factor_1, image_real, image_rendered, ctx=ctx, is_train=True)
            zoom_image_real = exe1.outputs[0].asnumpy()[0].transpose((1, 2, 0)) + pixel_means
            zoom_image_real[zoom_image_real < 0] = 0
            zoom_image_real[zoom_image_real > 255] = 255
            zoom_image_real = zoom_image_real.astype('uint8')
            fig = plt.figure(frameon=False, figsize=(8, 6), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(zoom_image_real[:, :, [2, 1, 0]])
            fig.gca().text(10, 25, title, color='green', bbox=dict(facecolor='white', alpha=0.8))
            fig.gca().text(10, legend_loc, legend, color='red', bbox=dict(facecolor='white', alpha=0.8))
            plt.legend(handles=patches, loc=4, borderaxespad=0.)
            save_d = os.path.join(save_dir, os.path.dirname(image_path).split('/')[-1])
            mkdir_if_missing(save_d)
            save_path = os.path.join(save_d, os.path.basename(image_path).replace('.png', '_1.png'))
            plt.savefig(save_path, aspect='normal')
            # plt.show()
            plt.close()

            ##################### (2/3)
            zoom_factor_2 = np.zeros((1, 4))
            zoom_factor_2[0, 0] = 1 - 2 * delta
            zoom_factor_2[0, 1] = 1 - 2 * delta
            zoom_factor_2[0, 2] = tx / 3 * 2
            zoom_factor_2[0, 3] = ty / 3 * 2

            simple_forward(exe1, zoom_factor_2, image_real, image_rendered, ctx=ctx, is_train=True)
            zoom_image_real = exe1.outputs[0].asnumpy()[0].transpose((1, 2, 0)) + pixel_means
            zoom_image_real[zoom_image_real < 0] = 0
            zoom_image_real[zoom_image_real > 255] = 255
            zoom_image_real = zoom_image_real.astype('uint8')
            fig = plt.figure(frameon=False, figsize=(8, 6), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(zoom_image_real[:, :, [2, 1, 0]])
            fig.gca().text(10, 25, title, color='green', bbox=dict(facecolor='white', alpha=0.8))
            fig.gca().text(10, legend_loc, legend, color='red', bbox=dict(facecolor='white', alpha=0.8))
            plt.legend(handles=patches, loc=4, borderaxespad=0.)
            save_d = os.path.join(save_dir, os.path.dirname(image_path).split('/')[-1])
            mkdir_if_missing(save_d)
            save_path = os.path.join(save_d, os.path.basename(image_path).replace('.png', '_2.png'))
            plt.savefig(save_path, aspect='normal')
            # plt.show()
            plt.close()

        ####################### (3/3)
        simple_forward(exe1, zoom_factor, image_real, image_rendered, ctx=ctx, is_train=True)
        zoom_image_real = exe1.outputs[0].asnumpy()[0].transpose((1, 2, 0)) + pixel_means
        zoom_image_real[zoom_image_real < 0] = 0
        zoom_image_real[zoom_image_real > 255] = 255
        zoom_image_real = zoom_image_real.astype('uint8')
        fig = plt.figure(frameon=False, figsize=(8, 6), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(zoom_image_real[:, :, [2, 1, 0]])
        fig.gca().text(10, 25, title, color='green', bbox=dict(facecolor='white', alpha=0.8))
        fig.gca().text(10, legend_loc, legend, color='red', bbox=dict(facecolor='white', alpha=0.8))
        plt.legend(handles=patches, loc=4, borderaxespad=0.)
        save_d = os.path.join(save_dir, os.path.dirname(image_path).split('/')[-1])
        mkdir_if_missing(save_d)
        if is_initial:
            save_path = os.path.join(save_d, os.path.basename(image_path).replace('.png', '_3.png'))
            # plt.show()
            plt.savefig(save_path, aspect='normal')
            plt.close()
        elif is_last:
            save_path_0 = os.path.join(save_d, os.path.basename(image_path).replace('.png', '_0.png'))
            save_path_1 = os.path.join(save_d, os.path.basename(image_path).replace('.png', '_1.png'))
            save_path_2 = os.path.join(save_d, os.path.basename(image_path).replace('.png', '_2.png'))
            plt.savefig(save_path_0, aspect='normal')
            plt.savefig(save_path_1, aspect='normal')
            plt.savefig(save_path_2, aspect='normal')
            # plt.show()
            plt.close()
        else:
            save_path = os.path.join(save_d, os.path.basename(image_path))
            plt.savefig(save_path, aspect='normal')
            # plt.show()
            plt.close()

    if process_images:
        use_first_zoom_factor = False
        print('saving processed images to {}'.format(save_dir))
        for image_path in tqdm(pose_path_list):
            # image_path = pose_path_list[0]
            if 'iter_00' in image_path:
                is_initial = True
            else:
                is_initial = False

            if 'iter_04' in image_path:
                is_last = True
            else:
                is_last = False
            # if 'iter_05' in image_path: # gt
            #     continue

            get_zoom_iamges(image_path, save_dir=save_dir, is_initial=is_initial, is_last=is_last,
                            use_first_zoom_factor=use_first_zoom_factor)


    ########################################
    # generate video with new images
    new_pose_dirs = [os.path.join(save_dir, d) for d in os.listdir(save_dir) if 'pose' in d]
    new_pose_dirs = sorted(new_pose_dirs)

    new_pose_path_list = []
    for new_pose_dir in new_pose_dirs:
        files = [os.path.join(new_pose_dir, fn) for fn in os.listdir(new_pose_dir) if '.png' in fn]
        files = sorted(files)
        for i in range(len(files)):
            if i == 0 or i == len(files) - 1:
                for j in range(1):
                    new_pose_path_list.append(files[i])
            else:
                new_pose_path_list.append(files[i])

    N = len(new_pose_path_list)
    images_dict = {k: [] for k in ['pose']}
    print('loading images...')
    for i in tqdm(range(N)):
        images_dict['pose'].append(cv2.imread(new_pose_path_list[i],
                                            cv2.IMREAD_COLOR))


    height, width, channel = images_dict['pose'][0].shape
    print(height, width)
    width = 800
    height = 600

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    video_pose_zoom = cv2.VideoWriter(os.path.join(exp_dir, '../video_full/pose_iter_zoom.avi'), fourcc, 2.0, (width, height))


    print('writing video...')
    for i in tqdm(range(N)):
        res_img = images_dict['pose'][i]
        if res_img.shape[0] == 480:
            im_scale = 600.0/480.0
            res_img = cv2.resize(res_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        video_pose_zoom.write(res_img)

    video_pose_zoom.release()

    os.popen('ffmpeg -i {} -vcodec mpeg4 -acodec copy -preset placebo -crf 1 -b:v 1550k {}'.format(
        os.path.join(exp_dir, '../video_full/pose_iter_zoom.avi'),
        os.path.join(exp_dir, '../video_full/pose_iter_zoom_compressed.avi')))


if __name__ == '__main__':
    main()



