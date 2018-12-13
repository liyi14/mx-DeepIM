# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
from glumpy import app, gl, gloo, data, log
import logging

log.setLevel(logging.WARNING)  # ERROR, WARNING, DEBUG, INFO
from lib.pair_matching import RT_transform

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 position;
attribute vec3 normal;
attribute vec2 texcoord;
varying vec3   v_normal;
varying vec3   v_position;
varying vec2   v_texcoord;

void main()
{
    // Assign varying variables
    v_normal   = normal;
    v_position = position;
    v_texcoord = texcoord;

    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
}
"""


def get_fragment(brightness_ratio=0.4):
    fragment = """
    uniform mat4      u_model;           // Model matrix
    uniform mat4      u_view;            // View matrix
    uniform mat4      u_normal;          // Normal matrix
    uniform mat4      u_projection;      // Projection matrix
    uniform sampler2D u_texture;         // Texture
    uniform vec3      u_light_position;  // Light position
    uniform vec3      u_light_intensity; // Light intensity

    varying vec3      v_normal;          // Interpolated normal (in)
    varying vec3      v_position;        // Interpolated position (in)
    varying vec2      v_texcoord;        // Interpolated fragment texture coordinates (in)
    void main()
    {{
        // Calculate normal in world coordinates
        vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

        // Calculate the location of this fragment (pixel) in world coordinates
        vec3 position = vec3(u_view*u_model * vec4(v_position, 1));
        //vec3 light_position = vec3(u_view*u_model * vec4(u_light_position, 1));
        // Calculate the vector from this pixels surface to the light source
        vec3 surfaceToLight = u_light_position - position;

        // Calculate the cosine of the angle of incidence (brightness)
        float brightness = dot(normal, surfaceToLight) /
                          (length(surfaceToLight) * length(normal));
        brightness = max(min(brightness,1.0),0.0);

        // Calculate final color of the pixel, based on:
        // 1. The angle of incidence: brightness
        // 2. The color/intensities of the light: light.intensities
        // 3. The texture and texture coord: texture(tex, fragTexCoord)

        // Get texture color
        vec4 t_color = vec4(texture2D(u_texture, v_texcoord).rgb, 1.0);

        // Final color
        gl_FragColor = t_color * ({} + {}*brightness * vec4(u_light_intensity, 1));
    }}
    """.format(1 - brightness_ratio, brightness_ratio)
    return fragment


class Render_Py_Light_MultiProgram():
    def __init__(self,
                 class_name_list,
                 model_folder_dict,
                 K,
                 width=640,
                 height=480,
                 zNear=0.25,
                 zFar=6.0,
                 brightness_ratios=[0.4, 0.3, 0.2]):
        self.width = width
        self.height = height
        self.zNear = zNear
        self.zFar = zFar
        self.K = K
        self.model_folder_dict = model_folder_dict
        self.class_name_list = class_name_list

        self.render_kernels = {
        }  # self.render_kernels[class_name][brightness_ratio]
        for cls_name in class_name_list:
            if cls_name == "__background__":
                continue
            model_folder = model_folder_dict[cls_name]
            log.info("Loading mesh")
            vertices, indices = data.objload(
                "{}/textured.obj".format(model_folder), rescale=False)
            if cls_name not in self.render_kernels.keys():
                self.render_kernels[cls_name] = []
            for brightness_ratio in brightness_ratios:
                print('class_name: {}, brightness_ratio: {}, model_folder: {}'.
                      format(cls_name, brightness_ratio, model_folder))
                fragment = get_fragment(brightness_ratio=brightness_ratio)
                render_kernel = gloo.Program(vertex, fragment)
                render_kernel.bind(vertices)

                log.info("Loading texture")
                render_kernel['u_texture'] = np.copy(
                    data.load(
                        "{}/texture_map.png".format(model_folder))[::-1, :, :])

                render_kernel['u_model'] = np.eye(4, dtype=np.float32)
                u_projection = self.my_compute_calib_proj(
                    K, width, height, zNear, zFar)
                render_kernel['u_projection'] = np.copy(u_projection)

                render_kernel['u_light_intensity'] = 1, 1, 1
                self.render_kernels[cls_name].append(render_kernel)

        self.class_name = class_name_list[-1]
        self.brightness_k = 0  # init

        self.window = app.Window(width=width, height=height, visible=False)

        @self.window.event
        def on_draw(dt):
            self.window.clear()
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            # print('brightness_k', self.brightness_k) # this function runs when running app.run()
            self.render_kernels[self.class_name][self.brightness_k].draw(
                gl.GL_TRIANGLES)

        @self.window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)

    def render(self,
               r,
               t,
               light_position,
               light_intensity,
               class_name,
               brightness_k=0,
               r_type='quat'):
        '''
        :param r:
        :param t:
        :param light_position:
        :param light_intensity:
        :param brightness_k: choose which brightness in __init__
        :param r_type:
        :return:
        '''
        if r_type == 'quat':
            R = RT_transform.quat2mat(r)
        elif r_type == 'mat':
            R = r
        self.class_name = class_name
        self.brightness_k = brightness_k
        self.render_kernels[
            self.class_name][brightness_k]['u_view'] = self._get_view_mtx(
                R, t)
        self.render_kernels[
            self.class_name][brightness_k]['u_light_position'] = light_position
        self.render_kernels[
            self.class_name][brightness_k]['u_normal'] = np.array(
                np.matrix(
                    np.dot(
                        self.render_kernels[self.class_name][brightness_k]
                        ['u_view'].reshape(4, 4), self.render_kernels[
                            self.class_name][brightness_k]['u_model'].reshape(
                                4, 4))).I.T)
        self.render_kernels[self.class_name][brightness_k][
            'u_light_intensity'] = light_intensity

        app.run(framecount=0, framerate=0)
        rgb_buffer = np.zeros((self.height, self.width, 4), dtype=np.float32)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT,
                        rgb_buffer)

        rgb_gl = np.copy(rgb_buffer)
        rgb_gl.shape = 480, 640, 4
        rgb_gl = rgb_gl[::-1, :]
        rgb_gl = np.round(rgb_gl[:, :, :3] * 255).astype(
            np.uint8)  # Convert to [0, 255]
        bgr_gl = rgb_gl[:, :, [2, 1, 0]]

        depth_buffer = np.zeros((self.height, self.width), dtype=np.float32)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_DEPTH_COMPONENT,
                        gl.GL_FLOAT, depth_buffer)
        depth_gl = np.copy(depth_buffer)
        depth_gl.shape = 480, 640
        depth_gl = depth_gl[::-1, :]
        depth_bg = depth_gl == 1
        depth_gl = 2 * self.zFar * self.zNear / (self.zFar + self.zNear -
                                                 (self.zFar - self.zNear) *
                                                 (2 * depth_gl - 1))
        depth_gl[depth_bg] = 0
        return bgr_gl, depth_gl

    def __del__(self):
        self.window.close()

    def my_compute_calib_proj(self, K, w, h, zNear, zFar):
        u0 = K[0, 2] + 0.5
        v0 = K[1, 2] + 0.5
        fu = K[0, 0]
        fv = K[1, 1]
        L = +(u0) * zNear / -fu
        T = +(v0) * zNear / fv
        R = -(w - u0) * zNear / -fu
        B = -(h - v0) * zNear / fv
        proj = np.zeros((4, 4))
        proj[0, 0] = 2 * zNear / (R - L)
        proj[1, 1] = 2 * zNear / (T - B)
        proj[2, 2] = -(zFar + zNear) / (zFar - zNear)
        proj[2, 0] = (R + L) / (R - L)
        proj[2, 1] = (T + B) / (T - B)
        proj[2, 3] = -1.0
        proj[3, 2] = -(2 * zFar * zNear) / (zFar - zNear)
        return proj

    def _get_view_mtx(self, R, t):
        u_view = np.eye(4, dtype=np.float32)
        u_view[:3, :3], u_view[:3, 3] = R, t.squeeze()
        yz_flip = np.eye(4, dtype=np.float32)
        yz_flip[1, 1], yz_flip[2, 2] = -1, -1
        u_view = yz_flip.dot(u_view)  # OpenCV to OpenGL camera system
        u_view = u_view.T  # OpenGL expects column-wise matrix format
        return u_view


if __name__ == "__main__":
    import cv2
    import os
    import sys
    import scipy.io as sio
    from tqdm import tqdm
    import random
    random.seed(2333)
    np.random.seed(2333)

    cur_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(1, os.path.join(cur_path, '../..'))

    def mat2quat(M):
        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
        # Fill only lower half of symmetric matrix
        K = np.array(
            [[Qxx - Qyy - Qzz, 0, 0, 0], [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
             [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
             [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K)
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1
        return q

    width = 640
    height = 480
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    ZNEAR = 0.25
    ZFAR = 6.0
    # idx = '000001'

    class_name_list = [
        '__background__', '002_master_chef_can', '003_cracker_box',
        '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
        '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
        '010_potted_meat_can', '011_banana', '019_pitcher_base',
        '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill',
        '036_wood_block', '037_scissors', '040_large_marker',
        '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'
    ]
    class_name = '002_master_chef_can'
    model_folder_dict = {}
    for class_name in class_name_list:
        if class_name == "__background__":
            continue
        model_folder_dict[class_name] = os.path.join(
            cur_path, '../../data/LOV/models/{}'.format(class_name))

    model_dir = os.path.join(cur_path,
                             '../../data/LOV/models/{}'.format(class_name))

    LOV_data_syn_root_dir = os.path.join(cur_path, '../..', 'data', 'LOV',
                                         'data_syn')
    LOV_data_syn_meta_path = "%s/{}-meta.mat" % (LOV_data_syn_root_dir)
    # get all real files
    real_prefix_list = [
        fn.split('-')[0] for fn in os.listdir(LOV_data_syn_root_dir)
        if '-color.png' in fn
    ]
    real_prefix_list = sorted(real_prefix_list)

    brightness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4]
    model_dir = os.path.join(
        cur_path, '../../data/LOV/models/{}'.format('002_master_chef_can'))
    model_folder_dict = {}
    for class_name in class_name_list:
        if class_name == "__background__":
            continue
        model_folder_dict[class_name] = os.path.join(
            cur_path, '../../data/LOV/models/{}'.format(class_name))

    render_machine = Render_Py_Light_MultiProgram(
        class_name_list, model_folder_dict, K, width, height, ZNEAR, ZFAR,
        brightness_ratios)
    for idx, real_index in enumerate(tqdm(real_prefix_list[10000:10003])):
        prefix = real_index  # real_prefix_list[idx]

        real_depth_file = os.path.join(LOV_data_syn_root_dir,
                                       prefix + '-depth.png')
        real_label_file = os.path.join(LOV_data_syn_root_dir,
                                       prefix + '-label.png')
        real_color_file = os.path.join(LOV_data_syn_root_dir,
                                       prefix + '-color.png')

        real_label = cv2.imread(real_label_file, cv2.IMREAD_UNCHANGED)

        meta_data = sio.loadmat(LOV_data_syn_meta_path.format(prefix))
        cls_indices = meta_data['cls_indexes']
        cls_indices = [int(cls_idx) for cls_idx in np.squeeze(cls_indices)]

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

        # randomly adjust color and intensity for light_intensity
        colors = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                           [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        intensity = np.random.uniform(0.9, 1.1, size=(3, ))
        colors_randk = random.randint(0, colors.shape[0] - 1)
        light_intensity = colors[colors_randk] * intensity

        # randomly choose a render machine(brightness_ratio)
        rm_randk = random.randint(0, len(brightness_ratios) - 1)

        # copyfile(real_label_file, render_real_label_file)
        # continue
        rgb_dict = {}

        for class_idx in cls_indices:
            cls_name = class_name_list[class_idx]
            # init render
            model_dir = os.path.join(cur_path,
                                     '../data/LOV/models/{}'.format(cls_name))
            print('model_dir: ', model_dir)

            inner_id = np.where(
                np.squeeze(meta_data['cls_indexes']) == class_idx)
            if len(meta_data['poses'].shape) == 2:
                pose = meta_data['poses']
            else:
                pose = np.squeeze(meta_data['poses'][:, :, inner_id])

            # adjust light position according to pose
            light_position = np.array(light_position) * 0.5
            # inverse yz
            light_position[0] += pose[0, 3]
            light_position[1] -= pose[1, 3]
            light_position[2] -= pose[2, 3]

            rgb_gl, depth_gl = render_machine.render(
                RT_transform.mat2quat(pose[:3, :3]),
                pose[:, -1],
                light_position,
                light_intensity,
                class_name=cls_name,
                brightness_k=rm_randk)

            rgb_dict[class_idx] = rgb_gl

            real_color_img = cv2.imread(real_color_file, cv2.IMREAD_COLOR)

            import matplotlib.pyplot as plt

            fig = plt.figure()
            plt.axis('off')
            fig.add_subplot(1, 3, 1)
            plt.imshow(rgb_gl[:, :, [2, 1, 0]])

            fig.add_subplot(1, 3, 3)
            plt.imshow(real_color_img[:, :, [2, 1, 0]])

            fig.suptitle(
                'light position: {}\n light_intensity: {}\n brightness: {}'.
                format(light_position, light_intensity,
                       brightness_ratios[rm_randk]))
            plt.show()
