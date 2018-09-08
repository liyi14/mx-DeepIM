# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import os
from shutil import copyfile

if __name__=='__main__':
    curtail_stride = 10
    file_type_list = ['-color.png', '-depth.png', '-label.png', '-meta.mat', '-box.txt']

    cur_path = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cur_path, '..', 'data', 'LOV_curtail', 'data')
    image_set_dir = os.path.join(cur_path, '..', 'data', 'LOV_curtail', 'image_set')
    print(data_dir)
    dir_list = os.listdir(data_dir)
    dir_list = sorted(dir_list)

    image_set = []
    for sub_dir in dir_list:
        src_file_dir = os.path.join(data_dir, sub_dir)

        data_list = os.listdir(src_file_dir)
        data_list = sorted(data_list)
        file_name_list = [name[:-9] for name in data_list if name.endswith('-meta.mat')]
        for x in file_name_list:
            image_set.append('{}/{}'.format(sub_dir, x))
        print("folder {} done".format(sub_dir))

    output_file_name = os.path.join(image_set_dir, "curtail_all.txt")
    with open(output_file_name, "w") as text_file:
        for x in image_set:
            text_file.write("{}\n".format(x))
