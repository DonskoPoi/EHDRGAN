#!usr/bin/env python3
# -*- encoding:utf-8 -*- 
# Created by donskopoi on 2022/05/03
# File for visualize 16bit image with related align ratio

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

import cv2
import util as F
import numpy as np
from data.util import get_path_from_txt
from data_io import imread_uint16_png
from tqdm import tqdm

def main():
    # Please modify the following parameters
    input_dir = '/Volumes/Azi/results/EHDRGAN/L1/test_1'
    ratio_dir = '/Volumes/Azi/results/EHDRGAN/L1/test_1'
    save_dir = '/Volumes/Azi/results/EHDRGAN/L1/test_1_visualize'

    assert os.path.exists(input_dir), 'input dir does not exist'
    assert os.path.exists(ratio_dir), 'ratio dir does not exist'

    _, input_paths = get_path_from_txt(data_type='img', root_path=input_dir)
    input_paths = tqdm(input_paths)
    for img_path in input_paths:
        # get alignratio path
        img_name = os.path.basename(img_path)
        alignratio_path = os.path.join(ratio_dir, os.path.splitext(img_name)[0] + '_alignratio.npy')
        #alignratio_path = os.path.join(ratio_dir, img_name.split('_')[0] + '_alignratio.npy')
        # read image
        img = imread_uint16_png(img_path, alignratio_path)
        # gamma correction
        liner_img = img ** 2.24
        # tanh mu tone mapping
        percentile_norm = np.percentile(liner_img, 99)
        img = F.tanh_norm_mu_tonemap(liner_img, percentile_norm)
        # to unit8
        img *= 255
        img = img.round().astype(np.uint8)
        # write image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    main()

