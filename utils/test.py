#!usr/bin/env python3
# -*- encoding:utf-8 -*- 
# Created by donskopoi on 2022/05/03
# File for calculate metrics

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

import logging
import time
import numpy as np

import util as M

from tqdm import tqdm
from data.util import get_path_from_txt
from data_io import imread_uint16_png

def main():
    # Please modify your dir paths here
    pred_img_dir = '/Volumes/Azi/results/EHDRGAN/L1/test_1'
    pred_ratio_dir = '/Volumes/Azi/results/EHDRGAN/l1/test_1'
    gt_img_dir = '/Volumes/Azi/dataset/NTIRE2021_HDR/val/val_gt_sub'
    gt_ratio_dir = '/Volumes/Azi/dataset/NTIRE2021_HDR/val/val_ratio'

    # get logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    size1, pred_img_paths = get_path_from_txt(data_type='img', root_path=pred_img_dir)
    size3, gt_img_paths = get_path_from_txt(data_type='img', root_path=gt_img_dir)

    assert size1 == size3, 'dataset size does not match'
    pred_img_paths = sorted(pred_img_paths)
    gt_img_paths = sorted(gt_img_paths)

    # ---------- calculate metrics -------------
    tqdm_bar = tqdm(total=size1)
    psnr, norm_psnr, mu_psnr = 0, 0, 0
    logger.info('starting calculate...')
    start_time = time.time()
    for i in range(size1):
        id1, sub1 = os.path.basename(pred_img_paths[i]).split('_')[0], os.path.basename(pred_img_paths[i]).split('_')[-2]
        id3, sub3 = os.path.basename(gt_img_paths[i]).split('_')[0], os.path.basename(gt_img_paths[i]).split('_')[-1].split('.')[0]
        pred_name = os.path.splitext(os.path.basename(pred_img_paths[i]))[0]

        assert id1 == id3, 'file id does not match'
        assert sub1 == sub3, 'file sub does not match'

        # get gt alignratio path
        gt_ratio_path = os.path.join(gt_ratio_dir, id1+'_alignratio.npy')
        pred_ratio_path = os.path.join(pred_ratio_dir, pred_name+'_alignratio.npy')

        # load image
        pred_img = imread_uint16_png(pred_img_paths[i], pred_ratio_path)
        gt_img = imread_uint16_png(gt_img_paths[i], gt_ratio_path)

        # calculate
        psnr += M.calculate_psnr(pred_img, gt_img)
        norm_psnr += M.calculate_normalized_psnr(pred_img, gt_img, np.max(gt_img))  # s-PSNR
        mu_psnr += M.calculate_tonemapped_psnr(pred_img, gt_img, percentile=99, gamma=2.24)  # µ-PSNR

        tqdm_bar.update(1)

    psnr /= size1
    norm_psnr /= size1
    mu_psnr /= size1
    # ---------- end calculate metrics -------------
    end_time = time.time()
    logger.info(f'end calculate, cost {end_time-start_time:.4f}s')
    logger.info(f'psnr: {psnr:.2f}, norm_psnr: {norm_psnr:.2f}, mu_psnr: {mu_psnr:.2f}')


if __name__ == "__main__":
    main()