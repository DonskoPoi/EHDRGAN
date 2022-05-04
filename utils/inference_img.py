#!usr/bin/env python3
# -*- encoding:utf-8 -*- 
# Created by donskopoi on 2022/05/03
# File for generate predicted images

import logging
import time
import util

from tqdm import tqdm
from basicsr.utils.options import parse_options, dict2str
from basicsr.utils import make_exp_dirs, get_root_logger
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model

import os
import sys
root_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_path)
import data
import models


def main():
    # get options
    opt, args = parse_options(root_path=root_path, is_train=False)
    # make result dirs
    make_exp_dirs(opt)
    # get logger
    logger = get_root_logger()
    logger.setLevel(logging.INFO)
    logger.info(dict2str(opt))
    # create data loaders and datasets
    val_dataloaders = list()
    for phase, dataset_opt in opt['datasets'].items():
        if phase.split('_')[0] in ['val', 'test']:
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_dataloaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    # create model
    model = build_model(opt)

    # ----------- generation -------------
    for val_dataloader in val_dataloaders:
        dataset_name = val_dataloader.dataset.opt['name']
        logger.info(f'Validate dataset {dataset_name}...')

        sava_dir = os.path.join(opt['path']['visualization'], dataset_name)
        os.makedirs(sava_dir)
        tqdm_bar = tqdm(total=len(val_dataloader))
        need_gt = False if val_dataloader.dataset.opt.get('dataroot_GT') is None else True

        start_time = time.time()
        for data in val_dataloader:
            img_name = os.path.basename(data['LQ_path'][0])
            img_name = os.path.splitext(img_name)[0]
            img_name += '_val'
            img_path = os.path.join(sava_dir, img_name + '.png')
            alignratio_path = os.path.join(sava_dir, img_name + '_alignratio.npy')

            # feed data
            model.feed_data(data=data, need_GT=need_gt)
            # inference
            fake_hdr = model.inference()
            # post process
            fake_hdr_img = util.tensor2numpy(fake_hdr.detach()[0].cpu().float())

            # save image
            util.save_img_with_ratio(img_path, fake_hdr_img, alignratio_path)
            # show log
            tqdm_bar.set_description(f'{img_name:20s}')
            tqdm_bar.update(1)

        end_time = time.time()
        logger.info(f'Ending validate {dataset_name}, cost {end_time-start_time:.3f}s')







if __name__ == "__main__":
    main()