import argparse
import os
import time

import cv2
import numpy as np
import torch.cuda

from basicsr.models import build_model
from scripts.visualize import visualize
from utils.util import tensor2numpy


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def get_paser_and_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='weights/gen_net_latest.pth')
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='data/output')
    # create opt
    net_opt = dict(type='R2AttU_Net', img_ch=3, output_ch=3, t=4)
    inf_opt = dict(
        name='EHDRGAN_inference',
        model_type='EHDRGANModel',
        is_train=False,
        num_gpu=1 if torch.cuda.is_available() else 0,
        dist=False,
        network_g=net_opt
    )
    return parser.parse_args(), inf_opt


def main():
    '''
    usage: python inference.py --checkpoint <checkpoint path> --image_dir <input image dir> --save_dir <save dir>
    '''
    args, opt = get_paser_and_opts()
    save_dir = args.save_dir
    checkpoint_path = args.checkpoint
    image_dir = args.image_dir
    os.makedirs(save_dir, exist_ok=True)
    assert os.path.exists(checkpoint_path) is True, "checkpoint path does not exist"
    assert os.path.exists(image_dir) is True, "input image dir does not exist"

    # create model
    model = build_model(opt)
    model.load_gen(checkpoint_path)

    # get all image path
    image_paths = list()
    for root, dirs, files in os.walk(image_dir):
        for name in files:
            ext = name.split('.')[-1].lower()
            if ext in ['png', 'jpg', 'jpeg']:
                image_paths.append(os.path.join(root, name))
    n_images = len(image_paths)
    print(f'{n_images} to process')

    # processing image
    # with aspect ratio un changed (i.e. 1920x1080 -> 512x288)
    long_edge = 512
    stride = 32
    for image_path in image_paths:
        ori_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_name = image_path.split('/')[-1]
        if ori_img is None:
            continue
        ori_h, ori_w = ori_img.shape[:2]  # h, w, 3
        if ori_h > ori_w:
            input_h = long_edge
            input_w = int(input_h / ori_h * ori_w)
            input_w = (input_w + stride - 1) // stride * stride
        else:
            input_w = long_edge
            input_h = int(input_w / ori_w * ori_h)
            input_h = (input_h + stride - 1) // stride  * stride
        img = cv2.resize(ori_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

        # normalize the image and to tensor
        img = img.astype(np.float32) / 255.0  # 255 can be changed with the alignratio
        img = img.transpose(2, 0, 1)  # h, w, 3 -> 3, h, w
        img = torch.from_numpy(img).unsqueeze(0)  # 1, 3, h, w
        input = dict(LQ=img)

        # inference
        start = time_synchronized()
        model.feed_data(input, need_GT=False)
        fake_hdr = model.inference()
        end = time_synchronized()
        print('cost %.2f ms' % ((end - start) * 1000))
        
        # save image
        # 1, 3, 288, 512
        fake_hdr = tensor2numpy(fake_hdr.squeeze(0).cpu())
        fake_hdr = cv2.resize(fake_hdr, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        fake_hdr = visualize(fake_hdr)
        cv2.imwrite(os.path.join(save_dir, image_name.split('.')[0]+'_inference.png'), fake_hdr)


if __name__ == '__main__':
    main()