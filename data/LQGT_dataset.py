import os
import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LQGTDataset(data.Dataset):

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None

        # 由于mac的文件夹里会出现.DS_Store以及一堆._*文件，这里要删掉
        os.system(f"rm -rf {opt['dataroot_LQ']}/.DS_Store")
        os.system(f"rm -rf {opt['dataroot_GT']}/.DS_Store")
        os.system(f"rm -rf {opt['dataroot_ratio']}/.DS_Store")
        os.system(f"rm -rf {opt['dataroot_LQ']}/._*")
        os.system(f"rm -rf {opt['dataroot_GT']}/._*")
        os.system(f"rm -rf {opt['dataroot_ratio']}/._*")

        self.sizes_LQ, self.paths_LQ = util.get_path_from_txt(self.data_type, opt['dataroot_LQ'])
        self.sizes_GT, self.paths_GT = util.get_path_from_txt(self.data_type, opt['dataroot_GT'])
        self.folder_ratio = opt['dataroot_ratio']

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt.get('GT_size', -1)

        # get LQ image
        LQ_path = self.paths_LQ[index]
        # print(LQ_path)
        img_LQ = util.read_imgdata(LQ_path, ratio=255.0) # -> lq都是normalized by 255的(/255)

        # get GT alignratio
        filename = osp.basename(LQ_path)[:4] + "_alignratio.npy"
        ratio_path = osp.join(self.folder_ratio, filename)
        alignratio = np.load(ratio_path).astype(np.float32)

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_imgdata(GT_path, ratio=alignratio)  # -> gt都是/alignratio的也就是还原到0-255色彩空间的图

        if self.opt['phase'] == 'train':

            H, W, C = img_LQ.shape
            H_gt, W_gt, C = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))
            LQ_size = GT_size // scale

            # randomly crop
            if GT_size != 0:
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])
            # print(f'LQ size:{img_LQ.shape}, GT size:{img_GT.shape}')
        # condition
        # if self.opt['condition'] == 'image':
            # cond = img_LQ.copy()
        # elif self.opt['condition'] == 'gradient':
            # cond = util.calculate_gradient(img_LQ)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            # cond = cond[:, :, [2, 1, 0]]

        H, W, _ = img_LQ.shape
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        # cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path, 'ratio_path': ratio_path}

    def __len__(self):
        return len(self.paths_GT)
