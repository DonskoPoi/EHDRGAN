import os.path

import cv2
import torch

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from collections import OrderedDict
from scripts import visualize as vs
import utils.util as util
import numpy as np
from archs.R2AttUNet import R2AttU_Net
from archs.HDRUNet_arch import HDRUNet
from .customize_loss import tanhL1
from .customize_loss import tanhL2


@MODEL_REGISTRY.register()
class EHDRGANModel(BaseModel):
    def __init__(self, opt):
        super(EHDRGANModel, self).__init__(opt)
        if self.is_train:
            train_opt = opt['train']  # check if the model is in training progress
        need_dis = True if 'network_d' in opt else False  # check if the model needs to initialize discriminator
        # 是否使用cond图层
        self.is_cond = True \
            if opt['network_g']['type'] == 'HDRUNet' \
            else False
        # get logger
        self.logger = get_root_logger()

        if not need_dis:
            self.logger.info('you will not initial discriminator network due to the lack of discriminator options')

        # init generator
        self.gen_net = build_network(opt['network_g'])
        self.gen_net = self.model_to_device(self.gen_net)
        # init discriminator
        if need_dis:
            self.dis_net = build_network(opt['network_d'])
            self.dis_net = self.model_to_device(self.dis_net)

        if 'debug' in opt['name'] and opt['show_network']:
            self.print_network(self.gen_net)
            if need_dis:
                self.print_network(self.dis_net)

        # load pretrained models for both networks
        if 'path' in opt:
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_g', 'params')
                self.load_network(self.gen_net, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None and need_dis:
                param_key = self.opt['path'].get('param_key_d', 'params')
                self.load_network(self.dis_net, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        # set networks in training model
        if self.is_train:
            self.gen_net.train()
            if need_dis:
                self.dis_net.train()

        # set iters of discriminator
        if need_dis:
            self.dis_iters = train_opt.get('net_d_iters', 1)
            self.dis_init_iters = train_opt.get('net_d_init_iters', 0)

        # define losses
        if self.is_train:
            if train_opt.get('pixel_opt'):
                self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            else:
                self.cri_pix = None

            if train_opt.get('perceptual_opt'):
                self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            else:
                self.cri_perceptual = None

            if train_opt.get('gan_opt'):
                self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # setup optimizer and schedulers
        if self.is_train:
            self.setup_optimizer(train_opt)
            self.setup_schedulers()

    def setup_optimizer(self, train_opt):
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.gen_net.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.dis_net.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data, need_GT=True):
        self.lq = data['LQ'].to(self.device)  # LQ
        if self.is_cond:
            self.cond = data['cond'].to(self.device)  # cond
        if need_GT:
            self.gt = data['GT'].to(self.device)  # GT
        if 'debug' in self.opt['name']:
            self.logger.info(f'feed data complete')

        # print(self.lq.shape)
        # print(self.gt.shape)

    def optimize_parameters(self, current_iter):
        # firstly optimizing generator
        # disable gradient in discriminator
        for p in self.dis_net.parameters():
            p.requires_grad = False
        # init grident
        self.optimizer_g.zero_grad()
        # generate output
        # training input: [batch_size, 3, GT_size, GT,size] with range 0-255 (unit8)
        # network output: [batch_size, 3, GT_size, GT,size] with range 0-65535 (unit16)
        if self.is_cond:
            self.output = self.gen_net((self.lq, self.cond))
        else:
            self.output = self.gen_net(self.lq)
        # init loss
        gen_loss_total = 0
        loss_dict = OrderedDict()
        # optimize generator
        if current_iter % self.dis_iters == 0 and current_iter > self.dis_init_iters:
            # pixel loss
            if self.cri_pix:
                gen_pix_loss = self.cri_pix(self.output, self.gt)
                gen_loss_total += gen_pix_loss
                loss_dict['gen_pix_loss'] = gen_pix_loss
            # perceptual loss
            if self.cri_perceptual:
                gen_percep_loss, gen_style_loss = self.cri_perceptual(self.output, self.gt)
                if gen_percep_loss is not None:
                    gen_loss_total += gen_percep_loss
                    loss_dict['gen_percep_loss'] = gen_percep_loss
                if gen_style_loss is not None:
                    gen_loss_total += gen_style_loss
                    loss_dict['gen_style_loss'] = gen_style_loss
            # gan loss (relativistic gan)
            real_d_pred = self.dis_net(self.gt).detach()
            fake_g_pred = self.dis_net(self.output)

            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            gen_gan_loss = (l_g_real + l_g_fake) / 2

            gen_loss_total += gen_gan_loss
            loss_dict['gen_gan_loss'] = gen_gan_loss

            gen_loss_total.backward()
            self.optimizer_g.step()

            # then optimize net_d
            # enable gradient
            for p in self.dis_net.parameters():
                p.requires_grad = True
            # init giadient
            self.optimizer_d.zero_grad()
            # gan loss (relativistic gan)

            # In order to avoid the error in distributed training:
            # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
            # the variables needed for gradient computation has been modified by
            # an inplace operation",
            # we separate the backwards for real and fake, and also detach the
            # tensor for calculating mean.

            # real
            fake_d_pred = self.dis_net(self.output).detach()
            real_d_pred = self.dis_net(self.gt)
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
            l_d_real.backward()
            # fake
            fake_d_pred = self.dis_net(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            l_d_fake.backward()
            self.optimizer_d.step()

            loss_dict['dis_real_loss'] = l_d_real
            loss_dict['dis_fake_loss'] = l_d_fake
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

            self.log_dict = self.reduce_loss_dict(loss_dict)

    def save(self, epoch, current_iter):
        self.save_network(self.gen_net, 'gen_net', current_iter)
        self.save_network(self.dis_net, 'dis_net', current_iter)
        self.save_training_state(epoch, current_iter)

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        '''
        use for calculate psnr for validate
        '''
        avg_psnr = 0.0
        avg_normalized_psnr = 0.0
        avg_tonemapped_psnr = 0.0
        idx = 0
        for val_data in dataloader:
            idx += 1

            # load data
            self.feed_data(val_data)

            # inference
            fake_hdr = self.inference()

            # transfer fake_hdr, gt to numpy
            fake_hdr = fake_hdr.detach()[0].float().cpu()
            gt = self.gt.detach()[0].float().cpu()

            fake_hdr_img = util.tensor2numpy(fake_hdr)
            gt_img = util.tensor2numpy(gt)  # -> float32, [h, w, 3]

            # calculate psnr
            avg_psnr += util.calculate_psnr(fake_hdr_img, gt_img)
            avg_normalized_psnr += util.calculate_normalized_psnr(fake_hdr_img, gt_img, np.max(gt_img))
            avg_tonemapped_psnr += util.calculate_tonemapped_psnr(fake_hdr_img, gt_img, percentile=99, gamma=2.24)

            # visualize
            if save_img:
                if idx % 1000 == 0:
                    lq_path = val_data['LQ_path'][0]
                    filename = lq_path.split('/')[-1][:16] + f'_{current_iter}_visualize.png'
                    result = vs.visualize_with_gt(gt_img, fake_hdr_img)
                    self.logger.info(f"saving visualize image to {os.path.join(self.opt['path']['visualization'], filename)}")
                    cv2.imwrite(os.path.join(self.opt['path']['visualization'], filename), result)


        # calculate avg
        avg_psnr /= idx
        avg_normalized_psnr /= idx
        avg_tonemapped_psnr /= idx

        # log
        self.logger.info(
            '# Validation # PSNR: {:.4e}, norm_PSNR: {:.4e}, mu_PSNR: {:.4e}'.format(
                avg_psnr,
                avg_normalized_psnr,
                avg_tonemapped_psnr
            )
        )
        if tb_logger:
            tb_logger.add_scalar('psnr', avg_psnr, current_iter)
            tb_logger.add_scalar('norm_PSNR', avg_normalized_psnr, current_iter)
            tb_logger.add_scalar('mu_PSNR', avg_tonemapped_psnr, current_iter)

    def inference(self):
        # get test data
        self.gen_net.eval()
        with torch.no_grad():
            if self.is_cond:
                fake_hdr = self.gen_net((self.lq, self.cond))
            else:
                fake_hdr = self.gen_net(self.lq)
        self.gen_net.train()
        return fake_hdr

    def load_gen(self, load_path, strict=True):
        '''
        load weights of generator
        :param load_path: weigths path (i.e. xxx.pth)
        :param strict: strict load
        :return: None
        '''
        self.logger.info(f'loading weights for generation using {load_path}')
        self.load_network(self.gen_net, load_path, strict=strict)