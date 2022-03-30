import torch

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
from basicsr.losses import build_loss
from collections import OrderedDict
from archs.R2AttUNet import R2AttU_Net


@MODEL_REGISTRY.register()
class EHDRGANModel(BaseModel):
    def __init__(self, opt):
        super(EHDRGANModel, self).__init__(opt)
        train_opt = opt['train']

        # init generator
        self.gen_net = build_network(opt['network_g'])
        self.gen_net = self.model_to_device(self.gen_net)
        # init discriminator
        self.dis_net = build_network(opt['network_d'])
        self.dis_net = self.model_to_device(self.dis_net)

        if 'debug' in opt['name']:
            self.print_network(self.gen_net)
            self.print_network(self.dis_net)

        # load pretrained models for both networks
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.gen_net, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.dis_net, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        # set networks in training model
        if self.is_train:
            self.gen_net.train()
            self.dis_net.train()

        # set iters of discriminator
        self.dis_iters = train_opt.get('net_d_iters', 1)
        self.dis_init_iters = train_opt.get('net_d_init_iters', 0)

        # define losses
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
        if need_GT:
            self.gt = data['GT'].to(self.device)  # GT

        # print(self.lq.shape)
        # print(self.gt.shape)

    def optimize_parameters(self, current_iter):
        # firstly optimizing generator
        # disable gradient in discriminator
        for p in self.dis_net.parameters():
            p.requires_grad = False
        # init grident
        self.optimizer_g.zero_grad()
        # get output
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

