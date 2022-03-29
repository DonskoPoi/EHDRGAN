from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
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