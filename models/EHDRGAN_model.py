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
