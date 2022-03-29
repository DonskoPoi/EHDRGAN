from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel


@MODEL_REGISTRY.register()
class EHDRGANModel(BaseModel):
    def __init__(self):
        pass