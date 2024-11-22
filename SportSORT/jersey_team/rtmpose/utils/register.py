from mmengine.registry import MODELS
from rtmpose.transform_utils import registry_transforms
import torch.nn as nn
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

registry_transforms()
MODELS.register_module(module=nn.SiLU, name='SiLU')
MODELS.register_module('SyncBN', module=SyncBatchNorm)