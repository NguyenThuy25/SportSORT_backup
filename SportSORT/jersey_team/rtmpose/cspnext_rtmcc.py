

import torch
import torch.nn as nn
# from src.human_pose.modules.rtmpose.cspnext import CSPNeXt
# from src.human_pose.modules.rtmpose.rtmcc_head import RTMCCHead
# from src.configs.dnp_common import HumanPoseConfig
from rtmpose.rtmcc_head import RTMCCHead
from rtmpose.cspnext import CSPNeXt
from mmengine.registry import MODELS

@MODELS.register_module(name='CSPNeXtRTMCC')
class CSPNeXtRTMCC(nn.Module):
    """Model class for pose estimation. Class include CSPNeXt backbone and RTMCC head
    """
    
    # Some configs to init model with Ultralytics backend
    yaml = {'yaml_file': "yolo.yaml"}
    task = "detect"
    names = "1+1"
    stride = torch.tensor([8, 16, 32])
    # pt_path = HumanPoseConfig.POSE_CHECKPOINT.value
    
    
    def __init__(self, backbone_cfg, head_cfg, device="cpu"):
        """Init CSPNeXtRTMCC model
        
        Args:
            backbone_cfg (dict): config to init CSPNeXt backbone
            head_cfg (dict): config to init RTMCC head
            device (str, optional): device to run model. Defaults to "cpu".
        """
        
        super(CSPNeXtRTMCC, self).__init__()
        
        # Init CSPNeXt backbone
        self.backbone = CSPNeXt(arch=backbone_cfg['arch'], 
                                deepen_factor=backbone_cfg['deepen_factor'],
                                widen_factor=backbone_cfg['widen_factor'],
                                expand_ratio=backbone_cfg['expand_ratio'],
                                channel_attention=backbone_cfg['channel_attention'],
                                norm_cfg=backbone_cfg['norm_cfg'],
                                act_cfg=backbone_cfg['act_cfg'],
                                init_cfg=backbone_cfg['init_cfg'])
        
        # Init RTMCC head
        self.head = RTMCCHead(in_channels=head_cfg['in_channels'],
                                       out_channels=head_cfg['out_channels'],
                                       input_size=head_cfg['input_size'],
                                       in_featuremap_size=head_cfg['in_featuremap_size'],
                                       simcc_split_ratio=head_cfg['simcc_split_ratio'],
                                       final_layer_kernel_size=head_cfg['final_layer_kernel_size'],
                                       gau_cfg=head_cfg['gau_cfg'],
                                       decoder=head_cfg['decoder'])

        # Init device
        self.device = torch.device(device=device)
    
    def forward(self, x):
        """
        forward function of CSPNeXtRTMCC model
        
        Args:
            x (torch.Tensor): input image
        
        Returns:
            x (torch.Tensor): output of model
        """
        
        # Inference CSPNeXt backbone
        x = self.backbone(x)
        # Inference RTMCC head
        x = self.head(x)
        return x

    def fuse(self):
        return self

