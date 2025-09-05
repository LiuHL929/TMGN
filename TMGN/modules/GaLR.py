# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .GaLR_utils import *
import copy
import ast


# from .mca import SA,SGA

class Fusion_MIDF(nn.Module):
    def __init__(self):
        super(Fusion_MIDF, self).__init__()
        self.opt = {
            'fusion': {'correct_local_hidden_dim': 768,
                       'correct_local_hidden_drop': 0.2,
                       'supplement_global_hidden_dim': 768,
                       'supplement_global_hidden_drop': 0.2,
                       'dynamic_fusion_dim': 768,
                       'dynamic_fusion_drop': 0.2,
                       'mca_DROPOUT_R': 0.15,
                       'mca_HIDDEN_SIZE': 768,
                       'mca_FF_SIZE': 1024,
                       'mca_MULTI_HEAD': 12,
                       'mca_HIDDEN_SIZE_HEAD': 64},
            'embed':
                {'embed_dim': 768}
        }

        # local trans
        self.l2l_SA = SA(self.opt)

        # global trans
        self.g2g_SA = SA(self.opt)

        # # local correction
        self.g2l_SGA = SGA(self.opt)
        #
        # # global supplement
        self.l2g_SGA = SGA(self.opt)

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(self.opt['embed']['embed_dim'], self.opt['fusion']['dynamic_fusion_dim']),
            nn.Sigmoid(),
            nn.Dropout(p=self.opt['fusion']['dynamic_fusion_drop']),
            nn.Linear(self.opt['fusion']['dynamic_fusion_dim'], 768),
        )

    def forward(self, global_feature, local_feature):
        # print('变换前global_feature的形状：', global_feature.shape)
        # global_feature = torch.unsqueeze(global_feature, dim=1)
        # print('变换后global_feature的形状：', global_feature.shape)
        # local_feature = torch.unsqueeze(local_feature, dim=1)

        # global trans
        global_feature = self.g2g_SA(global_feature)
        # local trans
        local_feature = self.l2l_SA(local_feature)

        # local correction
        local_feature = self.g2l_SGA(local_feature, global_feature)

        # global supplement
        global_feature = self.l2g_SGA(global_feature, local_feature)

        global_feature_t = torch.squeeze(global_feature, dim=1)
        local_feature_t = torch.squeeze(local_feature, dim=1)

        global_feature = F.sigmoid(local_feature_t) * global_feature_t
        local_feature = global_feature_t + local_feature_t

        # dynamic fusion
        feature_gl = global_feature + local_feature
        # print('feature_gl的形状：', feature_gl.shape)
        dynamic_weight = self.dynamic_weight(feature_gl)

        # print('dynamic_weight的形状：', dynamic_weight.shape)
        weight_global = dynamic_weight.expand_as(global_feature)

        weight_local = dynamic_weight.expand_as(global_feature)

        visual_feature = weight_global * global_feature + weight_local * local_feature
        # print('visual_feature的形状：', visual_feature.shape)

        return visual_feature


