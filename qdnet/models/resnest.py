# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QDNet Model effnet
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------
'''



import torch
import torch.nn as nn
from resnest.torch import resnest50
from resnest.torch import resnest101
from resnest.torch import resnest200
from resnest.torch import resnest269
import torchvision.models as models
from qdnet.models.metric_strategy import Swish_module, ArcMarginProduct_subcenter, ArcFaceLossAdaptiveMargin


class Resnest(nn.Module):
    '''
    '''
    def __init__(self, enet_type, out_dim, drop_nums=1, pretrained=False, metric_strategy=False):
        super(Resnest, self).__init__()
        if enet_type in ["resnest18","resnest50", "resnest101", "resnest200", "resnest269"]:
            # self.enet = locals()[enet_type](pretrained=pretrained)
            # self.enet = eval(enet_type)(pretrained=pretrained)
            print (f'enet_type : {enet_type}')
            # model_dict = {"resnest18": models.resnet18(pretrained=True)}
            # model_dict = {"resnest18": models.resnet18(pretrained=True),\
            #               "resnest50": models.resnet50(pretrained=True),\
            #               "resnest101": models.resnet101(pretrained=True),\
            #               "resnest200": models.resnet200(pretrained=True),\
            #                "resnest269": models.resnet269(pretrained=True)}
            model_dict = {"resnest18": models.resnet18(),\
                          "resnest50": models.resnet50(),\
                          "resnest101": models.resnet101()}
            # self.enet = models.resnet18(pretrained=True)
            self.enet = model_dict[enet_type]
            #判断是否pretrain,以及加载路径
            if pretrained:
                self.enet.load_state_dict(torch.load('./resnet18.pth'))

        self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(drop_nums) ])
        in_ch = self.enet.fc.in_features
        self.fc = nn.Linear(in_ch, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.classify = nn.Linear(in_ch, out_dim)
        self.enet.fc = nn.Identity()
        self.metric_strategy = metric_strategy
   
    def extract(self, x):
        x = self.enet(x)
        return x
   
    def forward(self, x):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.metric_strategy:
             out = self.metric_classify(self.swish(self.fc(x)))
        else:
             for i, dropout in enumerate(self.dropouts):
                 if i == 0:
                     out = self.classify(dropout(x))
                 else:
                     out += self.classify(dropout(x))
             out /= len(self.dropouts)
        return out
 
'''
config :{'data_dir': './data/', 'data_folder': './data/', 'image_size': 512, 'enet_type': 'resnest50', 
'metric_strategy': False, 'batch_size': 8, 'num_workers': 4, 'init_lr': '3e-5', 'out_dim': 2, 
'n_epochs': 15, 'drop_nums': 1, 'loss_type': 'ce_loss', 'use_amp': False, 'mixup_cutmix': False, 
'model_dir': './resnest101/weight/', 'log_dir': './resnest101/logs/', 'CUDA_VISIBLE_DEVICES': '0', 
'fold': '0,1,2,3,4', 'pretrained': True, 'eval': 'best', 'oof_dir': './resnest101/oofs/', 'auc_index': 'male'}
'''