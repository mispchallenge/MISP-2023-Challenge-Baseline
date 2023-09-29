#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as nf

from network.network_feature_extract import FeatureExtractor


class misp_BatchCalMSE_calcIRMLabel(nn.Module):
    def __init__(self, extractor_setting, mode='correct', **other_params):
        super(misp_BatchCalMSE_calcIRMLabel, self).__init__()
        self.mode = mode
        self.calcRealIRM = FeatureExtractor(extractor_type = 'mask', extractor_setting=extractor_setting)

    def forward(self, net_output, clean_wave, mixture_wave, length=None):
        
        noise_wave = mixture_wave - clean_wave
        label, length = self.calcRealIRM([clean_wave, noise_wave], length)
        
        label = label.transpose(1,2)
        if length is not None:
            mask = label.new_ones(label.shape)  
            for i in range(mask.shape[0]):
                mask[i, int(length[i]):] = 0.  #[b,t,c]
            mse_sum = nf.mse_loss(input=net_output*mask, target=label*mask, size_average=None, reduce=None,
                                  reduction='sum')
            time_num = length.sum().float()
            item_num = (mask == 1).sum().float()
        else:
            mse_sum = nf.mse_loss(input=net_output, target=label, size_average=None, reduce=None, reduction='sum')
            time_num = reduce(lambda x, y: x * y, label.shape[:2])
            item_num = reduce(lambda x, y: x * y, label.shape)
        if self.mode == 'correct':
            mse_loss = mse_sum / item_num
        elif self.mode == 'error':
            mse_loss = mse_sum / time_num
        else:
            raise ValueError('unknown mode: {}'.format(self.mode))
        return mse_loss, torch.tensor([(mse_sum / item_num).item()])
