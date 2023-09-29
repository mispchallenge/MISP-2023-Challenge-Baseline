#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import Pool

from network.network_feature_extract import ShortTimeFourierTransform
from tool.data_io import safe_load, safe_store


class ReconstructMask2Wave(nn.Module):  
    def __init__(self, n_fft, hop_length, win_type='hamming', win_length=None, rescale=False):
        super(ReconstructMask2Wave, self).__init__()
        self.stft = ShortTimeFourierTransform(n_fft=n_fft, hop_length=hop_length, win_type=win_type,
                                              win_length=win_length, is_complex=False)
        self.rescale = rescale

    def forward(self, predict_mask, mixture_wave, clean_wave=None):
        mixture_wave = torch.tensor(mixture_wave)
        
        mixture_wave = mixture_wave.to(torch.float32) / (2. ** 15)
        mixture_spectrum = self.stft(x=mixture_wave, inverse=False)
        if mixture_spectrum.ndim == 4:
            mixture_magnitude = mixture_spectrum[:, :, :, 0]
            mixture_angle = mixture_spectrum[:, :, :, 1]
            predict_mask = predict_mask.transpose(1, 2) 
        else:
            mixture_magnitude = mixture_spectrum[:, :, 0]
            mixture_angle = mixture_spectrum[:, :, 1]
            predict_mask = predict_mask.transpose(0, 1)
        
        # pad_length = mixture_magnitude.shape[1]-predict_mask.shape[1]
        # predict_mask = nn.ZeroPad2d(padding=(0, pad_length))(predict_mask)
        
        reconstruct_magnitude = mixture_magnitude * predict_mask
        reconstruct_wave = self.stft(x=torch.stack([reconstruct_magnitude, mixture_angle], dim=-1), inverse=True,
                                     length=mixture_wave.shape[-1])
        reconstruct_wave = reconstruct_wave * (2. ** 15)
        if reconstruct_wave.ndim == 1:
            if self.rescale:
                clean_wave = torch.tensor(clean_wave)
                clean_wave = clean_wave.to(torch.float32)
                rescale = torch.sum(clean_wave * reconstruct_wave) / torch.sum(reconstruct_wave * reconstruct_wave)
                reconstruct_wave = rescale * reconstruct_wave
            value_max = reconstruct_wave.abs().max()
            if value_max > 2. ** 15:
                reconstruct_wave = reconstruct_wave * (2. ** 14) / value_max
            reconstruct_wave = reconstruct_wave[:mixture_wave.shape[0]]

        if reconstruct_wave.ndim == 2:
            if self.rescale:
                clean_wave = clean_wave.float()
                rescale = (clean_wave * reconstruct_wave).sum(dim=1, keepdim=True) / (
                        reconstruct_wave * reconstruct_wave).sum(dim=1, keepdim=True)
                reconstruct_wave = rescale * reconstruct_wave
            value_max = reconstruct_wave.abs().max(dim=1, keepdim=True)[0]
            value_clip = torch.where(value_max > 2. ** 15, (2. ** 14) * value_max.new_ones(value_max.shape), value_max)
            reconstruct_wave = reconstruct_wave * value_clip / value_max
        
        return [reconstruct_wave.to(torch.int16)]


class ReconstructMask2Wave_forJointOptimization(nn.Module):  
    def __init__(self, n_fft, hop_length, win_type='hamming', win_length=None, rescale=False):
        super(ReconstructMask2Wave_forJointOptimization, self).__init__()
        self.stft = ShortTimeFourierTransform(n_fft=n_fft, hop_length=hop_length, win_type=win_type,
                                              win_length=win_length, is_complex=False)
        self.rescale = rescale

    def forward(self, predict_mask, mixture_wave, clean_wave=None):
        mixture_wave = torch.tensor(mixture_wave)
        
        mixture_wave = mixture_wave.to(torch.float32) / (2. ** 15)
        mixture_spectrum = self.stft(x=mixture_wave, inverse=False)
        if mixture_spectrum.ndim == 4:
            mixture_magnitude = mixture_spectrum[:, :, :, 0]
            mixture_angle = mixture_spectrum[:, :, :, 1]
            predict_mask = predict_mask.transpose(1, 2)  
        else:
            mixture_magnitude = mixture_spectrum[:, :, 0]
            mixture_angle = mixture_spectrum[:, :, 1]
            predict_mask = predict_mask.transpose(0, 1)
        
        # pad_length = mixture_magnitude.shape[1]-predict_mask.shape[1]
        # predict_mask = nn.ZeroPad2d(padding=(0, pad_length))(predict_mask)
        
        reconstruct_magnitude = mixture_magnitude * predict_mask
        reconstruct_wave = self.stft(x=torch.stack([reconstruct_magnitude, mixture_angle], dim=-1), inverse=True,
                                     length=mixture_wave.shape[-1])
        reconstruct_wave = reconstruct_wave * (2. ** 15)
        if reconstruct_wave.ndim == 1:
            if self.rescale:
                clean_wave = torch.tensor(clean_wave)
                clean_wave = clean_wave.to(torch.float32)
                rescale = torch.sum(clean_wave * reconstruct_wave) / torch.sum(reconstruct_wave * reconstruct_wave)
                reconstruct_wave = rescale * reconstruct_wave
            value_max = reconstruct_wave.abs().max()
            if value_max > 2. ** 15:
                reconstruct_wave = reconstruct_wave * (2. ** 14) / value_max
            reconstruct_wave = reconstruct_wave[:mixture_wave.shape[0]]

        if reconstruct_wave.ndim == 2:
            if self.rescale:
                clean_wave = clean_wave.float()
                rescale = (clean_wave * reconstruct_wave).sum(dim=1, keepdim=True) / (
                        reconstruct_wave * reconstruct_wave).sum(dim=1, keepdim=True)
                reconstruct_wave = rescale * reconstruct_wave
            value_max = reconstruct_wave.abs().max(dim=1, keepdim=True)[0]
            value_clip = torch.where(value_max > 2. ** 15, (2. ** 14) * value_max.new_ones(value_max.shape), value_max)
            reconstruct_wave = reconstruct_wave * value_clip / value_max
        return [reconstruct_wave]

def reconstruct_worker(data_dic, data_items, predict_dic, predict_items, reconstruct_folder, reconstruct_items, reconstruct_type,
                       reconstruct_settings, processing_id=None, processing_num=None):  
    data_keys = sorted([*data_dic.keys()])
    reconstruct_dic = {}
    if reconstruct_type == 'mack2wave':
        reconstructor = ReconstructMask2Wave(**reconstruct_settings)
    else:
        raise NotImplementedError('unknown reconstruct_type: {}'.format(reconstruct_type))
    for key_idx in tqdm(range(len(data_keys)), leave=True, desc='0' if processing_id is None else str(processing_id)):
        if processing_id is None: 
            processing_token = True
        else:
            if key_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            key = data_keys[key_idx]
            predict_input = []
            data_input = []
            for predict_item in predict_items:
                predict_input.append(safe_load(file=predict_dic[key][predict_item]))
            for data_item in data_items:
                data = safe_load(file=data_dic[key][data_item])
                data_input.append(data)
                if data == torch.Size([]):
                    print("key=",key)
            reconstruct_output = reconstructor(*predict_input, *data_input)
            temp_dic = {}
            for reconstruct_item, reconstruct_value in zip(reconstruct_items, reconstruct_output):
                temp_dic[reconstruct_item] = os.path.join(reconstruct_folder, reconstruct_item, '{}.wav'.format(key))  
                reconstruct_value=reconstruct_value.numpy()
                safe_store(file=temp_dic[reconstruct_item], data=reconstruct_value, mode='cover', ftype='wav')
            reconstruct_dic[key] = temp_dic
    return reconstruct_dic


def reconstruct_manager(data_jsons, data_items, predict_jsons, predict_items, reconstruct_json, reconstruct_items,
                        reconstruct_type, reconstruct_settings, processing_num=1):  
    reconstruct_folder = os.path.split(reconstruct_json)[0]
    
    data_dic = {}
    data_jsons = data_jsons if isinstance(data_jsons, list) else [data_jsons]
    data_items = data_items if isinstance(data_items, list) else [data_items]
    for data_json in data_jsons:
        sub_dic = safe_load(file=data_json)
        if 'key2path' in sub_dic:
            data_dic.update(sub_dic['key2path'])
        else:
            data_dic.update(sub_dic)
    
    predict_dic = {}
    predict_jsons = predict_jsons if isinstance(predict_jsons, list) else [predict_jsons]
    predict_items = predict_items if isinstance(predict_items, list) else [predict_items]
    for predict_json in predict_jsons:
        sub_dic = safe_load(file=predict_json)
        if 'key2path' in sub_dic:
            predict_dic.update(sub_dic['key2path'])
        else:
            predict_dic.update(sub_dic)
    
    reconstruct_items = reconstruct_items if isinstance(reconstruct_items, list) else [reconstruct_items]
    
    if processing_num > 1:
        pool = Pool(processes=processing_num)  
        all_result = []
        for i in range(processing_num):
            part_result = pool.apply_async(reconstruct_worker, kwds={
                'data_dic': data_dic, 'data_items': data_items, 'predict_dic': predict_dic, 'predict_items': predict_items,
                'reconstruct_folder': reconstruct_folder, 'reconstruct_items': reconstruct_items,
                'reconstruct_type': reconstruct_type, 'reconstruct_settings': reconstruct_settings, 'processing_id': i,
                'processing_num': processing_num})
            all_result.append(part_result)
        pool.close() 
        pool.join() 
        final_dic = {}
        for item in all_result:
            part_dic = item.get()
            final_dic.update(part_dic)
    else:
        final_dic = reconstruct_worker(data_dic=data_dic, data_items=data_items, predict_dic=predict_dic, predict_items=predict_items,
                                       reconstruct_folder=reconstruct_folder, reconstruct_items=reconstruct_items,
                                       reconstruct_type=reconstruct_type, reconstruct_settings=reconstruct_settings)
    
    if os.path.exists(reconstruct_json):
        reconstruct_dic = safe_load(file=reconstruct_json)
        for key, value in final_dic.items():
            if key in reconstruct_dic:
                reconstruct_dic[key].update(value)
            else:
                reconstruct_dic[key] = value
    else:
        reconstruct_dic = final_dic
    safe_store(file=reconstruct_json, data=reconstruct_dic, mode='cover', ftype='json')
    return None
