#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tool.data_io import safe_load
from loader.sampler_dynamic_distributed import DynamicBatchSampler, DistributedSamplerWrapper
from loader.loader_truncation_dynamic_distributed import BaseTruncationDataset, PaddedBatch
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
import json
from tool_map import (map_mandarin_phone179_token, map_mandarin_phone61_phone179, map_mandarin_phone61_token,
                      map_mandarin_phone32_phone61, map_mandarin_phone32_token, map_mandarin_place8_phone32,
                      map_mandarin_place8_token, map_english_phone39_token, map_english_viseme13_phone39,
                      map_english_viseme13_token, map_english_place10_phone39, map_english_place10_token,
                      map_mandarin_place10_token, map_mandarin_pinyin61_token,
                      map_english_phone41_place11, map_english_phone41_token, map_english_place11_token)

from sklearn.model_selection import KFold

map_mandarin_phone179_phone61, map_mandarin_phone61_phone32, map_mandarin_phone32_place8 = {}, {}, {}
for phone61, phone179 in map_mandarin_phone61_phone179.items():
    for mono_phone in phone179:
        map_mandarin_phone179_phone61[mono_phone] = phone61
for phone32, phone61 in map_mandarin_phone32_phone61.items():
    for mono_phone in phone61:
        map_mandarin_phone61_phone32[mono_phone] = phone32
for place8, phone32 in map_mandarin_place8_phone32.items():
    for mono_phone in phone32:
        map_mandarin_phone32_place8[mono_phone] = place8

map_english_phone39_viseme13, map_english_phone39_place10 = {}, {}
for viseme13, phone39 in map_english_viseme13_phone39.items():
    for mono_phone in phone39:
        map_english_phone39_viseme13[mono_phone] = viseme13
for place10, phone39 in map_english_place10_phone39.items():  
    for mono_phone in phone39:
        map_english_phone39_place10[mono_phone] = place10  

def read_token_list(token_list):
    if isinstance(token_list,str):
        with open(token_list,encoding='utf-8') as f:
            token_list = [line.rstrip() for line in f]
    tokenizer = build_tokenizer(token_type='char')
    token_id_converter = TokenIDConverter(
        token_list=token_list
    )
    return tokenizer, token_id_converter

tokenizer, token_id_converter = read_token_list("./asr_file/tokens.txt")

def text_process(tokenizer, token_id_converter, text):
    tokens = tokenizer.text2tokens(text)
    text_ints = token_id_converter.tokens2ids(tokens)
    data = np.array(text_ints, dtype=np.int64)
    data = torch.tensor(data)
    return data


class AudioVisualTruncationDataset(BaseTruncationDataset):
    def __init__(self, annotate, repeat_num, max_duration, hop_duration, items, duration_factor=None, deleted_keys=None,
                 key_output=False, logger=None):
        super(AudioVisualTruncationDataset, self).__init__(
            annotate=annotate, repeat_num=repeat_num, max_duration=max_duration, hop_duration=hop_duration, items=items,
            duration_factor=duration_factor, deleted_keys=deleted_keys, key_output=key_output, logger=logger)

    def _get_value(self, key, item, begin, duration, item2file):
        if item in ['lip_frames', 'clean_wave', 'mixture_wave', 'irm', 'noise_wave', 'gray_lip_frames', 'text']:
            item2sample_rate = {'lip_frames': 25, 'gray_lip_frames': 25, 'clean_wave': 16000, 'mixture_wave': 16000, 'irm': 100, 'noise_wave': 16000, 'text': 1}
            begin_point = int(np.around(begin * item2sample_rate[item]))
            end_point = int(np.around(begin_point + duration * item2sample_rate[item]))  
            if item in ['irm']:
                item_data = safe_load(file=item2file[item])[begin_point: end_point].float()
            elif item in ['gray_lip_frames']:
                if '.mp4' in item2file['lip_frames']:
                    new_lip_path = item2file['lip_frames'].replace("/train13/cv1/hangchen2/avse2_evalset","/train13/cv1/hangchen2/avse2_evalset/croped_lip").replace(".mp4",".pt")
                    item_data = safe_load(file=new_lip_path)[begin_point: end_point]
                    item_data = np.array(item_data)
                    frames = []
                    for frame in item_data:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    item_data = np.array(frames)
                    item_data = torch.tensor(item_data) 
                else:
                    item_data = safe_load(file=item2file['lip_frames'])[begin_point: end_point]
                    item_data = np.array(item_data)
                    frames = []
                    for frame in item_data:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    item_data = np.array(frames)
                    item_data = torch.tensor(item_data) 
            elif item in ['text']:
                item_data = item2file[item]
                if "@" in item_data:
                    item_data = torch.cat([text_process(tokenizer, token_id_converter,json.loads(f'"{path}"')) for path in item2file[item].split("@")])
                    item_data = item_data.squeeze()
                else:
                    item_data = item2file[item]
                    item_data = json.loads(f'"{item_data}"')
                    item_data = text_process(tokenizer, token_id_converter,item_data)
            else:
                item_data = safe_load(file=item2file[item])[begin_point:end_point]
                
        elif item in ['key']:
            item_data = key
        elif item.split('_')[0] in ['classification', 'posteriori']:
            item_type, language, grain, interval = item.split('_')  
            grain = int(grain)
            interval = float(interval)
            fa_information = safe_load(file=item2file['fa'])
            if language == 'mandarin':
                if grain in [9]:
                    characters = fa_information['phone_60']
                    end_timestamp = np.array(fa_information['phone_60_end_time']).astype(np.float32) - begin
                elif grain in [62]:  
                    if 'test' in item2file['fa']:
                        pinyin_path = "/train20/intern/permanent/cxwang15/AVSE_challenge2/MISP2021AVSR_with_Home_Noise/test_pinyin_label/" + item2file['fa'].split('/')[-1].split('_')[0] + ".pt"
                    else:
                        pinyin_path = "/train20/intern/permanent/cxwang15/AVSE_challenge2/MISP2021AVSR_with_Home_Noise/train_pinyin_label/" + item2file['fa'].split('/')[-1].split('_')[0] + ".pt"
                    fa_information = safe_load(file=pinyin_path)
                    characters = fa_information['pinyin_61']
                    end_timestamp = np.array(fa_information['pinyin_61_end_time']).astype(np.float32) - begin  #.astype('float64')
                if grain in [10]:
                    characters = fa_information['place_10']
                    end_timestamp = np.array(fa_information['place_10_end_time']).astype(np.float32) - begin
                else: 
                    characters = fa_information['CD-phone']
                    end_timestamp = np.array(fa_information['end_frame_40ms'])*0.04 - begin
            elif language == 'english':
                if grain in [39, 13, 10]:
                    characters = fa_information['phone_39']  
                    end_timestamp = np.array(fa_information['phone_39_end_time']) - begin 
                elif grain in [41, 11]:
                    characters = fa_information['phone_41']
                    end_timestamp = np.array(fa_information['phone_41_end_time'], dtype='float64') - begin
                elif grain in [2704, 2008, 1256, 448]:
                    characters = fa_information['senone_{}'.format(grain)]
                    end_timestamp = np.array(fa_information['senone_{}_end_time'.format(grain)]) - begin
                else:
                    raise ValueError('unknown grain {} for english'.format(grain))
            else:
                raise NotImplementedError('unknown language: {}'.format(language))
            if item_type == 'classification':
                item_data = (torch.ones(int(np.around(duration/interval))) * grain).long()  
                begin_timestamp = 0.
                for i in range(len(characters)):  
                    if end_timestamp[i] > 0:
                        begin_idx = int(np.around(begin_timestamp / interval))
                        if end_timestamp[i] <= duration:
                            end_idx = int(np.around(end_timestamp[i] / interval))
                            item_data[begin_idx:end_idx] = self.map_character_token(characters[i], grain, language)
                            begin_timestamp = end_timestamp[i]
                        else:
                            end_idx = int(np.around(duration / interval))
                            item_data[begin_idx:end_idx] = self.map_character_token(characters[i], grain, language)
                            break
                
            else:
                item_data = torch.zeros(int(np.around(duration/interval)), grain, dtype=torch.float)
                begin_timestamp = 0.
                for i in range(len(characters)):
                    if end_timestamp[i] > 0:
                        begin_idx = int(np.around(begin_timestamp / interval))
                        token = self.map_character_token(characters[i], grain, language)
                        if end_timestamp[i] <= duration:
                            end_idx = int(np.around(end_timestamp[i] / interval))
                            if token < grain:
                                item_data[begin_idx:end_idx, token] = 1.
                            begin_timestamp = end_timestamp[i]
                        else:
                            end_idx = int(np.around(duration / interval))
                            if token < grain:
                                item_data[begin_idx:end_idx, token] = 1.
                            break
        else:
            raise NotImplementedError('unknown output')
        
        return item_data

    @staticmethod
    def map_character_token(character, grain, language):
        if language == 'mandarin':
            if grain in [218095, 9004, 8004, 7004, 6004, 5004, 4004, 3004, 2004, 1004]:
                if character in map_mandarin_cdphone_senone:
                    token = map_mandarin_cdphone_senone[character][grain]
                else:
                    token = grain
            elif grain == 179:
                phone = character.split('-')[-1].split('+')[0]
                token = map_mandarin_phone179_token[phone]
            elif grain == 61:
                phone = character.split('-')[-1].split('+')[0]
                token = map_mandarin_phone61_token[map_mandarin_phone179_phone61[phone]]
            elif grain == 32:
                phone = character.split('-')[-1].split('+')[0]
                token = map_mandarin_phone32_token[map_mandarin_phone61_phone32[map_mandarin_phone179_phone61[phone]]]
            elif grain == 8:
                phone = character.split('-')[-1].split('+')[0]
                token = map_mandarin_place8_token[map_mandarin_phone32_place8[map_mandarin_phone61_phone32[
                    map_mandarin_phone179_phone61[phone]]]]
            elif grain == 9:
                phone = character
                token = map_mandarin_place10_token[phone]
            elif grain == 10:
                phone = character
                token = map_mandarin_place10_token[phone]
            elif grain == 62:
                pinyin = character
                token = map_mandarin_pinyin61_token[pinyin]
            else:
                raise NotImplementedError('unknown grain')
        elif language == 'english':
            if grain == 39:
                phone = character
                token = map_english_phone39_token[phone]
            elif grain == 13:
                phone = character
                token = map_english_viseme13_token[map_english_phone39_viseme13[phone]]
            elif grain == 10:
                phone = character
                token = map_english_place10_token[map_english_phone39_place10[phone]]  
            elif grain == 11:
                phone = character
                token = map_english_place11_token[map_english_phone41_place11[phone]]
            elif grain in [2704, 2008, 1256, 448]:
                token = character
            else:
                raise NotImplementedError('unknown grain')
        else:
            raise NotImplementedError('unknown language')
        return token  


def get_data_loader(
        annotate, items, batch_size, max_batch_size, target_shape, pad_value, repeat=1, max_duration=100,
        hop_duration=10, duration_factor=None, deleted_keys=None, key_output=False, dynamic=True,
        bucket_length_multiplier=1.1, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, seed=123456,
        epoch=0, logger=None, distributed=False, **other_params):
    dataset = AudioVisualTruncationDataset(annotate=annotate, repeat_num=repeat, max_duration=max_duration,
                                           hop_duration=hop_duration, items=items, duration_factor=duration_factor,
                                           deleted_keys=deleted_keys, key_output=key_output, logger=logger)

    data_sampler = DynamicBatchSampler(lengths_list=dataset.duration, batch_size=batch_size, dynamic=dynamic,
                                       max_batch_size=max_batch_size, epoch=epoch, drop_last=drop_last, logger=logger,
                                       bucket_length_multiplier=bucket_length_multiplier, shuffle=shuffle, seed=seed)

    if distributed:
        data_sampler = DistributedSamplerWrapper(sampler=data_sampler, seed=seed, shuffle=shuffle, drop_last=drop_last)
    
    collate_fn = PaddedBatch(items=dataset.items, target_shape=target_shape, pad_value=pad_value)

    data_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
                             batch_sampler=data_sampler)

    return data_loader, data_sampler

def get_KFold_data_loader(
        kfold_num, annotate, items, batch_size, max_batch_size, target_shape, pad_value, repeat=1, max_duration=100,
        hop_duration=10, duration_factor=None, deleted_keys=None, key_output=False, dynamic=True,
        bucket_length_multiplier=1.1, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, seed=123456,
        epoch=0, logger=None, distributed=False, **other_params):
    train_dataset = AudioVisualTruncationDataset(annotate=annotate, repeat_num=repeat, max_duration=max_duration,
                                           hop_duration=hop_duration, items=items, duration_factor=duration_factor,
                                           deleted_keys=deleted_keys, key_output=key_output, logger=logger)
    kf = KFold(n_splits=kfold_num, shuffle=True)
    train_loaders = []
    eval_loaders = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)


    
