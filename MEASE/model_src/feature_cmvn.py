#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse

from tool.data_io import safe_store
from network.network_feature_extract import FeatureExtractor
from loader_audio_visual_senone import get_data_loader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lps_cmvn(annotate, n_fft=400, hop_length=160, cmvn=None, **other_params):
    extractor = FeatureExtractor(extractor_type='lps', extractor_setting={
        'n_fft': n_fft, 'hop_length': hop_length, 'win_type': 'hamming', 'win_length': n_fft, 'cmvn': cmvn})
    extractor = nn.DataParallel(extractor).cuda()

    data_loader, checkout_data_loader = get_data_loader(
        annotate=annotate, items=['mixture_wave'], batch_size=96, max_batch_size=512, repeat=1, max_duration=100,
        hop_duration=6, key_output=False, dynamic=True, bucket_length_multiplier=1.1, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=False, seed=123456, epoch=0, logger='print', distributed=False, target_shape=[0], pad_value=[0])
    lps_mean, lps_std, frames_count = 0., 0., 0.

    checkout_data_loader.set_epoch(epoch=0)

    batch_num = len(data_loader)
    batch_bar = tqdm(total=batch_num)

    for batch_idx, batch_data in enumerate(data_loader):
        data = batch_data[0]
        length = batch_data[1]
        mixture_lps, frame_num = extractor(data.cuda(), length)
        frame_num = frame_num.long()
        for sample_idx in range(mixture_lps.size(0)):
            avail_lps = mixture_lps[sample_idx, :, :frame_num[sample_idx]]
            updated_count = frames_count + frame_num[sample_idx]
            lps_mean = lps_mean * (frames_count/updated_count) + avail_lps.sum(dim=1) / updated_count
            lps_std = lps_std * (frames_count/updated_count) + (avail_lps ** 2).sum(dim=1) / updated_count
            frames_count = updated_count
        batch_bar.update(1)
    lps_std = torch.sqrt(lps_std - lps_mean ** 2)
    return frames_count, lps_mean, lps_std

    


def fbank_cmvn(annotate, n_fft=512, hop_length=160, cmvn=None, f_min=0, f_max=8000, n_mels=40, sample_rate=16000,
               norm='slaney', preemphasis_coefficient=0.97, vtln=False, vtln_low=0, vtln_high=8000,
               vtln_warp_factor=1.):
    extractor = FeatureExtractor(extractor_type='fbank', extractor_setting={
        'n_fft': n_fft, 'hop_length': hop_length, 'win_type': 'hamming', 'win_length': n_fft, 'cmvn': cmvn,
        'f_min': f_min, 'f_max': f_max, 'n_mels': n_mels, 'sample_rate': sample_rate, 'norm': norm,
        'preemphasis_coefficient': preemphasis_coefficient, 'vtln': vtln, 'vtln_low': vtln_low,
        'vtln_high': vtln_high, 'vtln_warp_factor': vtln_warp_factor})
    extractor = nn.DataParallel(extractor).cuda()
    data_loader, checkout_data_loader = get_data_loader(
        annotate=annotate, items=['mixture_wave'], batch_size=96, max_batch_size=512, repeat=1, max_duration=100,
        hop_duration=6, key_output=False, dynamic=True, bucket_length_multiplier=1.1, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=False, seed=123456, epoch=0, logger='print', distributed=False, target_shape=[0], pad_value=[0])
    checkout_data_loader.set_epoch(epoch=0)
    
    fbank_mean, fbank_std, frames_count = 0., 0., 0.

    batch_num = len(data_loader)
    batch_bar = tqdm(total=batch_num)

    for batch_idx, batch_data in enumerate(data_loader):
        data = batch_data[0]
        length =batch_data[1]
        mixture_fbank, frame_num = extractor(data.cuda(), length)
        frame_num = frame_num.long()
        for sample_idx in range(mixture_fbank.size(0)):
            avail_fbank = mixture_fbank[sample_idx, :, :frame_num[sample_idx]]
            updated_count = frames_count + frame_num[sample_idx]
            fbank_mean = fbank_mean * (frames_count / updated_count) + avail_fbank.sum(dim=1) / updated_count
            fbank_std = fbank_std * (frames_count / updated_count) + (avail_fbank ** 2).sum(dim=1) / updated_count
            frames_count = updated_count
        batch_bar.update(1)

    fbank_std = torch.sqrt(fbank_std - fbank_mean ** 2)
    return frames_count, fbank_mean, fbank_std

if __name__ == '__main__':
    parser = argparse.ArgumentParser('feature_cmvn')
    parser.add_argument('--annotate', type=str, default=None, help='trainset jsonfile')
    args = parser.parse_args()
    stage = [0,1]
     
    annotate=[args.annotate]
    if 'sim' in args.annotate:
        print('sim')
        if 0 in stage:
            start_time = time.time()
            cmvn_output = fbank_cmvn(
                annotate=annotate, n_fft=400, hop_length=160, cmvn=None, f_min=0,
                f_max=8000, n_mels=40, sample_rate=16000, norm='slaney', preemphasis_coefficient=0.97, vtln=False,
                vtln_low=0, vtln_high=8000, vtln_warp_factor=1.)
            safe_store(file='./cmvn/cmvn_fbank_htk_misp_sim_trainset.pt',
                    data={'mean': cmvn_output[1].cpu(), 'std': cmvn_output[2].cpu()}, ftype='torch', mode='cover')
            print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
            start_time = time.time()
            cmvn_output = fbank_cmvn(
                annotate=annotate, n_fft=400, hop_length=160,
                cmvn='./cmvn/cmvn_fbank_htk_misp_sim_trainset.pt', f_min=0,
                f_max=8000, n_mels=40, sample_rate=16000, norm='slaney', preemphasis_coefficient=0.97, vtln=False,
                vtln_low=0, vtln_high=8000, vtln_warp_factor=1.)
            print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
            print('mean:', cmvn_output[1])
            print('std: ', cmvn_output[2])

        if 1 in stage:
            start_time = time.time()
            cmvn_output = lps_cmvn(annotate=annotate, n_fft=400, hop_length=160, cmvn=None)
            safe_store(file='./cmvn/cmvn_lps_misp_sim_trainset.pt',
                    data={'mean': cmvn_output[1].cpu(), 'std': cmvn_output[2].cpu()}, ftype='torch', mode='cover')
            print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
            start_time = time.time()
            cmvn_output = lps_cmvn(
                annotate=annotate, n_fft=400, hop_length=160,
                cmvn='./cmvn/cmvn_lps_misp_sim_trainset.pt')
            print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
            print('mean:', cmvn_output[1])
            print('std: ', cmvn_output[2])
    if 'real' in args.annotate:
        print('real')
        if 0 in stage:
            start_time = time.time()
            cmvn_output = fbank_cmvn(
                annotate=annotate, n_fft=400, hop_length=160, cmvn=None, f_min=0,
                f_max=8000, n_mels=40, sample_rate=16000, norm='slaney', preemphasis_coefficient=0.97, vtln=False,
                vtln_low=0, vtln_high=8000, vtln_warp_factor=1.)
            safe_store(file='./cmvn/cmvn_fbank_htk_misp_real_trainset.pt',
                    data={'mean': cmvn_output[1].cpu(), 'std': cmvn_output[2].cpu()}, ftype='torch', mode='cover')
            print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
            start_time = time.time()
            cmvn_output = fbank_cmvn(
                annotate=annotate, n_fft=400, hop_length=160,
                cmvn='./cmvn/cmvn_fbank_htk_misp_real_trainset.pt', f_min=0,
                f_max=8000, n_mels=40, sample_rate=16000, norm='slaney', preemphasis_coefficient=0.97, vtln=False,
                vtln_low=0, vtln_high=8000, vtln_warp_factor=1.)
            print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
            print('mean:', cmvn_output[1])
            print('std: ', cmvn_output[2])

        if 1 in stage:
            start_time = time.time()
            cmvn_output = lps_cmvn(annotate=annotate, n_fft=400, hop_length=160, cmvn=None)
            safe_store(file='./cmvn/cmvn_lps_misp_real_trainset.pt',
                    data={'mean': cmvn_output[1].cpu(), 'std': cmvn_output[2].cpu()}, ftype='torch', mode='cover')
            print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
            start_time = time.time()
            cmvn_output = lps_cmvn(
                annotate=annotate, n_fft=400, hop_length=160,
                cmvn='./cmvn/cmvn_lps_misp_real_trainset.pt')
            print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
            print('mean:', cmvn_output[1])
            print('std: ', cmvn_output[2])
