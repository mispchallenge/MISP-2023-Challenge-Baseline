#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch.nn as nn
import torch

from network_classifier import AudioClassifier, VisualClassifier, AudioVisualClassifier
from network.network_feature_extract import FeatureExtractor
from network_audio_visual_fusion import AudioVisualFuse
from network.network_common_module import expend_params, chose_norm
from network.asr import ICME_ASR
from pipeline_reconstruct import ReconstructMask2Wave_forJointOptimization

class MultimodalEmbeddingAwareSpeechEnhancement_ASR_jointOptimization(nn.Module):
    def __init__(self, out_channels, input_types, extractor_types, extractor_settings, encoder_types, encoder_settings,
                 fusion_type, fusion_setting, decoder_type, decoder_setting, reconstruct_settings, skip_convert=True):
        super(MultimodalEmbeddingAwareSpeechEnhancement_ASR_jointOptimization, self).__init__()

        assert input_types == ['mixture_wave', 'lip_frames'], 'unknown input_types: {}'.format(
            input_types)
        assert extractor_types == ['lps', 'multimodal_embedding'], 'unknown extractor_types: {}'.format(extractor_types)

        self.extractor = EmbeddingExtractor(  
            embedding_type=extractor_types[0], extractor_setting=extractor_settings[0])
        self.multimodal_embedding_extractor = EmbeddingExtractor(  
            embedding_type=extractor_types[1], extractor_setting=extractor_settings[1])

        self.encoder = Encoder( 
            encoder_type=encoder_types[0],
            encoder_setting={**encoder_settings[0], 'in_channels': self.extractor.embedding_size})
        self.multimodal_encoder = Encoder(  
            encoder_type=encoder_types[1],
            encoder_setting={**encoder_settings[1], 'in_channels': self.multimodal_embedding_extractor.embedding_size})

        self.fuse = AudioVisualFuse( 
            fuse_type=fusion_type,
            fuse_setting={**fusion_setting,
                          'in_channels': [self.encoder.out_channels, self.multimodal_encoder.out_channels]})

        self.decoder = Decoder( 
            decoder_type=decoder_type,
            decoder_setting={**decoder_setting, 'in_channels': self.fuse.out_channels})

        self.mask_convolution = nn.Conv1d(
            in_channels=self.decoder.out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.mask_activate = nn.Sigmoid()
        self.out_channels = out_channels
        self.skip_convert = expend_params(value=skip_convert, length=2)

        self.reconstruct_wave = ReconstructMask2Wave_forJointOptimization(**reconstruct_settings)
        self.ICME_ASR = ICME_ASR()
        for name, parameter in self.ICME_ASR.named_parameters():
            parameter.requires_grad = False


    def forward(self, mixture_wave, lip_frames, wav_length, video_length, text=None, text_length=None):
        if not self.skip_convert[0]:
            mixture_wave = mixture_wave.float()  
        representation, length = self.extractor(mixture_wave, wav_length)  
        if not self.skip_convert[1]:
            lip_frames = lip_frames.float()
        multimodal_representation, _ = self.multimodal_embedding_extractor(mixture_wave, lip_frames)
        encode_representation, length = self.encoder(representation, length)
        multimodal_encode_representation, _ = self.multimodal_encoder(multimodal_representation)
        fused_representation, length = self.fuse([encode_representation], [multimodal_encode_representation], length)
        output_representation, length = self.decoder(fused_representation, length)
        predict_mask = self.mask_activate(self.mask_convolution(output_representation))
        reconstruct_wav = self.reconstruct_wave(predict_mask.transpose(1, 2), mixture_wave)
        if text != None: 
            loss_ctc, loss_att = self.ICME_ASR(reconstruct_wav[0].to(torch.float32), wav_length, text, text_length)
            return loss_ctc, loss_att, predict_mask.transpose(1, 2), length
        else: 
            return predict_mask.transpose(1, 2), length

class MultimodalEmbeddingAwareSpeechEnhancement(nn.Module):
    def __init__(self, out_channels, input_types, extractor_types, extractor_settings, encoder_types, encoder_settings,
                 fusion_type, fusion_setting, decoder_type, decoder_setting, skip_convert=True):
        super(MultimodalEmbeddingAwareSpeechEnhancement, self).__init__()

        assert input_types == ['mixture_wave', 'lip_frames'], 'unknown input_types: {}'.format(
            input_types)
        assert extractor_types == ['lps', 'multimodal_embedding'], 'unknown extractor_types: {}'.format(extractor_types)

        self.extractor = EmbeddingExtractor(   
            embedding_type=extractor_types[0], extractor_setting=extractor_settings[0])
        self.multimodal_embedding_extractor = EmbeddingExtractor( 
            embedding_type=extractor_types[1], extractor_setting=extractor_settings[1])

        self.encoder = Encoder( 
            encoder_type=encoder_types[0],
            encoder_setting={**encoder_settings[0], 'in_channels': self.extractor.embedding_size})
        self.multimodal_encoder = Encoder(  
            encoder_type=encoder_types[1],
            encoder_setting={**encoder_settings[1], 'in_channels': self.multimodal_embedding_extractor.embedding_size})

        self.fuse = AudioVisualFuse( 
            fuse_type=fusion_type,
            fuse_setting={**fusion_setting,
                          'in_channels': [self.encoder.out_channels, self.multimodal_encoder.out_channels]})

        self.decoder = Decoder(
            decoder_type=decoder_type,
            decoder_setting={**decoder_setting, 'in_channels': self.fuse.out_channels})

        self.mask_convolution = nn.Conv1d(
            in_channels=self.decoder.out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.mask_activate = nn.Sigmoid()
        self.out_channels = out_channels
        self.skip_convert = expend_params(value=skip_convert, length=2)

    def forward(self, mixture_wave, lip_frames, length=None):
        if not self.skip_convert[0]:
            mixture_wave = mixture_wave.float()  
        representation, length = self.extractor(mixture_wave, length)
        if not self.skip_convert[1]:
            lip_frames = lip_frames.float()
        multimodal_representation, _ = self.multimodal_embedding_extractor(mixture_wave, lip_frames)
        encode_representation, length = self.encoder(representation, length)
        multimodal_encode_representation, _ = self.multimodal_encoder(multimodal_representation)
        fused_representation, length = self.fuse([encode_representation], [multimodal_encode_representation], length)
        output_representation, length = self.decoder(fused_representation, length)
        predict_mask = self.mask_activate(self.mask_convolution(output_representation))
        return predict_mask.transpose(1, 2), length

class EmbeddingExtractor(nn.Module): 
    def __init__(self, embedding_type, extractor_setting):
        """"
        :param embedding_type: 'mixture_lps', 'visual_embedding', 'audio_embedding', 'audio_visual_embedding'
        :param extractor_setting:
        """
        super(EmbeddingExtractor, self).__init__()
        self.embedding_type = embedding_type
        if 'requires_grad' in extractor_setting:
            requires_grad = extractor_setting.pop('requires_grad') 
        else:
            requires_grad = True
        if self.embedding_type in ['lps']:
            self.embedding_extractor = FeatureExtractor(  
                extractor_type=embedding_type, extractor_setting=extractor_setting)
            self.embedding_size = self.embedding_extractor.output_size
        elif self.embedding_type == 'visual_embedding':
            extractor_setting = {**extractor_setting, 'extract_feature': True, 'skip_convert': True}
            self.embedding_extractor = VisualClassifier(**extractor_setting) 
            self.embedding_size = self.embedding_extractor.backbone.out_channels
        elif self.embedding_type == 'audio_embedding':
            extractor_setting = {**extractor_setting, 'extract_feature': True, 'skip_convert': True}
            self.embedding_extractor = AudioClassifier(**extractor_setting)
            self.embedding_size = self.embedding_extractor.backbone.out_channels
        elif self.embedding_type == 'multimodal_embedding':
            extractor_setting = {**extractor_setting, 'extract_feature': True, 'skip_convert': True}
            self.embedding_extractor = AudioVisualClassifier(**extractor_setting)
            self.embedding_size = self.embedding_extractor.audio_visual_fusion.out_channels
        else:
            raise NotImplementedError('unknown embedding_type')

        if not requires_grad:
            for extractor_parameter in self.embedding_extractor.parameters():
                extractor_parameter.requires_grad = False

    def forward(self, *x):
        embedding, length = self.embedding_extractor(*x)
        return embedding, length


class Encoder(nn.Module):
    def __init__(self, encoder_type, encoder_setting):
        """
        :param encoder_type:
        :param encoder_setting:
        """
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type == 'none':
            self.encoder = None
            self.out_channels = encoder_setting['in_channels']
            requires_grad = True
        elif self.encoder_type == 'DSResConv':
            default_encoder_setting = {
                'layer_num': 5, 'out_channels': 1536, 'kernel': 5, 'stride': [1, 2, 1, 2, 1], 'dilation': 1,
                'norm_type': 'BN1d', 'requires_grad': True}
            encoder_setting = {**default_encoder_setting, **encoder_setting}
            requires_grad = encoder_setting.pop('requires_grad') if 'requires_grad' in encoder_setting else True
            self.encoder = DSResConvStack(**encoder_setting)
            self.out_channels = expend_params(
                value=encoder_setting['out_channels'], length=encoder_setting['layer_num'])[-1]
        else:
            raise NotImplementedError('unknown encoder_type')

        if not requires_grad:
            for encoder_parameter in self.encoder.parameters():
                encoder_parameter.requires_grad = False

    def forward(self, x, length=None): 
        if self.encoder_type == 'none':
            encoder_output = x
        elif self.encoder_type == 'DSResConv':
            encoder_output, length = self.encoder(x, length)
        else:
            raise NotImplementedError('unknown encoder_type')
        return encoder_output, length


class Decoder(nn.Module):
    def __init__(self, decoder_type, decoder_setting):
        super(Decoder, self).__init__()
        self.decoder_type = decoder_type
        if self.decoder_type == 'none':
            self.decoder = None
            self.out_channels = decoder_setting['in_channels']
            requires_grad = True
        elif self.decoder_type == 'DSResConv':
            default_decoder_setting = {  
                'layer_num': 15, 'out_channels': 1536, 'kernel': 5, 'requires_grad': True,
                'stride': [1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1], 'dilation': 1, 'norm_type': 'BN1d'}
            decoder_setting = {**default_decoder_setting, **decoder_setting}
            requires_grad = decoder_setting.pop('requires_grad') if 'requires_grad' in decoder_setting else True
            self.decoder = DSResConvStack(**decoder_setting)
            self.out_channels = expend_params(
                value=decoder_setting['out_channels'], length=decoder_setting['layer_num'])[-1]
        else:
            raise NotImplementedError('unknown decoder_type')

        if not requires_grad:
            for decoder_parameter in self.decoder.parameters():
                decoder_parameter.requires_grad = False

    def forward(self, x, length=None):
        if self.decoder_type == 'none':
            decoder_output = x
        elif self.decoder_type == 'DSResConv':
            decoder_output, length = self.decoder(x, length)
        else:
            raise NotImplementedError('unknown encoder_type')
        return decoder_output, length


class DSResConvStack(nn.Module):
    def __init__(
            self, in_channels, layer_num, out_channels, kernel, stride, dilation, norm_type='BN1d', **other_params):
        super(DSResConvStack, self).__init__()
        out_channels = expend_params(out_channels, layer_num)
        kernel = expend_params(kernel, layer_num)
        stride = expend_params(stride, layer_num)
        dilation = expend_params(dilation, layer_num)
        norm_type = expend_params(norm_type, layer_num)
        in_channel = in_channels
        stack = []
        self.layer_num = layer_num
        self.length_retract = 1
        for i in range(layer_num):
            stack.append(DSResConvolution(in_channels=in_channel, out_channels=out_channels[i], kernel_size=kernel[i],
                                          stride=stride[i], dilation=dilation[i], norm_type=norm_type[i]))
            in_channel = out_channels[i] 
            self.length_retract = self.length_retract*stride[i]  #缩放
        self.stack = nn.ModuleList(stack) 

    def forward(self, x, length=None):
        if length is not None:
            length = (length / self.length_retract).long()
        for i in range(self.layer_num):
            x = self.stack[i](x)
        return x, length


class DSResConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, norm_type="gLN"):
        """
        Args:
            in_channels: Number of channel in input feature
            out_channels: Number of channel in output feature
            kernel_size: Kernel size in D-convolution
            stride: stride in D-convolution
            dilation: dilation factor
            norm_type: BN1d, gLN1d, cLN1d, gLN1d is no causal
        """
        super(DSResConvolution, self).__init__()
        # Use `groups` option to implement depth-wise convolution
        # [M, H, K] -> [M, H, K]
        padding = int((kernel_size-1)*dilation/2)
        self.relu = nn.ReLU()
        self.norm = chose_norm(norm_type, in_channels)
        if in_channels != out_channels:
            self.res_convolution = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                             stride=1, padding=0,  dilation=1, groups=1, bias=False,
                                             padding_mode='zeros')
        else:
            self.res_convolution = None
        if stride >= 1:  
            self.d_convolution = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=in_channels,
                                           bias=False, padding_mode='zeros')
            if stride == 1:
                self.res_downsample = False
            else:
                self.res_downsample = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2,
                                                   ceil_mode=False, count_include_pad=True)
        elif stride > 0:
            stride = int(1./stride)
            self.d_convolution = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels,
                                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                                    output_padding=stride-1, groups=in_channels, bias=False,
                                                    dilation=dilation, padding_mode='zeros')
            self.res_downsample = nn.Upsample(size=None, scale_factor=stride, mode='linear', align_corners=False)
        else:
            raise ValueError('error stride {}'.format(stride))
        self.s_convolution = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                       padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        res = x
        x = self.relu(x)
        x = self.norm(x)
        x = self.d_convolution(x)
        x = self.s_convolution(x)
        if self.res_convolution: 
            res = self.res_convolution(res)
        if self.res_downsample:
            res = self.res_downsample(res)
        return x+res
