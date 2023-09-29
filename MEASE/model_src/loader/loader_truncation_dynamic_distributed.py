#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn.functional as nf
from torch.utils.data import Dataset

import json
import codecs
import logging
import numpy as np


class BaseTruncationDataset(Dataset):
    def __init__(self, annotate, repeat_num, max_duration, hop_duration, items, duration_factor=None, deleted_keys=None,
                 key_output=False, logger=None, **other_params):
        super(BaseTruncationDataset, self).__init__()
        deleted_keys = [] if deleted_keys is None else deleted_keys
        if isinstance(annotate, str):
            annotate = [annotate]
        elif isinstance(annotate, list) and all(isinstance(a, str) for a in annotate):
            annotate = annotate
        else:
            raise ValueError('unknown annotate type: {}.'.format(annotate))
        annotate_num = len(annotate)
        repeat_num = self.expend_params(value=repeat_num, length=annotate_num)
        max_duration = self.expend_params(value=max_duration, length=annotate_num)  
        hop_duration = self.expend_params(value=hop_duration, length=annotate_num)  
        duration_factor = self.expend_params(value=duration_factor, length=annotate_num)

        self.items = ['key', *items] if items[0] != 'key' and key_output else items
        self.keys = []
        self.duration = []
        self.begin = []
        self.key2path = {}
        for annotate_id in range(annotate_num):
            write_log(content='Load {} from {}, max_duration is {} s, hop_duration is {} s, repeat {} times.'.format(
                ','.join(items), annotate[annotate_id], max_duration[annotate_id], hop_duration[annotate_id],
                repeat_num[annotate_id]), level='info', logger=logger)
            with codecs.open(annotate[annotate_id], 'r') as handle: 
                data_dic = json.load(handle)
            data_keys = data_dic['keys']
            data_duration = data_dic['duration']
            data_key2path = data_dic['key2path']
            # del sample
            for key_idx in range(len(data_keys)-1, -1, -1):
                if data_keys[key_idx] in deleted_keys:  
                    data_key2path.pop(data_keys[key_idx])
                    data_keys.pop(key_idx)
                    data_duration.pop(key_idx)
            self.key2path.update(data_key2path)
            split_keys, split_duration, split_begin = self.cut_off(
                keys=data_keys, duration=data_duration, max_duration=max_duration[annotate_id],
                hop_duration=hop_duration[annotate_id], duration_factor=duration_factor[annotate_id])
            for _ in range(repeat_num[annotate_id]):
                self.keys = self.keys + split_keys
                self.duration = self.duration + split_duration
                self.begin = self.begin + split_begin
            del data_dic
        write_log(content='Delete samples: {}'.format(deleted_keys), level='info', logger=logger)
        write_log(content='All duration is {} h'.format(np.sum(self.duration) / 3600.), level='info', logger=logger)

    def __getitem__(self, index):
        main_key = self.keys[index]
        item2paths = self.key2path[main_key]
        value_lst = []
        for item in self.items:  
            value = self._get_value(key=main_key, item=item, begin=self.begin[index], duration=self.duration[index],
                                    item2file=item2paths)

            if isinstance(value, list):
                value_lst.extend(value)
            else:
                value_lst.append(value)
            del value
        return value_lst  

    def _get_value(self, key, item, begin, duration, item2file):
        return key

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def cut_off(keys, duration, max_duration, hop_duration, duration_factor=None):
        split_keys = []
        split_duration = []
        split_begin = []
        for idx in range(len(keys)):  
            idx_key = keys[idx]
            idx_duration = duration[idx]
            idx_begin = 0.
            while idx_duration > max_duration:
                split_keys.append(idx_key)
                split_duration.append(max_duration)
                split_begin.append(idx_begin)
                idx_duration = idx_duration - hop_duration
                idx_begin = idx_begin + hop_duration
            split_keys.append(idx_key)
            if duration_factor:
                final_duration = idx_duration - idx_duration % duration_factor
            else:
                final_duration = idx_duration
            split_duration.append(final_duration)
            split_begin.append(idx_begin)
        return split_keys, split_duration, split_begin

    @staticmethod
    def expend_params(value, length):
        if isinstance(value, list):
            if len(value) == length:
                return value
            else:
                raise ValueError('list have unmatched length: {}'.format(value))
        else:
            return [value for _ in range(length)]


class PaddedBatch(object):
    def __init__(self, items, target_shape, pad_value):
        self.items = items
        self.target_shape = target_shape
        self.pad_value = pad_value

    def __call__(self, dataset_outputs):
        pad_idx = 0
        batched_value = []
        for item, *batch_values in zip(self.items, *dataset_outputs):
            if item in ['key']:
                batched_value.append(batch_values)
            else:
                batched_value.extend(
                    self._batch_pad_right(target_shape=self.target_shape[pad_idx], pad_value=self.pad_value[pad_idx],
                                          tensors=batch_values))
                pad_idx = pad_idx + 1
        return batched_value

    def _batch_pad_right(self, target_shape, pad_value, tensors):
        if not len(tensors):
            raise IndexError("Tensors list must not be empty")

        if len(tensors) > 1 and not(any([tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))])):
            raise IndexError("All tensors must have same number of dimensions")

        shape_items = torch.zeros(len(tensors)+1, tensors[0].ndim, dtype=torch.long)
        shape_items[-1, :] = torch.tensor(target_shape)  
        for x_idx in range(len(tensors)):
            shape_items[x_idx] = torch.tensor(tensors[x_idx].shape)
        target_shape = shape_items.max(dim=0).values.tolist()
        length = shape_items[:-1, 0]

        batched = []
        for t in tensors:
            batched.append(self.pad_right_to(tensor=t, target_shape=target_shape, value=pad_value))
        batched = torch.stack(batched)
        return batched, length


    @staticmethod
    def pad_right_to(tensor, target_shape, mode="constant", value=0):
        """
        This function takes a torch tensor of arbitrary shape and pads it to target
        shape by appending values on the right.
        Parameters
        ----------
        tensor : input torch tensor
            Input tensor whose dimension we need to pad.
        target_shape:
            Target shape we want for the target tensor its len must be equal to tensor.ndim
        mode : str
            Pad mode, please refer to torch.nn.functional.pad documentation.
        value : float
            Pad value, please refer to torch.nn.functional.pad documentation.
        Returns
        -------
        tensor : torch.Tensor
            Padded tensor.
        valid_vals : list
            List containing proportion for each dimension of original, non-padded values.
        """
        assert len(target_shape) == tensor.ndim, 'target_shape is {}, but tensor shape is {}'.format(target_shape,
                                                                                                     tensor.shape)
        pads = []  # this contains the abs length of the padding for each dimension.
        i = len(target_shape) - 1  # iterating over target_shape ndims
        while i >= 0:
            assert (target_shape[i] >= tensor.shape[i]), 'Target shape must be >= original shape for every dim'
            pads.extend([0, target_shape[i] - tensor.shape[i]])
            i -= 1
        tensor = torch.tensor(tensor)
        tensor = nf.pad(tensor, pads, mode=mode, value=value)
        return tensor


def write_log(content, logger=None, level=None, **other_params):
    """
    write log
    :param content: content need to write
    :param level: level of content or None
    :param logger: False or logger
    :param other_params: reserved interface
    :return: None
    """
    if not logger:
        pass
    elif logger == 'print':
        print(content)
    elif isinstance(logger, logging.Logger):
        if not level:
            pass
        else:
            assert level in ['debug', 'info', 'warning', 'error', 'critical'], 'unknown level'
            getattr(logger, level)(content)
    else:
        raise NotImplementedError('unknown logger')
    return None


