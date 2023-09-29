#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import logging
import argparse

from pipeline_reconstruct import reconstruct_manager
from tool.data_io import safe_load, safe_store
from tool.match_search import search

logging.getLogger('matplotlib.font_manager').disabled = True


def init_logger(file_level_modes, **other_params):
    """
    initialize logger
    :param file_level_modes: list of (filepath, log_level, log_mode)
    :param other_params: reserved interface
    :return: logger
    """
    def handler_init(file_level):
        level_map = {
            'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR,
            'critical': logging.CRITICAL}
        sub_file, sub_level, sub_mode = file_level
        if '.log' in sub_file:
            os.makedirs(os.path.split(sub_file)[0], exist_ok=True)
            sub_handler = logging.FileHandler(sub_file, mode=sub_mode)
        elif sub_file == 'command_line':
            sub_handler = logging.StreamHandler()
        else:
            raise ValueError('unknown handle file')
        if sub_level in level_map:
            sub_handler.filter = lambda record: record.levelno == level_map[sub_level]
        else:
            raise ValueError('unknown logger level')
        sub_handler.setFormatter(logging.Formatter(fmt='%(asctime)s|%(message)s', datefmt='%Y%m%d-%H:%M:%S'))
        return sub_handler

    used_logger = logging.getLogger()
    used_logger.setLevel(level=logging.DEBUG)
    for handler_id in range(len(file_level_modes)):
        used_logger.addHandler(handler_init(file_level=file_level_modes[handler_id]))
    return used_logger


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


def reconstruct_interface(hparams, args, **other_params):
    exp_folder_candidates = search(root_dir=args.exp_dir, regular_expression=args.config)
    if len(exp_folder_candidates) == 0:
        raise ValueError('regular_expression {} is error, find no possible exp folder'.format(args.config))
    elif len(exp_folder_candidates) > 1:
        raise ValueError('regular_expression {} is error, find more than 1 possible exp folder: {}'.format(args.config, exp_folder_candidates))
    else:
        exp_folder = exp_folder_candidates[0]
    
    if args.used_model == -1:
        predict_store_dir = os.path.join(exp_folder, 'predict_best_test_{}'.format(args.predict_data)) #predict_best_avseeval_
    else:
        predict_store_dir = os.path.join(exp_folder, 'predict_epoch{}_{}'.format(args.used_model, args.predict_data))
    
    data_with_predicted_json = os.path.join(predict_store_dir, 'data_with_predicted.json')  
    reconstruct_json = os.path.join(predict_store_dir, 'reconstruct', 'reconstruct.json')
    reconstruct_yml = os.path.join(predict_store_dir, 'reconstruct', 'reconstruct.yml')
    logger = init_logger(file_level_modes=[[os.path.join(predict_store_dir, 'reconstruct', 'reconstruct.log'), 'info', 'w']])
    
    write_log(content='Load predict content and data from {}.'.format(data_with_predicted_json), level='info', logger=logger)
    write_log(content='Write content to {}.'.format(os.path.join(predict_store_dir, 'reconstruct')), level='info', logger=logger)
    safe_store(file=reconstruct_yml, data=hparams, mode='cover')
    
    reconstruct_manager(data_jsons=data_with_predicted_json, predict_jsons=data_with_predicted_json, reconstruct_json=reconstruct_json, 
                        processing_num=args.worker_num, **hparams['reconstruct'])
    
    # update data dic
    data_with_reconstructed_json = os.path.join(predict_store_dir, 'reconstruct', 'data_with_reconstructed.json')
    data_with_predicted_dic = safe_load(file=data_with_predicted_json)
    reconstruct_dic = safe_load(reconstruct_json)  
    data_with_reconstructed_dic = {'keys': data_with_predicted_dic['keys'],
                                   'duration': data_with_predicted_dic['duration'],
                                   'key2path': {}}
    for key, value in data_with_predicted_dic['key2path'].items():
        data_with_reconstructed_dic['key2path'][key] = {**value, **reconstruct_dic[key]}
    safe_store(file=data_with_reconstructed_json, data=data_with_reconstructed_dic, mode='cover')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('run_cpu')
    parser.add_argument('-r', '--exp_dir', type=str, default='./',
                        help='directory of experiment')
    parser.add_argument('-y', '--yaml_dir', type=str, default='./exp_yml', help='directory of config yaml')
    parser.add_argument('-c', '--config', type=str, required=True, help='config id')
    parser.add_argument('-m', '--mode', type=str, nargs='+', default=['reconstruct'], help='select run mode')

    # predict setting
    parser.add_argument('-pd', '--predict_data', type=str, default='test', help='predict data item')
    parser.add_argument('-um', '--used_model', type=int, default=-1, help='used model during predicting, -1 means best')

    # reconstruct & evaluate setting
    parser.add_argument('-wn', '--worker_num', type=int, default=1, help='number of worker used to reconstruct')
    input_args = parser.parse_args()

    # find config file based on config id
    config_file = search(root_dir=input_args.yaml_dir, regular_expression=input_args.config)
    input_hparams = safe_load(file=config_file[0])

    for stage_id, stage in enumerate(input_args.mode):
        if stage == 'reconstruct':
            reconstruct_interface(hparams=input_hparams, args=input_args)
        else:
            raise NotImplementedError('unknown stage')
