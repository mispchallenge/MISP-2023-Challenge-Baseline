#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import logging

from .loss_mse import misp_BatchCalMSE_calcIRMLabel

logging.getLogger('matplotlib.font_manager').disabled = True


class LossFunctionWorker(object):
    def __init__(self, log_type=None, logger=None):
        super(LossFunctionWorker, self).__init__()
        self.log_level = log_type
        self.logger = logger
        self._build_map()

    def __call__(self, loss_type, loss_setting, **other_params):
        loss_function = self.init_loss(loss_type=loss_type, **loss_setting)
        return loss_function

    def _build_map(self):
        self.name2loss = {
            'misp_mse_calcIRMLabel': misp_BatchCalMSE_calcIRMLabel
        }

    def init_loss(self, loss_type, **loss_setting):
        write_log(content='Using loss: {}'.format(loss_type), level=self.log_level, logger=self.logger)
        loss_function = self.name2loss[loss_type](**loss_setting)
        return loss_function


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


