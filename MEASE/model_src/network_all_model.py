#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
from network.network_model import BaseModelWorker
from network_ease import (MultimodalEmbeddingAwareSpeechEnhancement, MultimodalEmbeddingAwareSpeechEnhancement_ASR_jointOptimization)

class EASEWorker(BaseModelWorker):
    def __init__(self, log_type, logger=None):
        super(EASEWorker, self).__init__(log_type, logger)

    def _build_map(self):
        self.name2network = {
            'mease': MultimodalEmbeddingAwareSpeechEnhancement,
            'mease_asr_jo': MultimodalEmbeddingAwareSpeechEnhancement_ASR_jointOptimization
        }
        return None
