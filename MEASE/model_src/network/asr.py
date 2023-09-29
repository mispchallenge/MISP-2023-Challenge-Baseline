from espnet2.tasks.asr import ASRTask
from pathlib import Path
import torch.nn as nn
import torch

class ICME_ASR(nn.Module):
    def __init__(self,expdir="./asr_file/",device="cuda"):
        super().__init__()
        config = Path(expdir) / "config.yaml"
        model = Path(expdir) / "valid.acc.ave_5best.pth"
        asr_model, asr_train_args = ASRTask.build_model_from_file(str(config), model, device)
        self.model = asr_model
    def forward(self,wav,wav_length,text,text_lengths):
        #loss_ctc,loss_att = self.model(wav,wav_length,text,text_lengths,outloss=True)
        dict = self.model(wav,wav_length,text,text_lengths,outloss=True)
        loss_ctc=dict[1]['loss_ctc']
        loss_att=dict[1]['loss_att']
        return loss_ctc,loss_att
    
