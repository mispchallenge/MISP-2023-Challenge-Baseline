#!/usr/bin/env bash
# source ./bashrc #load customized enviroment
set -eou pipefail

train_set=None #ignore
valid_set=None #ignore
test_sets=       #customized kalid-like eval datadir dump/raw/dev_far_farlip
asr_exp=exp/a/a_conformer_farmidnear #model_path and the model config file is $asr_exp/confing.yaml
inference_config=conf/decode_asr.yaml #decode config
use_lm=false #LM is forbidden
use_word_lm=false

# stage 1 decoding    stage 2 scoring 


bash asr.sh                                  \
    --stage 1                               \
    --stop_stage 2                         \
    --asr_exp ${asr_exp}                 \
    --lang zh                              \
    --nj 8                                 \
    --speed_perturb_factors "0.9 1.0 1.1"  \
    --inference_asr_model  valid.acc.ave_5best.pth \
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 1                               \
    --token_type char                      \
    --feats_type raw                       \
    --use_lm ${use_lm}                     \
    --inference_config "${inference_config}" \
    --use_word_lm ${use_word_lm}           \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             




