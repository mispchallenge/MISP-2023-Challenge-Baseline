# MISP 2023 Challenge: Audio-Visual Target Speaker Extraction (Baseline)
**2023.9.29: Some explanations have been provided for the espnet environment of the back-end ASR system**

**2023.9.22: We have fixed several bugs. If you have downloaded the code before, please download it again to synchronize it.**

MISP 2023 challenge focuses on the  Audio-Visual Target Speaker Extraction (AVTSE) task. For a description of the task setting, dataset and baseline, please refer to the challenge overview [paper](https://arxiv.org/abs/2309.08348).

We use the ASR backend to decode the extracted speech and calculate CER as an evaluation index. The results for the baseline system are as follows:
## Results

| Systems    |  S  |  D  |  I  |  CER  |
| -----------|------|--------|----------|-------|
|   Beamforming      | 31.0 |  7.2 |   4.8  | 43.0 |
|   GSS      | 20.0 |   4.2 |    2.2  | 26.4 |
|   GSS+AEASE      | 21.5 |   4.9 |    2.2  | 28.6 |
|   GSS+MEASE      | 20.4 |   4.4 |    2.2  | 27.0 |
|   **GSS+MEASE+Finetune**      | 19.8 |   4.7 |    1.8  | **26.3** |

The results of GSS+MEASE+Finetune are used as the results of the baseline system. Furthermore, we calculated the DNSMOS P.835 [15] as a reference to explore the relationship between speech auditory quality and back-end tasks. Please refer to the overview paper.

The baseline code mainly includes three parts, namely data preparation and simulation, MEASE model training and decoding, and back-end ASR decoding. Among them, the MEASE model is an audio-visual speech enhancement model. You can refer to this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608021002355).

Before starting, you need to configure Kaldi, Espnet and GPU-GSS environments. You **must** configure the GPU-GSS environment according to [this link](https://github.com/rywang99/gss), otherwise errors will occur.

## Data preparation and Simulation

In this step, we perform data preparation, data simulation, and GSS.

- **Data preparation**

First, we cut relatively clean near-field audio based on the DNSMOS score, and then prepare related files. The list of clean audio is in /simulation/data/train_clean_config/segments. Please note that these data are not unique, you can also perform data cleaning according to your own standards.
```
cd simulation
bash prepare_data.sh   # (Please change your file path in the script.)
```

- **Data simulation**

Second, we use the clean near-field audio and noise data to simulate 6-channels far-field audio. The simulation script is mainly based on this [link](https://github.com/jsalt2020-asrdiar/jsalt2020_simulate). Please note that simulation methods are not unique, you can use other simulation methods in MISP 2023 challenge.
```
bash run_simulation.sh   # (Please change your file path in the script.)
```

- **GSS**

Then, we perform GSS on simulation data, real development set data, and real training set data.

```
cd simulation/gss
bash run.sh   # (Please change your file path in the script.)
cd simulation/gss_main/gss_main
bash prepare_gss_dev.sh
bash prepare_gss_train.sh
```

## MEASE model training and decoding

- **Pre-trained model**

The mee.tar in the directory /model_src/pretrainModels is a pre-trained multimodal embedding extractor. It is frozen during the training of MEASE. The mease.tar is a pre-trained MEASE model that can be used to initialize MEASE for joint optimization of MEASE and ASR backend. The mease_asr_jointOptimization.tar is the pre-trained MEASE_Finetune model. The valid.acc.ave_5best.pth is the pre-trained ASR model.

You can download these models in this link of Google drive: https://drive.google.com/drive/folders/16TuN4_ClSCSo4YRUOFwGOt-lmA16v5bg.

After that, you need to put mee.tar, mease.tar, mease_asr_jointOptimization.tar in MEASE/model_src/pretrainModels/, 

and put valid.acc.ave_5best.pth in MEASE /model_src/asr_file/ and asr/exp/a/a_conformer_farmidnear/.

- **MEASE data prepare**

First, prepare the json files for MEASE.
  ```
  cd MEASE/data_prepare
  bash MEASE/data_prepare/prepare_data.sh  # (Please change your file path in the script.)
  ```
For the training set, generate the cmvn files required for training .

  ```
  python feature_cmvn.py --annotate "../data_prepare/misp_sim_trainset.json"  #generate cmvn files for training MEASE
  python feature_cmvn.py --annotate "../data_prepare/misp_real_trainset.json"  #generate cmvn files for joint optimization of MEASE and ASR backend
  ```
  Please change some file paths in the feature_cmvn.py to your own path.

The directory /model_src/experiment_config contains the following four files:

  ```python
  0_1_MISP_Mease.yml  ##Configuration file for training MEASE
  0_1_MISP_Mease_test.yml  ##Configuration file for decoding real test set with the MEASE.
  1_1_MISP_Mease_ASR_jointOptimization.yml  ##Configuration file for joint optimization of MEASE and ASR backend
  1_2_MISP_Mease_ASR_jointOptimization_test.yml  ##Configuration file for decoding real test set with MEASE after joint training
  ```

Please change some file paths in the configuration files to your own path.

- **MEASE model training**

```python
cd MEASE/model_src
## Train the MEASE model
source run.sh 0.1
## Joint optimization of MEASE and ASR backend
source run.sh 0.3
```

Please change some file paths in the script to your own path.

- **Decoding the real test set with the official pre-trained models**

```python
cd model_src
## Decoding real test set with MEASE.
source run.sh 0.0.1
## Decoding real test set with MEASE after joint training.
source run.sh 0.0.2
```

- **Decoding the real test set with the your models**

```python
cd model_src
## Decoding real test set with MEASE.
source run.sh 0.2
## Decoding real test set with MEASE after joint training.
source run.sh 0.4
```

Please change some file paths in the script to your own path.

## Back-end ASR system decoding
- **Env Setup**

  After installing espnet and kaldi, you are supposed to replace the directories named "espnet" and "espnet2" in your customized Espnet package with the corresponding directories from the [repository](https://github.com/mispchallenge/MISP-ICME-AVSR/tree/master). Then enter the workplace directory ./asr and set the standard espnet directories like utils, steps, scripts, and pyscripts and customize .path and .barsh files. Here is a [reference script](asr/reference_env.sh).

- **Prepare Kaldi-format data directory**

  As shown in the example in the [dir](asr/dump/raw/dev_far_farlip), you need to prepare 5 kaldi files, namely `wav.scp, utt2spk, text, text_shape.char, and speech_shape`. **In addition, in order to keep everyone consistent, we offer an stardard [UID list](asr/dump/raw/dev_far_farlip/test.list) for final test.** Please note that the timestamp in the UID is rounded up to the nearest multiple of 4. Therefore, you need to round up the timestamp to match it with the UID. If you use our baseline code, you don't need to think about it since the rounding is already done when doing GSS.

- **Released ASR model**

  You can download the pre-training ASR models from [here](https://drive.google.com/drive/folders/16TuN4_ClSCSo4YRUOFwGOt-lmA16v5bg)(valid.acc.ave_5best.pth) and place it in the `asr/exp/a_conformer_farmidnear` directory with the name `valid.acc.ave_5best.pth`. **Please note that any update to the ASR model is not allowed during test-time (back-end frozen)**. Here are some details about the ASR model. It consists of a 12-layer  conformer as the encoder and a 6-layer Transformer as the decoder. You  can find more information about the model's architecture in the [train_asr.yaml](asr/conf/train_asr.yaml) file. During the ASR training, utterances from far, middle, and near fields are used. GPU-gss is applied to the far and middle utterances. Various  data augmentation techniques are applied to far-field audios, including  adding noise+Room Impulse Response (RIR) convolution (3-fold), speed  perturbation (2-fold with speeds 0.9 and 1.1), and concatenating nearby  segments (2-fold) to create a 10-fold training set.

- **Run Decoding**

```
./decoder_asr.sh 
# options:
        test_sets= #customized kalid-like eval datadir dump/raw/dev_far_farlip
        asr_exp=exp/a/a_conformer_farmidnear #model_path and the model config file is $asr_exp/confing.yaml
        inference_config=conf/decode_asr.yaml #decode config
        use_lm=false #LM is forbidden
        # stage 1 decoding    stage 2 scoring 
        
# If you encounter a "Permission denied" issue, you can resolve it using the chmod a+x command.
```

# Requirments

- Kaldi
- Espnet
- pyrirgen
- GPU-GSS

- Python Package:

  pytorch
  
  argparse
  
  numpy
  
  tqdm
  
  s3prl 

  opencv-python

  resampy

  einops
  
  g2p_en
  
  scipy
  
  prefetch_generator
  
  matplotlib

  soundfile

  opencv-python
  
  matplotlib 

  tensorboard

# Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
@misc{wu2023multimodal,
      title={The Multimodal Information Based Speech Processing (MISP) 2023 Challenge: Audio-Visual Target Speaker Extraction}, 
      author={Shilong Wu and Chenxi Wang and Hang Chen and Yusheng Dai and Chenyue Zhang and Ruoyu Wang and Hongbo Lan and Jun Du and Chin-Hui Lee and Jingdong Chen and Shinji Watanabe and Sabato Marco Siniscalchi and Odette Scharenborg and Zhong-Qiu Wang and Jia Pan and Jianqing Gao},
      year={2023},
      eprint={2309.08348},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}

@article{chen2021correlating,
  title={Correlating subword articulation with lip shapes for embedding aware audio-visual speech enhancement},
  author={Chen, Hang and Du, Jun and Hu, Yu and Dai, Li-Rong and Yin, Bao-Cai and Lee, Chin-Hui},
  journal={Neural Networks},
  volume={143},
  pages={171--182},
  year={2021},
  publisher={Elsevier}
}

@inproceedings{dai2023improving,
  title={Improving Audio-Visual Speech Recognition by Lip-Subword Correlation Based Visual Pre-training and Cross-Modal Fusion Encoder},
  author={Dai, Yusheng and Chen, Hang and Du, Jun and Ding, Xiaofei and Ding, Ning and Jiang, Feijun and Lee, Chin-Hui},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={2627--2632},
  year={2023},
  organization={IEEE}
}
