#!/usr/bin/env bash
source ./bashrc
set -eou pipefail

# 1. install espnet and kaldi
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    #the following is a version of reference
    #1.1 install base envs
    espnetpath=
    conda create -y -n new38cu11 python=3.8
    conda activate new38cu11

    pip install torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl 
    pip install torchvision-0.11.3+cu113-cp38-cp38-linux_x86_64.whl 
    pip install torchaudio-0.10.2+cu113-cp38-cp38-linux_x86_64.whl 
    pip install s3prl scipy soundfile tqdm lmdb matplotlib tensorboard

    #1.2 install espnet
    #make sure pythonpath and cuda match with installed torch
    export PATH="python_path:$PATH" 
    export CUDNNV2_PATH=cuda/cudnn_v2
    export CUDAROOT=cuda-11.3.0/
    export CUDA_HOME=$CUDAROOT
    export CUDA_PATH=$CUDAROOT
    export MODULEPATH=/opt/tool/modulefiles
    module load cuda/11.3.0-cudnn-v8.2.1
    module load cmake/3.20.1

    echo `nvcc --version`
    echo `which python`

    cd $espnet_path/tools
    python3 -m pip install -e "..[recipe]"
    python check_install.py

    #1.3 link installed tools to espent tools
    ln -s kaldi ./kaldi
    ln -s kaldi/tools/openfst ./openfst
    ln -s kaldi/tools/sctk ./sctk
    ln -s kaldi/tools/sph2pipe_v2.5 ./sph2pipe_v2.5
fi

# 2.create link in program directory misp/avsr
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    
    cd asr
    for file in scripts steps utils pyscripts path.sh db.sh;do 
        unlink ./$file
        ln -s $espnetpath/egs2/TEMPLATE/asr1/$file ./$file
    done 
fi

# 3.create customized enviroment loader(./bashrc & ./path) to prepare the running environment, they will be execute in decode_avsr.sh
# e.g  CUDA_PATH | PATH | CPLUS_INCLUDE_PATH | LD_LIBRARY_PATH | PYTHON_PATH 
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    cd asr
    touch ./bashrc
fi  