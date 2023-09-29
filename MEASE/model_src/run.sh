#!/bin/bash

# Begin configuration section
agi=0,1
# End configuration section

echo "$0 $@"  # Print the command line for logging

. ./parse_options.sh || exit 1

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <step>"
  echo "e.g.: $0 0"
  echo "Options: "
  echo "--agi <available gpu idxes>"
fi

step=$1
gpu_num=`echo ${agi//,/} | wc -L`

if [ $step == 0.0.1 ]; then 
  test_folder=./experiment/0_2_MISP_Mease_test/
  testmodel=./pretrainModels/mease.tar
  if [ ! -f "$testmodel" ]; then  
    echo "The testmodel does not exist."
    exit 0
  fi 
  if [ ! -d "$test_folder" ]; then  
      mkdir "$test_folder" 
  fi 
  cp $testmodel $test_folder
  mv ${test_folder}"mease.tar" ${test_folder}"best.tar"
  CUDA_VISIBLE_DEVICES=$agi\
  python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=14245 \
  run_gpu.py -r ./experiment -y ./experiment_config \
  -c 0_2 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train test -ms mse -si 1 -ci 0 -co min -pd test -um -1

  python run_cpu.py -r ./experiment -y ./experiment_config \
  -c 0_2 -m reconstruct -pd test -um -1 -wn 8

elif [ $step == 0.0.2 ]; then 
  test_folder=./experiment/1_2_MISP_Mease_ASR_jointOptimization_test/
  testmodel=./pretrainModels/mease_asr_jointOptimization.tar
  if [ ! -f "$testmodel" ]; then  
    echo "The testmodel does not exist."
    exit 0
  fi 
  if [ ! -d "$test_folder" ]; then  
      mkdir "$test_folder" 
  fi 
  cp $testmodel $test_folder
  mv ${test_folder}"mease_asr_jointOptimization.tar" ${test_folder}"best.tar"
  CUDA_VISIBLE_DEVICES=$agi\
  python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=14245 \
  run_gpu.py -r ./experiment -y ./experiment_config \
  -c 1_2 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train test -ms mse -si 1 -ci 0 -co min -pd test -um -1

  python run_cpu.py -r ./experiment -y ./experiment_config \
  -c 1_2 -m reconstruct -pd test -um -1 -wn 8
fi

if [ $step == 0.1 ]; then
  CUDA_VISIBLE_DEVICES=$agi\
  python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=14245 \
  run_gpu.py -r ./experiment -y ./experiment_config \
  -c 0_1 -m train -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train test -ms mse -si 1 -ci 0 -co min -pd test -um -1

elif [ $step == 0.2 ]; then  
  test_folder=./experiment/0_2_MISP_Mease_test/
  trained_folder=./experiment/0_1_MISP_Mease
  testmodel=./experiment/0_1_MISP_Mease/best.tar 
  if [ ! -f "$testmodel" ]; then  
    echo "The testmodel does not exist."
    exit 0
  fi 
  if [ ! -d "$test_folder" ]; then  
      mkdir "$test_folder" 
  fi 
  cp $testmodel $test_folder
  
  CUDA_VISIBLE_DEVICES=$agi\
  python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=14245 \
  run_gpu.py -r ./experiment -y ./experiment_config \
  -c 0_2 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train test -ms mse -si 1 -ci 0 -co min -pd test -um -1

  python run_cpu.py -r ./experiment -y ./experiment_config \
  -c 0_2 -m reconstruct -pd test -um -1 -wn 8

elif [ $step == 0.3 ]; then  
  CUDA_VISIBLE_DEVICES=$agi\
  python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=14245 \
  run_gpu.py -r ./experiment -y ./experiment_config \
  -c 1_1 -m train -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train test -ms mse -si 1 -ci 0 -co min -pd test -um -1

elif [ $step == 0.4 ]; then  
  test_folder=./experiment/1_2_MISP_Mease_ASR_jointOptimization_test/
  trained_folder=./experiment/1_1_MISP_Mease_ASR_jointOptimization
  testmodel=./experiment/1_1_MISP_Mease_ASR_jointOptimization/best.tar 
  if [ ! -f "$testmodel" ]; then  
    echo "The testmodel does not exist."
    exit 0
  fi 
  if [ ! -d "$test_folder" ]; then  
      mkdir "$test_folder" 
  fi 
  cp $testmodel $test_folder

  CUDA_VISIBLE_DEVICES=$agi\
  python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=14245 \
  run_gpu.py -r ./experiment -y ./experiment_config \
  -c 1_2 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train test -ms mse -si 1 -ci 0 -co min -pd test -um -1

  python run_cpu.py -r ./experiment -y ./experiment_config \
  -c 1_2 -m reconstruct -pd test -um -1 -wn 8
fi
