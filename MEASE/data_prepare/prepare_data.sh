#!/bin/bash

ROOTDIR=/disk4/hblan/MISP-2023-Challenge-Baseline-main/
output_folder=$ROOTDIR/MEASE/data_prepare
stage=0

if [ $stage -le 0 ]; then
    # Simulation data timestamp alignment
    echo "stage 0"
    label_audio=$ROOTDIR/simulation/data/label/audio
    label_video=$ROOTDIR/simulation/data/label/video
    gss_audio=$ROOTDIR/simulation/gss/exp/gss/train/enhanced/
    python $ROOTDIR/MEASE/data_prepare/re_name.py --label_audio $label_audio --label_video $label_video --gss_audio $gss_audio

    python check_audio.py --gss_data $gss_audio --audio_label $label_audio --output_folder $output_folder  
fi

if [ $stage -le 1 ]; then
    # prepare wav.scp for sim_trainingset real_trainingset real_devset
    echo "stage 1"
    dataset=sim_trainingset
    input_folder=/disk4/hblan/MISP-2023-Challenge-Baseline-main/simulation/gss/exp/gss/train/enhanced/
    python $ROOTDIR/MEASE/data_prepare/2wavscp.py --input_folder $input_folder --dataset $dataset --output_folder $output_folder

    dataset=real_trainingset
    input_folder=/disk4/hblan/MISP-2023-Challenge-Baseline-main/simulation/gss_main/train_wave/far/wpe/gss_new/enhanced/
    python $ROOTDIR/MEASE/data_prepare/2wavscp.py --input_folder $input_folder --dataset $dataset --output_folder $output_folder

    dataset=real_devset
    input_folder=/disk4/hblan/MISP-2023-Challenge-Baseline-main/simulation/gss_main/dev_wave/far/wpe/gss_new/enhanced/
    python $ROOTDIR/MEASE/data_prepare/2wavscp.py --input_folder $input_folder --dataset $dataset --output_folder $output_folder
fi

if [ $stage -le 2 ]; then
    # split testset
    echo "stage 2"
    input_file=$ROOTDIR/MEASE/data_prepare/sim_trainingset_wav.scp
    output_file=$ROOTDIR/MEASE/data_prepare/sim_trainingset_test_wav.scp
    python $ROOTDIR/MEASE/data_prepare/split_train_dev.py --input_file $input_file --output_file $output_file --number 900

    input_file=$ROOTDIR/MEASE/data_prepare/real_trainingset_wav.scp
    output_file=$ROOTDIR/MEASE/data_prepare/real_trainingset_test_wav.scp
    python $ROOTDIR/MEASE/data_prepare/split_train_dev.py --input_file $input_file --output_file $output_file --number 5000

fi

if [ $stage -le 3 ]; then
    # prapare simulation data json
    echo "stage 3"
    data=sim
    trainset_mixture_wavpath=$ROOTDIR/simulation/gss/exp/gss/train/enhanced/
    trainset_clean_wavpath=$ROOTDIR/simulation/data/label/audio
    trainset_videos_path=$ROOTDIR/simulation/data/label/video
    deleted_file=$ROOTDIR/MEASE/data_prepare/check.txt
    trainset_output_json_path=$ROOTDIR/MEASE/data_prepare/
    split_test=$ROOTDIR/MEASE/data_prepare/sim_trainingset_test_wav.scp
    python $ROOTDIR/MEASE/data_prepare/create_json.py --data $data --trainset_mixture_wavpath $trainset_mixture_wavpath --trainset_clean_wavpath $trainset_clean_wavpath --trainset_videos_path $trainset_videos_path --trainset_output_json_path $trainset_output_json_path --deleted_file $deleted_file --split_test=$split_test
fi

if [ $stage -le 4 ]; then
    # prapare real_devset json
    echo "stage 4"
    data=real_dev
    #trainset_mixture_wavpath=/disk3/chime/simulation/gss_main/dev_wave/far/wpe/gss_new/enhanced/
    trainset_mixture_wavpath=$ROOTDIR/simulation/gss_main/dev_wave/far/wpe/gss_new/enhanced/
    trainset_videos_path=$ROOTDIR/simulation/gss_main/dev_video/
    trainset_output_json_path=$ROOTDIR/MEASE/data_prepare/
    python $ROOTDIR/MEASE/data_prepare/create_json.py --data $data --trainset_mixture_wavpath $trainset_mixture_wavpath --trainset_videos_path $trainset_videos_path --trainset_output_json_path $trainset_output_json_path
fi


if [ $stage -le 5 ]; then
    # prapare real_trainingset json
    echo "stage 5"
    data=real_train
    trainset_mixture_wavpath=$ROOTDIR/simulation/gss_main/train_wave/far/wpe/gss_new/
    trainset_videos_path=$ROOTDIR/simulation/gss_main/train_video/
    trainset_output_json_path=$ROOTDIR/MEASE/data_prepare/
    split_test=$ROOTDIR/MEASE/data_prepare/real_trainingset_test_wav.scp
    text=$ROOTDIR/simulation/gss_main/gss_main/data/train_far/text
    python $ROOTDIR/MEASE/data_prepare/create_json.py --data $data --trainset_mixture_wavpath $trainset_mixture_wavpath --trainset_videos_path $trainset_videos_path --trainset_output_json_path $trainset_output_json_path --split_test=$split_test --text $text
fi
