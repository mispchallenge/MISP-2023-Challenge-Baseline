#!/bin/bash

ROOTDIR=/disk4/hblan/MISP-2023-Challenge-Baseline-main/simulation
audio_path=/mnt/201_disk6/mkhe/dataset/misp2021/audio_v2/train_near_audio/wav/ # Raw training set near-field speech path
clean_config=$ROOTDIR/data/train_clean_config/
noise_config=$ROOTDIR/data/train_noise_config
clean_segments=$clean_config/segments # Segments file path of clean_audio
clean_data=$ROOTDIR/data/clean_data # clean audio store path 
clean_information=$ROOTDIR/data/clean_information.json # clean_information.json store path
noise_segments=$noise_config/segments # Segments file path of noise_audio
noise_data=$ROOTDIR/data/noise_data # noise audio store path
noise_file=$ROOTDIR/data/noise_file # # noise files store path
video_path=/mnt/201_disk6/mkhe/dataset/misp2021/video/train/middle # Raw training set middle_field video path 
roi_json_dir=/mnt/201_disk6/mkhe/dataset/misp2021/audio_v2/train_middle_detection_result # Raw training set middle_field ROI information path 
roi_store_dir=$ROOTDIR/data/video # clean data video store path
stage=1


if [ $stage -le 1 ]; then
    # clean speech
    echo "stage 1"
    mkdir -p $clean_data
    python $ROOTDIR/tools/segment.py --input_file $clean_segments --input_folder $audio_path --output_folder $clean_data
fi

if [ $stage -le 2 ]; then
    # clean_information.json
    echo "stage 2"
    python $ROOTDIR/tools/create_json.py --clean_folder $clean_data --output $clean_information
fi

if [ $stage -le 3 ]; then
    # noise speech
    echo "stage 3"
    mkdir -p $noise_data
    python $ROOTDIR/tools/segment.py --input_file $noise_segments --input_folder $audio_path --output_folder $noise_data
fi

if [ $stage -le 4 ]; then
    # noise files for simulation
    echo "stage 4"
    mkdir -p $noise_file
    python $ROOTDIR/tools/prepare_files.py --noise_folder $noise_data --output $noise_file
fi

if [ $stage -le 5 ]; then
    # prepare clean data videos
    echo "stage 5"
    python $ROOTDIR/tools/segment_video_roi.py --data_root $clean_config --video_path $video_path --roi_json_dir $roi_json_dir --roi_store_dir $roi_store_dir
    echo "done!"
fi
