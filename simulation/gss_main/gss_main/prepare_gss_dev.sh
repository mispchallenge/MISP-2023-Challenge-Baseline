#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Dalision)
# Apache 2.0
. ./cmd.sh
. ./path.sh
set -eou pipefail
cmd=run.pl
set -e
stage=0 #start stage  e.g. stage=0
. ./utils/parse_options.sh
stop_stage=8 #stop stage  e.g.stage=1


nj=4 # Number of shards
need_channel=true
data_type=dev # dataset type

# channel_dir=  # multi-channel audio data dir      e.g./disk/dataset/audio/dev/far 
# gss_dir= # multi-channel audio data dir       e.g./disk/dataset/audio/dev/far 
# video_dir=None # video dir (ignore) 
# transcription_dir= # transcription dir      e.g./disk/dataset/transcription_revise/dev
# store_dir=  # stage 0 output :kaldi format data root        e.g./disk/gss_main/data/dev_far  
# Manifest_root= # file after gss     e.g./disk/gss_main/data_after_gss

channel_dir=/mnt/201_disk6/mkhe/dataset/misp2022/Released/audio/dev_new/far/  # multi-channel audio data dir      e.g./disk/dataset/audio/dev/far 
gss_dir=/mnt/201_disk6/mkhe/dataset/misp2022/Released/audio/dev_new/far/  # multi-channel audio data dir == channel_dir      e.g./disk/dataset/audio/dev/far 
video_dir=None # video dir (ignore) 
transcription_dir=/mnt/201_disk6/mkhe/dataset/misp2022/Released/transcription_revise/dev_new/ # transcription dir      e.g./disk/dataset/transcription_revise/dev, you can generate from transcription by: mkdir -p transcription_revise for l in `ls transcription/ | grep TextGrid`;do cat transcription/$l | uniq > transcription_revise/$l done
store_dir=/disk3/chime/simulation/gss_main/gss_main/data/dev_far  # stage 0 output :kaldi format data root        e.g./disk/gss_main/data/dev_far  
Manifest_root=/disk3/chime/simulation/gss_main/ # file after gss     e.g./disk/gss_main/data_after_gss


# transform misp data to kaldi format
# you can change --without_mp4,enhancement_wav,video_path to combine different audio and video filed as you like

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    _opt=""
    echo "prepare wav.scp segments text_sentence utt2spk"
    echo "$need_channel"
    [[ -n $need_channel ]] && _opts+="--channel_dir $channel_dir --without_wav False"
    python prepare_gss_data.py  $gss_dir $video_dir $transcription_dir $data_type $store_dir $_opts --without_wav True
    for file in wav.scp channels.scp mp4.scp segments utt2spk text_sentence;do
        if [ -f $store_dir/temp/$file ];then
            if [ $file == "text_sentence" ];then 
                cat $store_dir/temp/$file | sort -k 1 | uniq > $store_dir/text 
            else
                cat $store_dir/temp/$file | sort -k 1 | uniq > $store_dir/$file
            fi
        fi
    done
    
    [[ -e $store_dir/temp/channels.scp ]] && cp $store_dir/channels.scp $store_dir/wav.scp
    rm -r $store_dir/temp
    echo "prepare done"

    utils/utt2spk_to_spk2utt.pl $store_dir/utt2spk | sort -k 1 | uniq > $store_dir/spk2utt
    touch data/nlsyms.txt
    utils/fix_data_dir.sh $store_dir
    echo "prepare_gss_data.sh succeeded"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "prepare video"
    middle_field_video_data=/disk3/chime/export/corpus/misp2022/Released/video/dev_new/middle/
    raw_segments=$store_dir/segments
    roi_json_dir=/disk3/chime/dev_middle_detection_result/
    roi_store_dir=/disk3/chime/simulation/gss_main/dev_video
    python prepare_files.py --data_root $middle_field_video_data --file_root $store_dir --segments $raw_segments
    python video_roi.py -ji 0 -nj 4 --data_root $middle_field_video_data --file_root $store_dir --roi_json_dir $roi_json_dir --roi_store_dir $roi_store_dir
    echo "prepare video succeeded"
fi


# prepare data for gss training

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

    for dset in dev;do    
        echo "Stage 1: create $dset mainfest"   
        Manifest_dir=$Manifest_root/${dset}_wave/far/wpe/gss_new
        
        [[ ! -e $Manifest_dir ]] && mkdir -p $Manifest_dir
        python prepare_misp.py  \
        --data_dir $store_dir \
        --sampling_rate 16000 \
        --manifest_dir $Manifest_dir \
        --channels 6 

        echo "Stage 2: generate cuts"
        lhotse cut simple  \
        --force-eager \
        -r $Manifest_dir/recordings.jsonl.gz \
        -s $Manifest_dir/supervisions.jsonl.gz \
        $Manifest_dir/cuts.jsonl.gz

        echo "Stage 3: trim cuts"
        lhotse cut trim-to-supervisions --discard-overlapping --keep-all-channels \
        $Manifest_dir/cuts.jsonl.gz $Manifest_dir/cuts_per_segment.jsonl.gz
        echo "stage 3.5: clean cuts"
        python prepare_misp.py --mode filter_cut --cutpath $Manifest_dir/cuts_per_segment.jsonl.gz    
        echo "Stage 4: Split segments into $nj parts"
        gss utils split $nj $Manifest_dir/cuts_per_segment.jsonl.gz $Manifest_dir/split$nj

    done
fi

# Running GPU GSS
# you can change nj= CUDA_VISIBLE_DEVICES=  $cmd JOB=   according to your device
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    
    for dset in dev;do 
        echo "Stage 2: Runing GPU GSS"
        Manifest_dir=$Manifest_root/${dset}_wave/far/wpe/gss_new
        CUDA_VISIBLE_DEVICES=0 $cmd JOB=1:2 $Manifest_dir/log/enhance.JOB.log \
        gss enhance cuts \
          $Manifest_dir/cuts.jsonl.gz $Manifest_dir/split$nj/cuts_per_segment.JOB.jsonl.gz \
          $Manifest_dir/enhanced \
          --min-segment-length 0.1 \
          --max-segment-length 20.0 \
          --context-duration 15.0 \
          --use-garbage-class \
          --bss-iterations 20 \
          --max-batch-duration 10.0 \
          --num-buckets 3 \
          --num-workers 6 \
          --duration-tolerance 3.0 \
          --max-batch-cuts 1 \
          --force-overwrite \
          --enhanced-manifest $Manifest_dir/split$nj/cuts_enhanced.JOB.jsonl.gz &
    done

    for dset in dev;do 
        echo "Stage 2: Runing GPU GSS"
        Manifest_dir=$Manifest_root/${dset}_wave/far/wpe/gss_new
        CUDA_VISIBLE_DEVICES=1 $cmd JOB=3:4 $Manifest_dir/log/enhance.JOB.log \
        gss enhance cuts \
          $Manifest_dir/cuts.jsonl.gz $Manifest_dir/split$nj/cuts_per_segment.JOB.jsonl.gz \
          $Manifest_dir/enhanced \
          --min-segment-length 0.1 \
          --max-segment-length 20.0 \
          --context-duration 15.0 \
          --use-garbage-class \
          --bss-iterations 20 \
          --max-batch-duration 10.0 \
          --num-buckets 3 \
          --num-workers 6 \
          --duration-tolerance 3.0 \
          --max-batch-cuts 1 \
          --force-overwrite \
          --enhanced-manifest $Manifest_dir/split$nj/cuts_enhanced.JOB.jsonl.gz &
    done
fi
wait

# Rounding audio timestamp
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    for dset in dev;do     
        Manifest_dir=$Manifest_root/${dset}_wave/far/wpe/gss_new/enhanced
        python re_name.py --base_folder $Manifest_dir
    done
fi

# export gss files as kaldi format

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then

    for dset in dev;do   
        echo "$dset: cuts.js.gz to dumpdir"
        Manifest_dir=$Manifest_root/${dset}_wave/far/wpe/gss_new # 
        manifest_file=gssgpu_${dset}_far
        lhotse combine $Manifest_dir/split$nj/cuts_enhanced.*.jsonl.gz $Manifest_dir/cuts_enhanced.jsonl.gz
        python gss2lhotse.py -i $Manifest_dir -o $Manifest_dir/${manifest_file}
        lhotse kaldi export $Manifest_dir/${manifest_file}_recordings.jsonl.gz $Manifest_dir/${manifest_file}_supervisions.jsonl.gz dump/raw/${manifest_file}
        ./utils/utt2spk_to_spk2utt.pl dump/raw/${manifest_file}/utt2spk > dump/raw/${manifest_file}/spk2utt
        ./utils/fix_data_dir.sh dump/raw/${manifest_file}
    done 
fi

# prepare speech shape and text_shape.char

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "stage 6"
    for dset in dev;do 
        echo "$dset: speech shape and text_shape.char"
        Manifest_dir=$Manifest_root/${dset}_wave/far/wpe/gss_new
        tgtdir=gssgpu_${dset}_far
        [[ ! -e dump/raw/$tgtdir ]] && mkdir -p $tgtdir
        python kaldi_datadir_pre_v1.py  --mode ptdir2scp --suffix ".wav" \
        --filename wav.scp --pt_dir $Manifest_dir --tgtdir dump/raw/$tgtdir
        #sort
        cat dump/raw/$tgtdir/wav.scp | sort -k1 > dump/raw/$tgtdir/wav.scp.tmp && mv dump/raw/$tgtdir/wav.scp.tmp dump/raw/$tgtdir/wav.scp
        #other file
        bash creat_shapefile.sh --nj 8 --input "dump/raw/$tgtdir/text" --output "dump/raw/$tgtdir/text_shape.char"
        bash creat_shapefile.sh --nj 8 --input "dump/raw/$tgtdir/wav.scp" --output "dump/raw/$tgtdir/speech_shape" --dimnum 1
        utils/utt2spk_to_spk2utt.pl dump/raw/$tgtdir/utt2spk > dump/raw/$tgtdir/spk2utt
    done
fi

# 
#align_id for wav.scp
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "stage 7"
    for dset in dev;do 
        tgtdir=gssgpu_${dset}_far
        outdir=dump/raw/$tgtdir
        for file_name in speech_shape text text_shape.char utt2spk wav.scp;do
            inputdir=dump/raw/$tgtdir
            outdir=dump/raw/${dset}
            mkdir -p $outdir
            python ./align_id_simple.py --input_path $inputdir --output_path $outdir --filename $file_name
        done   
    done
fi


if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "stage 8" 
    for dset in dev;do 
        tgtdir=gssgpu_${dset}_far
        for file_name in spk2utt;do
            inputdir=dump/raw/$tgtdir
            outdir=dump/raw/${dset}
            python ./align_spk2utt.py --input_path $inputdir --output_path $outdir --filename $file_name
        done   
    done
fi
