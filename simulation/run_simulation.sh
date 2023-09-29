#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
. ./path.sh

function print_usage_and_exit {
    CMD=`basename $0`    
    echo ""
    echo "    ''"
    echo "    < $CMD >"
    echo ""
    echo "    Usage: $CMD [--split N] [--roomcfg FILE] [--dyncfg FILE] [--vad] [--save_image] [--save_channels_separately] [--help] dest-dir set"
    echo ""
    echo "    Description: Preprocess the original LibriSpeech data."
    echo ""
    echo "    Args: "
    echo "        dest-dir: Destination directory, relative to \$EXPROOT/data."
    echo "        set: {train|dev|eval}, subset to process."
    echo ""
    echo "    Options: "
    echo "        --split N                  : Split the data set into N subsets for parallel processing. N defaults to 32."    
    echo "        --roomcfg FILE             : Room acoustics configuration file. FILE defaults to <repo-root>/configs/common/meeting_reverb.json."
    echo "        --dyncfg FILE              : Room acoustics configuration file. FILE defaults to <repo-root>/configs/common/meeting_dynamics.json."
    echo "        --vad                      : Use VAD-segmented signals. Not recommended."
    echo "        --save_image               : Save source images instead of anechoic signals and RIRs."
    echo "        --save_channels_separately : Save each output channel separately."
    echo "        --help                     : Show this message."
    echo "    ''"
    echo ""
    exit 1
}

stage=1
# Import $EXPROOT. 
ROOTDIR=`dirname $0`/
ROOTDIR=`realpath $ROOTDIR`
source $ROOTDIR/path.sh
############################# Output path ####################################
tgtroot=$EXPROOT/data/train


if [ $stage -le 1 ]; then
    echo "stage 1"

    nj=24  # default parallel jobs

    # Subset of Kaldi utils
    KALDI_UTILS=$ROOTDIR/tools/kaldi_utils

    # Environment
    export PATH=${KALDI_UTILS}:${PATH}
    . $ROOTDIR/configs/cmd.sh

    # Scripts
    splitjson=$ROOTDIR/tools/splitjson.py
    mergejson=$ROOTDIR/tools/mergejsons.py
    gen_filelist=$ROOTDIR/tools/gen_filelist.py
    list2json=$ROOTDIR/tools/list2json_librispeech.py
    mixspec=$ROOTDIR/tools/gen_mixspec_mtg.py
    mixer=$ROOTDIR/tools/mixaudio_mtg.py

    # Directories
    set=train
    if [ -v vad ]; then
        srcdir=$EXPROOT/data/${set}/wav_newseg  # VAD-segmented signals
    else
        srcdir=$EXPROOT/data/${set}/wav  # original signals
    fi

    ############################# Your roomcfg ####################################
    if [ ! -v roomcfg ]; then
        roomcfg=$ROOTDIR/configs/common/meeting_reverb_misp.json
    fi
    ############################## Your dyncfg ####################################
    if [ ! -v dyncfg ]; then
        dyncfg=$ROOTDIR/configs/common/meeting_dynamics_misp_no_overlap.json
    fi

    # Split datalist for parallel processing
    splitdir=${tgtroot}/split${nj}
    mkdir -p ${splitdir}/log

    ################### Your clean_information.json ###############################
    datajson=$ROOTDIR/data/clean_information.json

    # Generate mixture specs. 
    tgtdir=$tgtroot/wav
    mkdir -p $tgtdir

    specjson=$tgtroot/mixspec.json
    python $mixspec --inputfile $datajson --outputfile $specjson --targetdir $tgtdir --random_seed 50 --config $dyncfg

    # Split $tgtroot/mixspec.json into several smaller json files: $splitdir/mixspec.JOB.json
    python $splitjson --inputfile $specjson --number_splits $nj --outputdir $splitdir

    # Generate mixed audio files. 
    mixlog=$tgtroot/mixlog.json
    if [ ! -v save_channels_separately ]; then
        opts='--save_each_channel_in_onefile'
    else
        opts=''
    fi
    if [ -v save_image ]; then
    opts="$opts --save_image"
    fi
    ${gen_cmd} JOB=1:${nj} ${splitdir}/log/mixlog.JOB.log \
        python $mixer $opts --iolist ${splitdir}/mixspec.JOB.json --cancel_dcoffset --random_seed JOB --sample_rate 16000 --log ${splitdir}/mixlog.JOB.json --mixers_configfile $roomcfg
    python $mergejson $(for j in $(seq ${nj}); do echo ${splitdir}/mixlog.${j}.json; done) > $mixlog

    # prepare clean files
    input_folder=$tgtdir
    output=$ROOTDIR/data/clean_file
    mkdir -p $output
    python $ROOTDIR/tools/prepare_files2.py --input_folder $input_folder --output $output

    # prepare rttm
    python $ROOTDIR/tools/mtg2rttm.py --mixspec $tgtroot
    cat $tgtroot/rttm/*.rttm > $tgtroot/train.rttm
fi

####################### generate label ########################
if [ $stage -le 2 ]; then
    echo "stage 2"
    python   $ROOTDIR/tools/name.py --mixlog $tgtroot --clean_data $ROOTDIR/data/clean_data --video_data $ROOTDIR/data/video --label_data $ROOTDIR/data/label/
fi

####################### Add noise ########################
if [ $stage -le 3 ]; then
    echo "stage 3"
    for snr in $(seq -10 5 20); do
    python $ROOTDIR/steps/data/augment_data_multi_channel_dir.py --utt-suffix "noise" --fg-interval 0 --fg-snrs "$snr" --random-seed 50 --fg-noise-dir "$ROOTDIR/data/noise_file" $ROOTDIR/data/clean_file $ROOTDIR/data/clean_file/${snr}db
    done
fi
##################### generate wav #########################
if [ $stage -le 4 ]; then
    echo "stage 4"
    for snr in $(seq -10 5 20); do
        input=$ROOTDIR/data/clean_file/${snr}db/wav.scp
        output=$ROOTDIR/data/clean_file/${snr}db/output.scp
        mkdir -p $ROOTDIR/data/simulation_data/${snr}db
        awk '{temp=$1; $1=""; print $0, temp}' $input > $output
        #awk "{last_column=$NF; $NF=""; print $0, "$ROOTDIR/data/sim_gss_dev_noise/${snr}db/" last_column}" $output > temp.txt && mv temp.txt $output
        awk -v path="$ROOTDIR" -v snr="$snr" '{last_column=$NF; $NF=""; print $0, path "/data/simulation_data/" snr "db/" last_column}' "$output" > temp.txt && mv temp.txt "$output"
        sed -i 's/$/.wav/' $output
        sed -i "s/- |//g; s/-noise.wav$/_${snr}db.wav/" $output
        while IFS= read -r line; do eval "$line"; done < $output
    done
fi
