#!/usr/bin/env bash
# Copyright 2021 USTC (Authors: Dilision)
# Apache 2.0
# ./local/create_shapefile.sh  --nj 16 --input "${_tag_set}/wav.scp" --output "${_tag_set}/speech_shape" --dimnum 1
input=
output=
dimnum= #for video pt
mode=
token_list= #for bpe
token_type=
bpemodel=
nj=4
cmd=run.pl
pdfflag=false
local_dirname=
set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

infile=$(basename $input )
infilename=${infile%%.*}
outfile=$(basename $output )
outfilename=${outfile%%.*}
outdir=${output%/*}
[[ -n $local_dirname ]] || local_dirname=utils
logdir=$outdir/tmpshape
mkdir -p $logdir
rm -rf $logdir/*

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/${infile}.${n}"
done

utils/split_scp.pl ${input} ${split_scps}

_opt=""
[[ -n $dimnum ]] && _opt+="--dimnum $dimnum "
[[ -n $mode ]] && _opt+="--mode $mode "
[[ -n $token_list ]] && _opt+="--token_list $token_list "
[[ -n $token_type ]] && _opt+="--token_type $token_type "
[[ -n $bpemodel ]] && _opt+="--bpemodel $bpemodel "
[[ -n $extra_output ]] && _opt+="--extra_output $extra_output "


${cmd} JOB=1:${nj} ${logdir}/$outfile.JOB.log \
    python $local_dirname/create_shapefile.py --input ${logdir}/${infile}.JOB --output ${logdir}/${outfile}.JOB $_opt --pdfflag $pdfflag

for n in $(seq ${nj}); do
    cat ${logdir}/${outfile}.$n
done > ${output}

rm -rf ${logdir}