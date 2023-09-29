#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=4 # adjust based on number of your GPUs
stage=1
stop_stage=100

manifests_dir=
dset_name=
dset_part=
exp_dir=
cmd=run.pl #if you use gridengine: "queue-freegpu.pl --gpu 1 --mem 8G --config conf/gpu.conf"
max_batch_duration=180 # adjust based on your GPU VRAM, here 40GB
max_segment_length=200

gss_iterations=20

. ./path.sh
. parse_options.sh

mkdir -p ${exp_dir}


recordings=${manifests_dir}/recordings.jsonl.gz
supervisions=${manifests_dir}/supervisions.jsonl.gz

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare cut set"
  lhotse cut simple --force-eager \
      -r $recordings \
      -s $supervisions \
      ${exp_dir}/cuts.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Trim cuts to supervisions (1 cut per supervision segment)"
  lhotse cut trim-to-supervisions --discard-overlapping \
       ${exp_dir}/cuts.jsonl.gz  \
       ${exp_dir}/cuts_per_segment.jsonl.gz
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Split segments into $nj parts"
  for part in dev eval; do
    gss utils split $nj  ${exp_dir}/cuts_per_segment.jsonl.gz \
     ${exp_dir}/split$nj
  done
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Enhance segments using GSS"
  mkdir -p ${exp_dir}/log/
  for ((i=1; i<=$nj;i++));do
    (
      export CUDA_VISIBLE_DEVICES=`expr $i - 1` 
      gss enhance cuts \
        ${exp_dir}/cuts.jsonl.gz  ${exp_dir}/split$nj/cuts_per_segment.$i.jsonl.gz \
        ${exp_dir}/enhanced \
        --bss-iterations $gss_iterations \
        --context-duration 15.0 \
        --use-garbage-class \
        --min-segment-length 0.0 \
        --max-segment-length $max_segment_length \
        --max-batch-duration $max_batch_duration \
        --max-batch-cuts 1 \
        --num-buckets 4 \
        --num-workers 4 \
        --force-overwrite \
        --max-batch-cuts 1 \
        --force-overwrite \
        --enhanced-manifest $exp_dir/split$nj/cuts_enhanced.$i.jsonl.gz \
        --duration-tolerance 3.0  > $exp_dir/log/enhance.$i.log 2>&1

    )&
  done
fi
wait
