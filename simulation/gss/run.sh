#!/usr/bin/env bash
# DATAPREP CONFIG
manifests_root=./data/lhotse # dir where to save lhotse manifests
cmd_dprep=run.pl
gss_dump_root=./exp/gss
ngpu=4  # set equal to the number of GPUs you have, used for GSS and ASR training
# GSS CONFIG
gss_max_batch_dur=180 # set accordingly to your GPU VRAM, A100 40GB you can use 360
cmd_gss=run.pl # change to suit your needs e.g. slurm !

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

mkdir -p data

src_train_rttm='/disk3/chime/simulation/simulate_data/data/train/train.rttm' # rttm file (generate from run_simulation.sh stage1)

mkdir -p data/rttm
rm -rf data/rttm/train.rttm
suffixes=("0db" "10db" "-10db" "15db" "20db" "5db" "-5db")
for suffix in "${suffixes[@]}";do
  awk -v s="_$suffix" '{$2 = $2 s}1' $src_train_rttm >> data/rttm/train.rttm
done

misp_se_root_path=/disk3/chime/simulation/data/simulation_data # simulation_data (generate from run_simulation.sh stage4)
input_rttm=data/rttm/train.rttm  # new rttm file 
outdir=data/lhotse/train # lhotse output path



python get_lhotse_train.py --misp_se_root_path $misp_se_root_path --input_rttm $input_rttm --outdir $outdir

local/run_gss.sh --manifests-dir ${manifests_root}/train \
        --exp-dir $gss_dump_root/train \
        --cmd "$cmd_gss" \
        --nj $ngpu \
        --stage 1 \
        --stop_stage 4
wait

