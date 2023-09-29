export LD_LIBRARY_PATH=/disk4/hblan/MISP-2023-Challenge-Baseline-main/simulation/external_tools/rir-generator/python
export PYTHONPATH=/disk4/hblan/MISP-2023-Challenge-Baseline-main/simulation/external_tools/rir-generator/python
export EXPROOT=/disk4/hblan/MISP-2023-Challenge-Baseline-main/simulation/simulate_data

export KALDI_ROOT=/disk1/chime/espnet/tools/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ -f $KALDI_ROOT/tools/config/common_path.sh ] && . $KALDI_ROOT/tools/config/common_path.sh

export LC_ALL=C
