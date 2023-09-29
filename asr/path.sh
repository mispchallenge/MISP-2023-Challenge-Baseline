
MAIN_ROOT= #/path/to/espnet/     e.g. /disk1/code/espnet/espnet-master
export KALDI_ROOT= #/path/to/kaldi/    e.g. /home/backman/kaldi 
export PYTHONPATH=$PYTHONPATH:/path/to/python/ # /path/to/python/  e.g /disk1/conda/envs/pytorch/bin/python3.8
export PATH=/path/to/python/:$MAIN_ROOT:${PATH} # /path/to/python e.g /disk1/conda/envs/pytorch/bin/python3.8


export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst-1.7.2/bin:$KALDI_ROOT/tools/sph2pipe:/disk3/hblan/2019/sctk-20159b5/bin:${PATH} # /path/to/kaldi/tools/
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present - > Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH
. "${MAIN_ROOT}"/tools/extra_path.sh
export OMP_NUM_THREADS=1
# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
# NOTE(kamo): Source at the last to overwrite the setting
