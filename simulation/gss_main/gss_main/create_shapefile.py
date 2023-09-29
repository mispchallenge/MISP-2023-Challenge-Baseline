from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
from pathlib import Path 
import argparse
from espnet2.utils.types import str2bool
from espnet2.fileio.read_text import read_2column_text
from espnet2.fileio.npy_scp import NpyScpReader,PtScpReader,BpeScpReader
from espnet2.fileio.sound_scp import SoundScpReader
from tqdm import tqdm
import numpy as np
# import pdb;pdb.set_trace()
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--pdfflag',type=str2bool, default=False)
    #pt reader
    parser.add_argument('--dimnum',type=int, default='1')
    ##bpe reader
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--token_list', type=str, default='')
    parser.add_argument('--token_type', type=str, default='')
    parser.add_argument('--bpemodel', type=str, default='')
    
    


    args = parser.parse_args()

    TYPEMAP = {"wav":SoundScpReader,"flac":SoundScpReader,"pt":PtScpReader,"npy":NpyScpReader,"bpe":BpeScpReader}
    input = Path(args.input)
    output = Path(args.output)
    dimnum = args.dimnum
    
    # 1. detect suffix of input file
    # textfile: <uid> char1char2char3
    # ptfile: <uid> xxx.pt
    if args.mode == None:
        with input.open("r") as f:
            value = f.readline().split(" ")[1]
            seglist = value.strip().split(".")
            if args.pdfflag:
                mode = "pdf" #pdf is the alignment generated from trained GMM-HMM
                func= PtScpReader
            elif len(seglist) == 1: 
                func = read_2column_text 
                mode = "text"
            else:
                func = TYPEMAP[seglist[-1]]
                if seglist[-1] == "wav" or seglist[-1] == "flac":
                    mode = "wav"
                else:
                    mode = "pt_arr"
    else:
        mode = args.mode
        if mode == "bpe":
            filedic = TYPEMAP[mode](args.input,args.token_list,args.token_type,args.bpemodel)
    
    #2. generate shapefile 
    tgtdir = output.parent
    with DatadirWriter(tgtdir) as writer:    
        subwriter = writer[output.name]
        if mode != "bpe":
            filedic = func(input)
        for key,vaule in tqdm(filedic.items()):
            if mode == "text":
                    subwriter[key] = str(len(vaule))
            elif mode == "pdf":
                pdf_fps= 25
                characters = vaule['pdf']
                end_timestamp = np.array(vaule['stamp'])
                subwriter[key] = str(int(np.around(end_timestamp[-1]*pdf_fps)))
            elif mode == "pt_arr":
                vstr = ""
                if dimnum > len(vaule.shape):
                    vstr = "1,"
                    for i in range(dimnum-1):
                        vstr = vstr+str(vaule.shape[i]) + ","  
                else:
                    for i in range(dimnum):
                        vstr = vstr+str(vaule.shape[i]) + ","  
                subwriter[key] = vstr[:-1]
            elif mode == "wav":            
                subwriter[key] = str(vaule[1].shape[0])
            elif mode == "bpe":
                subwriter[key] = str(len(vaule[0]))+","+str(filedic.token_listnum) #

            else:
                raise ValueError(f"there is no mode named: {mode} ")

        

