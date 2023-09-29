from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
import argparse
import json
import codecs
import os 
import re
def round_to_nearest_4(n):
    if n % 4 == 0:
        return n
    elif n % 4 != 0:
        res = n % 4
        n= n+4-res
        return n

def align_spk2utt(input_path,output_path,filename):
    filepath_in = Path(input_path) / filename
    filepath_out = Path(output_path) / filename
    with open(filepath_in, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.split()
        new_parts = []
        for part in parts:
            # import pdb;pdb.set_trace()
            if '-' in part:
                split_parts = re.split(r'[_-]', part)
                start_time = int(split_parts[-2])
                end_time =int(split_parts[-1])
                new_first_num = round_to_nearest_4(start_time)
                new_second_num = round_to_nearest_4(end_time)
                start_time = str(new_first_num ).zfill(6)
                end_time = str(new_second_num).zfill(6)
                # new_part = f"{sub_parts[0][:-9]}{new_first_num:06}-{new_second_num:06}"
                new_part = split_parts[0]+'_'+split_parts[1]+'_'+split_parts[2]+'_'+split_parts[3]+'_'+split_parts[4]+'_'+start_time+'-'+end_time
                new_parts.append(new_part)
                # import pdb;pdb.set_trace()
            else:
                new_parts.append(part)
        new_line = ' '.join(new_parts)+ '\n'
        new_lines.append(new_line)

    with open(filepath_out, 'w') as f:
        f.writelines(new_lines)
if __name__ == "__main__":
    parser = argparse.ArgumentParser("align id")
    parser.add_argument( "--input_path", type=str, default="/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/gss_train_far/wav.scp",)
    parser.add_argument( "--output_path", type=str, default="/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/gss_train_far/wav.scp",)
    parser.add_argument( "--filename", type=str, default="/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/gss_train_far/wav.scp",)        
    args = parser.parse_args()
    align_spk2utt(args.input_path,args.output_path,args.filename)