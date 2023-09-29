from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
import argparse
import json
import codecs
import os 
import re
def align_ids(input_path,output_path,filename):
    filepath_in = Path(input_path) / filename
    filepath_out = Path(output_path) / filename
    # import pdb;pdb.set_trace()
    with open(filepath_in, 'r') as input_file:
        with open(filepath_out, 'w') as output_file:
            for line in input_file:
                line_parts=line.split(" ")
                split_parts = re.split(r'[_-]', line_parts[0])

                # for part in split_parts:
                #     print(part)
                # name_parts = line_parts[0].split('-')
                # import pdb;pdb.set_trace()
                start_time = int(split_parts[-2])
                end_time =int(split_parts[-1])
                if start_time % 4 != 0:
                    res = start_time % 4
                    start_time = start_time+4-res
                if end_time % 4 != 0:
                    res = end_time % 4
                    end_time = end_time + 4 - res
                start_time = str(start_time).zfill(6)
                end_time = str(end_time).zfill(6)
                new_name = split_parts[0]+'_'+split_parts[1]+'_'+split_parts[2]+'_'+split_parts[3]+'_'+split_parts[4]+'_'+start_time+'-'+end_time
                new_path = line_parts[1]
                line = '{} {}'.format(new_name,new_path)
                
                output_file.write(line)
if __name__ == "__main__":
    parser = argparse.ArgumentParser("align id")
    parser.add_argument( "--input_path", type=str, default="/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/gss_train_far/wav.scp",)
    parser.add_argument( "--output_path", type=str, default="/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/gss_train_far/wav.scp",)
    parser.add_argument( "--filename", type=str, default="/yrfs2/cv1/hangchen2/espnet/mispi/avsr/dump/raw/gss_train_far/wav.scp",)        
    args = parser.parse_args()
    align_ids(args.input_path,args.output_path,args.filename)
