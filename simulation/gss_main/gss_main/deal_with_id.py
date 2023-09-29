from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
import argparse
import json
import codecs
import os 
def change_name_id(tgtdir):
    roifile = Path(tgtdir) / "wav.scp"
    with open(roifile, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        if "_Far" in line:
            new_line_raw = line
            parts = new_line_raw.replace('_', '-',5)
            new_line = parts.replace('-','_',6)
            new_lines.append(new_line)  
        elif "_Middle" in line:
            new_line_raw = line.replace('_Middle', '', 1)
            parts = new_line_raw.replace('_', '-',5)
            new_line = parts.replace('-','_',5)
            new_lines.append(new_line)  
    with open(roifile, 'w') as f:
        f.writelines(new_lines)      
if __name__ == '__main__':
    parser = argparse.ArgumentParser("run_wpe")
    parser.add_argument( "--tgtdir", type=str, default="/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/org/eval_mid_lip", help="the path where roi.scp storened")
    # import pdb; pdb.set_trace();
    args = parser.parse_args()        
    change_name_id(args.tgtdir)
