import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Prepare rttm.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mixspec', metavar='<file>', required=True, help='mixspec.')
args = parser.parse_args()

mixspec = os.path.join(args.mixspec, 'mixspec.json')

os.makedirs(os.path.dirname(mixspec)+"/rttm", exist_ok=True)

def write_speaker_line(OUT, session, start_time, end_time, speaker_id):
    OUT.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(session, start_time, end_time, speaker_id))

with open(mixspec) as IN:
    mixs = json.load(IN)

for mix in mixs:
    session = os.path.basename(mix["output"].split(".")[0])
    output_path = f"{os.path.dirname(mixspec)}/rttm/{session}.rttm"
    
    rttm = {}
    for utt in mix["inputs"]:
        speaker_id = utt["speaker_id"]
        if speaker_id not in rttm:
            rttm[speaker_id] = []
        rttm[speaker_id].append(utt)
    
    with open(output_path, "w") as OUT:
        for speaker_id in sorted(rttm.keys()):
            for utt in rttm[speaker_id]:
                start_time = utt["offset"]
                end_time = utt["length_in_seconds"]
                write_speaker_line(OUT, session, start_time, end_time, speaker_id)
