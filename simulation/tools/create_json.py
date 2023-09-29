import os
import json
import wave
import argparse

parser = argparse.ArgumentParser(description='Prepare clean information json.', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--clean_folder', metavar='<dir>', required=True,
                    help='Clean audio directory name.')
parser.add_argument('--output', metavar='<file>', required=True,
                    help='Output directory name.')
args = parser.parse_args()


root_folder = args.clean_folder


json_data = []


for root, dirs, files in os.walk(root_folder):
    sorted_files = sorted([file for file in files if file.endswith('.wav')])
    for file in sorted_files:
        file_path = os.path.join(root, file)
        speaker = file.split("_")[0][1:]
        with wave.open(file_path, 'rb') as wf:
            num_samples = wf.getnframes()
            sampling_rate = wf.getframerate()
            length_seconds = num_samples / float(sampling_rate)

        json_obj = {
            "utterance_id": os.path.splitext(file)[0], 
            "path": file_path,
            "speaker_id": speaker,  
            "number_of_samples": num_samples,
            "sampling_rate": sampling_rate,
            "length_in_seconds": round(length_seconds, 2)
        }    
        json_data.append(json_obj)

with open(args.output, 'w') as json_file:
     json.dump(json_data, json_file, indent=2)

print("done")


