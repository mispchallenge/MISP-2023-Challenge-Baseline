import json
import os
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser(description='Prepare rttm.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mixlog', metavar='<file>', required=True, help='mixlog.')
parser.add_argument('--clean_data', metavar='<dir>', required=True, help='clean_data path.')
parser.add_argument('--video_data', metavar='<dir>', required=True, help='video_data path.')
parser.add_argument('--label_data', metavar='<dir>', required=True, help='label_data path.')
args = parser.parse_args()

mixlog = os.path.join(args.mixlog, 'mixlog.json')

with open(mixlog) as f:
    data = json.load(f)


#clean_audio_folder = '/disk4/cyzhang/misp2023/data/train/video'
label_dir = args.label_data
clean_audio_folder = args.clean_data
output_folder = os.path.join(label_dir, 'audio')

os.makedirs(output_folder)

total_entries = 0
missing_files = 0
missing_file_list = []

for entry in data:
    total_entries += 1
    output_name = os.path.basename(entry['output'])[:-4]  
    #print(output_name)
    for i, input_entry in enumerate(entry['inputs']):
        speaker_id = input_entry['speaker_id']
        #print(speaker_id)
        offset_start = input_entry['offset']
        length_in_seconds = input_entry['length_in_seconds']
        rounded_num_1 = round(offset_start, 2)
        rounded_num_2 = round(length_in_seconds, 2)
        offset_end = rounded_num_1 + rounded_num_2


        int_num = int(round(rounded_num_1 * 100))  
        offset_start_1 = f"{int_num:06d}"  

        int_num = int(round(offset_end * 100)) 
        offset_end_1 = f"{int_num:06d}" 

        input_filename = os.path.basename(input_entry['path'])
        #input_filename_pt = input_filename.replace(".wav", ".pt")
        
        timestamp = f"{offset_start_1}_{offset_end_1}"
        new_filename = f"{speaker_id}-{output_name}-{timestamp}.wav"

        input_path = os.path.join(clean_audio_folder, input_filename)
        output_path = os.path.join(output_folder, new_filename)
        if not os.path.exists(input_path):
            print(f"Warning: File {input_path} not found, skipping...")
            missing_files += 1
            missing_file_list.append(input_filename)
            continue
        else:
            copyfile(input_path, output_path)



with open(mixlog) as f:
    data = json.load(f)
video_folder = args.video_data
output_folder = os.path.join(label_dir, 'video')

os.makedirs(output_folder)

total_entries = 0
missing_files = 0
missing_file_list = []

for entry in data:
    total_entries += 1
    output_name = os.path.basename(entry['output'])[:-4]  
    #print(output_name)
    for i, input_entry in enumerate(entry['inputs']):
        speaker_id = input_entry['speaker_id']
        #print(speaker_id)
        offset_start = input_entry['offset']
        length_in_seconds = input_entry['length_in_seconds']
        rounded_num_1 = round(offset_start, 2)
        rounded_num_2 = round(length_in_seconds, 2)
        offset_end = rounded_num_1 + rounded_num_2


        int_num = int(round(rounded_num_1 * 100))  
        offset_start_1 = f"{int_num:06d}"  

        int_num = int(round(offset_end * 100)) 
        offset_end_1 = f"{int_num:06d}" 

        input_filename = os.path.basename(input_entry['path'])
        input_filename_pt = input_filename.replace(".wav", ".pt")
        
        timestamp = f"{offset_start_1}_{offset_end_1}"
        new_filename = f"{speaker_id}-{output_name}-{timestamp}.pt"

        input_path = os.path.join(video_folder, input_filename_pt)
        output_path = os.path.join(output_folder, new_filename)
        if not os.path.exists(input_path):
            print(f"Warning: File {input_path} not found, skipping...")
            missing_files += 1
            missing_file_list.append(input_filename)
            continue
        else:
            copyfile(input_path, output_path)
    print(f"Total entries: {total_entries}")
    print(f"Missing files: {missing_files}")

    with open('{}missing_files.txt'.format(label_dir), 'w') as f:
        for filename in missing_file_list:
            f.write(f"{filename}\n")

