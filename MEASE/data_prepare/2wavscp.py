import os
import argparse
parser = argparse.ArgumentParser(description='Prepare clean information json.', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_folder', metavar='<dir>', required=True,
                    help='enhanced flac.')
parser.add_argument('--output_folder', metavar='<dir>', required=True,
                    help='enhanced wav.')
parser.add_argument('--dataset', metavar='<file>', required=True,
                    help='sim_trainingset')
args = parser.parse_args()

audio_folder = args.input_folder
output_file = os.path.join(args.output_folder, args.dataset + '_wav.scp')

wav_scp = {}


for root, dirs, files in os.walk(audio_folder):
    for file in files:
        if file.endswith(".flac") or file.endswith(".wav"):
            file_path = os.path.join(root, file)
            audio_id = os.path.splitext(file)[0]
            wav_scp[audio_id] = file_path
            
with open(output_file, 'w') as f:
    for audio_id, file_path in wav_scp.items():
        f.write(f'{audio_id}\n')
