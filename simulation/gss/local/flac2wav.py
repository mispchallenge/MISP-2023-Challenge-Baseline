from pydub import AudioSegment
import os

import argparse
parser = argparse.ArgumentParser(description='Prepare clean information json.', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_folder', metavar='<dir>', required=True,
                    help='enhanced flac.')
parser.add_argument('--output_folder', metavar='<file>', required=True,
                    help='enhanced wav.')
args = parser.parse_args()

def convert_flac_to_wav(flac_path, wav_path):
    if os.path.getsize(flac_path) == 0:
        print(f"Skipping conversion for {flac_path} as the file is empty.")
        return
    flac_audio = AudioSegment.from_file(flac_path, format="flac")
    flac_audio.export(wav_path, format="wav")



input_folder = args.input_folder
output_folder = args.output_folder

os.makedirs(output_folder, exist_ok=True) 
gDirPath = os.walk(input_folder)
for root, dirs, files in gDirPath:
    for flac_file in files:
        print(flac_file)
        print(root)
        if flac_file.endswith(".flac"):
            #base_name = os.path.splitext(flac_file)[0]
            os.makedirs(root.replace('enhanced','enhanced_wav').replace('-far',''), exist_ok=True)
            wav_file = os.path.join(root.replace('enhanced','enhanced_wav').replace('-far',''), flac_file.replace('.flac','.wav').replace('-far',''))
            flac_path = os.path.join(root, flac_file)
            convert_flac_to_wav(flac_path, wav_file)
            print(f"Converted {flac_file} to {wav_file}")

print("Conversion complete!")