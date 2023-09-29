import os
import soundfile as sf
import argparse

parser = argparse.ArgumentParser('check_audio')
parser.add_argument('--gss_data', type=str,  help='root directory of gss data')
parser.add_argument('--audio_label', type=str,  help='root directory of label data')
parser.add_argument('--output_folder', type=str,  help='root directory of gss data')
args = parser.parse_args()

def generate_reco2dur_for_subfolder(subfolder_path, output_file):
    reco2dur = {}
    for dir_name in os.listdir(subfolder_path):
        
        dir_path = os.path.join(subfolder_path, dir_name)
        #print(subfolder_path)
        if os.path.isdir(subfolder_path) and "_0db" in subfolder_path:
            for filename in os.listdir(subfolder_path):
                print(filename)
                if filename.endswith('.wav'):
                    audio_path = os.path.join(subfolder_path, filename)
                    
                    audio, sample_rate = sf.read(audio_path)
                    num_samples = len(audio)
                
                    duration = num_samples / sample_rate
                    audio_id = os.path.splitext(filename)[0]
                    
                    reco2dur[audio_id] = (num_samples, duration)
    
    with open(output_file, 'a') as f:
        for audio_id in sorted(reco2dur.keys()):
            
            num_samples, duration = reco2dur[audio_id]
            print(f"{audio_id} {num_samples} {duration:.6f}\n")
            f.write(f"{audio_id} {num_samples} {duration:.6f}\n")


main_folder = args.gss_data
output_file = os.path.join(args.output_folder, 'gss_dur')

for subfolder_name in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder_name)
    #print(subfolder_path)
    if os.path.isdir(subfolder_path):
        generate_reco2dur_for_subfolder(subfolder_path, output_file)

import os
import soundfile as sf

def generate_reco2dur(audio_folder, output_file):
    reco2dur = {}
    for filename in os.listdir(audio_folder):
        if filename.endswith('.wav'):
            audio_path = os.path.join(audio_folder, filename)
            audio, sample_rate = sf.read(audio_path)
            num_samples = len(audio)
            duration = num_samples / sample_rate
            audio_id = os.path.splitext(filename)[0]
            reco2dur[audio_id] = (num_samples, duration)
    
    with open(output_file, 'w') as f:
        for audio_id in sorted(reco2dur.keys()):
            num_samples, duration = reco2dur[audio_id]
            print(f"{audio_id} {num_samples} {duration:.6f}\n")
            f.write(f"{audio_id} {num_samples} {duration:.6f}\n")

# 设置音频文件夹和输出文件路径
audio_folder = args.audio_label
output_file = os.path.join(args.output_folder, 'label_dur')

# 生成reco2dur文件
generate_reco2dur(audio_folder, output_file)

def read_file(filename):
    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            key = parts[0].replace('_0db','').replace('S','')
            value = int(parts[1])
            data[key] = value
    return data

# 读取两个文件的内容
file1_data = read_file(os.path.join(args.output_folder, 'gss_dur'))
file2_data = read_file(os.path.join(args.output_folder, 'label_dur'))

# 找到不同的项并输出
output_filename = os.path.join(args.output_folder, 'check.txt')
with open(output_filename, 'w') as output_file:
    for key in file1_data:
        if key in file2_data and file1_data[key] != file2_data[key]:
            output_line = f"Key: {key}, File1 Value: {file1_data[key]}, File2 Value: {file2_data[key]}\n"
            output_file.write(output_line)
