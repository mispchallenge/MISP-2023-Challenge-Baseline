import os
import argparse

parser = argparse.ArgumentParser(description='Prepare video files.', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', metavar='<dir>', required=True,
                    help='Video directory name.')
parser.add_argument('--file_root', metavar='<dir>', required=True,
                    help='Output directory name.')
parser.add_argument('--segments', metavar='<dir>', required=True,
                    help='segments directory name.')
args = parser.parse_args()

input_dir = args.data_root
output_file = os.path.join(args.file_root, 'mp4.scp')

wav_entries = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        wav_path = os.path.join(root, file)
        identifier = os.path.splitext(file)[0]
        wav_entries.append(f"{identifier} {wav_path}")

sorted_wav_entries = sorted(wav_entries)

with open(output_file, "w") as f_out:
    for entry in sorted_wav_entries:
        f_out.write(f"{entry}\n")

print(f"Generated and sorted {output_file}")


with open(args.segments, "r") as file:
    lines = file.readlines()


modified_lines = []

for line in lines:
    parts = line.split('_')
    num_part = parts[0]  
    num = num_part[1:] 
    
    modified_line = line.replace("_Far", f"_Middle_{num}", 1)
    modified_lines.append(modified_line)

output_file = os.path.join(args.file_root, 'mp4_segments')

with open(output_file, "w") as file:
    file.write("".join(modified_lines))




# input_wav_scp = os.path.join(args.output, 'mp4.scp')
# output_utt2spk = os.path.join(args.output, 'mp4_utt2spk')

# with open(input_wav_scp, "r") as f_in:
#     wav_entries = f_in.readlines()

# with open(output_utt2spk, "w") as f_out:
#     for wav_entry in wav_entries:
#         identifier, _ = wav_entry.strip().split(" ", 1)
#         f_out.write(f"{identifier} {identifier}\n")

# print(f"Generated {output_utt2spk}")
