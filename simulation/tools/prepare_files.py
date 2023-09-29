import os
import argparse

parser = argparse.ArgumentParser(description='Prepare noise information.', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--noise_folder', metavar='<dir>', required=True,
                    help='Noise audio directory name.')
parser.add_argument('--output', metavar='<dir>', required=True,
                    help='Output directory name.')
args = parser.parse_args()

input_dir = args.noise_folder
output_file = os.path.join(args.output, 'wav.scp')

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


input_wav_scp = os.path.join(args.output, 'wav.scp')
output_reco2dur = os.path.join(args.output, 'reco2dur')

with open(input_wav_scp, "r") as f_in:
    wav_entries = f_in.readlines()

with open(output_reco2dur, "w") as f_out:
    for wav_entry in wav_entries:
        identifier, wav_path = wav_entry.strip().split(" ", 1)
        duration_command = f"soxi -D '{wav_path}'"
        duration = float(os.popen(duration_command).read().strip())
        f_out.write(f"{identifier} {duration:.2f}\n")

print(f"Generated {output_reco2dur}")

input_wav_scp = os.path.join(args.output, 'wav.scp')
output_utt2spk = os.path.join(args.output, 'utt2spk')

with open(input_wav_scp, "r") as f_in:
    wav_entries = f_in.readlines()

with open(output_utt2spk, "w") as f_out:
    for wav_entry in wav_entries:
        identifier, _ = wav_entry.strip().split(" ", 1)
        f_out.write(f"{identifier} {identifier}\n")

print(f"Generated {output_utt2spk}")

input_wav_scp = os.path.join(args.output, 'wav.scp')
output_spk2utt = os.path.join(args.output, 'spk2utt')

with open(input_wav_scp, "r") as f_in:
    wav_entries = f_in.readlines()

with open(output_spk2utt, "w") as f_out:
    for wav_entry in wav_entries:
        identifier, _ = wav_entry.strip().split(" ", 1)
        f_out.write(f"{identifier} {identifier}\n")

print(f"Generated {output_spk2utt}")