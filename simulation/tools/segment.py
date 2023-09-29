import os
import soundfile as sf
import argparse
from tqdm import tqdm

def split_and_save_audio(input_folder, input_file, output_folder):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing"):
        parts = line.strip().split()
        audio_name = parts[1] + '.wav'
        start_time = float(parts[2])
        end_time = float(parts[3])

        audio_path = os.path.join(input_folder, audio_name)
        audio, sample_rate = sf.read(audio_path)
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        sliced_audio = audio[start_sample:end_sample]

        output_audio_name = parts[0] + '.wav'
        
        output_path = os.path.join(output_folder, output_audio_name)

        sf.write(output_path, sliced_audio, sample_rate)


def main(args):
    # file setting
    input_folder = args.input_folder
    input_file = args.input_file
    output_folder = args.output_folder

    # segment
    split_and_save_audio(input_folder, input_file, output_folder)


def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Segment audio.', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_file', metavar='<file>', required=True,
                        help='Input corpus file in segments.')
    parser.add_argument('--input_folder', metavar='<dir>', required=True,
                        help='Input directory name.')
    parser.add_argument('--output_folder', metavar='<dir>', required=True,
                        help='Output directory name.')
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
