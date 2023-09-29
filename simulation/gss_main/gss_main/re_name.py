import os
import argparse

parser = argparse.ArgumentParser('Rounding audio timestamps')
parser.add_argument('--base_folder', type=str,  help='root directory of gss data')
args = parser.parse_args()

def round_to_nearest_multiple(number, multiple):
    return ((number + multiple - 1) // multiple) * multiple

base_folder = args.base_folder


for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".wav"):
                parts = filename.split("-")
                time_parts = parts[-1].split("_")
                
                start_time = int(time_parts[0])
                end_time = int(time_parts[1].split(".")[0])
                
                if start_time % 4 != 0:
                    start_time = round_to_nearest_multiple(start_time, 4)
                if end_time % 4 != 0:
                    end_time = round_to_nearest_multiple(end_time, 4)
                
                new_filename = f"{parts[0]}-{parts[1]}-{int(start_time):06d}_{int(end_time):06d}.wav"
                
                old_filepath = os.path.join(subfolder_path, filename)
                new_filepath = os.path.join(subfolder_path, new_filename)
                
                os.rename(old_filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")

# for filename in os.listdir(folder_path):
#     if filename.endswith(".wav"):
#         parts = filename.split("-")
#         time_parts = parts[-1].split("_")
        
#         start_time = int(time_parts[0])
#         end_time = int(time_parts[1].split(".")[0])
#         #print(end_time)
#         if start_time % 4 != 0:
#             start_time = round_to_nearest_multiple(start_time, 4)
#         if end_time % 4 != 0:
#             end_time = round_to_nearest_multiple(end_time, 4)
        
#         new_filename = f"{parts[0]}-{parts[1]}-{int(start_time):06d}_{int(end_time):06d}.wav"
        
#         old_filepath = os.path.join(folder_path, filename)
#         new_filepath = os.path.join(folder_path, new_filename)
        
#         os.rename(old_filepath, new_filepath)
#         print(f"Renamed: {filename} -> {new_filename}")


