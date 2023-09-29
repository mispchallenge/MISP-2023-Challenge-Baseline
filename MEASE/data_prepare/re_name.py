import os
import argparse

parser = argparse.ArgumentParser('Rounding audio timestamps')
parser.add_argument('--label_audio', type=str,  help='root directory of label data')
parser.add_argument('--label_video', type=str,  help='root directory of label data')
parser.add_argument('--gss_audio', type=str,  help='root directory of gss data')
args = parser.parse_args()

def round_to_nearest_multiple(number, multiple):
    return ((number + multiple - 1) // multiple) * multiple

label_audio = args.label_audio
label_video = args.label_video
gss_audio = args.gss_audio
folders = [label_audio, label_video, gss_audio]

for folder in folders:
  print(folder)
  for subfolder in os.listdir(folder):
      subfolder_path = os.path.join(folder, subfolder)
      if os.path.isdir(subfolder_path):
          for filename in os.listdir(subfolder_path):
              if filename.endswith(".wav"):
                  parts = filename.rsplit('-', 1)
                  time_parts = parts[-1].split("_")
                
                  start_time = int(time_parts[0])
                  end_time = int(time_parts[1].split(".")[0])
                
                  if start_time % 4 != 0:
                      start_time = round_to_nearest_multiple(start_time, 4)
                  if end_time % 4 != 0:
                      end_time = round_to_nearest_multiple(end_time, 4)
                
                  new_filename = f"{parts[0]}-{int(start_time):06d}_{int(end_time):06d}.wav"
                
                  old_filepath = os.path.join(subfolder_path, filename)
                  new_filepath = os.path.join(subfolder_path, new_filename)
                
                  os.rename(old_filepath, new_filepath)
                  print(f"Renamed: {filename} -> {new_filename}")
                  
              if filename.endswith(".pt"):
                  parts = filename.rsplit('-', 1)
                  time_parts = parts[-1].split("_")
                
                  start_time = int(time_parts[0])
                  end_time = int(time_parts[1].split(".")[0])
                
                  if start_time % 4 != 0:
                      start_time = round_to_nearest_multiple(start_time, 4)
                  if end_time % 4 != 0:
                      end_time = round_to_nearest_multiple(end_time, 4)
                
                  new_filename = f"{parts[0]}-{int(start_time):06d}_{int(end_time):06d}.pt"
                
                  old_filepath = os.path.join(subfolder_path, filename)
                  new_filepath = os.path.join(subfolder_path, new_filename)
                
                  os.rename(old_filepath, new_filepath)
                  print(f"Renamed: {filename} -> {new_filename}")

  
      else:
          for filename in os.listdir(folder):
              if filename.endswith(".wav"):
                  parts = filename.rsplit('-', 1)
                  time_parts = parts[-1].split("_")
                
                  start_time = int(time_parts[0])
                  end_time = int(time_parts[1].split(".")[0])
                
                  if start_time % 4 != 0:
                      start_time = round_to_nearest_multiple(start_time, 4)
                  if end_time % 4 != 0:
                      end_time = round_to_nearest_multiple(end_time, 4)
                
                  new_filename = f"{parts[0]}-{int(start_time):06d}_{int(end_time):06d}.wav"
                
                  old_filepath = os.path.join(folder, filename)
                  new_filepath = os.path.join(folder, new_filename)
                
                  os.rename(old_filepath, new_filepath)
                  print(f"Renamed: {filename} -> {new_filename}")
                  
              if filename.endswith(".pt"):
                  parts = filename.rsplit('-', 1)
                  time_parts = parts[-1].split("_")
                
                  start_time = int(time_parts[0])
                  end_time = int(time_parts[1].split(".")[0])
                
                  if start_time % 4 != 0:
                      start_time = round_to_nearest_multiple(start_time, 4)
                  if end_time % 4 != 0:
                      end_time = round_to_nearest_multiple(end_time, 4)
                
                  new_filename = f"{parts[0]}-{int(start_time):06d}_{int(end_time):06d}.pt"
                
                  old_filepath = os.path.join(folder, filename)
                  new_filepath = os.path.join(folder, new_filename)
                
                  os.rename(old_filepath, new_filepath)
                  print(f"Renamed: {filename} -> {new_filename}")
          break
  

