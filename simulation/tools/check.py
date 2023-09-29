import os

folder1_path = "/disk3/chime/simulation/data/noise_data/"
folder2_path = "/disk3/chime/jsalt2020_simulate/misp_noise/"

folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)


files_only_in_folder1 = [file for file in folder1_files if file not in folder2_files]


files_to_delete = files_only_in_folder1


wav_scp_path = "/disk3/chime/simulation/data/train_noise_config/segments"

with open(wav_scp_path, "r") as f:
    lines = f.readlines()

# 创建一个新的wav.scp文件，其中不包含要删除的行
with open('/disk3/chime/simulation/data/train_noise_config/segments_new', "w") as f:
    for line in lines:
        # 使用os.path.basename()获取文件名，然后检查是否在要删除的列表中
        if os.path.basename(line.strip().split()[0]) not in files_to_delete:
            f.write(line)

for filename in files_only_in_folder1:
    file_path = os.path.join(folder1_path, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted {filename} from folder1.")
    else:
        print(f"{filename} not found in folder1.")
