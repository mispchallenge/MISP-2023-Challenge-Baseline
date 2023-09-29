# 打开文件以读取每一行内容
with open("/disk3/chime/misp_dev_video/dev_middle/segments", "r") as file:
    lines = file.readlines()

# 创建一个用于存储修改后行的列表
modified_lines = []

# 遍历每一行内容
for line in lines:
    parts = line.split('_')
    num_part = parts[0]  # 提取开头的数字部分
    num = num_part[1:]  # 去除开头的字符（例如，S）
    
    # 找到包含 "_Far" 的部分并替换为 "_Middle_数字"
    modified_line = line.replace("_Far", f"_Middle_{num}", 1)
    modified_lines.append(modified_line)

# 将修改后的行写入新的文件
with open("/disk3/chime/misp_dev_video/dev_middle/1.txt", "w") as file:
    file.write("".join(modified_lines))
