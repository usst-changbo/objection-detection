"""json of MMWHS2017-mr"""


import os
import re
import json

# 数据集文件夹路径
dataset_folder = r"/home/changbo/DATASET/MM-WHS2017/mr_train/mr_train"

# 生成的 JSON 文件路径
json_file_path = "dataset.json"

# 创建包含训练数据的列表
training_data = []

# 遍历数据集文件夹中的每个图像文件
for image_filename in os.listdir(dataset_folder):
    if image_filename.endswith("image.nii.gz"):
        # 使用正则表达式从文件名中搜索数字
        match = re.search(r"\d+", image_filename)

        if match:
            # 提取病例编号
            case_number = match.group()

            # 构建标签文件名
            label_filename = f"mr_train_{case_number}_label.nii.gz"

            # 图像路径
            image_path = os.path.join(dataset_folder, image_filename)

            # 标签路径
            label_path = os.path.join(dataset_folder, label_filename)

            # 检查标签文件是否存在
            if os.path.exists(label_path):
                # 构建包含图像和标签路径的字典
                data = {"image": image_path, "label": label_path}

                # 将字典添加到训练数据列表
                training_data.append(data)

# 构建包含训练数据的字典
json_data = {"training": training_data}

# 将字典转换为 JSON 字符串
json_string = json.dumps(json_data, indent=4)

# 将 JSON 字符串写入文件
with open(json_file_path, "w") as json_file:
    json_file.write(json_string)

print("JSON 文件已生成：", json_file_path)