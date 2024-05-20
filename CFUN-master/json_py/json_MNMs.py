"""json of MNMS"""

import os
import json

def generate_json_files(folder_path):
    training_data = []

    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 用于存储图像和标签的映射关系
    image_label_map = {}

    # 遍历文件夹中的文件
    for file in files:
        file_path = os.path.join(folder_path, file)

        # 检查文件是否为普通文件
        if os.path.isfile(file_path):
            # 获取文件名和扩展名
            file_name, file_ext = os.path.splitext(file)

            # 检查文件名是否符合要求
            if file_ext == '.gz':
                split_name = file_name.split('_')
                if len(split_name) >= 2 and split_name[-1] in ['image.nii', 'label.nii']:
                    prefix = '_'.join(split_name[:-1])
                    if prefix not in image_label_map:
                        image_label_map[prefix] = {'image': '', 'label': ''}
                    if split_name[-1] == 'image.nii':
                        image_label_map[prefix]['image'] = file_path
                    else:
                        image_label_map[prefix]['label'] = file_path

    # 构建训练数据列表
    for data in image_label_map.values():
        training_data.append(data)

    # 构建 JSON 数据
    json_data = {'training': training_data}

    # 生成 JSON 文件
    with open('data.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

# 指定文件夹路径
folder_path = r"/home/changbo/DATASET/MNMs-object detection"

# 调用函数生成 JSON 文件
generate_json_files(folder_path)