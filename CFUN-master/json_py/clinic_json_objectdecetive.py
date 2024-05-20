'''为实验室数据生成用于fasterrcnn训练的数据'''


import os
import json

def create_file_list(folder_path):
    file_list = []

    image_files = []
    label_files = []

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if "image" in file_name:
                image_files.append(file_name)
            elif "label" in file_name:
                label_files.append(file_name)

    for image_file in image_files:
        file_entry = {"image": "", "label": ""}
        image_prefix = image_file.split("_image")[0]
        label_file = f"{image_prefix}_label.nii.gz"

        if label_file in label_files:
            file_entry["image"] = os.path.join(folder_path, image_file)
            file_entry["label"] = os.path.join(folder_path, label_file)

            file_list.append(file_entry)

    return file_list

def save_file_list_to_json(file_list, json_file):
    data = {"training": file_list}
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

# 示例用法
folder_path = "/home/changbo/DATASET/clinic-object detection/train"  # 替换为实际的文件夹路径
json_file = "/CFUN-master/dataset.json"  # 替换为输出 JSON 文件的路径

file_list = create_file_list(folder_path)
save_file_list_to_json(file_list, json_file)







