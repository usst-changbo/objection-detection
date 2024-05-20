import os
import json

def generate_json(folder_path):
    data = {"training": []}


    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_image.nii.gz"):
                image_path = os.path.join(root, file)
                label_path = os.path.join(root, file.replace("_image.nii.gz", "_label.nii.gz"))

                entry = {"image": image_path, "label": label_path}
                data["training"].append(entry)


    with open("../dataset_ACDC_testing.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


folder_path = '/home/changbo/DATASET/ACDC_object_detection/testing'


generate_json(folder_path)