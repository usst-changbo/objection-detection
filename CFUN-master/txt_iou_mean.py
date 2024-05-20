"""使用正则计算txt文件中rpn_roi_iou_max:后面的直的平均值"""

import re

filename = "/home/changbo/Object_detection/faster_rcnn_unet/CFUN-master/log/train_text_1008_355.txt"
iou_values = []
with open(filename, "r") as file:
    content = file.read()
    matches = re.findall(r"rpn_roi_iou_max:\s*([\d.]+)", content)
    iou_values = [float(match) for match in matches]

if iou_values:
    mean_iou = sum(iou_values) / len(iou_values)
    mean_iou = round(mean_iou, 5)
    print("Mean IOU:", mean_iou)
else:
    print("No IOU values found in the file.")