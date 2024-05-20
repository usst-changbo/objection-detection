"""
主要的CFUN模型实现。
"""

import time
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import utils
import backbone
import mask_branch

############################################################
#  日志功能
############################################################


def log(text, array=None):
     """打印文本信息。并且，可选地，如果提供了Numpy数组打印它的形状、最小值和最大值。"""

     if not os.path.exists('log'):
         os.makedirs('log')
     log_file = os.path.join('log', 'train_text.txt')
     if array is not None:
         text = text.ljust(25)
         text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
             str(array.shape),
             array.min() if array.size else "",
             array.max() if array.size else ""))
     print(text)
     with open(log_file, 'a') as f:
         f.write("\r"+text)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=''):
    """ 调用一个循环来创建终端进度条
#     @params:
#         iteration   - Required  : 当前迭代(Int)
#         total       - Required  : 总迭代数(Int)
#         prefix      - Optional  : 前缀字符串(Str)
#         suffix      - Optional  : 后缀字符串(Str)
#         decimals    - Optional  : 百分位小数的positive数(Int)
#         length      - Optional  : bar字符长度(Int)
#         fill        - Optional  : bar填充字符(Str)(Str)
#     """
    if not os.path.exists('log'):
        os.makedirs('log')
    log_file = os.path.join('log', 'train_text.txt')

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '\u2589' * (filled_length)
    progress_text = '\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
    progress_text_new = '\r%s  %s' % (prefix, suffix)
    print(progress_text, end='\n')

    log_file = os.path.join('log', 'train_text.txt')
    with open(log_file, 'a') as f:
        f.write(progress_text_new )

    if iteration == total:
        print()

############################################################
#  Pytorch Functions
############################################################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.detach()]


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).detach()]


def log2(x):
    """log2的实现。Pytorch没有本地实现。"""
    ln2 = torch.log(torch.FloatTensor([2.0]))
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2


def compute_backbone_shapes(config, image_shape):
    """计算特征提取网络各阶段的深度、宽度和高度。
    Returns:
        [N, (depth, height, width)]. N是stages数
    """
    H, W, D = image_shape[:3]
    return np.array(
        [[int(math.ceil(D / stride)),
          int(math.ceil(H / stride)),
          int(math.ceil(W / stride))]
         for stride in config.BACKBONE_STRIDES])


############################################################
#  FPN Graph
############################################################

class TopDownLayer(nn.Module):
    """生成金字塔特征图。
    返回[p2_out, p3_out, cθ_out]，其中p2_out和p3_out用于RPN,
    cO_out用于mrcnn mask分支。
    """
    def __init__(self, in_channels, out_channels):
        super(TopDownLayer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        y = F.interpolate(y, scale_factor=2)
        x = self.conv1(x)
        return self.conv2(x + y)


class FPN(nn.Module):
    def __init__(self, C1, C2, C3, out_channels, config):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.P3_conv1 = nn.Conv3d(config.BACKBONE_CHANNELS[1] * 4, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P2_conv1 = nn.Conv3d(config.BACKBONE_CHANNELS[0] * 4, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x

        p3_out = self.P3_conv1(c3_out)
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        return [p2_out, p3_out]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """将给定的增量应用于给定的boxes。
    boxes: [N, 6] 每一行是 z1 y1 x1 z2 y2 x2
    deltas: [N, 6] 每一行是 [dz, dy, dx, log(dd), log(dh), log(dw)]
    """
    # Convert to z, y, x, d, h, w
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]
    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    # Apply deltas
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= torch.exp(deltas[:, 3])
    height *= torch.exp(deltas[:, 4])
    width *= torch.exp(deltas[:, 5])
    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([z1, y1, x1, z2, y2, x2], dim=1)
    return result


def clip_boxes(boxes, window):
    """boxes: [N, 6] each col is z1, y1, x1, z2, y2, x2
    window: [6] in the form z1, y1, x1, z2, y2, x2
    """
    boxes = torch.stack(
        [boxes[:, 0].clamp(float(window[0]), float(window[3])),
         boxes[:, 1].clamp(float(window[1]), float(window[4])),
         boxes[:, 2].clamp(float(window[2]), float(window[5])),
         boxes[:, 3].clamp(float(window[0]), float(window[3])),
         boxes[:, 4].clamp(float(window[1]), float(window[4])),
         boxes[:, 5].clamp(float(window[2]), float(window[5]))], 1)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """接收anchor分数，并选择一个子集作为proposals传递给第二阶段。
    Filter是基于anchor分数和非最大抑制来消除重叠。
    它还将边界框细化增量应用于锚点。
    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    Returns:
        在标准化坐标系中的Proposals [batch, rois, (z1, y1, x1, z2, y2, x2)]
    """

    # 目前只支持batchsize为1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box的分数。使用前景类置信度 [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 6]
    deltas = inputs[1]
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 6])).float()
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # 通过按分数调整到最顶端的anchor，并在较小的子集上同样计算，用来提高性能。
    pre_nms_limit = min(config.PRE_NMS_LIMIT, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.detach(), :]
    anchors = anchors[order.detach(), :]

    # 将delta应用于anchor以获得精细的anchor。
    # [batch, N, (z1, y1, x1, z2, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # 裁剪到图像边界. [batch, N, (z1, y1, x1, z2, y2, x2)]
    height, width, depth = config.IMAGE_SHAPE[:3]
    window = np.array([0, 0, 0, depth, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # 非最大抑制
    keep = utils.non_max_suppression(boxes.cpu().detach().numpy(),
                                     scores.cpu().detach().numpy(), nms_threshold, proposal_count)
    keep = torch.from_numpy(keep).long()
    boxes = boxes[keep, :]

    # 将维度标准化到0到1的范围。
    norm = torch.from_numpy(np.array([depth, height, width, depth, height, width])).float()
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # 添加回batch尺寸
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes


############################################################
#  ROIAlign Layer
############################################################

def RoI_Align(feature_map, pool_size, boxes):
    """实现3D RoI Align(实际上它只是pooling而不是Align)。
    feature_map: [channels, depth, height, width]. 生成 FPN.
    pool_size: [D, H, W]. The shape of the output.
    boxes: [num_boxes, (z1, y1, x1, z2, y2, x2)].
    """
    boxes = utils.denorm_boxes_graph(boxes, (feature_map.size()[1], feature_map.size()[2], feature_map.size()[3]))
    boxes[:, 0] = boxes[:, 0].floor()
    boxes[:, 1] = boxes[:, 1].floor()
    boxes[:, 2] = boxes[:, 2].floor()
    boxes[:, 3] = boxes[:, 3].ceil()
    boxes[:, 4] = boxes[:, 4].ceil()
    boxes[:, 5] = boxes[:, 5].ceil()
    boxes = boxes.long()
    output = torch.zeros((boxes.size()[0], feature_map.size()[0], pool_size[0], pool_size[1], pool_size[2])).cuda()
    for i in range(boxes.size()[0]):
        try:
            output[i] = F.interpolate((feature_map[:, boxes[i][0]:boxes[i][3], boxes[i][1]:boxes[i][4], boxes[i][2]:boxes[i][5]]).unsqueeze(0),
                                      size=pool_size, mode='trilinear', align_corners=True).cuda()
        except:
            print("RoI_Align error!")
            print("box:", boxes[i], "feature_map size:", feature_map.size())
            pass

    return output.cuda()


def pyramid_roi_align(inputs, pool_size, test_flag=False):
    """在特征金字塔的多个层上实现 ROI Pooling
    Params:
    - pool_size: [depth, height, width] 输出pooled 区域. 通常是 [7, 7, 7]
    - image_shape: [height, width, depth, channels]. 输入图像的形状，以像素为单位
    Inputs:
    - boxes: [batch, num_boxes, (z1, y1, x1, z2, y2, x2)] 在标准化坐标系中。
    - Feature maps: 来自金字塔不同层的特征图列表。
                    Each is [batch, channels, depth, height, width]
    Output:
    Pooled 区域的形状: [num_boxes, channels, depth, height, width].
    宽度、高度和深度是在层构造函数的pool_shape中指定的。
    """

    # 目前只支持batchsize为1
    if test_flag:
        for i in range(0, len(inputs)):
            inputs[i] = inputs[i].squeeze(0)
    else:
        for i in range(1, len(inputs)):
            inputs[i] = inputs[i].squeeze(0)

    # 裁剪 boxes [batch, num_boxes, (y1, x1, z1, y2, x2, z2)] 在标准化坐标系中
    boxes = inputs[0]
    # 特征图。来自特征金字塔不同层的特征映射列表。
    # 每个是[batch, channels, depth, height, width]
    feature_maps = inputs[1:]

    # 根据ROI数量将每个ROI分配到金字塔中的一个层。
    z1, y1, x1, z2, y2, x2 = boxes.chunk(6, dim=1)
    d = z2 - z1
    h = y2 - y1
    w = x2 - x1

    # 特征金字塔论文中的方程1。
    # 考虑到这里的坐标是归一化的。
    # TODO: change the equation here
    roi_level = 4 + (1. / 3.) * log2(h * w * d)
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 3)

    # 循环各层并将 ROI pooling 应用到 P2 或 P3.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 4)):
        ix = (roi_level == level)
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.detach(), :]

        # 追踪具体哪一个box 映射到哪一个 level
        box_to_level.append(ix.detach())

        # 停止梯度传播到ROI proposals
        level_boxes = level_boxes.detach()

        # 裁剪和调整大小
        # 来自Mask R-CNN论文:对四个常规位置进行采样，这样就可以评估最大或平均池化。
        # 事实上，在每个bin中心只插入一个值(没有池化)几乎同样有效。”
        # 在这里，使用简化的方法，即每个bin使用一个值。
        # Result: [batch * num_boxes, channels, pool_depth, pool_height, pool_width]
        pooled_features = RoI_Align(feature_maps[i], pool_size, level_boxes)
        pooled.append(pooled_features)

    # 将汇集的特征放进一个张量
    pooled = torch.cat(pooled, dim=0)

    # 将box_to_level映射打包到一个数组中，
    # 并添加另一列，表示pooled boxes的顺序
    box_to_level = torch.cat(box_to_level, dim=0)

    # 重新排列pooled的特征以匹配原始boxes的顺序
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :, :]

    return pooled


############################################################
#  检测目标层
############################################################

def bbox_overlaps(boxes1, boxes2):
    """计算两组boxes之间的IoU重叠。
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].
    """
    # 1. 平铺boxes2并重复boxes1。
    # 这允许在没有循环的情况下比较每个boxes1和每个boxes2。
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 6)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    # 2. 计算交集
    b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = boxes1.chunk(6, dim=1)
    b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = boxes2.chunk(6, dim=1)
    z1 = torch.max(b1_z1, b2_z1)[:, 0]
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    z2 = torch.min(b1_z2, b2_z2)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(z1.size()[0]), requires_grad=False)
    if z1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros) * torch.max(z2 - z1, zeros)

    # 3. 计算并集
    b1_volume = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_volume = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_volume[:, 0] + b2_volume[:, 0] - intersection

    # 4. 计算IoU并reshape为 [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """对proposals进行子样本并为每个proposals生成细化后的目标框、class_id和mask。
    Inputs:
    proposals: [batch, N, (z1, y1, x1, z2, y2, x2)]归一化坐标。
               如果没有足够的proposals，可能是零填充。
    gt_class_ids: [batch, MAX_GT_INSTANCES] 整型类id.
    gt_boxes: [batch, MAX_GT_INSTANCES, (z1, y1, x1, z2, y2, x2)]在标准化坐标系中。
    gt_masks: [batch, MAX_GT_INSTANCES, depth, height, width] np.int32 类型
    Returns: 目标roi和相应的类id，BBox移位和mask。
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (z1, y1, x1, z2, y2, x2)] 在标准化坐标系中
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. 整型类id.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dz, dy, dx, log(dd), log(dh), log(dw), class_id)]
                   特定于类的bbox细化。
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, depth, height, width)
                 mask裁剪为bbox边界并调整为神经网络输出大小。
    """

    # 目前只支持batchsize为1
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)

    # 计算重叠矩阵 [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    # 确定positive 和 negative ROIs
    global roi_iou_max
    roi_iou_max = torch.max(overlaps, dim=1)[0]
    print("rpn_roi_iou_max:", roi_iou_max.max())



    # 1. Positive ROIs 是指带有GT box的IoU >= 0.7
    positive_roi_bool = roi_iou_max >= config.DETECTION_TARGET_IOU_THRESHOLD

    # 子例ROIs.目标是33% positive
    # Positive ROIs
    if torch.nonzero(positive_roi_bool).size()[0] != 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.detach(), :]
        # 为GT boxes匹配 positive ROIs.
        positive_overlaps = overlaps[positive_indices.detach(), :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.detach(), :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.detach()]

        # 为positive ROIs计算细化bbox
        deltas = Variable(utils.box_refinement(positive_rois.detach(), roi_gt_boxes.detach()), requires_grad=False)
        std_dev = torch.from_numpy(config.BBOX_STD_DEV).float()
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev
        # 为GT masks匹配positive ROIs
        # 将mask排列为[N, depth, height, width]
        # 为每个ROI选择正确的mask
        roi_gt_masks = np.zeros((positive_rois.shape[0], 8,) + config.MASK_SHAPE)
        for i in range(0, positive_rois.shape[0]):
             z1 = int(gt_masks.shape[1]*positive_rois[i, 0])
             z2 = int(gt_masks.shape[1]*positive_rois[i, 3])
             y1 = int(gt_masks.shape[2]*positive_rois[i, 1])
             y2 = int(gt_masks.shape[2]*positive_rois[i, 4])
             x1 = int(gt_masks.shape[3]*positive_rois[i, 2])
             x2 = int(gt_masks.shape[3]*positive_rois[i, 5])
             crop_mask = gt_masks[:, z1:z2, y1:y2, x1:x2].cpu().numpy()
             crop_mask = utils.resize(crop_mask, (8,) + config.MASK_SHAPE, order=0, preserve_range=True)
             roi_gt_masks[i, :, :, :, :] = crop_mask
        roi_gt_masks = torch.from_numpy(roi_gt_masks).cuda()
        roi_gt_masks = roi_gt_masks.type(torch.DoubleTensor)
    else:
        positive_count = 0

    # 2. Negative ROIs 是每个GT box < 0.5 .
    negative_roi_bool = roi_iou_max < config.DETECTION_TARGET_IOU_THRESHOLD
    negative_roi_bool = negative_roi_bool
    # Negative ROIs. 添加足够的数量保持 positive:negative比例.
    if torch.nonzero(negative_roi_bool).size()[0] != 0 and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.detach(), :]
    else:
        negative_count = 0


    # 附加 negative roi，并用0填充不用于negative roi的bbox的deltas和mask。
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).long()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids.long(), zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 6), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.MASK_SHAPE[2]),
                         requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = roi_gt_masks
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        positive_rois = Variable(torch.FloatTensor(), requires_grad=False)
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
            positive_rois = positive_rois.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 6), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.MASK_SHAPE[2]),
                         requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros
    else:
        positive_rois = Variable(torch.FloatTensor(), requires_grad=False)
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            positive_rois = positive_rois.cuda()
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()
    return positive_rois, rois, roi_gt_class_ids, deltas, masks


############################################################
#  检测层
############################################################

def clip_to_window(window, boxes):
    """window: (z1, y1, x1, z2, y2, x2). window 在要剪辑到的图像中.
        boxes: [N, (z1, y1, x1, z2, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[3]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[4]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[2]), float(window[5]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[0]), float(window[3]))
    boxes[:, 4] = boxes[:, 4].clamp(float(window[1]), float(window[4]))
    boxes[:, 5] = boxes[:, 5].clamp(float(window[2]), float(window[5]))

    return boxes


def refine_detections(rois, probs, deltas, window, config):
    """细化分类proposals，过滤重叠部分，并返回最终检测结果。
    Inputs:
        rois: [N, (z1, y1, x1, z2, y2, x2)] 在标准化坐标系中
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dz, dy, dx, log(dd), log(dh), log(dw))].
                特定于类的边界框deltas.
        window:图像坐标中的(z1, y1, x1, z2, y2, x2)。
               图像中包含图像的部分，但不包括填充。
    返回检测形状: [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
    """

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # 每个ROI top类别的类别概率
    # 特定于类的bboxes 的deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
    class_scores = probs[idx, class_ids.detach()]
    deltas_specific = deltas[idx, class_ids.detach()]

    # 应用 bounding box deltas
    # Shape: [boxes, (z1, y1, x1, z2, y2, x2)] 在标准化坐标系中
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 6])).float()
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

    # 将坐标转换为图像域
    height, width, depth = config.IMAGE_SHAPE[:3]
    scale = torch.from_numpy(np.array([depth, height, width, depth, height, width])).float()
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois *= scale

    # 将boxes 剪贴到image window
    refined_rois = clip_to_window(window, refined_rois)

    # 舍入并强制转换为int，因为现在处理的是像素
    refined_rois = torch.round(refined_rois)

    # 过滤掉背景 boxes
    keep_bool = class_ids > 0

    # 过滤掉低可靠性的 boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:, 0]

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep.detach()]
    pre_nms_scores = class_scores[keep.detach()]
    pre_nms_rois = refined_rois[keep.detach()]

    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        # 选择该类的检测
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]

        # Sort
        ix_rois = pre_nms_rois[ixs.detach()]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.detach(), :]

        class_keep = utils.non_max_suppression(ix_rois.cpu().detach().numpy(), ix_scores.cpu().detach().numpy(),
                                               config.DETECTION_NMS_THRESHOLD, config.DETECTION_MAX_INSTANCES)
        class_keep = torch.from_numpy(class_keep).long()

        # Map 索引
        class_keep = keep[ixs[order[class_keep].detach()].detach()]

        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)

    # 保持top检测
    roi_count = config.DETECTION_MAX_INSTANCES
    roi_count = min(roi_count, keep.size()[0])
    top_ids = class_scores[keep.detach()].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.detach()]

    # 将输出定为 [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
    # 坐标在图像域中。
    result = torch.cat((refined_rois[keep.detach()],
                        class_ids[keep.detach()].unsqueeze(1).float(),
                        class_scores[keep.detach()].unsqueeze(1)), dim=1)

    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """获取分类 proposal boxes和 bbox deltas，返回最终检测框。
    Returns:
    [batch, num_detections, (z1, y1, x1, z2, y2, x2, class_score)] 以像素为单位
    """

    # 目前只支持batchsize为1
    rois = rois.squeeze(0)

    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)

    return detections


############################################################
#  Region Proposal Network
############################################################

class RPN(nn.Module):
    """构建Region Proposal Network模型.
    anchors_per_location: 特征图中每个像素的anchor数
    anchor_stride: 控制anchor的密度。通常为1(特征图中的每个像素的anchor)或2(其他每个像素)。
    Returns:
        rpn_logits: [batch, D, H, W, 2] Anchor 分类器 logits (在softmax之前)
        rpn_probs: [batch, D, H, W, 2] Anchor 分类器概率
        rpn_bbox: [batch, D, H, W, (dz, dy, dx, log(dd), log(dh), log(dw))]将Deltas 应用于anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, channel, conv_channel):
        super(RPN, self).__init__()
        self.conv_shared = nn.Conv3d(channel, conv_channel, kernel_size=3, stride=anchor_stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv3d(conv_channel, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv3d(conv_channel, 6 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # RPN的共享卷积base
        x = self.relu(self.conv_shared(x))

        # Anchor Score. [batch, anchors per location * 2, depth, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 4, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax 在最后一个维度上 BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # 细化bbox. [batch, anchors per location * 6, D, H, W]
        # where 6 == delta [z, y, x, log(d), log(h), log(w)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, anchors, 6]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 4, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 6)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Feature Pyramid Network Heads
############################################################

class Classifier(nn.Module):
    def __init__(self, channel, pool_size, image_shape, num_classes, fc_size, test_flag=False):
        super(Classifier, self).__init__()
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.fc_size = fc_size
        self.test_flag = test_flag

        self.conv1 = nn.Conv3d(channel, fc_size, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm3d(fc_size, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv3d(fc_size, fc_size, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm3d(fc_size, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(fc_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.linear_bbox = nn.Linear(fc_size, num_classes * 6)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.test_flag)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, self.fc_size)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 6)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class Mask(nn.Module):
    def __init__(self, channel, pool_size, num_classes, conv_channel, stage, test_flag=False):
        super(Mask, self).__init__()
        self.pool_size = pool_size
        self.test_flag = test_flag

        self.modified_u_net = mask_branch.Modified3DUNet(channel, num_classes, stage, conv_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.test_flag)
        x = self.modified_u_net(x)
        output = self.softmax(x)

        return x, output


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor 分类损失.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # 为简化压缩最后一个维度
    rpn_match = rpn_match.squeeze(2)

    # 获取anchor类. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive 和 Negative anchors 造成损失
    # 但是neutral anchors不能是 (match value = 0) .
    indices = torch.nonzero(rpn_match != 0)

    # 选择导致损失的行，并过滤掉其余的行。
    rpn_class_logits = rpn_class_logits[indices.detach()[:, 0], indices.detach()[:, 1], :]
    anchor_class = anchor_class[indices.detach()[:, 0], indices.detach()[:, 1]]

    # Cross-entropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """返回RPN bounding box loss graph.
    target_bbox: [batch, max positive anchors, (dz, dy, dx, log(dd), log(dh), log(dw))].
        用0填充未使用的bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """

    # 为简化，在最后一个维度压缩
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors 造成损失
    # 但是nagative 和 neutral anchors不能是 (match value = 0) .
    indices = torch.nonzero(rpn_match == 1)

    # 选取导致损失的bbox delta
    rpn_bbox = rpn_bbox[indices.detach()[:, 0], indices.detach()[:, 1]]

    # 将目标边界box delta修剪为与rpn_bbox相同的长度。
    target_bbox = target_bbox[0, :rpn_bbox.size()[0], :]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """classifier head of Mask RCNN 损失.
    target_class_ids: [batch, num_rois]. 整型类id。使用零填充填充数组。
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if target_class_ids.size()[0] != 0:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Mask R-CNN bounding box 细化损失.
    target_bbox: [batch, num_rois, (dz, dy, dx, log(dd), log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. 整型类id.
    pred_bbox: [batch, num_rois, num_classes, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """

    if target_class_ids.size()[0] != 0:
        # 只有positive的roi才会造成损失。
        # 并且只有每个正确class_id的ROI，得到它们的指标。
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.detach()].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        # *收集导致损失的delta(正确的预测)
        target_bbox = target_bbox[indices[:, 0].detach(), :]
        pred_bbox = pred_bbox[indices[:, 0].detach(), indices[:, 1].detach(), :]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss

def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                       target_deltas, mrcnn_bbox):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    target_class_ids = target_class_ids.cpu()
    mrcnn_class_loss = compute_mrcnn_class_loss(torch.from_numpy(np.where(target_class_ids > 0, 1, 0)).cuda(),
                                                mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas,
                                              torch.from_numpy(np.where(target_class_ids > 0, 1, 0)).cuda(), mrcnn_bbox)

    # mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask_logits)
    # if stage == 'finetune':
    #     mrcnn_mask_edge_loss = compute_mrcnn_mask_edge_loss(target_mask, target_class_ids, mrcnn_mask)
    # else:
    #     mrcnn_mask_edge_loss = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()

    # return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss.cuda(), mrcnn_mask_edge_loss]
    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss]


############################################################
#  Data Generator
############################################################

def load_image_gt(image, mask, angle, dataset, config, anchors):
    """ 加载并返回图像的真实数据。
    angle: 旋转图像和mask进行增强
    anchors:用于生成rpn_match和rpn_bbox
    Returns:
    image: [1, depth, height, width]
    class_ids: [instance_count] 整型类id
    bbox: [instance_count, (z1, y1, x1, z2, y2, x2)]
    mask: [depth, height, width, instance_count]
    rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    rpn_bbox: [batch, N, (dz, dy, dx, log(dd), log(dh), log(dw))] Anchor bbox deltas
    """
    # Augmentation
    import imgaug
    from imgaug import augmenters as iaa
    augment = iaa.Affine(rotate=angle, order=0)  # randomly rotate the image between -20 degree and 20 degree
    if augment is not None:
        # 可以应用于masks的增强
        # 比如说 Affine, have settings 使得他们不安全,并总是如此
        # 在masks上测试增强效果
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """确定要应用于mask的增强器。"""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # 在增强之前存储形状以进行比较
        image_shape = image.shape
        mask_shape = mask.shape
        # 将图像调整为[height, width, depth] 这样就可以在上面实现图像增强。.
        # 将图像的每一个[height, width, 1]视为一个图像，并对其进行成像。
        image = np.squeeze(image, 3)
        # 使增强器应用于相似的图像和mask
        det = augment.to_deterministic()
        image = det.augment_image(image)
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # 将图像重塑为 [height, width, length, 1]
        image = np.expand_dims(image, -1)
        # 确认形状没有改变
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        #将mask改回 np.int32
        mask = mask.astype(np.int32)
    # 将图像和mask转换到正确的形状。
    # 转换之后, image: [C, D, H, W], mask: [D, H, W, C]
    image = image.transpose((3, 2, 0, 1))
    mask = mask.transpose((2, 0, 1))
    # bbox: [num_instances, (z1, y1, x1, z2, y2, x2)]
    bbox = utils.extract_bboxes(np.expand_dims(mask, -1))
    z1, y1, x1, z2, y2, x2 = bbox[0, :]
    depth = z2 - z1
    height = y2 - y1
    width = x2 - x1
    z1 -= depth * 0.1
    z2 += depth * 0.1
    y1 -= height * 0.1
    y2 += height * 0.1
    x1 -= width * 0.1
    x2 += width * 0.1
    z1 = np.floor(max(0, z1))
    z2 = np.ceil(min(mask.shape[0], z2))
    y1 = np.floor(max(0, y1))
    y2 = np.ceil(min(mask.shape[1], y2))
    x1 = np.floor(max(0, x1))
    x2 = np.ceil(min(mask.shape[2], x2))
    bbox[0, :] = z1, y1, x1, z2, y2, x2
    bbox = np.tile(bbox.astype(np.int32), (config.NUM_CLASSES - 1, 1))
    # 获取整个心脏的mask、特定实例的mask和class_id。
    masks, class_ids = dataset.process_mask(mask)

    # RPN Targets
    rpn_match, rpn_bbox = build_rpn_targets(anchors, np.array([bbox[0]]), config)

    #添加到 batch
    rpn_match = rpn_match[:, np.newaxis]
    image = mold_image(image.astype(np.float32))

    return image, rpn_match, rpn_bbox, class_ids, bbox, masks


def build_rpn_targets(anchors, gt_boxes, config):
    """给定anchor和GT box，计算重叠，识别positive anchor和delta，
    并对其进行细化，以匹配相应的GT box。
    anchors: [num_anchors, (z1, y1, x1, z2, y2, x2)]
    gt_class_ids: [num_gt_boxes] 整型类id.
    gt_boxes: [num_gt_boxes, (z1, y1, x1, z2, y2, x2)]
    Returns:
    rpn_match: [N] (int32) anchor和GT box之间的匹配.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dz, dy, dx, log(dd), log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dz, dy, dx, log(dd), log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6))
    # 计算重叠 [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # 将anchor与GT box匹配
    # 如果anchor与GT box重叠 IoU >= 0.7，那么它就是 positive.
    # 如果anchor与GT box重叠 IoU < 0.3 那么它就是negative.
    # Neutral anchors 是那些不符合上述条件的，它们不影响损失函数。
    # 然而，不要让任何GT box不匹配(rare, but happens).
    # 相反，将它与最近的anchor匹配 (即便他们的最大IoU < 0.3).

    # 1. 首先设置negative anchor。如果GT box与它们匹配，它们将被覆盖在下面。避开拥挤区域的box。
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[anchor_iou_max < 0.3] = -1

    # 2.为每个GT box设置一个anchor(无论IoU值如何)。
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1

    # 3. 将高重叠的anchor设置为positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # 平衡positive和negative anchor的子样本不要让positive anchor超过一半
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # 将多余的重置为neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # negative proposals也是相同的
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # 把多余的放在neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # 对于positive anchors, 计算转换它们以匹配相应的GT box所需的位移和比例
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # 索引到rpn_bbox
    for i, a in zip(ids, anchors[ids]):
        # 最接近的 gt box ( IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # 将坐标转换为中心加 width/height.
        # GT Box
        gt_d = gt[3] - gt[0]
        gt_h = gt[4] - gt[1]
        gt_w = gt[5] - gt[2]
        gt_center_z = gt[0] + 0.5 * gt_d
        gt_center_y = gt[1] + 0.5 * gt_h
        gt_center_x = gt[2] + 0.5 * gt_w
        # Anchor
        a_d = a[3] - a[0]
        a_h = a[4] - a[1]
        a_w = a[5] - a[2]
        a_center_z = a[0] + 0.5 * a_d
        a_center_y = a[1] + 0.5 * a_h
        a_center_x = a[2] + 0.5 * a_w

        # 计算RPN应该预测的细化bbox。
        rpn_bbox[ix] = [
            (gt_center_z - a_center_z) / a_d,
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log2(gt_d / a_d),
            np.log2(gt_h / a_h),
            np.log2(gt_w / a_w),
        ]
        # 归一化
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config):
        """一个生成器，返回图像和相应的目标类id;
            bounding box deltas,和masks.
            dataset: 要从中选择数据的Dataset对象
            config: 模型配置对象
            shuffle: 如果为True，则在每个epoch之前对样本进行清洗
            augment: 如果不是None，则将给定的图像增强应用于图像
            返回一个Python生成器。在它上面调用next()时，生成器返回两个列表，输入和输出。
            列表的内容根据接收到的参数而不同:
            inputs list:
            - images: [batch, H, W, D, C]
            - image_metas: [batch, size of image meta]
            - mask: [batch, H, W, D]
            outputs list:常规训练时通常为空。
            但是如果detection_targets为True，则输出列表包含目标class_id、bbox增量和mask。
            """
        self.b = 0  # batch 索引
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.dataset = dataset
        self.config = config

    def __getitem__(self, image_index):
        image_id = self.image_ids[image_index]
        # 首先加载图像，即[H, W, D, C]。
        image = self.dataset.load_image(image_id)
        # 加载mask，首先是[H, W, D]。
        mask = self.dataset.load_mask(image_id)
        # 注意这里的window已经是(z1, y1, x1, z2, y2, x2) (z1, y1, x1, z2, y2, x2) here.
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            max_dim=self.config.IMAGE_MAX_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            mode=self.config.IMAGE_RESIZE_MODE)
        mask = utils.resize_mask(mask, scale, padding, max_dim=self.config.IMAGE_MAX_DIM,
                                 min_dim=self.config.IMAGE_MIN_DIM, crop=crop, mode=self.config.IMAGE_RESIZE_MODE)

        # Active classes
        # 不同的数据集有不同的类，所追踪该图像的数据集中支持的类。
        active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
        source_class_ids = self.dataset.source_class_ids[self.dataset.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1
        # 图像元数据
        image_meta = compose_image_meta(image_id, image.shape, window, active_class_ids)

        return image, image_meta, mask

    def __len__(self):
        return self.image_ids.shape[0]


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN(nn.Module):
    """封装3D-Mask-RCNN模型功能。"""

    def __init__(self, config, model_dir, test_flag=False):
        """config: Config类的子类
        model_dir: 保存训练日志和训练权重的目录
        """
        super(MaskRCNN, self).__init__()
        self.epoch = 0
        self.config = config
        self.model_dir = model_dir
        self.build(config=config, test_flag=test_flag)
        self.initialize_weights()

    def build(self, config, test_flag=False):
        """“构建3D-Mask-RCNN架构。"""

        # 图像大小必须能被2整除多次
        h, w, d = config.IMAGE_SHAPE[:3]
        if h / 16 != int(h / 16) or w / 16 != int(w / 16) or d / 16 != int(d / 16):
            raise Exception("Image size must be dividable by 16. Use 256, 320, 512, ... etc.")

        # 构建共享卷积层。
        # 返回每个stage最后一层的列表，总共5层。
        P3D_Resnet = backbone.P3D19(config=config)
        C1, C2, C3 = P3D_Resnet.stages()

        # Top-down Layers
        self.fpn = FPN(C1, C2, C3, out_channels=config.TOP_DOWN_PYRAMID_SIZE, config=config)

        # Generate Anchors
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                compute_backbone_shapes(
                                                                                    config, config.IMAGE_SHAPE),
                                                                                config.BACKBONE_STRIDES,
                                                                                config.RPN_ANCHOR_STRIDE)).float(),
                                requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, config.TOP_DOWN_PYRAMID_SIZE, config.RPN_CONV_CHANNELS)

        # FPN Classifier
        self.classifier = Classifier(config.TOP_DOWN_PYRAMID_SIZE, config.POOL_SIZE, config.IMAGE_SHAPE,
                                     2, config.FPN_CLASSIFY_FC_LAYERS_SIZE, test_flag)

        # FPN Mask
        self.mask = Mask(1, config.MASK_POOL_SIZE, config.NUM_CLASSES, config.UNET_MASK_BRANCH_CHANNEL, self.config.STAGE, test_flag)

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        if not config.TRAIN_BN:
            self.apply(set_bn_fix)

    def initialize_weights(self):
        """初始化模型权重"""

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex):
        """如果模型层的名称与给定的正则表达式匹配，则将其设置为可训练的。"""
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def load_weights(self, file_path):
        """修改了相应的Keras函数的版本，增加了多gpu支持，并能够从加载中排除某些层。
        exclude: 要排除的层名的列表
        """
        if os.path.exists(file_path):
            pretrained_dict = torch.load(file_path)
            self.load_state_dict(pretrained_dict, strict=True)
            print("Weight file loading success!")
        else:
            print("Weight file not found ...")

    def detect(self, images):
        """运行检测pipeline.
        images: 图像列表，可能有不同的大小。 [1, height, width, depth, channels]
        返回一个字典列表，每个图像一个字典。该词典包含:
        rois: [N, (y1, x1, z1, y2, x2, z2)]检测bbox
        class_ids: [N] 整形格式的类的IDs
        scores: [N]类id的浮动概率分数
        masks: [H, W, D, N] 实例二值mask
        将所有输出从pytorch形状转换为normal形状。
        """

        # 神经网络期待输入的mold格式
        # 已被转换为 pytorch 形式.
        start_time = time.time()
        molded_images, image_metas, windows = self.mold_inputs(images)

        # 将图像转化为torch 张量
        molded_images = torch.from_numpy(molded_images).float()

        # To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        # 换行变量
        with torch.no_grad():
            molded_images = Variable(molded_images)

        # 运行目标检测
        detections, mrcnn_mask = self.predict([molded_images, image_metas], mode='inference')
        # 转换为numpy
        detections = detections.detach().cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 5, 2).detach().cpu().numpy()
        print("detect done, using time", time.time() - start_time)

        # 检测过程
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_mask = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       [image.shape[3], image.shape[2], image.shape[0], image.shape[1]],
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "mask": final_mask,
            })

        return results

    def predict(self, inputs, mode):
        molded_images = inputs[0]
        image_metas = inputs[1]

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            # 设置batchnorm在训练期间始终处于eval模式
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        # 特征提取
        p2_out, p3_out = self.fpn(molded_images)

        rpn_feature_maps = [p2_out, p3_out]
        mrcnn_classifier_feature_maps = [p2_out, p3_out]
        mrcnn_mask_feature_maps = [molded_images, molded_images]

        # 遍历金字塔层
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # 输出连接层
        # 将level输出列表的列表转换为across levels输出列表的列表。
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # 生成proposals
        # Proposals 是 [batch, N, (z1, y1, x1, z2, y2, x2)] 在标准化坐标系中并用0填充
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                  proposal_count=proposal_count,
                                  nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                  anchors=self.anchors,
                                  config=self.config)
        if mode == 'inference':
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_classifier_feature_maps, rpn_rois)

            # 检测
            # 输出是[batch, num_detections, (z1, y1, x1, z2, y2, x2, class_id, score)] 在图像坐标中
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

            # 将box转换为标准化坐标
            h, w, d = self.config.IMAGE_SHAPE[:3]
            scale = torch.from_numpy(np.array([d, h, w, d, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :6] / scale

            # 添加回batch维度
            detection_boxes = detection_boxes.unsqueeze(0)

            # 为检测创建mask
            _, mrcnn_mask = self.mask(mrcnn_mask_feature_maps, detection_boxes)

            # 添加回batch维度
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            return [detections, mrcnn_mask]

        elif mode == 'training':
            gt_class_ids = inputs[2]  # [1, 2, ..., num_classes - 1]
            gt_boxes = inputs[3]  # [(num_classes - 1) * one_class_bbox]
            gt_masks = inputs[4]  # multi_classes masks [D, H, W, num_classes - 1]

            # 标准坐标系
            h, w, d = self.config.IMAGE_SHAPE[:3]
            scale = torch.from_numpy(np.array([d, h, w, d, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            # 生成检测目标
            # 对proposal进行抽样并生成训练的目标输出
            p_rois, rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if rois.size()[0] == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_mask_logits = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_mask_logits = mrcnn_mask_logits.cuda()
            elif p_rois.size()[0] == 0:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_classifier_feature_maps, rois)
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_mask_logits = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_mask_logits = mrcnn_mask_logits.cuda()

            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_classifier_feature_maps, rois)

                # 为检测创建mask
                mrcnn_mask_logits, mrcnn_mask = self.mask(mrcnn_mask_feature_maps, p_rois)

            return [rpn_class_logits, rpn_bbox,
                    target_class_ids, mrcnn_class_logits,
                    target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs):
        """训练模型.
        train_dataset, val_dataset: 训练和验证数据集对象。
        learning_rate:训练的学习率
        epochs: 训练周期数。请注意，之前的训练阶段被认为已经完成，
                所以这实际上决定了总的训练阶段，而不是在这个特定的调用中。
        """
        layers = ".*"  # set all the layers trainable

        # 数据生成器
        train_set = Dataset(train_dataset, self.config)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=23)
        val_set = Dataset(val_dataset, self.config)
        val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=10)

        # Train
        self.set_trainable(layers)

        # 优化器
        # 加入L2正则化
        # 跳过批处理归一化层的gamma和beta权重。
        trainables_wo_bn = [param for name, param in self.named_parameters()
                            if param.requires_grad and 'bn' not in name]
        trainables_only_bn = [param for name, param in self.named_parameters()
                              if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        total_start_time = time.time()
        start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if not os.path.exists("./logs/heart/" + str(start_datetime)):
            os.makedirs("./logs/heart/" + str(start_datetime))
        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}".format(epoch))
            start_time = time.time()
            # Training
            angle = np.random.randint(-20, 21)
            loss, loss_rpn_class, loss_rpn_bbox, \
                loss_mrcnn_class, loss_mrcnn_bbox = \
                self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH, angle, train_dataset)

            print("One Training Epoch time:", int(time.time() - start_time),
                  "Total time:", int(time.time() - total_start_time))
            torch.cuda.empty_cache()


            best_val_loss = float('inf')
            best_model_path = ""

            if epoch % 5 == 0:
                # Validation
                val_loss, val_loss_rpn_class, val_loss_rpn_bbox, \
                val_loss_mrcnn_class, val_loss_mrcnn_bbox = \
                    self.valid_epoch(val_generator, self.config.VALIDATION_STEPS, angle, val_dataset)

            for epoch in range(epoch % 5 == 0):
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = "./logs/heart/best_model.pth"
                        torch.save(self.state_dict(), best_model_path)

            if best_model_path:
                self.load_state_dict(torch.load(best_model_path))

        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps, angle, dataset):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_mrcnn_mask_edge_sum = 0
        step = 0

        optimizer.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            image = inputs[0]
            image_metas = inputs[1]
            mask = inputs[2]

            image = image.squeeze(0).cpu().numpy()
            mask = mask.squeeze(0).cpu().numpy()

            images, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = load_image_gt(image, mask, angle, dataset,
                                                                                          self.config,
                                                                                          self.anchors.cpu().numpy())

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # Wrap in variables
            images = Variable(torch.from_numpy(images).float().unsqueeze(0))
            rpn_match = Variable(torch.from_numpy(rpn_match).unsqueeze(0))
            rpn_bbox = Variable(torch.from_numpy(rpn_bbox).float().unsqueeze(0))
            gt_class_ids = Variable(torch.from_numpy(gt_class_ids).unsqueeze(0))
            gt_boxes = Variable(torch.from_numpy(gt_boxes).float().unsqueeze(0))
            gt_masks = Variable(torch.from_numpy(gt_masks).float().unsqueeze(0))

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, \
                mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')


            # Compute losses
            rpn_class_loss, rpn_bbox_loss, \
            mrcnn_class_loss, mrcnn_bbox_loss = \
                compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                               mrcnn_class_logits, target_deltas, mrcnn_bbox)

            loss = self.config.LOSS_WEIGHTS["rpn_class_loss"] * rpn_class_loss + \
                self.config.LOSS_WEIGHTS["rpn_bbox_loss"] * rpn_bbox_loss + \
                self.config.LOSS_WEIGHTS["mrcnn_class_loss"] * mrcnn_class_loss + \
                self.config.LOSS_WEIGHTS["mrcnn_bbox_loss"] * mrcnn_bbox_loss #+ \


            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0


            print_progress_bar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                               suffix="rpn_roi_iou_max: {:.5f} - Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f}"
                                      " - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f}"

                               .format(roi_iou_max.max(),loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["rpn_class_loss"] * rpn_class_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["rpn_bbox_loss"] * rpn_bbox_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["mrcnn_class_loss"] * mrcnn_class_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["mrcnn_bbox_loss"] * mrcnn_bbox_loss.detach().cpu().item()
                                       ),
                               length=45)


            #统计数据
            loss_sum += loss.detach().cpu().item() / steps
            loss_rpn_class_sum += rpn_class_loss.detach().cpu().item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.detach().cpu().item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.detach().cpu().item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.detach().cpu().item() / steps

            #在“steps”之后break
            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, \
            loss_mrcnn_class_sum, loss_mrcnn_bbox_sum
    def valid_epoch(self, datagenerator, steps, angle, dataset):

        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0


        for inputs in datagenerator:

            image = inputs[0]
            image_metas = inputs[1]
            mask = inputs[2]

            image = image.squeeze(0).cpu().numpy()
            mask = mask.squeeze(0).cpu().numpy()

            images, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = load_image_gt(image, mask, angle, dataset,
                                                                                          self.config,
                                                                                          self.anchors.cpu().numpy())

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # Wrap in variables
            images = Variable(torch.from_numpy(images).float().unsqueeze(0))
            rpn_match = Variable(torch.from_numpy(rpn_match).unsqueeze(0))
            rpn_bbox = Variable(torch.from_numpy(rpn_bbox).float().unsqueeze(0))
            gt_class_ids = Variable(torch.from_numpy(gt_class_ids).unsqueeze(0))
            gt_boxes = Variable(torch.from_numpy(gt_boxes).float().unsqueeze(0))
            gt_masks = Variable(torch.from_numpy(gt_masks).float().unsqueeze(0))

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, \
                mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            if target_class_ids.size()[0] == 0:
                continue


            # Compute losses
            rpn_class_loss, rpn_bbox_loss, \
                mrcnn_class_loss, mrcnn_bbox_loss = \
                compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                               mrcnn_class_logits, target_deltas, mrcnn_bbox)


            loss = self.config.LOSS_WEIGHTS["rpn_class_loss"] * rpn_class_loss + \
                   self.config.LOSS_WEIGHTS["rpn_bbox_loss"] * rpn_bbox_loss + \
                   self.config.LOSS_WEIGHTS["mrcnn_class_loss"] * mrcnn_class_loss + \
                   self.config.LOSS_WEIGHTS["mrcnn_bbox_loss"] * mrcnn_bbox_loss


            # Progress
            print_progress_bar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                               suffix="rpn_roi_iou_max: {:.5f} - Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f}"
                                      " - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f}"

                               .format(roi_iou_max.max(),loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["rpn_class_loss"] * rpn_class_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["rpn_bbox_loss"] * rpn_bbox_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["mrcnn_class_loss"] * mrcnn_class_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["mrcnn_bbox_loss"] * mrcnn_bbox_loss.detach().cpu().item()
                                       ),
                               length=10)


            # 数据统计
            loss_sum += loss.detach().cpu().item() / steps
            loss_rpn_class_sum += rpn_class_loss.detach().cpu().item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.detach().cpu().item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.detach().cpu().item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.detach().cpu().item() / steps


            # 在step之后break
            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, \
              loss_mrcnn_class_sum, loss_mrcnn_bbox_sum

    def mold_inputs(self, images):
        """获取一个图像列表，并将其修改为预期的格式，作为神经网络的输入。
        images: 图像矩阵列表[height, width, depth, channels]. 图像可以有不同大小
        返回3个Numpy矩阵::
        molded_images: [N, 1, d, h, w]. 图像大小调整和标准化。.
        image_metas: [N, length of meta data]. 每个图像的详细信息。
        windows: [N, (z1, y1, x1, z2, y2, x2)]. 图像中包含原始图像的部分（不包括填充）
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # 调整图像大小以适应模型期望的大小
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image)
            molded_image = molded_image.transpose((3, 2, 0, 1))  # [C, D, H, W]
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # 打包成数组
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """将一个图像的检测从神经网络输出的格式重新格式化为适合应用程序其余部分使用的格式。
        detections: [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
        mrcnn_mask: [N, depth, height, width, num_classes]
        image_shape: [channels, depth, height, width] 调整大小前的图像原始大小
        window: [z1, y1, x1, z2, y2, x2] Box中除去填充的真实图像。
        Returns:
        boxes: [N, (y1, x1, z1, y2, x2, z2)] 以像素为单位的bbox
        class_ids: [N] 每个bbox的整数类id
        scores: [N] 浮点型class_id的概率分数
        masks: [height, width, depth]正常形状的完整mask
        """
        start_time = time.time()
        # 有多少次检测?
        # 检测数组用零填充。查找第一个class_id == 0。
        zero_ix = np.where(detections[:, 6] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :6].astype(np.int32)
        class_ids = detections[:N, 6].astype(np.int32)
        scores = detections[:N, 7]
        masks = mrcnn_mask[np.arange(N), :, :, :, :]

        # 计算缩放和移位，将bbox转换到图像域。
        d_scale = image_shape[1] / (window[3] - window[0])
        h_scale = image_shape[2] / (window[4] - window[1])
        w_scale = image_shape[3] / (window[5] - window[2])
        shift = window[:3]  # z, y, x
        scales = np.array([d_scale, h_scale, w_scale, d_scale, h_scale, w_scale])
        shifts = np.array([shift[0], shift[1], shift[2], shift[0], shift[1], shift[2]])
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # 过滤掉零区域的检测。通常只发生在训练的早期阶段，此时网络权重仍然有点随机。
        exclude_ix = np.where(
            (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)

        # 将mask大小调整为原始图像大小。
        full_masks = utils.unmold_mask(masks[0], boxes[0], image_shape)
        full_mask = np.argmax(full_masks, axis=3)

        # 将bbox的形状转换为正常形状。
        boxes[:, [0, 1, 2, 3, 4, 5]] = boxes[:, [1, 2, 0, 4, 5, 3]]
        print("unmold done, using time", time.time() - start_time)

        return boxes, np.arange(1, 8), scores, full_mask.transpose((1, 2, 0))


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """获取图像的属性并将其放入一个1维数组中。使用parse_image_meta()解析返回的值。
    image_id: 图像的int型ID。用于调试。
    image_shape: [channels, depth, height, width]
    window: (z1, y1, x1, z2, y2, x2)以像素为单位。真实图像所在的图像体积(不包括填充)
    active_class_ids:图像来自的数据集中可用的class_ids列表。
        如果训练来自多个数据集的图像，而不是所有的类都存在于所有数据集中，则非常有用。
    """
    meta = np.array(
        [image_id] +            # size = 1
        list(image_shape) +     # size = 4
        list(window) +          # size = 6: (z1, y1, x1, z2, y2, x2) 在图像坐标中
        list(active_class_ids)  # size = num_classes
    )
    return meta


def parse_image_meta(meta):
    """将图像信息Numpy数组解析为其组件。
    参见compose_image_meta()了解更多细节。
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:5]
    window = meta[:, 5:11]   # (z1, y1, x1, z2, y2, x2) window 以像素为单位的图像
    active_class_ids = meta[:, 11:]
    return image_id, image_shape, window, active_class_ids


def mold_image(images):
    """将输入图像归一化，使其均值= 0,std = 1。"""
    return (images - images.mean()) / images.std()
