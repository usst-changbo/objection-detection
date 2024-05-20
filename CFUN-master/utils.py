"""
该模型中常用的函数和类。
"""

import torch
import numpy as np
import nibabel as nib
import skimage.transform
from distutils.version import LooseVersion
import torch.nn.functional as F


############################################################
#  边界框
############################################################

def compute_3d_iou(box1, box2):
    """计算三维数据的iou。box1和box2是坐标，格式（x1,y1,z1,x2,y2,z2）"""
    box1 = np.array(box1)
    box2 = np.array(box2)
    vol1 = (box1[3] - box1[0] + 1) * (box1[4] - box1[1] + 1) * (box1[5] - box1[2] + 1)
    vol2 = (box2[3] - box2[0] + 1) * (box2[4] - box2[1] + 1) * (box2[5] - box2[2] + 1)
    inter_x1 = np.maximum(box1[0], box2[0])
    inter_y1 = np.maximum(box1[1], box2[1])
    inter_z1 = np.maximum(box1[2], box2[2])
    inter_x2 = np.minimum(box1[3], box2[3])
    inter_y2 = np.minimum(box1[4], box2[4])
    inter_z2 = np.minimum(box1[5], box2[5])
    inter_vol = np.clip(inter_x2 - inter_x1 + 1, a_min=0, a_max=None) * np.clip(inter_y2 - inter_y1 + 1, a_min=0,
        a_max=None) * np.clip(inter_z2 - inter_z1 + 1, a_min=0, a_max=None)
    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-6)

    return iou


def compute_3d_bounding_box(label):
    """计算nii三维数据标签的边框"""

    indices = np.nonzero(label)
    x1 = np.min(indices[0])
    y1 = np.min(indices[1])
    z1 = np.min(indices[2])
    x2 = np.max(indices[0])
    y2 = np.max(indices[1])
    z2 = np.max(indices[2])
    return (x1, y1, z1, x2, y2, z2)


def extract_bboxes(mask):
    """ 从mask中提取边界框，存在一个数组中
        mask: [depth, height, width, num_instances]. mask像素为0或1.
        Returns: bbox 数组 [num_instances, (z1, y1, x1, z2, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 6], dtype=np.int32)
    for i in range(mask.shape[-1]):
        ix = np.where(np.sum(mask, axis=2) > 0)
        z1 = ix[0].min()
        z2 = ix[0].max()
        y1 = ix[1].min()
        y2 = ix[1].max()
        ix = np.where(np.sum(mask, axis=0) > 0)
        x1 = ix[1].min()
        x2 = ix[1].max()
        # X2 y2 z2不应该是box的一部分。增量为1。
        if z1 != z2:
            z2 += 1
            x2 += 1
            y2 += 1
        else:
            # 实例没有mask，可能由于调整大小或裁剪而发生。将bbox设置为零
            x1, x2, y1, y2, z1, z2 = 0, 0, 0, 0, 0, 0
        boxes[i] = np.array([z1, y1, x1, z2, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_volume, boxes_volume):
    """ 用给定box和boxes数组计算IoU。
        box: 一维向量 [z1, y1, x1, z2, y2, x2]
        boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
        box_volume: float. box的体积
        boxes_volume: boxes的体积
    """
    z1 = np.maximum(box[0], boxes[:, 0])
    z2 = np.minimum(box[3], boxes[:, 3])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[4], boxes[:, 4])
    x1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / (union + 1e-6)
    return iou


def compute_overlaps(boxes1, boxes2):
    """计算两组box之间重叠的iou。
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].
    为了获得更好的性能，首先传递最大的集合，然后传递较小的集合。
    """
    # anchors 和 GT boxes的volumes
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    # 计算重叠矩阵 [boxes1 count, boxes2 count]
    # 每个cell包含IoU值。
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, volume2[i], volume1)
    return overlaps


def box_refinement(box, gt_box):
    """计算将box转化为gt_box所需的细化偏移。
       box和gt_box为[N， (z1, y1， ×1, z2, y2, x2)]
    """

    depth = box[:, 3] - box[:, 0]
    height = box[:, 4] - box[:, 1]
    width = box[:, 5] - box[:, 2]
    center_z = box[:, 0] + 0.5 * depth
    center_y = box[:, 1] + 0.5 * height
    center_x = box[:, 2] + 0.5 * width

    gt_depth = gt_box[:, 3] - gt_box[:, 0]
    gt_height = gt_box[:, 4] - gt_box[:, 1]
    gt_width = gt_box[:, 5] - gt_box[:, 2]
    gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height
    gt_center_x = gt_box[:, 2] + 0.5 * gt_width

    dz = (gt_center_z - center_z) / depth
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dd = torch.log(gt_depth / depth)
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dz, dy, dx, dd, dh, dw], dim=1)
    return result


def non_max_suppression(boxes, scores, threshold, max_num):
    """ 执行非最大抑制并返回保留box的索引。
        boxes: [N, (z1, y1, x1, z2, y2, x2)]. 注意(z2, y2, x2)在box外面.
        scores: box分数的一维数组.
        threshold: 浮点型. 用于优化器的IoU阈值
        max_num: 整型. 要保留box的最大数量
        返回保留box的索引值.
    """
    # 计算box体积
    z1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x1 = boxes[:, 2]
    z2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x2 = boxes[:, 5]
    volume = (z2 - z1) * (y2 - y1) * (x2 - x1)

    # 获取按分数排序的box的索引(最高优先)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # 选择top box并将其索引添加到列表中
        i = ixs[0]
        pick.append(i)
        if len(pick) >= max_num:
            break
        # 计算选中的box的IoU和其余的
        iou = compute_iou(boxes[i], boxes[ixs[1:]], volume[i], volume[ixs[1:]])
        # 识别IoU超过阈值的box。返回索引到ixs[1:]，
        # 加1得到索引为ixs。
        remove_ixs = np.where(iou > threshold)[0] + 1
        # 移除选中重叠框的索引。
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def denorm_boxes_graph(boxes, size):
    """ 将box从归一化坐标转换为像素坐标（反归一化）。
        boxes: [..., (z1, y1, x1, z2, y2, x2)] 在标准化坐标系中
        size: [depth, height, width], 调整到的大小.
        Note: 在像素坐标中(z2, y2, x2)在box外。但是在标准化坐标系中，它在box内。
        Returns:[..., (z1, y1, x1, z2, y2, x2)] 以像素坐标表示
    """
    d, h, w = size
    scale = torch.Tensor([d, h, w, d, h, w]).cuda()
    denorm_boxes = torch.mul(boxes, scale)
    return denorm_boxes


############################################################
#  数据集
############################################################

class Dataset(object):
    """ 数据集类的标准类。要使用它需要创建一个新类，
        该类添加特定于要使用的数据集的函数。例如:
        class CatsAndDogsDataset(Dataset):
            def load_cats_and_dogs(self):
            ...
            def load_mask(self, image_id):
            ...
            def image_reference(self, image_id):
            ...
        参见COCO数据集和Shapes数据集作为示例。
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # 背景永远是第一个类
        self.class_info = [{"source": "heart", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        # 添加类别
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """ 返回到图片源网站的链接或图片的详细信息，以帮助查找或调试图片。
            覆盖数据集，但如果遇到不在数据集中的图像，则传递给此函数。
        """
        return ""

    def prepare(self, class_map=None):
        """ 准备Dataset类以供使用。
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """返回对象名称的较短版本，以便更清晰地显示。"""
            return ",".join(name.split(",")[:1])

        # 从信息字典中构建(或重建)其他所有内容。
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # 将source映射到它们支持的class_ids
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # 循环遍历数据集
        for source in self.sources:
            self.source_class_ids[source] = []
            # 查找属于此数据集的类
            for i, info in enumerate(self.class_info):
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """获取source类ID并返回分配给它的整型ID。
            例如:
            dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """ 将内部class ID映射到源数据集中相应的类ID"""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # 将外部图像id映射到内部id。
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """返回图像的路径或URL。覆盖它以返回一个URL到图像。
           如果它是在线可用的，以便于调试。
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """加载指定的图像并返回一个[H, W, D, 1] Numpy数组。"""
        # 加载图像
        image = nib.load(self.image_info[image_id]['path']).get_data().copy()
        return np.expand_dims(image, -1)

    def load_mask(self, image_id):
        """加载指定的mask并返回一个[H, W, D] Numpy数组"""
        # 重写此函数以从数据集加载mask。否则，它返回一个空的 mask。
        mask = np.empty([0, 0, 0])
        return mask


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None):
    """ Scikit-Image resize()的包装器。.
        Scikit-Image在每次调用resize()时，如果没有接收到正确的参数，就会产生警告。
        正确的参数取决于skimage的版本。这通过在每个版本中使用不同的参数解决了这个问题。
        它还提供了一个控制默认大小调整的中心位置。
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """ 调整图像的大小，保持长宽比。
        min_dim: 调整图像的大小，使其更小边 == min_dim
        max_dim: 确保图像最长边不超过此值。
        min_scale: 如果提供，即使min_dim不需要，也要确保图像至少按这个百分比缩放。
        mode: 调整模式。
            none:   没有调整。返回原图像。
            square: 调整大小并填充0以获得大小为[max_dim, max_dim, max_dim]的方形图像。
            pad64:  用0填充宽度和高度，使它们成为64的倍数。
                    如果提供了min_dim或min_scale，则在填充之前将图像向上缩放。在这种模式下，Max_dim被忽略。
                    需要64的倍数来确保特征映射在FPN金字塔的6个层次(2**6=64)上下平滑缩放。
            crop:   从图像中选择随机裁剪。首先，根据min_dim和min_scale缩放图像，
                    然后选择大小为min_dim x min_dim的随机裁剪。只能用于训练。该模式不使用Max_dim。
            self:   自行设计的调整策略。
                    将图像大小调整为[IMAGE_MAX_DIM, IMAGE_MAX_DIM, IMAGE_MIN_DIM]

        Returns:
        image: 调整图像[height, width, depth, channels]
        window: (z1, y1 x1, z2, y2, x2)如果提供了max_dim，则可以在返回的图像中插入填充。
                如果是，则此窗口是完整图像的图像部分的坐标(不包括填充)。不包括x2, y2, z2像素。
        scale:  用于调整图像大小的比例因子
        padding: 为图像添加填充[(top, bottom), (left, right), (front, back), (0, 0)]
    """
    # 追踪图像 dtype 并以相同的 dtype 返回结果
    image_dtype = image.dtype
    # 默认window (z1, y1, x1, z2, y2, x2)并且默认 scale == 1.
    h, w, d = image.shape[:3]
    window = (0, 0, 0, d, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # 自行设计的调整策略。
    if mode == "self":
        image = resize(image, (max_dim, max_dim, min_dim, 1),
                       order=1, mode="constant", preserve_range=True)
        return image.astype(image_dtype), (0, 0, 0, min_dim, max_dim, max_dim), -1, \
            [(0, 0), (0, 0), (0, 0), (0, 0)], crop


def resize_mask(mask, scale, padding, max_dim=0, min_dim=0, crop=None, mode="square"):
    """ 使用给定的比例和填充调整mask的大小。
        通常，可以从resize_image()中获取缩放和填充，以确保图像和mask都被一致地调整大小。
        scale: mask比例系数
        padding: 填充添加到mask的形式[(top, bottom), (left, right), (front, back), (0, 0)]
    """
    # 自行设计的调整策略。
    if mode == "self":
        mask = resize(mask, (max_dim, max_dim, min_dim), order=0, mode='constant', preserve_range=True)
        return np.round(mask).astype(np.int32)


def minimize_mask(bbox, mask, mini_shape):
    """ 将mask大小调整较小以减少内存负载。
        然后Mini-masks可以使用expand_mask()将大小调整回图像比例
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, :, i]
        z1, y1, x1, z2, y2, x2 = bbox[i][:6]
        m = m[z1:z2, y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with volume of zero")
        m = skimage.transform.resize(m, mini_shape, order=0, mode="constant", preserve_range=True)
        mini_mask[:, :, :, i] = np.around(m).astype(np.int32)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """将mini_masks调整回图像大小。反转minimize_mask()的更改。"""
    mask = np.zeros(image_shape[:3] + (mini_mask.shape[-1],), dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, :, i]
        z1, y1, x1, z2, y2, x2 = bbox[i][:6]
        d = z2 - z1
        h = y2 - y1
        w = x2 - x1
        m = skimage.transform.resize(m, (d, h, w), order=1, mode="constant", preserve_range=True)
        mask[z1:z2, y1:y2, x1:x2, i] = np.around(m).astype(np.int32)
    return mask


def unmold_mask(mask, bbox, image_shape):
    """ 将给定的mask根据边框还原到指定图像尺寸上
        mask: [depth, height, width, num_instances] 是float类型. 典型的是28x28的mask。
        bbox: [z1, y1, x1, z2, y2, x2]. 用来匹配mask的box.
        image_shape: [channels, depth, height, width]
        返回与原始图像大小相同的tf.int32的mask。
    """
    z1, y1, x1, z2, y2, x2 = bbox
    mask = torch.from_numpy(mask).float().cuda()
    mask = mask.permute(3, 0, 1, 2).unsqueeze(0)
    mask = F.interpolate(mask, size=(z2 - z1, y2 - y1, x2 - x1), mode='trilinear', align_corners=False)
    mask = mask.squeeze(0).detach().cpu().numpy().transpose(1, 2, 3, 0)
    # 把mask放在正确的位置。
    full_mask = np.zeros((image_shape[1], image_shape[2], image_shape[3], mask.shape[-1]), dtype=np.float32)
    full_mask[z1:z2, y1:y2, x1:x2, :] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """ 生成anchors的坐标
        scales: 以像素为单位的anchor大小的1D数组。示例:[32,64,128]
        ratios: anchor宽度/高度比例的一维阵列。例如:[1]
        shape: [depth, height, width]要在其上生成anchor的特征图的空间形状。
        feature_stride: 特征映射相对于图像的步幅(以像素为单位)。
        anchor_stride: 特征映射上的anchor的步数。例如，如果值为2，则为每个其他特征图像素生成anchor。
    """
    # 获得所有比例和比率的组合
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # 从scales和ratios中列举高度和宽度
    # TODO: conditions when we have different ratios?
    depths = scales
    heights = scales
    widths = scales

    # 枚举特征空间中的位移
    shifts_z = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_y = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[2], anchor_stride) * feature_stride
    shifts_z, shifts_y, shifts_x = np.meshgrid(shifts_z, shifts_y, shifts_x)

    # 枚举移位、宽度和高度的组合
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape得到一个列表(z, y, x)和一个列表(d, h, w)
    box_centers = np.stack(
        [box_centers_z, box_centers_y, box_centers_x], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_depths, box_heights, box_widths], axis=2).reshape([-1, 3])

    # 转换为坐标(z1, y1， ×1, z2, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """ 在特征金字塔的不同层次上生成anchor。
        每个比例都与金字塔的一个层相关联，但每个比例在金字塔的所有层中都使用。
        Returns:
        anchors: [N， (z1, y1, x1, z2, y2, x2)]。
                所有生成的anchor都在一个数组中。按给定尺度的相同顺序排序。
                因此，scale为[0]的anchor首先出现，然后是scale为[1]的anchor，以此类推。
    """

    anchors = [] # [anchor_count, (z1, y1, x1, z2, y2, x2)]
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  其他函数
############################################################

# 批处理切片一些自定义层只支持batch size大小为1，并且需要大量的工作来支持大于1的batch size
# 这个函数在批处理维度上分割输入张量，并提供大小为1的批。
# 实际上是一种简单的方法，只需少量代码修改即可快速支持batch size> 1。
# 从长远来看，修改代码以支持大规模批处理并去掉这个函数会更有效。将此视为临时解决方案
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """ 将输入分割成片，并将每个片提供给给定计算图的副本，然后组合结果。
        它允许在一批输入上运行图形，即使图形只支持一个实例。
        inputs: 张量的列表。所有都必须具有相同的第一维尺寸
        graph_fn:一个函数返回一个TF张量，它是图的一部分。
        batch_size: 将数据划分到的片数。
        names: 如果提供，则为结果张量分配名称。
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # 将输出从一个片列表(每个片都是一个输出列表)更改为一个输出列表(每个输出列表都有一个片列表)
    outputs = list(zip(*outputs))
    if names is None:
        names = [None] * len(outputs)
    result = [torch.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def compute_per_class_mask_iou(gt_masks, pred_masks):
    """ 计算两组masks之间的per_class_IoU重叠。
        gt_masks, pred_masks: [Height, Width, Depth, instances], 如果没有这样的实例，则用0填充.
        返回每个实例的ious。
    """
    # 展开 masks 并计算面积
    gt_masks = np.reshape(gt_masks > .5, (-1, gt_masks.shape[-1])).astype(np.float32)
    pred_masks = np.reshape(pred_masks > .5, (-1, pred_masks.shape[-1])).astype(np.float32)
    area1 = np.sum(gt_masks, axis=0)
    area2 = np.sum(pred_masks, axis=0)

    # 计算交集和并集
    intersections = np.array([np.dot(gt_masks.T, pred_masks)[i, i] for i in range(gt_masks.shape[-1])])
    union = area1 + area2 - intersections
    ious = intersections / (union + 1e-6)

    return ious


def compute_mask_iou(gt_masks, pred_masks):
    """ 计算两组masks之间的IoU重叠。把不同的类别看得一样。
        gt_masks, pred_masks: [Height, Width, Depth].
        返回两个mask中的一个。
    """
    # 展开masks并计算他们的面积
    gt_masks[gt_masks > 0] = 1
    pred_masks[pred_masks > 0] = 1
    gt_masks = np.reshape(gt_masks, (-1)).astype(np.int32)
    pred_masks = np.reshape(pred_masks, (-1)).astype(np.int32)
    area1 = np.sum(gt_masks)
    area2 = np.sum(pred_masks)
    # 计算交集和并集
    intersections = np.dot(gt_masks.T, pred_masks)
    union = area1 + area2 - intersections
    ious = intersections / (union + 1e-6)

    return ious
