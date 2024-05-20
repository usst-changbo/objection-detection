"""
基本配置
"""

import numpy as np

class Config(object):

    NAME = "heart"
    GPU_COUNT = 1

    # 在每个GPU上训练的图像数量。
    # 一个12GB的GPU通常可以处理2张1024x1024pix的图像。
    # 根据GPU内存和图像大小进行调整。使用GPU可以处理的最高数字以获得最佳性能。
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 40
    VALIDATION_STEPS = 10
    BACKBONE = "P3D19"

    # FPN金字塔每一层的步幅。这些值基于Resnet101特征提取网络。
    BACKBONE_STRIDES = [8, 16]
    BACKBONE_CHANNELS = [16,32]

    # classification graph中全连接层的大小
    FPN_CLASSIFY_FC_LAYERS_SIZE = 128

    UNET_MASK_BRANCH_CHANNEL = 20

    # Mask graph中conv输出的通道
    FPN_MASK_BRANCH_CHANNEL = 256

    # 用于构建FPN特征金字塔的自顶向下层的大小
    TOP_DOWN_PYRAMID_SIZE = 128

    # RPN卷积层的通道数
    RPN_CONV_CHANNELS = 256

    # 类别数目(包括背景)
    NUM_CLASSES = 4

    # 正方形anchor锚边的长度(以像素为单位)
    RPN_ANCHOR_SCALES = (96, 128)

    # 每个cell中anchor的比率
    # 值1表示正方形anchor
    RPN_ANCHOR_RATIOS = [1]

    # 如果是1，则为backbone feature map中的每个cell创建anchor
    # 如果是2，则为每个其他cell创建anchor，以此类推
    RPN_ANCHOR_STRIDE = 1

    # 过滤RPN proposals的非最大抑制阈值。
    RPN_NMS_THRESHOLD = 0.7

    # 每个图像使用多少个anchor用于RPN训练
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128
    PRE_NMS_LIMIT = 1000

    # 非最大抑制(训练和测试)后roi数量
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 200

    # 如果启用，则将instance masks大小调整为较小的大小，以减少内存负载。
    # 使用高分辨率图像时推荐使用
    USE_MINI_MASK = False

    # 一般来说，使用“正方形”调整大小进行训练和预测，
    # 使得小边= IMAGE_MIN_DIM，但确保缩放不会使长边> IMAGE_MAX_DIM。
    # 然后用零填充图像，使其成为一个正方形，以便将多个图像放入一个批次中。
    # 可用的调整大小模式:
    # none:   没有调整大小或填充，返回图像不变。
    # square: 调整大小并填充零以获得大小为[max_dim, max_dim, max_dim]的方形图像。
    # pad64:  用零填充宽度和高度，使它们成为64的倍数。
    #         如果IMAGE_MIN_DIM或IMAGE_MIN_SCALE不是None，则在填充之前按比例增大。
    #         在此模式下忽略IMAGE_MAX_DIM。
    #         需要64的倍数来保证特征的平滑缩放在FPN金字塔的6个层次上下映射(2**6=64)。
    # crop:   从图像中选择随机裁剪。
    #         首先，根据IMAGE_MIN_DIM和IMAGE_MIN_SCALE缩放图像，
    #         然后随机选择大小为IMAGE_MIN_DIM x IMAGE_MIN_DIM的裁剪。
    #         只能用于训练模式，此模式不使用IMAGE_MAX_DIM。
    # self:   自行设计的调整策略。
    #         将图像大小调整为[IMAGE_MAX_DIM, IMAGE_MAX_DIM, IMAGE_MIN_DIM, 1]
    IMAGE_RESIZE_MODE = "self"
    IMAGE_MIN_DIM = 16
    IMAGE_MAX_DIM = 256
    # 最小缩放比。在MIN_IMAGE_DIM之后检查，可以强制进一步向上缩放。
    # 如果设置为2，则图像将缩放为宽度和高度的两倍，甚至更多，即使MIN_IMAGE_DIM不需要它。
    # 然而，在'square'模式下，它可以被IMAGE_MAX_DIM推翻。
    IMAGE_MIN_SCALE = 0
    IMAGE_CHANNEL_COUNT = 1

    # 输入到分类器/mask heads的每张图像的roi数
    # mask-rcnn论文使用512，但RPN通常不能生成足够多的positive proposals来填充这个值，
    # 并保持positive:negative=1:3的正负比,可以增加proposals数，调整RPN的NMS阈值。
    TRAIN_ROIS_PER_IMAGE = 10

    # 用于训练分类器/mask heads的positive roi百分比
    ROI_POSITIVE_RATIO = 0.3

    # Pooled ROIs
    POOL_SIZE = [12, 12, 12]
    MASK_POOL_SIZE = [96, 96, 96]

    # 在一张图像中使用的ground truth实例的最大数量
    MAX_GT_INSTANCES = 32

    # RPN和最终检测的BBox细化标准偏差。
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])

    # 最终检测的最大数目
    DETECTION_MAX_INSTANCES = 32

    # 接受检测到的实例的最小概率值，低于该阈值的roi将被跳过
    DETECTION_MIN_CONFIDENCE = 0.4

    # 检测的非最大抑制阈值
    DETECTION_NMS_THRESHOLD = 0.3

    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    # 权衰减正则化
    WEIGHT_DECAY = 0.0001
    LOSS_WEIGHTS = {
        "rpn_class_loss": 100.,
        "rpn_bbox_loss": 50.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 50.,
        "mrcnn_mask_loss": 0.,
        "mrcnn_mask_edge_loss": 0.
    }

    # 使用RPN roi或外部生成的roi进行培训,大多数情况下都是True。
    # 如果想训练由代码生成的ROI而不是来自RPN的ROI的head分支，则将其设置为False。
    # 例如，调试classifier head而不必训练RPN。
    USE_RPN_ROIS = True

    # 训练或者冻批归一化层
    # None: 训练BN层，这是普通模式
    # False: 冻结BN层。当使用小批量时效果很好
    # True: (不使用)，即使在预测时也将图层设置为训练模式
    TRAIN_BN = False  # 默认为False，因为批处理大小通常很小

    # 梯度范数裁剪
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self, stage):
        """设置计算属性的值。"""
        # 有效batch的大小
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # 输入图像尺寸: [height, width, depth, channels]
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 1])
        elif self.IMAGE_RESIZE_MODE == "self":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_MIN_DIM, 1])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 1])

        # 元数据长度
        # 参考compose_image_meta()了解详细信息
        self.IMAGE_META_SIZE = 1 + 4 + 4 + 6 + 1 + self.NUM_CLASSES
        self.STAGE = stage
        if stage == 'finetune':
            self.MINI_MASK_SHAPE = (192, 192, 192)
            self.MASK_SHAPE = (192, 192, 192)
            self.DETECTION_TARGET_IOU_THRESHOLD = 0.5
        else:
            self.MINI_MASK_SHAPE = (96, 96, 96)
            self.MASK_SHAPE = (96, 96, 96)
            self.DETECTION_TARGET_IOU_THRESHOLD = 0.5

    def display(self):
        """显示配置"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
