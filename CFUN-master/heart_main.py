"""
    执行代码入口，包括训练、验证和测试
"""
import os
import json
import time
import numpy as np
import nibabel as nib
from config import Config
import model
import utils
import warnings
warnings.filterwarnings('ignore')

############################################################
# 配置，需要和config.py匹配
############################################################

class HeartConfig(Config):

    NAME = "heart"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 4  # 背景＋类别
    STEPS_PER_EPOCH = 40
    VALIDATION_STEPS = 10
    BACKBONE = "P3D19"

    # FPN金字塔每一层的步幅
    # 这些值基于P3D-Resnet骨干网。
    BACKBONE_STRIDES = [8, 16]
    BACKBONE_CHANNELS = [16, 32]

    # classification graph中全连接层的大小
    FPN_CLASSIFY_FC_LAYERS_SIZE = 128  # 1024

    # mrcnn mask branch中U-net通道数
    UNET_MASK_BRANCH_CHANNEL = 20

    # 用于构建特征金字塔的自顶向下层的大小
    TOP_DOWN_PYRAMID_SIZE = 128  # 256

    RPN_CONV_CHANNELS = 256  # 512

    # 正方形anchor的长度（以像素为单位）
    RPN_ANCHOR_SCALES = (96, 128)

    # 如果是1，则为backbone feature map中的每个cell创建anchor
    # 如果是2，则为每个其他cell创建anchor，以此类推
    RPN_ANCHOR_STRIDE = 1  # 此处尽量使用少量的anchors

    # 每个cell中anchor的比率
    # 1表示anchor为正方形
    RPN_ANCHOR_RATIOS = [1]

    # 每个图像使用多少个anchor用于RPN训练
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128  # 256

    # 在非最大抑制之前的ROI数量
    PRE_NMS_LIMIT = 1000  # 6000

    # 在非最大抑制之后ROI数量 (训练阶段和测试（实例）阶段)
    POST_NMS_ROIS_TRAINING = 500  # 2000
    POST_NMS_ROIS_INFERENCE = 200  # 1000

    # 如果启用，则将实例 masks大小调整为较小的大小，以减少内存负载。使用高分辨率图像时推荐使用
    USE_MINI_MASK = False  # True

    # 调整输入图像大小
    # 一般来说，使用“正方形”进行训练和预测，然后用零填充图像，使其成为一个正方形，以便将多个图像放入一个批次中。
    # 可用的调整大小模式:
    # none:   没有调整大小或填充，返回图像不变。
    # square: 调整大小并填充零以获得大小为[max_dim, max_dim, max_dim]的方形图像。
    # pad64:  用零填充宽度和高度，使它们成为64的倍数。
    #         如果IMAGE_MIN_DIM或IMAGE_MIN_SCALE不是None，则在填充之前按比例增大。
    #         在此模式下忽略IMAGE_MAX_DIM,
    #         需要64的倍数来保证特征的平滑缩放在FPN金字塔的6个层次上下映射(2**6=64)。
    # crop:   从图像中选择随机裁剪。
    #         首先，根据IMAGE_MIN_DIM和IMAGE_MIN_SCALE缩放图像，
    #         然后随机选择大小为IMAGE_MIN_DIM x IMAGE_MIN_DIM的裁剪。
    #         只能用于训练模式，此模式不使用IMAGE_MAX_DIM。
    # self:   自行设计的调整策略。
    #         将图像大小调整为[IMAGE_MAX_DIM, IMAGE_MAX_DIM, IMAGE_MIN_DIM, 1]
    # 图像大小范围 [512, 512, 200-400]
    IMAGE_RESIZE_MODE = "self"
    IMAGE_MIN_DIM = 16
    IMAGE_MAX_DIM = 256
    # 最小缩放比。在MIN_IMAGE_DIM之后检查，可以强制缩放。
    # 如果设置为2，则图像将缩放为宽度和高度的两倍。
    IMAGE_MIN_SCALE = 0
    # 每个图像的颜色通道数。灰度= 1,RGB = 3, RGB-d = 4
    IMAGE_CHANNEL_COUNT = 1

    # 输入到分类器/mask heads的每张图像的roi数
    # mask-rcnn论文使用512，但RPN通常不能生成足够多的positive proposals来填充这个值，
    # 并保持positive:negative=1:3的正负比,可以增加proposals数，调整RPN的NMS阈值。
    TRAIN_ROIS_PER_IMAGE = 10  # 200

    # Pooled ROIs
    POOL_SIZE = [12, 12, 12]  # [7, 7, 7]
    MASK_POOL_SIZE = [96, 96, 96]

    # 接受检测到的实例的最小概率值，低于该阈值的roi将被跳过
    DETECTION_MIN_CONFIDENCE = 0.4

    # 检测时非最大抑制阈值
    DETECTION_NMS_THRESHOLD = 0.3

    # 在一张图像中使用的ground truth实例的最大数量
    MAX_GT_INSTANCES = 32  # 100

    # 最终检测的最大数目
    DETECTION_MAX_INSTANCES = 32  # 100

    # 为更精确的优化Loss weights，可用R-CNN训练设置。
    LOSS_WEIGHTS = {
        "rpn_class_loss": 100.,  # 在RPN中正确的检测是非常重要的!
        "rpn_bbox_loss": 50.,
        "mrcnn_class_loss": 20.,
        "mrcnn_bbox_loss": 30.,
        "mrcnn_mask_loss": 0.,
        "mrcnn_mask_edge_loss": 0.
    }

    # 训练或者冻结批归一化层
    #     None: 训练BN层，这是普通模式
    #     False: 冻结BN层。当使用小批量时效果很好
    #     True: (不使用)，即使在预测时也将图层设置为训练模式
    TRAIN_BN = False  # 默认为False，因为批处理大小通常很小


############################################################
#  Dataset
############################################################

class HeartDataset(utils.Dataset):

    def load_heart(self, subset):
        """加载心脏数据集的一个子集。
        dataset_dir: 数据集的根目录。
        subset: 要加载的子集:train或val
        """
        # 添加类
        self.add_class("heart", 1, "RV")
        self.add_class("heart", 2, "MYo")
        self.add_class("heart", 3, "LV")

        # Train 或者 validation数据集?
        assert subset in ["train", "val"]

        # 加载数据集信息
        info = json.load(open(args.data + "dataset.json"))
        info = list(info['training'])

        if subset == "train":
            info = info[40:]
        else:
            info = info[:40]

        # 添加图像和对应的标签
        for a in info:
            image = nib.load(a['image']).get_fdata().copy()
            height, width, depth = image.shape
            self.add_image(
                "heart",
                image_id=a['image'],  # 使用文件名作为图像id
                path=a['image'],
                width=width, height=height, depth=depth,
                mask=a['label'])  # 保存对应标签的路径

    def load_image(self, image_id):
        """加载指定的图像并返回一个[H,W,D,C]的Numpy数组。."""
        # 加载图像
        image = nib.load(self.image_info[image_id]['path']).get_fdata().copy()
        return np.expand_dims(image, axis=-1)

    def process_mask(self, mask):
        """给定的[D，H，W]mask可能包含许多注释类。
        生成实例标签和一个只包含心脏的标签。
        返回:
        masks:一个np.int32数组，形状[depth, height, width, instance_count]，每个实例一个标签。
        class_ids:一个包含实例标签的class_ids的一维数组.
        """
        masks = np.zeros((self.num_classes, mask.shape[0], mask.shape[1], mask.shape[2]))
        for i in range(self.num_classes):
            masks[i][mask == i] = 1

        # 返回每个实例的masks和class_id的数组。
        return masks.astype(np.int32), np.arange(1, self.num_classes, 1, dtype=np.int32)

    def load_mask(self, image_id):
        """加载指定的mask，返回一个[H,W,D] Numpy数组。
       Returns:
        masks:一个形状为 [height, width, depth]的 np.int32数组.
        """
        # 如果不是心脏数据集图像，则委托给父类。
        image_info = self.image_info[image_id]
        if image_info["source"] != "heart":
            return super(self.__class__, self).load_mask(image_id)

        # 将mask转换为形状为[height, width, depth, instance_count]的bitmap mask
        # 加载mask.
        mask = nib.load(self.image_info[image_id]['mask']).get_fdata().copy()
        return mask

    def image_reference(self, image_id):
        """返回图像路径"""
        info = self.image_info[image_id]
        if info["source"] == "heart":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """训练模型"""
    # 训练数据集
    dataset_train = HeartDataset()
    dataset_train.load_heart("train")
    dataset_train.prepare()

    # 验证数据集
    dataset_val = HeartDataset()
    dataset_val.load_heart("val")
    dataset_val.prepare()

    print("Train all layers")
    model.train_model(dataset_train, dataset_val,
                      learning_rate=config.LEARNING_RATE,
                      epochs=1000)


############################################################
#  Detection
############################################################

def new_test(model, limit, save, bbox):
    """ 测试模型.
        model: 要测试的模型.
        limit: 要使用的图像.
        save: 是否保存mask.
        bbox: 是否绘制边框.
    """
    # 加载测试数据集
    info = json.load(open(args.data + "dataset.json"))
    info = list(info['testing'])
    # 开始检测
    detect_time = 0
    # 该列表用于保存原始标签与预测标签的二维IOU
    iou_2d_list = []
    # 遍历检测所有的测试集
    for path in info[:limit]:
        path_image = path['image']
        path_label = path['label']
        image = nib.load(path_image).get_fdata().copy() # 加载并复制原始图像
        label = nib.load(path_label)                    # 加载原始图像的标签
        affine = label.affine
        label = label.get_fdata().copy()                # 复制原始标签
        label_1 = np.expand_dims(label, -1)             # 对标签在最后一个维度进行维度扩充
        image = np.expand_dims(image, -1)               # 对原图在最后一个维度进行维度扩充
        start_time = time.time()                        # 计时每一次检测
        result = model.detect([image])[0]               # 获得检测结果
        detect_time += time.time() - start_time         # 获得检测时间
        print("该图检测时间:", time.time() - start_time)
        rois = result["rois"]
        # 冻结最后一个维度用于后面计算
        image = np.squeeze(image, -1)
        label_1 = np.squeeze(label_1, -1)

        # 保存结果
        if save == "true":
            # 绘制 bboxes
            if bbox == "true":
                y1, x1, z1, y2, x2, z2 = rois[0, :]
                z2 = z2-1
                # 若为0,则用于后续计算相似度，若为255,则用于观察边框
                image[y1, x1:x2, z1] = 0
                image[y1, x1:x2, z2] = 0
                image[y2, x1:x2, z1] = 0
                image[y2, x1:x2, z2] = 0
                image[y1:y2, x1, z1] = 0
                image[y1:y2, x2, z1] = 0
                image[y1:y2, x1, z2] = 0
                image[y1:y2, x2, z2] = 0
                image[y1, x1, z1:z2] = 0
                image[y1, x2, z1:z2] = 0
                image[y2, x1, z1:z2] = 0
                image[y2, x2, z1:z2] = 0

                # 如果测试阶段没有标签文件，则需要删掉这些代码
                # 若为0,则用于后续计算相似度，若为1,则用于观察边框
                label_1[y1, x1:x2, z1] = 0
                label_1[y1, x1:x2, z2] = 0
                label_1[y2, x1:x2, z1] = 0
                label_1[y2, x1:x2, z2] = 0
                label_1[y1:y2, x1, z1] = 0
                label_1[y1:y2, x2, z1] = 0
                label_1[y1:y2, x1, z2] = 0
                label_1[y1:y2, x2, z2] = 0
                label_1[y1, x1, z1:z2] = 0
                label_1[y1, x2, z1:z2] = 0
                label_1[y2, x1, z1:z2] = 0
                label_1[y2, x2, z1:z2] = 0
            # 保存标注坐标框后的nii图像
            vol = nib.Nifti1Image(image.astype(np.int32), affine)
            # 按照bboxes裁剪原始图像
            cropped_image = vol.get_fdata()
            cropped_image = cropped_image[y1:y2, x1:x2, z1:(z2+1)]
            cropped_image = nib.Nifti1Image(cropped_image.astype(np.int32),
                                            affine=vol.affine, header=vol.header)
            # 保存标注坐标框后的nii标签
            vol_label = nib.Nifti1Image(label_1.astype(np.int32), affine)
            # 按照bboxes裁剪原始标签
            cropped_label = vol_label.get_fdata()
            label_2 = cropped_label
            cropped_label = cropped_label[y1:y2, x1:x2, z1:(z2+1)]
            cropped_label = nib.Nifti1Image(cropped_label.astype(np.int32),
                                            affine=vol_label.affine, header=vol_label.header)

            # 按照bboxes在原始图像中将ROI区域外置为0
            def set_outside_pixels_to_zero(image):
                # 将图像转化为int32类型
                image = image.astype(np.int32)
                y1, x1, z1, y2, x2, z2 = rois[0, :]
                # 将y1:y2, x1:x2, z1:z2范围之外的像素置0
                image[:y1, :, :] = 0
                image[y2:, :, :] = 0
                image[:, :x1, :] = 0
                image[:, x2:, :] = 0
                image[:, :, :z1] = 0
                image[:, :, z2:] = 0
                outside_zero = nib.Nifti1Image(image, affine)
                return outside_zero

            # 调用置0函数
            outside_zero = set_outside_pixels_to_zero(image)
            # 保存置0文件
            if not os.path.exists("./results/image_zero"):
                os.makedirs("./results/image_zero")
            nib.save(outside_zero, "./results/image_zero/"  + path_image[-31:])


            # 计算原始标签和预测标签二维层面的iou
            label_2[:y1, :, :] = 0
            label_2[y2:, :, :] = 0
            label_2[:, :x1, :] = 0
            label_2[:, x2:, :] = 0
            label_2[:, :, :z1] = 0
            label_2[:, :, (z2+1):] = 0
            iou_2d = utils.compute_mask_iou(label, label_2)
            iou_2d = round(iou_2d,3)
            iou_2d = 1.0 if iou_2d > 0.99 else iou_2d
            iou_2d_list.append(iou_2d)
            print("IOU_2d: ", iou_2d)

            # 保存带有边框的原始图像
            if not os.path.exists("./results/image"):
                os.makedirs("./results/image")
            nib.save(vol, "./results/image/"  + path_image[-31:])
            # 保存带有边框的原始标签
            if not os.path.exists("./results/label"):
                os.makedirs("./results/label")
            nib.save(vol_label, "./results/label/"  + path_label[-31:])
            # 保存裁剪后的原始图像
            if not os.path.exists("./results/cropped_image"):
                os.makedirs("./results/cropped_image")
            nib.save(cropped_image, "./results/cropped_image/"  + path_image[-31:])
            # 保存裁剪后的原始标签
            if not os.path.exists("./results/cropped_label"):
                os.makedirs("./results/cropped_label")
            nib.save(cropped_label, "./results/cropped_label/"  + path_label[-31:])
        print(path_image[-31:] + " 检测完成 " )
    print("检测全部完成")
    # 输出iou结果.
    iou_2d_list = np.array(iou_2d_list)
    print("IOU_2d均值： ",iou_2d_list.mean())
    print("全部检测时间:", detect_time)


############################################################
#  训练入口
############################################################

if __name__ == '__main__':
    import argparse
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Train the Faster-Unet model to apply whole heart segmentation.')
    parser.add_argument("--command",
                        default="new_test",
                        help="'train' or 'new_test'")
    parser.add_argument('--weights', required=False,
                        default="./weight/heart-late/best_model_1008_355.pth",
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file")
    parser.add_argument('--stage', default="beginning", required=False,
                        help="The training_stages now, 'beginning' or 'finetune'")
    parser.add_argument('--logs', required=False,
                        default="./logs/",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--data', required=False,
                        default="./",
                        metavar="/path/to/data/",
                        help='Dataset directory (default=./data/)')
    parser.add_argument('--limit', required=False,
                        default='200',
                        help='The number of images used for testing (default=5)')
    parser.add_argument('--save', required=False,
                        default="True",
                        help='Whether to save the detected masks (default=False)')
    parser.add_argument('--bbox', required=False,
                        default="True",
                        help='Whether to draw the bboxes (default=False)')

    args = parser.parse_args()

    assert args.stage in ['beginning', 'finetune']

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # 配置
    if args.command == "train":
        config = HeartConfig(args.stage.lower())
    else:
        class InferenceConfig(HeartConfig):
            # 将batch size大小设置为1，因为我们将一次对一个图像运行推断。
            # 批处理大小= GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.1
            DETECTION_MAX_INSTANCES = 1
        config = InferenceConfig(args.stage.lower())

    config.display()

    # 创建模型
    if args.command == "train":
        model = model.FasterRCNN(config=config, model_dir=args.logs, test_flag=False)
    else:
        model = model.FasterRCNN(config=config, model_dir=args.logs, test_flag=True)

    if config.GPU_COUNT:
        model = model.cuda()

    if args.weights.lower() != "none":
        # 选择加载权重文件
        weights_path = args.weights
        # 加载权重
        print("Loading weights ", weights_path)
        model.load_weights(weights_path)

    # 训练或评估
    if args.command == "train":
        print("Training...")
        train(model)
    elif args.command == "new_test":
        print("Testing...")
        new_test(model, int(args.limit.lower()), args.save.lower(), args.bbox.lower())
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
