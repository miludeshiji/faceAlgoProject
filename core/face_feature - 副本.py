# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
# This file has been completed by Gemini.
"""

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 将项目根目录添加到系统路径，以支持模块的绝对导入
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从项目结构中导入必需的模块
try:
    from feature.net.model_resnet import IR_18, IR_50
    from feature.net.mobilenet_v2 import MobileNetV2
    from face_matcher import EmbeddingMatching
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure that this script is run from the 'core' directory or the project root is in the Python path.")
    sys.exit(1)


# 定义模型文件路径
# 假设此文件 (face_feature.py) 位于 `core` 目录下
root = os.path.dirname(__file__)
MODEL_FILE = {
    "resnet50": os.path.join(root, "feature/weight/pth/resnet50.pth"),
    "resnet18": os.path.join(root, "feature/weight/pth/resnet18.pth"),
    "mobilenet_v2": os.path.join(root, "feature/weight/pth/mobilenet_v2.pth")
}

def draw_text_chinese(img, text, position, font_path, font_size, color):
    """
    在OpenCV图像上绘制包含中文的文本。
    
    :param img: OpenCV图像 (BGR格式)
    :param text: 要绘制的文本字符串
    :param position: 文本左上角的(x, y)坐标
    :param font_path: ttf字体文件的路径
    :param font_size: 字体大小
    :param color: 文本颜色 (B, G, R)
    :return: 绘制了文本的OpenCV图像
    """
    # 将OpenCV图像从BGR转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 创建一个Pillow图像对象
    pil_img = Image.fromarray(img_rgb)

    # 创建一个绘制对象
    draw = ImageDraw.Draw(pil_img)
    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"警告: 字体文件未找到 {font_path}，使用默认字体。")
        font = ImageFont.load_default()

    # 在图像上绘制文本 (Pillow使用RGB颜色)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

    # 将Pillow图像转换回OpenCV图像格式
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

class FaceFeature(object):
    def __init__(self, net_name="resnet50", input_size=(112, 112), embedding_size=512, device=torch.device('cpu')):
        """
        :param net_name: 使用的网络名称 ("resnet50", "resnet18", "mobilenet_v2")
        :param input_size: 模型输入尺寸
        :param embedding_size: 特征向量维度
        :param device: 计算设备 (torch.device)
        """
        if net_name not in MODEL_FILE:
            raise ValueError(f"错误: 不支持网络 '{net_name}'。支持的模型为 {list(MODEL_FILE.keys())}")

        model_file = MODEL_FILE[net_name]
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件未找到: {model_file}")

        self.net_name = net_name
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.device = device
        self.model = self.build_net(model_file, net_name, input_size, embedding_size)
        self.transform = self.default_transform(input_size)
        self.matcher = EmbeddingMatching()

    def build_net(self, model_file, net_name, input_size, embedding_size):
        """
        构建并加载模型
        """
        if net_name.lower() == "resnet18":
            model = IR_18(input_size, embedding_size)
        elif net_name.lower() == "resnet50":
            model = IR_50(input_size, embedding_size)
        elif net_name.lower() == "mobilenet_v2":
            # 根据demo.py，此处的MobileNetV2实现可能不接受input_size
            model = MobileNetV2(num_classes=embedding_size)
        else:
            raise NotImplementedError(f"错误: 网络 {net_name} 未实现")

        state_dict = torch.load(model_file, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def forward(self, image_tensor):
        """
        执行前向传播
        """
        with torch.no_grad():
            embeddings = self.model(image_tensor.to(self.device))
            embeddings = self.post_process(embeddings)
        return embeddings.cpu().numpy()

    def get_faces_embedding(self, faces_list):
        """
        获取多个人脸图像的特征向量
        :param faces_list: 人脸图像(BGR格式的numpy数组)列表, 必须是已裁剪对齐的人脸
        :return: 返回N张人脸的特征向量, shape=[-1, 512]
        """
        if not faces_list:
            return np.array([])

        face_tensors = self.pre_process(faces_list, self.transform)
        embeddings = self.forward(face_tensors)
        return embeddings

    def predict(self, faces_list, dist_threshold=1.2):
        """
        提取特征并预测人脸ID
        :param faces_list: 人脸图像(BGR)列表
        :param dist_threshold: 人脸识别的欧式距离阈值
        :return: 返回预测ID列表和对应的距离分数
        """
        embeddings = self.get_faces_embedding(faces_list)
        if embeddings.shape[0] == 0:
            return [], []
        pred_ids, pred_scores = self.get_embedding_matching(embeddings, dist_threshold=dist_threshold)
        return pred_ids, pred_scores

    def set_database(self, dataset_embeddings, dataset_id_list):
        """
        设置人脸底库数据到匹配器中
        :param dataset_embeddings: 人脸底库特征向量
        :param dataset_id_list: 人脸底库ID(标签)列表
        """
        self.matcher.dataset_embeddings = np.array(dataset_embeddings)
        self.matcher.dataset_id_list = list(dataset_id_list)
        print(f"人脸数据库已设置，包含 {len(self.matcher.dataset_id_list)} 条记录。")


    def get_embedding_matching(self, face_embedding, dist_threshold=1.2, use_fast=False):
        """
        比较特征向量，返回匹配结果
        :param face_embedding: 人脸特征, shape=(num_faces, embedding_size)
        :param dist_threshold: 距离阈值
        :param use_fast: 是否使用快速匹配 (此处未实现，作为保留参数)
        :return: 预测ID列表和距离分数列表
        """
        # 注意: 阈值是针对欧式距离的，值越小表示越相似。
        # demo.py中的get_scores函数将距离映射为相似度分数，一个1.2的距离阈值是比较合理的默认值。
        if use_fast:
            # K-means快速匹配功能需要额外实现，此处回退到标准匹配
            return self.matcher.embedding_matching(face_embedding, score_threshold=dist_threshold)
        else:
            return self.matcher.embedding_matching(face_embedding, score_threshold=dist_threshold)

    @staticmethod
    def get_label_name(pred_id, pred_scores, dataset_id_list):
        """
        解码ID，返回对应的标签名称。ID为-1时返回"Unknown"。
        """
        return EmbeddingMatching.decode_label(pred_id, pred_scores, dataset_id_list)

    @staticmethod
    def default_transform(input_size, rgb_mean=[0.5, 0.5, 0.5], rgb_std=[0.5, 0.5, 0.5]):
        """
        人脸识别模型的默认图像预处理方法
        """
        return transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[1] / 112)]),
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ])

    @staticmethod
    def pre_process(faces_list, transform):
        """
        对一批图像进行预处理
        :param faces_list: numpy数组格式的BGR图像列表
        :param transform: torchvision的transform流水线
        :return: 图像张量
        """
        tensors = []
        for face in faces_list:
            # 将cv2读取的BGR图像 (H, W, C) 转换为PIL的RGB图像
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            tensor = transform(pil_image)
            tensors.append(tensor)
        # 将多张图像堆叠成一个batch
        return torch.stack(tensors)

    @staticmethod
    def post_process(embeddings, axis=1):
        """
        对特征向量进行L2归一化
        """
        norm = torch.norm(embeddings, 2, axis, True)
        output = torch.div(embeddings, norm)
        return output


if __name__ == "__main__":
    # 导入测试脚本所需的模块
    import glob
    from face_detector import FaceDetector
    from pybaseutils import image_utils, file_utils

    print("="*40)
    print("开始执行人脸特征提取与识别测试...")
    print("="*40)

    # --- 1. 配置参数 ---
    # 假设项目根目录在本文件(core/face_feature.py)的上两级
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "portrait")
    TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "database-test")

    NET_NAME = "resnet50"  # 可选: "resnet18", "mobilenet_v2"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DIST_THRESHOLD = 1.2  # 欧式距离匹配阈值，越小要求越严格

    print(f"配置: [网络: {NET_NAME}], [设备: {DEVICE}], [距离阈值: {DIST_THRESHOLD}]")

    # --- 2. 初始化模型 ---
    try:
        print("\n正在初始化人脸检测器和特征提取器...")
        face_detector = FaceDetector(net_name='RBF', input_size=[640, 640], device=DEVICE)
        face_feature_extractor = FaceFeature(net_name=NET_NAME, device=DEVICE)
        print("模型初始化成功。")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        sys.exit(1)

    # --- 3. 构建人脸数据库 ---
    print(f"\n正在从路径构建人脸数据库: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库路径不存在: {DB_PATH}，请检查路径。")
        sys.exit(1)

    db_embeddings = []
    db_labels = []

    # 支持 .jpg 和 .png 格式的图片
    image_files = glob.glob(os.path.join(DB_PATH, '*.[jp][pn]g'))
    for image_path in image_files:
        # 从文件名解析人物标签 (例如: "胡歌-0001.jpg" -> "胡歌")
        label = os.path.basename(image_path).split('-')[0]
        print(f"  正在处理: {os.path.basename(image_path)} -> 标签: {label}")

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            print(f"    - 警告: 无法读取图片 {image_path}")
            continue

        # 检测并裁剪人脸 (假设每张底库图片只有一张清晰人脸)
        faces, _, _, _ = face_detector.detect_crop_faces(image, alignment=False)

        if faces:
            # 提取第一张检测到的人脸的特征
            embedding = face_feature_extractor.get_faces_embedding([faces[0]])
            if embedding.size > 0:
                db_embeddings.append(embedding[0])
                db_labels.append(label)
        else:
            print(f"    - 警告: 在 {os.path.basename(image_path)} 中未检测到人脸")

    if not db_labels:
        print("\n错误: 人脸数据库为空，无法继续测试。请检查底库图片。")
        sys.exit(1)

    # 将构建好的数据库加载到特征提取器的匹配器中
    face_feature_extractor.set_database(db_embeddings, db_labels)
    

    # --- 4. 用户选择测试图片 ---
    print(f"\n请从测试集选择一张图片: {TEST_DATA_PATH}")
    test_image_paths = sorted(file_utils.get_files_lists(TEST_DATA_PATH))

    if not test_image_paths:
        print(f"错误: 在 {TEST_DATA_PATH} 中未找到任何测试图片。")
        sys.exit(1)

    for i, path in enumerate(test_image_paths):
        display_path = os.path.relpath(path, PROJECT_ROOT)
        print(f"  {i+1}: {display_path}")

    while True:
        try:
            choice = int(input(f"\n请输入图片编号进行测试 (1-{len(test_image_paths)}): "))
            if 1 <= choice <= len(test_image_paths):
                selected_image_path = test_image_paths[choice - 1]
                break
            else:
                print("无效编号，请重试。")
        except ValueError:
            print("输入无效，请输入一个数字。")

    print(f"已选择: {selected_image_path}")

    # --- 5. 执行识别 ---
    test_image = cv2.imdecode(np.fromfile(selected_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if test_image is None:
        print("错误: 加载所选测试图片失败。")
        sys.exit(1)

    # 检测测试图中的所有人脸
    faces, bboxes, scores, _ = face_detector.detect_crop_faces(test_image, alignment=False)
    vis_image = test_image.copy()

    if not faces:
        print("\n在测试图片中未检测到任何人脸。")
    else:
        print(f"\n检测到 {len(faces)} 张人脸，正在进行识别...")
        # 对所有检测到的人脸进行批量预测
        pred_ids, pred_dists = face_feature_extractor.predict(faces, dist_threshold=DIST_THRESHOLD)
        # 解码预测结果
        pred_info = face_feature_extractor.get_label_name(pred_ids, pred_dists, db_labels)

        # === 在这里定义字体路径 ===
        # 请根据您的系统和字体位置修改此路径
        CHINESE_FONT_PATH = 'C:/Windows/Fonts/simhei.ttf'
        
        # --- 6. 可视化并显示结果 ---
        print("\n识别结果:")
        for i, bbox in enumerate(bboxes):
            label, dist = pred_info[i]
            if label == "-1":
                display_text = f"Unknown (Dist: {dist:.2f})"
            else:
                # 使用demo.py中的公式将距离转换为更直观的相似度分数(0-1)
                score = 1.0 / (1.0 + np.exp((dist - 1.4) / 0.2))
                display_text = f"{label} ({score*100:.1f}%)"
            
            print(f"  - 人脸 {i+1}: {display_text}")
            # 在图片上绘制包围框和文本
            # 新的代码:
            # 1. 先用OpenCV画出矩形框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
            # 2. 再用我们自己的函数画中文
            # 文本位置设置在框的左上角
            text_position = (int(x1) + 5, int(y1) + 5)
            vis_image = draw_text_chinese(vis_image, display_text, text_position, CHINESE_FONT_PATH, 20, (0, 255, 0))
    
    print("\n结果图片已生成，按任意键关闭窗口。")
    image_utils.cv_show_image("Recognition Result", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()