import os
import traceback
import numpy as np
import json
import cv2
from typing import Dict, List, Tuple
from pybaseutils import file_utils, image_utils
from face_matcher import EmbeddingMatching
from PIL import Image, ImageDraw, ImageFont

def draw_text_chinese(img, text, position, font_path, font_size, color):

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


class FaceRegister(object):
    def __init__(self, data_file=""):
        """
        :param data_file: 人脸数据库的文件路径。
        """
        self.data_file = data_file
        self.database: Dict[str, List[List[float]]] = {} # {人脸ID: [特征1, 特征2, ...]}
        self.matcher = None
        self.load()

    def add_face(self, face_id: str, face_fea: np.ndarray, update=False):
        """
        注册（添加）一个新的人脸。
        :param face_id: 人脸的ID（例如，姓名）。
        :param face_fea: 人脸的特征向量。
        :param update: 是否立即更新数据库文件。
        """
        face_fea_list = self.database.get(face_id, [])
        face_fea_list.append(face_fea.tolist())
        self.database[face_id] = face_fea_list
        if update:
            self.save()

    def del_face(self, face_id: str, update=False):
        """
        注销（删除）一个人脸。
        :param face_id: 要删除的人脸的ID。
        :param update: 是否立即更新数据库文件。
        """
        if face_id in self.database:
            del self.database[face_id]
            if update:
                self.save()

    def get_database_list(self) -> Tuple[np.ndarray, List[str]]:
        """
        将人脸数据库作为平均特征和对应ID的列表获取。
        :return: 一个元组，包含 (嵌入向量的numpy数组, ID列表)。
        """
        ids = []
        embeddings = []
        if not self.database:
            return np.empty((0, 512)), []
            
        for face_id, face_feas_list in self.database.items():
            # 对一个人的所有特征取平均，以获得更鲁棒的表示
            avg_fea = np.mean(np.array(face_feas_list), axis=0)
            embeddings.append(avg_fea)
            ids.append(face_id)
        return np.array(embeddings), ids

    def save(self, file=None):
        """将人脸数据库保存到JSON文件。"""
        file_path = file if file else self.data_file
        if not file_path:
            return
        print(f"正在保存数据库到 {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, indent=4, ensure_ascii=False)

    def load(self, file=None):
        """从JSON文件加载人脸数据库。"""
        file_path = file if file else self.data_file
        if file_path and os.path.exists(file_path):
            print(f"正在从 {file_path} 加载数据库")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
        else:
            self.database = {}

    def search_face(self, face_fea: np.ndarray, score_thresh: float, use_fast=False) -> Tuple[str, float]:
        """
        执行 1:N 人脸搜索。
        :param face_fea: 要搜索的人脸的特征向量，形状=(512,)
        :param score_thresh: 人脸识别的阈值。
        :return: 一个元组 (预测的ID, 相似度分数)。
        """
        if not self.database:
            return "unknown", 0.0
            
        db_embeddings, db_ids = self.get_database_list()
        
        if db_embeddings.shape[0] == 0:
            return "unknown", 0.0

        # 使用当前数据库创建一个匹配器
        matcher = EmbeddingMatching(db_embeddings, db_ids)
        pred_idx, score = matcher.embedding_matching(face_fea, score_thresh)
        
        if pred_idx == -1:
            return "unknown", score
            
        return db_ids[pred_idx], score

    def compare_feature(self, face_fea1: np.ndarray, face_fea2: np.ndarray) -> float:
        """
        执行 1:1 人脸比对。
        :param face_fea1: 人脸特征向量1。
        :param face_fea2: 人脸特征向量2。
        :return: 相似度分数。
        """
        return EmbeddingMatching.compare_embedding_scores(face_fea1, face_fea2)


if __name__ == "__main__":
    # --- 导入测试所需的模块 ---
    import torch
    from face_detector import FaceDetector
    from face_feature import FaceFeature
    from alignment.face_alignment import face_alignment as FaceAligner

    # --- 1. 配置 ---
    # 将路径调整为相对于项目根目录
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PORTRAIT_DIR = os.path.join(ROOT_PATH, "data/database/portrait")
    DB_TEST_DIR = os.path.join(ROOT_PATH, "data/database-test")
    DB_FILE = os.path.join(ROOT_PATH, "data/database/face_database.json")
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SCORE_THRESH = 0.70 # 相似度阈值

    # --- 2. 初始化模型 ---
    print("正在初始化模型...")
    face_detector = FaceDetector(net_name='RBF', input_size=[640, 640], conf_thresh=0.85, device=DEVICE)
    face_feature = FaceFeature(net_name="resnet50", device=DEVICE)
    face_register = FaceRegister(data_file=DB_FILE)
    print("模型初始化完成。")

    # --- 3. 建立/注册人脸数据库 ---
    # 询问用户是否要重建数据库
    rebuild_db = input(f"是否要从 '{DB_PORTRAIT_DIR}' 重建数据库? (y/n): ").lower()
    if rebuild_db == 'y':
        print("\n正在建立人脸数据库...")
        # 清空现有数据库
        face_register.database = {}
        image_files = file_utils.get_files_lists(DB_PORTRAIT_DIR)
        for image_path in image_files:
            try:
                face_id = os.path.basename(image_path).split('-')[0].split('.')[0]
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"警告：无法读取图像 {image_path}。正在跳过。")
                    continue
                
                # 检测人脸
                bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
                if bboxes is None or len(bboxes) == 0:
                    print(f"警告：在 {image_path} 中未检测到属于 {face_id} 的人脸。正在跳过。")
                    continue
                
                # 使用置信度最高的检测结果
                best_idx = np.argmax(scores)
                landmark = landmarks[best_idx]
                
                # 对齐人脸
                aligned_face = FaceAligner(image, landmark, None, vis=False)
                
                # 提取特征 (该函数期望接收一个含有人脸图像的列表)
                face_embedding = face_feature.get_faces_embedding([aligned_face])[0]
                
                # 添加到注册器
                face_register.add_face(face_id, face_embedding)
                print(f"已从 {os.path.basename(image_path)} 注册人脸: {face_id}")
            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")
        
        face_register.save()
        print("人脸数据库建立完成。\n")

    # --- 4. 测试人脸识别 ---
    if not face_register.database:
        print("错误：数据库为空。请先重新运行脚本并选择 'y' 来建立数据库。")
    else:
        print("\n--- 开始人脸识别测试 ---")
        test_images = sorted(file_utils.get_files_lists(DB_TEST_DIR))
        
        # 让用户选择一张图片进行测试
        print("请选择一张图片进行测试:")
        for i, path in enumerate(test_images):
            display_path = os.path.relpath(path, ROOT_PATH)
            print(f"  {i+1}: {display_path}")
        
        try:
            choice = int(input(f"请输入编号 (1-{len(test_images)}): "))
            test_image_path = test_images[choice-1]
        except (ValueError, IndexError):
            print("无效的选择。正在退出。")
            exit()
            
        print(f"正在使用 {os.path.basename(test_image_path)} 进行测试...")
        test_image = cv2.imdecode(np.fromfile(test_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # 检测测试图像中的所有人脸
        bboxes, scores, landmarks = face_detector.detect_face_landmarks(test_image)
        FONT_FILE = os.path.join(ROOT_PATH, "simhei.ttf")
        if bboxes is None or len(bboxes) == 0:
            print("在测试图像中未检测到人脸。")
        else:
            # 处理每个检测到的人脸
            for i in range(len(bboxes)):
                box, score, lms = bboxes[i], scores[i], landmarks[i]
                
                # 对齐人脸
                aligned_face = FaceAligner(test_image, lms, None, vis=False)
                
                # 获取嵌入特征
                face_embedding = face_feature.get_faces_embedding([aligned_face])[0]
                
                # 在数据库中搜索
                pred_id, pred_score = face_register.search_face(face_embedding, SCORE_THRESH)
                
                # 在图像上绘制结果
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if pred_id != "unknown" else (0, 0, 255)
                cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
                
                text = f"{pred_id} ({pred_score:.2f})"
                # 【新代码】调用我们新的中文绘制函数
                # 注意：为了让文字在框的上方，y坐标需要向上移动，移动距离约等于字体大小
                font_size = 25
                test_image = draw_text_chinese(test_image, text, (x1, y1 - font_size - 5), FONT_FILE, font_size, color)

            # 显示最终图像
            print("\n识别完成。正在显示结果...")
            cv2.imshow(f"Recognition Result - {os.path.basename(test_image_path)}", test_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()