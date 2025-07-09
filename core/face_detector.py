# face_detector.py

# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""
import os, sys
import torch

# Fix import path
# Assuming face_detector.py is in the root directory and 'detection' is a sub-directory
sys.path.insert(0, os.path.dirname(__file__))
import cv2
from detection.detector import FaceDetector as Detector
# Assuming you have a face_alignment module, otherwise this line needs to be adapted
# from alignment.face_alignment import face_alignment
from pybaseutils import image_utils, file_utils
import numpy as np

# A placeholder for face_alignment if the module is not available
# You should replace this with your actual face alignment implementation
def face_alignment(image, landmarks):
    print("Warning: face_alignment is a placeholder. Please implement it.")
    # A simple crop based on landmarks for demonstration
    if landmarks is not None and len(landmarks) > 0:
        pts = np.array(landmarks, dtype=np.float32).mean(axis=0)
        w = h = 112
        x_c, y_c = pts
        x1 = int(x_c - w/2)
        y1 = int(y_c - h/2)
        return image_utils.get_box_image(image, [x1,y1,x1+w,y1+h])
    return None

class FaceDetector(object):
    def __init__(self, net_name, input_size, conf_thresh=0.5, nms_thresh=0.3, device = torch.device('cpu')):
        self.detector = Detector(net_name=net_name,
                                 input_size=input_size,
                                 conf_thresh=conf_thresh,
                                 nms_thresh=nms_thresh,
                                 device=device)

    def detect_face_landmarks(self, bgr, vis=False):
        """
        :param bgr: input BGR image
        :param vis: whether to visualize
        :return: bboxes, scores, landms
        """
        return self.detector.detect(bgr, vis=vis)

    def face_alignment(self, image, landmarks):
        """
        :param image:
        :param landmarks: landmarks for a single face
        :return: aligned face image
        """
        return face_alignment(image, landmarks)

    def crop_faces_alignment(self, image, bboxes, landmarks, alignment=True):
        if landmarks is not None and alignment:
            face_list = []
            for i in range(len(landmarks)):
                face = self.face_alignment(image, landmarks[i])
                if face is not None:
                    face_list.append(face)
            return face_list
        else:
            return image_utils.get_bboxes_image(image, bboxes, size=(112, 112))

    def detect_crop_faces(self, bgr_image, alignment=True):
        """
        :param bgr_image:  input bgr-image
        :param alignment: True(default) or False
        :return: list of cropped faces, bboxes, scores, landmarks
        """
        bboxes, scores, landmarks = self.detect_face_landmarks(bgr_image)
        faces = self.crop_faces_alignment(bgr_image, bboxes, landmarks, alignment=alignment)
        return faces, bboxes, scores, landmarks

    def detect_image_dir(self, image_dir, vis=True):
        image_list = file_utils.get_files_lists(image_dir)
        for image_path in image_list:
            image = cv2.imread(image_path)
            if image is None:
                continue
            self.detect_face_landmarks(image, vis=vis)

    @staticmethod
    def show_landmark_boxes(win_name, image, bboxes, scores, landms):
        """
        显示landmark和boxes
        :param win_name: window name
        :param image:
        :param landms: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        """
        Detector.show_landmark_boxes(win_name, image, bboxes, scores, landms)


if __name__ == '__main__':
    # 测试图像路径
    test_images = r"F:/workspace/test/faceAlgoProject/core/detection/light_detector/test_image"

    # 初始化检测器
    detector = FaceDetector(net_name='RBF', input_size=[640, 640], device='cpu')

    for filename in os.listdir(test_images):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(test_images, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法加载图片: {image_path}")
                continue

        # 读取并调整图像尺寸
        image = cv2.imread(image_path)
        image = cv2.resize(image, (500, 500))

        # 执行人脸检测
        faces, bboxes, scores, landms = detector.detect_crop_faces(image, alignment=False)

        # 显示带标注的原图
        vis_image = image.copy()
        detector.show_landmark_boxes(f"Detection Results - {os.path.basename(image_path)}", vis_image, bboxes, scores, landms)

        # 显示每个裁剪的人脸
        for i, face in enumerate(faces):
            image_utils.cv_show_image(f"Face {i+1} - {os.path.basename(image_path)}", face)

        # 键盘控制
        key = cv2.waitKey(0)
        if key == 27:  # ESC退出
            break

    cv2.destroyAllWindows()