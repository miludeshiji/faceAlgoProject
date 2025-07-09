# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""

import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from alignment import cv_face_alignment
from pybaseutils import image_utils


def show_landmark_boxes(title, image, landmarks, boxes, color=(0, 255, 0)):
    '''
    在图像上绘制人脸框和关键点，用于可视化检测结果
    :param title: 窗口标题
    :param image: 输入图像
    :param landmarks: 关键点列表, e.g. [[[x1, y1], [x2, y2]], [[x1, y1], [x2, y2]]]
    :param boxes:     边框列表, e.g. [[x1, y1, x2, y2], [x1, y1, x2, y2]]
    :return: None
    '''
    vis_image = image.copy()
    # 绘制关键点
    if landmarks is not None:
        for landmark in landmarks:
            for point in landmark:
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_image, (x, y), 2, (0, 0, 255), -1)
    # 绘制人脸框
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

    cv2.imshow(title, vis_image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def face_alignment(image, landmarks, boxes, vis=False):
    """
    face alignment and crop face ROI
    接收输入图像和人脸关键点，实现基于关键点的仿射变换，将人脸标准化到112x112大小
    :param image: 输入RGB/BGR图像
    :param landmarks: 人脸关键点landmarks(5个点), shape=(5, 2)
    :param vis: 是否可视化矫正效果
    :return: 返回对齐后的112x112人脸图像
    """
    # 1. 设置输出尺寸为112x112，并获取对应的标准参考点
    face_size = [112, 112]
    # 调用 get_reference_facial_points 获取方形脸(112x112)的参考点
    kpts_ref = cv_face_alignment.get_reference_facial_points(square=True, vis=False)

    # 2. 执行人脸对齐和裁剪
    # 调用 alignment_and_crop_face 计算变换矩阵并应用仿射变换
    aligned_face = cv_face_alignment.alignment_and_crop_face(image,face_size=face_size,kpts=landmarks,kpts_ref=kpts_ref,align_type="estimate")

    # 3. 根据vis参数决定是否可视化
    if vis:
        # 显示原始图像和关键点
        print("显示原始图像和关键点...")
        show_landmark_boxes("Original Image with Landmarks", image, [landmarks], boxes)
        # 显示对齐裁剪后的人脸
        print("显示对齐后的112x112人脸图像...")
        cv2.imshow("Aligned Face (112x112)", aligned_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return aligned_face


if __name__ == "__main__":
    # 创建一个用于测试的空白图像
    image_path = "core/alignment/test.jpg"
    test_image = cv2.imread(image_path)

    if test_image is None:
        print(f"错误：无法加载图像，请确认文件 '{image_path}' 是否存在于脚本所在的目录中。")
    else:
        # !!! 重要 !!!
        # 您必须在此处提供 'text.jpg' 中人脸的实际关键点坐标。
        # 下面是一组示例坐标，请用您自己的坐标替换它们。
        # 坐标顺序: [左眼], [右眼], [鼻尖], [左嘴角], [右嘴角]
        dummy_landmarks = np.array([
            [285.46, 304],   # 示例: 左眼坐标 (x, y)
            [393, 271],   # 示例: 右眼坐标 (x, y)
            [378, 357],   # 示例: 鼻尖坐标 (x, y)
            [328, 408],   # 示例: 左嘴角坐标 (x, y)
            [418, 383]    # 示例: 右嘴角坐标 (x, y)
        ], dtype=np.float32)
        # 【新增】定义一个虚拟的人脸边界框 [x1, y1, x2, y2]
        dummy_boxes = np.array([
            [465, 227.5, 221, 477]
        ])

        print("开始进行人脸对齐测试...")
        # 调用 face_alignment 函数，并开启可视化
        aligned_face_image = face_alignment(test_image, dummy_landmarks,dummy_boxes, vis=True)

        print("\n人脸对齐处理完成。")

