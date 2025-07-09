# detection/detector.py

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
import cv2
import torch
from pybaseutils import image_utils, file_utils
from light_detector.light_detector import UltraLightFaceDetector
from mtcnn.mtcnn_detector import MTCNNDetector

class FaceDetector(object):
    def __init__(self, net_name, input_size, conf_thresh=0.5, nms_thresh=0.3, device = torch.device('cpu')):
        """
        :param net_name: "mtcnn", "RFB", "slim", "mobilenet0.25"
        :param input_size:
        :param conf_thresh:
        :param nms_thresh:
        :param device:
        """
        self.net_name = net_name
        self.device = device

        model_path_map = {
            "Slim": "slim_Final.pth",
            "RBF": "face_detection_rbf.pth",
            "MNet": "mobilenet0.25_Final.pth"
        }
        model_file = os.path.join(
            os.path.dirname(__file__),
            "light_detector",
            "pretrained",
            "pth",
            model_path_map[net_name]
        )

        self.detector = UltraLightFaceDetector(
            model_file=model_file,
            net_name=net_name,
            input_size=input_size,
            conf_thresh=conf_thresh,
            iou_thresh=nms_thresh,
            device=device
        )


    def detect(self, bgr, vis=False):
        """
        :param bgr:
        :param vis:
        :return: bboxes, scores, landms
        """
        return self.detector.detect(bgr, vis)

    def detect_image_dir(self, image_dir, vis=True):
        image_list = file_utils.get_files_lists(image_dir)
        for image_file in image_list:
            image = cv2.imread(image_file)
            if image is None:
                continue
            self.detect(image, vis=vis)

    @staticmethod
    def show_landmark_boxes(title, image, bboxes, scores, landms):
        """
        显示landmark和boxes
        :param title:
        :param image:
        :param landms: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        """
        image_ = image.copy()
        if bboxes is not None and len(bboxes) > 0:
            image_ = image_utils.draw_landmark(image_, landms)
            text = [f"{s:.3f}" for s in scores.flatten()]
            image_ = image_utils.draw_image_bboxes_text(image_, bboxes, text)
        image_utils.cv_show_image(title, image_)
        return image_


if __name__ == '__main__':
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_image"))
    detector = FaceDetector(net_name="RBF", input_size=[640, 640], conf_thresh=0.99)
    detector.detect_image_dir(test_dir, vis=True)