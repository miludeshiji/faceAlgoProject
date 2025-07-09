# detection/light_detector/light_detector.py

# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2018-04-03 18:38:34
# --------------------------------------------------------
"""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import cv2
import numpy as np
from models.config.config import cfg_mnet, cfg_slim, cfg_rbf
from models.nets.retinaface import RetinaFace
from models.nets.net_slim import Slim
from models.nets.net_rbf import RBF
from models.layers.functions.prior_box import PriorBox
from models.layers.box_utils import decode, decode_landm
from models.nms.py_cpu_nms import py_cpu_nms
from pybaseutils import image_utils, file_utils

root = os.path.dirname(__file__)


class UltraLightFaceDetector(object):
    def __init__(self,
                 model_file: str = "",
                 net_name: str = "RBF",
                 input_size: list = [320, None],
                 conf_thresh: float = 0.5,
                 iou_thresh: float = 0.3,
                 top_k: int = 500,
                 keep_top_k: int = 750,
                 device="cuda:0"):
        """
        :param model_file: model file
        :param net_name:"RBF", "slim", "mobilenet0.25"
        :param input_size:input_size,
        :param conf_thresh: confidence_threshold
        :param iou_thresh: nms_threshold
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param keep_top_k:
        :param device:
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net_name = net_name
        self.cfg = self.get_model_cfg(net_name)
        self.model = self.build_model(self.cfg, net_name, model_file)
        self.input_size = self.get_input_size(input_size, self.cfg)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.variance = self.cfg['variance']


    def get_input_size(self, input_size, cfg):
        if not input_size or input_size[0] is None:
            return [cfg['image_size'], cfg['image_size']]
        if input_size[1] is None:
            input_size[1] = input_size[0]
        return input_size


    def build_model(self, cfg: dict, network: str, model_path: str):
        """
        :param cfg: <dict> model config
        :param network: "mobilenet0.25", "slim" or "RBF"
        :param model_path: model path
        :return:
        """
        if network == "mobilenet0.25":
            net = RetinaFace(cfg=cfg, phase='test')
        elif network == "slim":
            net = Slim(cfg=cfg, phase='test')
        elif network == "RBF":
            net = RBF(cfg=cfg, phase='test')
        else:
            raise NotImplementedError(f"network:{network} not supported")
        model = self.load_model(net, model_path)
        model.to(self.device)
        model.eval()
        return model

    def get_model_cfg(self, network: str):
        """
        get model config
        :param network: "mobilenet0.25", "slim" or "RBF"
        :return:
        """
        supported = {
            "mobilenet0.25": cfg_mnet,
            "slim": cfg_slim,
            "RBF": cfg_rbf
        }
        if network in supported:
            return supported[network]
        else:
            raise NotImplementedError(f"network:{network} not supported")

    def load_model(self, model, model_path: str):
        """
        :param model: model
        :param model_path: model file
        :return:
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model_path:{model_path} not found")
        print(f'Loading model from {model_path}')
        state_dict = torch.load(model_path, map_location=self.device)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model

    def pre_process(self, image: np.ndarray, input_size: list, img_mean=(104, 117, 123)):
        """
        :param image:
        :param input_size: model input size [W,H]
        :param img_mean:
        :return:image_tensor: out image tensor[1,channels,W,H]
                orig_size  : original image size [W,H]
        """
        orig_h, orig_w, _ = image.shape
        img = cv2.resize(image, (input_size[0], input_size[1]))
        img = img.astype(np.float32)
        img -= img_mean
        img = img.transpose(2, 0, 1) # HWC to CHW
        img_tensor = torch.from_numpy(img).unsqueeze(0) # add batch dimension
        img_tensor = img_tensor.to(self.device)
        return img_tensor, [orig_w, orig_h]

    @staticmethod
    def get_priorbox(cfg, input_size):
        """
        :param cfg: model config
        :param input_size: model input size [W,H]
        :return:
        """
        priorbox = PriorBox(cfg, image_size=(input_size[1], input_size[0]))
        priors = priorbox.forward()
        return priors

    def pose_process(self, loc, conf, landms, image_size, input_size, variance):
        """
        :param loc: location predictions
        :param conf: confidence predictions
        :param landms: landmark predictions
        :param image_size: input orig-image size [W,H]
        :param input_size: model input size [W,H]
        :param variance:
        :return: boxes, scores, landms
        """
        priors = self.get_priorbox(self.cfg, input_size).to(self.device)
        scale = torch.Tensor([image_size[0], image_size[1], image_size[0], image_size[1]]).to(self.device)
        scale_landms = torch.Tensor([image_size[0], image_size[1]] * 5).to(self.device)

        boxes = decode(loc.data.squeeze(0), priors, variance)
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        scores = conf.data.squeeze(0)[:, 1].cpu().numpy()

        landms = decode_landm(landms.data.squeeze(0), priors, variance)
        landms = landms * scale_landms
        landms = landms.cpu().numpy()
        return boxes, scores, landms

    @staticmethod
    def nms_process(boxes, scores, landms, conf_threshold, iou_threshold, top_k, keep_top_k):
        """
        :param boxes: face boxes, (xmin,ymin,xmax,ymax)
        :param scores:scores
        :param landms: face landmark
        :param conf_threshold:
        :param iou_threshold:
        :param top_k:keep top_k results. If k <= 0, keep all the results.
        :param keep_top_k:
        :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
                 landms:(num_bboxes,10),[x0,y0,x1,y1,...,x4,y4]
        """
        # ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, iou_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K after NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        return dets, landms

    def inference(self, img_tensor):
        with torch.no_grad():
            loc, conf, landms = self.model(img_tensor)
        return loc, conf, landms

    def adapter_bbox_score_landmarks(self, dets, landms):
        bboxes = dets[:, :4]
        scores = dets[:, 4:5]
        # reshape landmarks
        landms = landms.reshape(-1, 5, 2)
        return bboxes, scores, landms

    @staticmethod
    def get_square_bboxes(bboxes, fixed="W"):
        '''
        :param bboxes:
        :param fixed: (W)width (H)height
        :return:
        '''
        if len(bboxes) == 0:
            return []
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        cx = bboxes[:, 0] + w / 2
        cy = bboxes[:, 1] + h / 2
        if fixed == "W":
            max_s = w
        else:
            max_s = h
        xmin = cx - max_s / 2
        ymin = cy - max_s / 2
        xmax = cx + max_s / 2
        ymax = cy + max_s / 2
        square_bboxes = np.vstack([xmin, ymin, xmax, ymax]).T
        return square_bboxes

    def detect(self, bgr, vis=False):
        """
        :param bgr:
        :return:
            bboxes: <np.ndarray>: (num_boxes, 4)
            scores: <np.ndarray>: (num_boxes, 1)
            landms: <np.ndarray>: (num_boxes, 5, 2)
        """
        # 1. pre-process
        img_tensor, orig_size = self.pre_process(bgr, self.input_size)
        # 2. inference
        loc, conf, landms_pred = self.inference(img_tensor)
        # 3. post-process
        boxes, scores, landms = self.pose_process(loc, conf, landms_pred, orig_size, self.input_size, self.variance)
        # 4. nms-process
        dets, landms = self.nms_process(boxes, scores, landms, self.conf_thresh, self.iou_thresh,
                                        self.top_k, self.keep_top_k)
        # 5. adapter
        bboxes, scores, landms = self.adapter_bbox_score_landmarks(dets, landms)

        if vis:
            self.show_landmark_boxes("UltraLight-Detector", bgr, bboxes, scores, landms)
        return bboxes, scores, landms

    @staticmethod
    def show_landmark_boxes(title, image, bboxes, scores, landms):
        """
        显示landmark和boxes
        :param title:
        :param image:
        :param landms: (num, 5, 2)
        :param bboxes: (num, 4)
        :return:
        """
        image_ = image.copy()
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image_, (x1, y1), (x2, y2), (0, 0, 255), 2)
            score = scores[i][0]
            cv2.putText(image_, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if landms is not None:
                for lm in landms[i]:
                    x, y = lm.astype(int)
                    cv2.circle(image_, (x, y), 2, (0, 255, 255), -1)
        cv2.imshow(title, image_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = UltraLightFaceDetector(
        model_file=r"F:/workspace/test/faceAlgoProject/core/detection/light_detector/pretrained/pth/face_detection_rbf.pth",
        net_name="RBF",
        input_size=[640, 640],
        conf_thresh=0.5
    )
    
    # 测试data目录下的所有图片
    test_dir = r"F:/workspace/test/faceAlgoProject/data/test_image"
    # 在测试代码部分添加判断逻辑
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(test_dir, filename)
            bgr = cv2.imread(image_path)
            if bgr is None:
                print(f"无法加载图片: {image_path}")
                continue
                
            print(f"\n正在检测: {filename}")
            bboxes, scores, landmarks = detector.detect(bgr, vis=True)
            
            # 新增窗口控制逻辑
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if len(bboxes) == 0:
                print(f" 未检测到人脸 (置信度阈值: {detector.conf_thresh})")
            else:
                print(f" 检测到 {len(bboxes)} 张人脸，平均置信度: {np.mean(scores):.2f}")
    bboxes, scores, landmarks = detector.detect(bgr, vis=True)