# core/face_feature.py

# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""


import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
# 调整导入路径以适应项目结构
from feature.net.model_resnet import IR_18, IR_50
from feature.net.mobilenet_v2 import MobileNetV2
from face_matcher import EmbeddingMatching
from pybaseutils import image_utils

# 获取当前文件所在目录的绝对路径
root = os.path.dirname(__file__)

# 定义模型文件路径
MODEL_FILE = {
    "resnet50": os.path.join(root, "feature/weight/pth/resnet50.pth"),
    "resnet18": os.path.join(root, "feature/weight/pth/resnet18.pth"),
    "mobilenet_v2": os.path.join(root, "feature/weight/pth/mobilenet_v2.pth")
}


class FaceFeature(object):
    def __init__(self, model_file="", net_name="resnet18", input_size=(112, 112), embedding_size=512, device = torch.device('cpu')):
        """
        :param model_file: model files
        :param net_name: a string selecting from "resnet18", "resnet50", "mobilenet_v2"
        :param input_size: [112,112] or [64,64]
        :param embedding_size: embedding feature size
        :param device: "cpu" or "cuda:0"
        """
        self.device = device
        self.model = self.build_net(model_file, net_name, input_size, embedding_size)
        self.transform = self.default_transform(input_size)
        self.embedding_matching = None

    def build_net(self, model_file, net_name, input_size, embedding_size):
        """
        :param model_file: Pytorch model file path
        :param net_name: a string selecting from "resnet18", "resnet50", "mobilenet_v2"
        :param input_size: [112,112] or [64,64]
        :param embedding_size: embedding feature size
        :return: Pytorch model
        """
        if not model_file:
            model_file = MODEL_FILE[net_name]
        
        if net_name.lower() == "resnet18":
            model = IR_18(input_size, embedding_size)
        elif net_name.lower() == "resnet50":
            model = IR_50(input_size, embedding_size)
        elif net_name.lower() == "mobilenet_v2":
            model = MobileNetV2(input_size, embedding_size)
        else:
            raise Exception(f"Error: Invalid network name '{net_name}'")
            
        state_dict = torch.load(model_file, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def forward(self, image_tensor):
        """
        Forward pass to get embeddings.
        :param image_tensor: pre-processed image tensor
        :return: embeddings
        """
        with torch.no_grad():
            embeddings = self.model(image_tensor.to(self.device))
        return embeddings

    def get_faces_embedding(self, faces_list):
        """
        Extract features from a list of face images.
        :param faces_list: A list of cropped face images (RGB format, numpy array), shape=[-1, 112, 112, 3]
        :return: A numpy array of 512-dimensional face features, shape=[-1, 512]
        """
        if not faces_list:
            return np.empty((0, 512))

        # Pre-process images
        faces_tensor = self.pre_process(faces_list, self.transform)
        
        # Forward pass
        embeddings = self.forward(faces_tensor)
        
        # Post-process (L2 normalization)
        embeddings = self.post_process(embeddings)
        
        return embeddings.cpu().numpy()

    def set_database(self, dataset_embeddings, dataset_id_list):
        """
        Set the face database for matching.
        :param dataset_embeddings: Face embeddings from the database.
        :param dataset_id_list: Corresponding IDs for the embeddings.
        """
        self.embedding_matching = EmbeddingMatching(dataset_embeddings, dataset_id_list)

    def get_embedding_matching(self, face_embedding, score_threshold=0.75):
        """
        Compare the similarity of embeddings.
        :param face_embedding: A single face feature vector, shape=(1, 512)
        :param score_threshold: The threshold for face recognition.
        :return: Predicted ID and similarity score.
        """
        if self.embedding_matching is None:
            raise Exception("Database not set. Please call set_database() first.")
        
        pred_id, pred_score = self.embedding_matching.embedding_matching(face_embedding, score_threshold)
        label_name = self.get_label_name(pred_id, pred_score, self.embedding_matching.dataset_id_list)
        return label_name, pred_score

    @staticmethod
    def get_label_name(pred_id, pred_scores, dataset_id_list):
        """
        Decode the ID to get the corresponding label name.
        :param pred_id: Predicted index.
        :param pred_scores: Similarity score.
        :param dataset_id_list: List of all IDs in the database.
        :return: Label name, or "unknown" if not found.
        """
        if pred_id == -1:
            return "unknown"
        return dataset_id_list[pred_id]

    @staticmethod
    def default_transform(input_size, RGB_MEAN=[0.5, 0.5, 0.5], RGB_STD=[0.5, 0.5, 0.5]):
        """
        Default pre-processing for face recognition.
        :param input_size: resize size.
        :param RGB_MEAN: mean.
        :param RGB_STD: standard deviation.
        :return: transform object.
        """
        return transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

    @staticmethod
    def pre_process(faces_list, transform):
        """
        :param faces_list: A list of numpy array images.
        :param transform: transform object.
        :return: A tensor of images.
        """
        tensors = [transform(Image.fromarray(face)) for face in faces_list]
        return torch.stack(tensors)

    @staticmethod
    def post_process(embeddings, axis=1):
        """
        L2 normalization.
        :param embeddings: The raw embeddings from the model.
        :param axis: The axis to apply normalization.
        :return: Normalized embeddings.
        """
        norm = torch.norm(embeddings, 2, axis, True)
        output = torch.div(embeddings, norm)
        return output