# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""
import os
import sys
import time
import math
import numpy as np
from sklearn.cluster import KMeans
import threading


def singleton(cls):
    _instances = {}
    def get_instance(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return get_instance


@singleton
class FaceFeatureKmeans():
    """FaceFeature K-means(设置为单实例类)"""

    def __init__(self, embeddings, emb_id_list):
        self.embeddings = embeddings
        self.emb_id_list = emb_id_list
        self.kmeans = None
        if embeddings is not None:
            self._kmeans()

    def _kmeans(self):
        self.kmeans = KMeans(n_clusters=len(set(self.emb_id_list)), random_state=42)
        self.kmeans.fit(self.embeddings)

    def fast_embedding_matching(self, face_emb, score_threshold):
        """
        通过欧式距离,比较embeddings特征相似程度,如果未找到小于dist_threshold的ID,则返回pred_idx为-1
        :param face_embedding: 人脸特征
        :param score_threshold: 人脸识别阈值
        :param numpyData:
        :return:返回预测pred_id的ID和距离分数pred_scores
        :return:
        """
        pred_id = []
        pred_scores = []
        for emb in face_emb:
            distances = np.linalg.norm(self.embeddings - emb, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            if min_dist < score_threshold:
                pred_id.append(min_idx)
                pred_scores.append(min_dist)
            else:
                pred_id.append(-1)
                pred_scores.append(min_dist)
        return pred_id, pred_scores

    @staticmethod
    def get_scores(x, meam=1.40, std=0.2):
        return np.exp(-(x - meam) ** 2 / (2 * std ** 2))



class EmbeddingMatching(object):
    """人脸特征匹配算法"""

    def __init__(self, dataset_embeddings=None, dataset_id_list=None):
        """
        :param dataset_embeddings:人脸底库数据embeddings特征
        :param dataset_id_list:人脸底库数据的ID
        """
        self.dataset_embeddings = dataset_embeddings
        self.dataset_id_list = dataset_id_list

    def embedding_matching(self, face_emb, score_threshold):
        """
        通过欧式距离,比较embeddings特征相似程度,如果未找到小于dist_threshold的ID,则返回pred_idx为-1
        :param face_embedding: 人脸特征
        :param score_threshold: 人脸识别阈值
        :param numpyData:
        :return:返回预测pred_id的ID和距离分数pred_scores
        :return:
        """
        pred_id = []
        pred_scores = []
        for emb in face_emb:
            distances = np.linalg.norm(self.dataset_embeddings - emb, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            if min_dist < score_threshold:
                pred_id.append(min_idx)
                pred_scores.append(min_dist)
            else:
                pred_id.append(-1)
                pred_scores.append(min_dist)
        return pred_id, pred_scores

    def fast_embedding_matching(self, face_emb, score_threshold):
        """
        通过欧式距离,比较embeddings特征相似程度,如果未找到小于dist_threshold的ID,则返回pred_idx为-1
        :param face_embedding: 人脸特征
        :param score_threshold: 人脸识别阈值
        :param numpyData:
        :return:返回预测pred_id的ID和距离分数pred_scores
        :return:
        """
        return self.embedding_matching(face_emb, score_threshold)

    def frame_embedding_matching(self, face_emb, score_threshold):
        """
        比较一帧数据中,所有人脸的embeddings特征相似程度,如果相似程度小于dist_threshold的ID,则返回pred_idx为-1
        PS:由于是同一帧的人脸,因此需要考虑人脸ID的互斥性
        :param face_emb: 输入的人脸特征必须是同一帧的人脸特征数据,shape=(nums_face, embedding_size)
        :param score_threshold: 人脸识别阈值
        :return:返回预测pred_id的ID和距离分数pred_scores
        """
        pred_id = []
        pred_scores = []
        used_idx = set()
        for emb in face_emb:
            distances = np.linalg.norm(self.dataset_embeddings - emb, axis=1)
            sorted_idx = np.argsort(distances)
            for idx in sorted_idx:
                if idx not in used_idx and distances[idx] < score_threshold:
                    used_idx.add(idx)
                    pred_id.append(idx)
                    pred_scores.append(distances[idx])
                    break
            else:
                pred_id.append(-1)
                pred_scores.append(np.min(distances))
        return pred_id, pred_scores

    @staticmethod
    def get_scores(x, meam=1.40, std=0.2):
        return np.exp(-(x - meam) ** 2 / (2 * std ** 2))

    @staticmethod
    def compare_embedding_scores(vect1, vect2):
        """
        compare two faces
        Args:
            face1_vector: vector of face1
            face2_vector: vector of face2

        Returns: similarity

        """
        return np.linalg.norm(vect1 - vect2)

    @staticmethod
    def decode_label(pred_id, pred_scores, dataset_id_list):
        """
        对ID进行解码,返回对应的lable名称,当pred_id为-1时,返回"-1"
        :param pred_id:
        :param pred_scores:
        :param dataset_id_list:
        :return:
        """
        label_names = []
        for idx, id_ in enumerate(pred_id):
            if id_ == -1:
                label_names.append(("-1", pred_scores[idx]))
            else:
                label_names.append((dataset_id_list[id_], pred_scores[idx]))
        return label_names


if __name__ == "__main__":
    pass
