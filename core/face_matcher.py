# core/face_matcher.py

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
import numpy as np

class EmbeddingMatching(object):
    """Face Feature Matching Algorithm"""

    def __init__(self, dataset_embeddings, dataset_id_list):
        """
        :param dataset_embeddings: Face embeddings from the database.
        :param dataset_id_list: Corresponding IDs for the embeddings.
        """
        self.dataset_embeddings = dataset_embeddings
        self.dataset_id_list = dataset_id_list

    def embedding_matching(self, face_emb, score_threshold):
        """
        Compares embedding similarity using Euclidean distance.
        If no ID with a distance below the threshold is found, returns pred_idx as -1.
        :param face_emb: A single face embedding to be compared.
        :param score_threshold: The threshold for face recognition.
        :return: The index of the predicted ID (pred_idx) and the similarity score (pred_scores).
        """
        if self.dataset_embeddings.size == 0:
            return -1, 0.0
            
        # Calculate Euclidean distance
        diff = self.dataset_embeddings - face_emb
        dist = np.sum(np.power(diff, 2), axis=1)
        
        # Find the minimum distance and its index
        pred_idx = np.argmin(dist)
        min_dist = dist[pred_idx]
        
        # Convert distance to a similarity score
        pred_score = self.get_scores(min_dist)
        
        if pred_score < score_threshold:
            pred_idx = -1
            
        return pred_idx, pred_score

    @staticmethod
    def get_scores(x, meam=1.40, std=0.2):
        """
        Maps Euclidean distance to a similarity score.
        :param x: Euclidean distance value.
        :param meam: Mean, default=1.40.
        :param std: Standard deviation, default=0.2.
        :return: Face similarity score (0 to 1), higher is more similar.
        """
        x = -(x - meam) / std
        # Sigmoid function
        scores = 1.0 / (1.0 + np.exp(-x))
        return scores

    @staticmethod
    def compare_embedding_scores(vect1, vect2):
        """
        Compares two face vectors.
        Args:
            vect1: vector of face1.
            vect2: vector of face2.
        Returns: similarity score.
        """
        diff = vect1 - vect2
        dist = np.sum(np.power(diff, 2))
        score = EmbeddingMatching.get_scores(dist)
        return score

    @staticmethod
    def decode_label(pred_id, pred_scores, dataset_id_list):
        """
        Decodes the ID to get the corresponding label name. Returns "-1" if pred_id is -1.
        :param pred_id: Predicted index.
        :param pred_scores: Similarity score.
        :param dataset_id_list: List of all IDs in the database.
        :return: The name of the person.
        """
        if pred_id == -1:
            return "unknown", -1
        return dataset_id_list[pred_id], pred_scores