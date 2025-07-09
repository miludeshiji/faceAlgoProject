# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""
import os
import cv2
import argparse
import numpy as np
import traceback
from core.face_recognizer import FaceRecognizer
from pybaseutils import image_utils, file_utils

def parse_opt():
    parser = argparse.ArgumentParser(description="1:1 Face Comparison")
    parser.add_argument("--image_file1", type=str, required=True, help="Path to the first image.")
    parser.add_argument("--image_file2", type=str, required=True, help="Path to the second image.")
    parser.add_argument("--score_thresh", type=float, default=0.75, help="Similarity score threshold.")
    return parser.parse_args()


def show_result(image1, face_info1, image2, face_info2, score, score_thresh):
    """Displays the comparison result."""
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    
    # Draw bounding boxes
    if face_info1 and face_info1["boxes"] is not None:
        x1, y1, x2, y2 = map(int, face_info1["boxes"][0])
        cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    if face_info2 and face_info2["boxes"] is not None:
        x1, y1, x2, y2 = map(int, face_info2["boxes"][0])
        cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Create a combined image to display side-by-side
    gap = 50
    max_h = max(h1, h2)
    vis_img = np.zeros((max_h, w1 + w2 + gap, 3), dtype=np.uint8)
    vis_img[:h1, :w1] = image1
    vis_img[:h2, w1 + gap:] = image2

    # Add text
    text = f"Similarity: {score:.4f}"
    is_same = score >= score_thresh
    result_text = "Result: SAME PERSON" if is_same else "Result: DIFFERENT PEOPLE"
    color = (0, 255, 0) if is_same else (0, 0, 255)

    cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_img, result_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("1:1 Face Comparison", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """1:1 Face Comparison: useful for applications like identity verification."""
    opt = parse_opt()

    # Database path is not needed for 1:1 comparison, so we pass None
    recognizer = FaceRecognizer(database=None)

    # Read images
    if not os.path.exists(opt.image_file1):
        raise FileNotFoundError(f"Image file not found: {opt.image_file1}")
    if not os.path.exists(opt.image_file2):
        raise FileNotFoundError(f"Image file not found: {opt.image_file2}")
        
    image1 = cv2.imdecode(np.fromfile(opt.image_file1, dtype=np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.fromfile(opt.image_file2, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Perform comparison
    print("Comparing faces...")
    score, face_info1, face_info2 = recognizer.compare_face(image1, image2)
    
    # Print result to console
    print(f"Similarity Score: {score:.4f}")
    if score >= opt.score_thresh:
        print(f"Result: The two images are of the SAME person (Score > {opt.score_thresh}).")
    else:
        print(f"Result: The two images are of DIFFERENT people (Score <= {opt.score_thresh}).")

    # Show visual result
    show_result(image1, face_info1, image2, face_info2, score, opt.score_thresh)