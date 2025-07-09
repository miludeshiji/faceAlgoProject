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
import traceback
from configs import configs
from core.face_recognizer import FaceRecognizer
from pybaseutils import image_utils, file_utils

def parse_opt():
    parser = argparse.ArgumentParser(description="1:N Face Search")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of test images.")
    parser.add_argument("--video_file", type=str, default=None, help="Path to video file or camera index (e.g., '0').")
    parser.add_argument("--out_dir", type=str, default="output/", help="Directory to save detection results.")
    parser.add_argument("--database", type=str, default="data/database/database.json", help="Path to the face database file.")
    parser.add_argument("--save_video", type=str, default=None, help="Path to save the processed video (e.g., output/result.avi).")
    parser.add_argument("--detect_freq", type=int, default=1, help="Detection frequency for video processing (process every Nth frame).")
    return parser.parse_args()


if __name__ == "__main__":
    """1:N Face Search: suitable for applications like face check-in, access control, personnel information query, and security monitoring."""
    opt = parse_opt()
    
    # Initialize the recognizer with the specified database
    recognizer = FaceRecognizer(database=opt.database)
    
    if opt.image_dir:
        print(f"--- Starting face search on image directory: {opt.image_dir} ---")
        recognizer.detect_image_dir(image_dir=opt.image_dir, out_dir=opt.out_dir, vis=True)
        print(f"--- Finished processing. Results saved to: {opt.out_dir} ---")
        
    elif opt.video_file:
        print(f"--- Starting face search on video: {opt.video_file} ---")
        recognizer.start_capture(
            video_file=opt.video_file, 
            save_video=opt.save_video, 
            detect_freq=opt.detect_freq, 
            vis=True
        )
        print(f"--- Finished video processing. ---")
        
    else:
        print("No input specified. Please provide --image_dir or --video_file.")