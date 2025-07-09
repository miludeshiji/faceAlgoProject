# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""
import os
import argparse
from configs import configs
from core.face_recognizer import FaceRecognizer


def parse_opt():
    parser = argparse.ArgumentParser(description="Register faces to build a database.")
    parser.add_argument("--portrait_dir", type=str, default="data/database/portrait", help="Directory of portrait images for registration.")
    parser.add_argument("--db_file", type=str, default="data/database/database.json", help="Path to save the face database JSON file.")
    return parser.parse_args()


def main(opt):
    """
    Registers faces and generates a face database.
    portrait: Directory of face database images, with the following requirements:
              (1) Images should be named as [ID-XXXX.jpg], e.g., JohnDoe-image.jpg, to be used as the base image for face recognition.
              (2) Portrait photos should be clear, front-facing, and contain only a single face.
    """
    recognizer = FaceRecognizer(database=opt.db_file)
    recognizer.create_database(portrait_dir=opt.portrait_dir, vis=False)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)