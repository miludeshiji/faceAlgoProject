"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""
import os
import traceback
import cv2
import torch
from typing import Dict, List, Tuple
import numpy as np
from .face_detector import FaceDetector
from .face_feature import FaceFeature
from .face_register import FaceRegister, draw_text_chinese
from .alignment.face_alignment import face_alignment as FaceAligner
from configs import configs as cfg
from pybaseutils import image_utils, file_utils

class FaceRecognizer(object):
    def __init__(self, database, det_thresh=cfg.det_thresh, rec_thresh=cfg.rec_thresh, device="auto"):
        """
        @param database: Path to the face database file.
        @param det_thresh: Face detection threshold. Bounding boxes with a score lower than this will be discarded.
        @param rec_thresh: Face recognition threshold. A recognition result with a score lower than this will be labeled 'unknown'.
        @param device: Computation device, "auto" will use CUDA if available, otherwise CPU.
        """
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize core components
        self.face_detector = FaceDetector(net_name='RBF', input_size=[640, 640],conf_thresh=det_thresh, device=self.device)
        self.face_feature = FaceFeature(net_name="resnet50", device=self.device)
        self.face_register = FaceRegister(data_file=database)
        self.aligner = FaceAligner
        
        self.rec_thresh = rec_thresh
        self.font_path = os.path.join(os.path.dirname(__file__), "simhei.ttf") # Path to a ttf font for Chinese characters

    def crop_faces_alignment(self, bgr: np.ndarray, boxes: np.ndarray, landm: np.ndarray) -> List[np.ndarray]:
        """
        Crop and align faces from an image.
        @param bgr: The input BGR image.
        @param boxes: Array of bounding boxes.
        @param landm: Array of landmarks.
        @return: A list of aligned face images.
        """
        face_list = []
        if boxes is None or len(boxes) == 0:
            return face_list
        for i in range(len(boxes)):
            aligned_face = self.aligner(bgr, landm[i], boxes[i])
            face_list.append(aligned_face)
        return face_list

    def extract_feature(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of face images (without detection and alignment).
        @param faces: List of BGR face images.
        @return: A numpy array of face features.
        """
        if not faces:
            return np.array([])
        return self.face_feature.get_faces_embedding(faces)

    def detect_extract_feature(self, bgr: np.ndarray, max_face=-1, vis=False) -> Dict:
        """
        Perform face detection, alignment, and feature extraction.
        @param bgr: Input BGR image.
        @param max_face: Maximum number of faces to process. -1 means all faces.
        @param vis: Whether to visualize detection results.
        @return: A dictionary containing face info: {"boxes", "landm", "feature", "face"}.
        """
        # Detect faces and landmarks
        boxes, scores, landms = self.detector(bgr, max_face, vis)
        
        # Crop and align faces
        faces = self.crop_faces_alignment(bgr, boxes, landms)
        
        # Extract features
        features = self.extract_feature(faces)
        
        face_info = {"boxes": boxes, "landm": landms, "scores": scores, "feature": features, "face": faces}
        return face_info

    def detector(self, bgr: np.ndarray, max_face=-1, vis=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform face and landmark detection.
        @param bgr: Input BGR image.
        @param max_face: Max number of faces. -1 for all.
        @param vis: Whether to visualize the detection.
        @return: A tuple of (boxes, scores, landmarks).
        """
        boxes, scores, landms = self.face_detector.detect_face_landmarks(bgr)
        if max_face > 0 and boxes is not None:
            boxes = boxes[:max_face]
            scores = scores[:max_face]
            landms = landms[:max_face]
        if vis and boxes is not None:
            self.face_detector.draw_face_landmarks(bgr, boxes, scores, landms)
        return boxes, scores, landms

    def detect_search(self, bgr: np.ndarray, max_face=-1, vis=True) -> Tuple[Dict, np.ndarray]:
        """
        Perform face detection and 1:N face search.
        @param bgr: Input BGR image.
        @param max_face: Max number of faces to process.
        @param vis: Whether to draw the results on the image.
        @return: A tuple of (face_info dictionary, image_with_results).
        """
        face_info = self.detect_extract_feature(bgr, max_face, vis=False)
        pred_ids, pred_scores = self.search_face(face_info["feature"], self.rec_thresh)
        face_info["pred_id"] = pred_ids
        face_info["pred_score"] = pred_scores
        
        bgr_draw = bgr.copy()
        if vis:
            bgr_draw = self.draw_result("Face Search", bgr_draw, face_info)

        return face_info, bgr_draw

    def create_database(self, portrait_dir: str, vis=True):
        """
        Generate the Face Database.
        @param portrait_dir: The directory of portrait images.
        @param vis: Whether to visualize the face detection process.
        """
        print(f"Starting to build face database from: {portrait_dir}")
        self.face_register.database = {}  # Clear existing database
        image_files = file_utils.get_files_lists(portrait_dir)
        
        for image_path in image_files:
            try:
                face_id = os.path.basename(image_path).split('-')[0]
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Warning: Could not read image {image_path}. Skipping.")
                    continue
                
                face_info = self.detect_extract_feature(image, max_face=1, vis=vis)
                
                if face_info["feature"].shape[0] == 0:
                    print(f"Warning: No face detected in {image_path} for ID {face_id}. Skipping.")
                    continue

                if face_info["feature"].shape[0] > 1:
                    print(f"Warning: Multiple faces detected in {image_path}. Using the first one for ID {face_id}.")

                self.face_register.add_face(face_id, face_info["feature"][0])
                print(f"Registered face from {os.path.basename(image_path)} with ID: {face_id}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        self.face_register.save()
        print("Face database creation complete.")

    def draw_result(self, title: str, image: np.ndarray, face_info: Dict, thickness=2, fontScale=0.8, delay=0, vis=True) -> np.ndarray:
        """
        Draw results on the image.
        @param title: Window title.
        @param image: Image to draw on.
        @param face_info: Dictionary with face information.
        @return: Image with drawn results.
        """
        if "boxes" not in face_info or face_info["boxes"] is None:
            return image
            
        for i, box in enumerate(face_info["boxes"]):
            x1, y1, x2, y2 = map(int, box)
            pred_id = face_info["pred_id"][i]
            pred_score = face_info["pred_score"][i]
            
            color = (0, 255, 0) if pred_id != "unknown" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            text = f"{pred_id} ({pred_score:.2f})"
            font_size = int(20 * fontScale)
            image = draw_text_chinese(image, text, (x1, y1 - font_size - 5), self.font_path, font_size, color)

        if vis:
            cv2.imshow(title, image)
            cv2.waitKey(delay)
        return image

    def add_face(self, face_id: str, bgr: np.ndarray, vis=False):
        """
        Add a single face to the database.
        @param face_id: The ID for the face.
        @param bgr: The original image containing the face.
        @param vis: Whether to visualize the detection.
        """
        face_info = self.detect_extract_feature(bgr, max_face=1, vis=vis)
        if face_info["feature"].shape[0] > 0:
            self.face_register.add_face(face_id, face_info["feature"][0], update=True)
            print(f"Successfully added face with ID: {face_id}")
        else:
            print(f"Failed to add face: No face detected for ID: {face_id}")

    def del_face(self, face_id: str):
        """
        Delete a face from the database.
        @param face_id: The ID of the face to delete.
        """
        self.face_register.del_face(face_id, update=True)
        print(f"Deleted face with ID: {face_id}")

    def search_face(self, face_feas: np.ndarray, rec_thresh: float) -> Tuple[List[str], List[float]]:
        """
        1:N search for multiple face features.
        @param face_feas: Numpy array of face features, shape=(num_faces, embedding_size).
        @param rec_thresh: Recognition threshold.
        @return: A tuple of (predicted_ids, predicted_scores).
        """
        pred_ids = []
        pred_scores = []
        if face_feas.shape[0] == 0:
            return pred_ids, pred_scores
            
        for fea in face_feas:
            pred_id, score = self.face_register.search_face(fea, rec_thresh)
            pred_ids.append(pred_id)
            pred_scores.append(score)
        return pred_ids, pred_scores

    def compare_face(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[float, Dict, Dict]:
        """
        1:1 face comparison.
        @param image1: BGR image 1.
        @param image2: BGR image 2.
        @return: A tuple of (similarity score, face_info1, face_info2).
        """
        face_info1 = self.detect_extract_feature(image1, max_face=1)
        face_info2 = self.detect_extract_feature(image2, max_face=1)
        
        score = 0.0
        if face_info1["feature"].shape[0] > 0 and face_info2["feature"].shape[0] > 0:
            feature1 = face_info1["feature"][0]
            feature2 = face_info2["feature"][0]
            score = self.compare_feature(feature1, feature2)
            
        return score, face_info1, face_info2

    def compare_feature(self, face_fea1: np.ndarray, face_fea2: np.ndarray) -> float:
        """
        1:1 feature comparison.
        @param face_fea1: Face feature vector 1.
        @param face_fea2: Face feature vector 2.
        @return: Similarity score.
        """
        return self.face_register.compare_feature(face_fea1, face_fea2)

    def detect_image_dir(self, image_dir: str, out_dir=None, vis=True):
        """
        Process a directory of images.
        @param image_dir: Directory of images.
        @param out_dir: Directory to save results.
        @param vis: Whether to display results.
        """
        if out_dir:
            file_utils.create_dir(out_dir)
            
        for image_path in file_utils.get_files_lists(image_dir):
            try:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None: continue
                
                _, res_image = self.detect_search(image, vis=False)

                if vis:
                    cv2.imshow("Result", res_image)
                    cv2.waitKey(0)
                if out_dir:
                    out_path = os.path.join(out_dir, os.path.basename(image_path))
                    cv2.imwrite(out_path, res_image)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        if vis:
            cv2.destroyAllWindows()
            
    def start_capture(self, video_file: str, save_video=None, detect_freq=1, vis=True):
        """
        Start capturing and processing a video stream.
        @param video_file: Path to video file or camera index ('0').
        @param save_video: Path to save the output video.
        @param detect_freq: Detection frequency (process every Nth frame).
        @param vis: Whether to display the video stream.
        """
        try:
            video_file = int(video_file)
        except ValueError:
            pass
        
        cap = cv2.VideoCapture(video_file)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if save_video:
            writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'XVID'), 25, (frame_width, frame_height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % detect_freq == 0:
                _, res_frame = self.detect_search(frame, vis=False)
            
            if vis:
                cv2.imshow("Face Search", res_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer:
                writer.write(res_frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if vis:
            cv2.destroyAllWindows()