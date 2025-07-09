import os
import json
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import base64
from io import BytesIO
from PIL import Image

# --- 核心逻辑导入 ---
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.face_detector import FaceDetector
from core.face_feature import FaceFeature
from core.face_register import FaceRegister, draw_text_chinese
from core.alignment.face_alignment import face_alignment as FaceAligner


# --- 应用配置 ---
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# --- 全局变量和模型初始化 ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DB_FILE = os.path.join("data", "database", "face_database.json")
DB_PORTRAIT_DIR = os.path.join("data", "database", "portrait")
FONT_FILE = os.path.join(os.path.dirname(__file__), "simhei.ttf")
SCORE_THRESH = 0.9

face_detector = None
face_feature = None
face_register = None

def initialize_models():
    """加载所有必要的AI模型"""
    global face_detector, face_feature, face_register
    print("Initializing models...")
    try:
        face_detector = FaceDetector(net_name='RBF', input_size=[640, 640], conf_thresh=0.9, device=DEVICE)
        face_feature = FaceFeature(net_name="resnet50", device=DEVICE)
        face_register = FaceRegister(data_file=DB_FILE)
        print("Models initialized successfully.")
    except Exception as e:
        print(f"Error initializing models: {e}")
        
def allowed_file(filename):
    """检查上传的文件扩展名是否合法"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_image_from_request(file_key='file'):
    """从请求中读取图片文件并解码"""
    if file_key not in request.files:
        raise ValueError("请求中缺少图片文件")
    file = request.files[file_key]
    if file.filename == '':
        raise ValueError("未选择文件")
    if file and allowed_file(file.filename):
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        image_array = np.frombuffer(in_memory_file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图片")
        return image, file.filename
    raise ValueError("文件类型不允许")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    try:
        image, filename = read_image_from_request()
        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(filename)}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        cv2.imwrite(upload_path, image)
        bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
        if bboxes is None or len(bboxes) == 0:
            return jsonify({
                "status": "success", 
                "message": "No faces detected.",
                "original_url": url_for('static', filename=f'uploads/{unique_filename}'),
                "processed_url": url_for('static', filename=f'uploads/{unique_filename}')
            })
        result_image = image.copy()
        for i in range(len(bboxes)):
            box, score, lms = bboxes[i], scores[i], landmarks[i]
            aligned_face = FaceAligner(result_image, lms, None, vis=False)
            face_embedding = face_feature.get_faces_embedding([aligned_face])[0]
            pred_id, pred_score = face_register.search_face(face_embedding, SCORE_THRESH)
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if pred_id != "unknown" else (0, 0, 255)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            text = f"{pred_id} ({pred_score:.2f})"
            font_size = 25
            text_y = y1 - font_size - 5 if y1 - font_size - 5 > 0 else y1 + 10
            result_image = draw_text_chinese(result_image, text, (x1, text_y), FONT_FILE, font_size, color)
        processed_filename = f"processed_{unique_filename}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, result_image)
        return jsonify({
            "status": "success",
            "original_url": url_for('static', filename=f'uploads/{unique_filename}'),
            "processed_url": url_for('static', filename=f'processed/{processed_filename}')
        })
    except Exception as e:
        print(f"An error occurred during recognition: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def api_detect():
    try:
        image, _ = read_image_from_request()
        bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
        if bboxes is None:
            return jsonify({"status": "success", "faces": []})
        faces = []
        for i in range(len(bboxes)):
            faces.append({
                'box': [int(c) for c in bboxes[i]],
                'score': float(scores[i]),
                'landmarks': landmarks[i].tolist()
            })
        return jsonify({"status": "success", "faces": faces})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/detect_and_draw', methods=['POST'])
def api_detect_and_draw():
    try:
        image, _ = read_image_from_request()
        result_image = image.copy()
        bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
        faces_data = []
        if bboxes is not None and len(bboxes) > 0:
            for i in range(len(bboxes)):
                box, lms, score = bboxes[i], landmarks[i], scores[i]
                faces_data.append({'box': [int(c) for c in box], 'score': float(score), 'landmarks': lms.tolist()})
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{float(score):.2f}"
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(result_image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                for j in range(len(lms)):
                    px, py = int(lms[j][0]), int(lms[j][1])
                    cv2.circle(result_image, (px, py), 3, (0, 0, 255), -1)
        _, buffer = cv2.imencode('.jpg', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"status": "success", "annotated_image": "data:image/jpeg;base64," + image_base64, "faces": faces_data})
    except Exception as e:
        print(f"An error occurred during detection and drawing: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/align', methods=['POST'])
def api_align():
    try:
        image, _ = read_image_from_request()
        landmarks_str = request.form.get('landmarks')
        if not landmarks_str:
            return jsonify({"status": "error", "message": "缺少 'landmarks' 参数"}), 400
        try:
            landmarks = np.array(json.loads(landmarks_str))
        except (json.JSONDecodeError, Exception) as e:
            return jsonify({"status": "error", "message": f"转换landmarks失败: {str(e)}"}), 400
        aligned_face = FaceAligner(image, landmarks, None, vis=False)
        _, buffer = cv2.imencode('.jpg', aligned_face)
        aligned_face_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            "status": "success",
            "aligned_image": "data:image/jpeg;base64," + aligned_face_b64
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        image, _ = read_image_from_request()
        face_id = request.form.get('id')
        if not face_id:
            return jsonify({"status": "error", "message": "缺少 'id' 参数"}), 400
        bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
        if bboxes is None or len(bboxes) == 0:
            return jsonify({"status": "error", "message": "未检测到人脸"}), 400
        best_idx = np.argmax(scores)
        landmark = landmarks[best_idx]
        aligned_face = FaceAligner(image, landmark, None, vis=False)
        face_embedding = face_feature.get_faces_embedding([aligned_face])[0]
        face_register.add_face(face_id, face_embedding, update=True)
        return jsonify({"status": "success", "message": f"人脸 '{face_id}' 注册成功"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/compare', methods=['POST'])
def api_compare():
    try:
        image1, _ = read_image_from_request('file1')
        image2, _ = read_image_from_request('file2')
        def get_embedding(image):
            bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
            if bboxes is None or len(bboxes) == 0:
                return None
            best_idx = np.argmax(scores)
            landmark = landmarks[best_idx]
            aligned_face = FaceAligner(image, landmark, None, vis=False)
            return face_feature.get_faces_embedding([aligned_face])[0]
        embedding1 = get_embedding(image1)
        embedding2 = get_embedding(image2)
        if embedding1 is None or embedding2 is None:
            return jsonify({"status": "error", "message": "至少有一张图片未检测到人脸"}), 400
        score = face_register.compare_feature(embedding1, embedding2)
        return jsonify({"status": "success", "similarity": float(score)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# =================================================================
# ===== API: 1:N 人脸搜索 (修改为支持多个人脸) =====
# =================================================================
@app.route('/api/search', methods=['POST'])
def api_search():
    """
    API: 1:N 人脸搜索。
    修改为支持检测图片中的所有的人脸，并对每张脸进行搜索。
    """
    try:
        image, _ = read_image_from_request()
        
        bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
        if bboxes is None or len(bboxes) == 0:
            return jsonify({"status": "success", "results": []}) # 未检测到人脸，返回空列表

        all_results = []
        threshold = float(request.form.get('threshold', SCORE_THRESH))

        # 遍历所有检测到的人脸
        for i in range(len(bboxes)):
            landmark = landmarks[i]
            
            # 对每张脸进行对齐和特征提取
            aligned_face = FaceAligner(image, landmark, None, vis=False)
            face_embedding = face_feature.get_faces_embedding([aligned_face])[0]
            
            # 对每张脸的特征进行搜索
            pred_id, pred_score = face_register.search_face(face_embedding, threshold)

            # 将结果存入列表
            all_results.append({
                "id": pred_id,
                "score": float(pred_score),
                "box": [int(c) for c in bboxes[i]]
            })

        # 返回包含所有结果的列表
        return jsonify({
            "status": "success",
            "results": all_results
        })
    except Exception as e:
        print(f"An error occurred during multi-face search: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
# =================================================================

# =================================================================
# ===== API: 智能监控识别 (识别+绘制) =====
# =================================================================
@app.route('/api/monitor_recognize', methods=['POST'])
def api_monitor_recognize():
    """
    API: 专用于安防监控页面。
    接收一张图片，返回带有标注框和ID的Base64图片，以及事件列表。
    """
    try:
        image, _ = read_image_from_request()
        
        # 1. 人脸检测
        bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
        
        # 2. 准备绘制和结果记录
        result_image = image.copy() # 复制原图用于绘制
        events = []
        threshold = float(request.form.get('threshold', SCORE_THRESH))
        
        if bboxes is None or len(bboxes) == 0:
            # 未检测到人脸，直接返回原图的Base64
            _, buffer = cv2.imencode('.jpg', result_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                "status": "success",
                "annotated_image": "data:image/jpeg;base64," + image_base64,
                "events": []
            })

        # 3. 遍历每个人脸进行识别和绘制
        for i in range(len(bboxes)):
            box, score, landmark = bboxes[i], scores[i], landmarks[i]
            
            # 3.1 提取特征并搜索
            aligned_face = FaceAligner(image, landmark, None, vis=False)
            face_embedding = face_feature.get_faces_embedding([aligned_face])[0]
            pred_id, pred_score = face_register.search_face(face_embedding, threshold)
            
            # 3.2 准备绘制信息
            x1, y1, x2, y2 = map(int, box)
            is_known = pred_id != "unknown"
            color = (0, 255, 0) if is_known else (0, 0, 255) # 绿色代表已知，红色代表未知
            text = f"{pred_id} ({pred_score:.2f})"
            font_size = 25

            # 3.3 在复制的图片上绘制矩形框和文字
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - font_size - 5 if y1 - font_size - 5 > 0 else y1 + 10
            # 调用你已有的中文绘制函数
            result_image = draw_text_chinese(result_image, text, (x1, text_y), FONT_FILE, font_size, color)
            
            # 3.4 记录事件
            events.append({
                "personId": pred_id,
                "status": "允许进入" if is_known else "陌生访客",
                "score": float(pred_score)
            })

        # 4. 将绘制好的图片编码为Base64
        _, buffer = cv2.imencode('.jpg', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 5. 返回最终结果
        return jsonify({
            "status": "success",
            "annotated_image": "data:image/jpeg;base64," + image_base64,
            "events": events
        })
    except Exception as e:
        print(f"An error occurred during monitor recognition: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/rebuild_db', methods=['POST'])
def rebuild_database():
    if not face_detector or not face_feature or not face_register:
        return jsonify({"status": "error", "message": "Models are not initialized."}), 500
    print("\nRebuilding face database...")
    try:
        face_register.database = {}
        image_files = [os.path.join(DB_PORTRAIT_DIR, f) for f in os.listdir(DB_PORTRAIT_DIR) if allowed_file(f)]
        registered_count = 0
        for image_path in image_files:
            try:
                face_id = os.path.basename(image_path).split('.')[0].split('-')[0]
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None: continue
                bboxes, scores, landmarks = face_detector.detect_face_landmarks(image)
                if bboxes is None or len(bboxes) == 0: continue
                best_idx = np.argmax(scores)
                landmark = landmarks[best_idx]
                aligned_face = FaceAligner(image, landmark, None, vis=False)
                face_embedding = face_feature.get_faces_embedding([aligned_face])[0]
                face_register.add_face(face_id, face_embedding)
                registered_count += 1
                print(f"Registered face from {os.path.basename(image_path)}: {face_id}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        face_register.save()
        message = f"Face database rebuilt successfully. Registered {registered_count} faces."
        print(message)
        return jsonify({"status": "success", "message": message})
    except Exception as e:
        print(f"Error during database rebuild: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# =================================================================
# ===== API: 人员库管理 =====
# =================================================================

@app.route('/api/persons', methods=['GET'])
def api_get_persons():
    """获取所有已注册人员的ID列表"""
    if not face_register:
        return jsonify({"status": "error", "message": "人脸注册模块未初始化"}), 500
    try:
        # face_register.database 的键就是所有注册的 person_id
        person_ids = list(face_register.database.keys())
        return jsonify({"status": "success", "persons": person_ids})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/persons/<string:person_id>', methods=['DELETE'])
def api_delete_person(person_id):
    """根据ID删除一个已注册的人员"""
    if not face_register:
        return jsonify({"status": "error", "message": "人脸注册模块未初始化"}), 500
    try:
        # 检查该ID是否存在于数据库中
        if person_id in face_register.database:
            # 从内存中的数据库字典里删除
            del face_register.database[person_id]
            # 将改动保存回 aface_database.json 文件
            face_register.save()
            return jsonify({"status": "success", "message": f"人员 '{person_id}' 已成功删除"})
        else:
            return jsonify({"status": "error", "message": f"未找到人员ID: {person_id}"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

        
if __name__ == '__main__':
    initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=True)