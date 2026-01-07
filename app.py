import os
import io
import base64
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

import cv2
import numpy as np

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100MB limit
app.config['MAX_FORM_MEMORY_SIZE'] = 100 * 1024 * 1024 # Allow large forms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'static', 'processed')
DB_PATH = os.path.join(BASE_DIR, 'database.db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            gender TEXT,
            age TEXT,
            image_filename TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

# Load models once
# ... (models loaded here) ...
FACE_PROTO = os.path.join(BASE_DIR, 'opencv_face_detector.pbtxt')
FACE_MODEL = os.path.join(BASE_DIR, 'opencv_face_detector_uint8.pb')
AGE_PROTO = os.path.join(BASE_DIR, 'age_deploy.prototxt')
AGE_MODEL = os.path.join(BASE_DIR, 'age_net.caffemodel')
GENDER_PROTO = os.path.join(BASE_DIR, 'gender_deploy.prototxt')
GENDER_MODEL = os.path.join(BASE_DIR, 'gender_net.caffemodel')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)


# Prefer OpenCV DNN backend on CPU for stability
try:
    for net in (faceNet, ageNet, genderNet):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
except Exception:
    pass


def highlight_face(net, frame, conf_threshold=0.5):
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), max(1, h // 150))
    return frame_copy, boxes


def classify_face_tta(face_img):
    """
    Perform inference using Test Time Augmentation (TTA).
    We predict on the original image and a horizontally flipped version,
    then average the probabilities.
    """
    # 1. Prepare blobs
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    face_flipped = cv2.flip(face_img, 1)
    blob_flipped = cv2.dnn.blobFromImage(face_flipped, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    # 2. Gender Prediction (Average)
    genderNet.setInput(blob)
    g1 = genderNet.forward()
    genderNet.setInput(blob_flipped)
    g2 = genderNet.forward()
    avg_g = (g1 + g2) / 2.0
    gender = GENDER_LIST[int(np.argmax(avg_g[0]))]
    
    # 3. Age Prediction (Average)
    ageNet.setInput(blob)
    a1 = ageNet.forward()
    ageNet.setInput(blob_flipped)
    a2 = ageNet.forward()
    avg_a = (a1 + a2) / 2.0
    age = AGE_LIST[int(np.argmax(avg_a[0]))]
    
    return gender, age


def run_inference(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Unable to read image')

    result_img, face_boxes = highlight_face(faceNet, img)
    if not face_boxes:
        return {
            'gender': 'Unknown',
            'age': 'Unknown',
            'result_img': result_img,
            'boxes': []
        }

    padding = 35 # Increased from 20 to capture more context (hair) for better gender detection
    last_gender = 'Unknown'
    last_age = 'Unknown'

    for (x1, y1, x2, y2) in face_boxes:
        y1c = max(0, y1 - padding)
        y2c = min(y2 + padding, img.shape[0] - 1)
        x1c = max(0, x1 - padding)
        x2c = min(x2 + padding, img.shape[1] - 1)
        face = img[y1c:y2c, x1c:x2c]
        
        if face.size == 0:
            continue
            
        # Use TTA for better accuracy
        last_gender, last_age = classify_face_tta(face)

        cv2.putText(
            result_img,
            f'{last_gender}, {last_age}',
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return {
        'gender': last_gender,
        'age': last_age,
        'result_img': result_img,
        'boxes': face_boxes,
    }


def save_image_from_dataurl(data_url, dest_folder):
    header, encoded = data_url.split(',', 1)
    binary = base64.b64decode(encoded)
    filename = datetime.utcnow().strftime('%Y%m%d%H%M%S%f') + '.png'
    path = os.path.join(dest_folder, filename)
    with open(path, 'wb') as f:
        f.write(binary)
    return filename, path


def run_inference_on_array(img_bgr):
    # This now returns a list of dicts: [{ 'box': [x1, y1, x2, y2], 'gender': 'Male', 'age': '(25-32)' }, ...]
    result_img, face_boxes = highlight_face(faceNet, img_bgr)
    if not face_boxes:
        return []

    padding = 20
    detections = []

    for (x1, y1, x2, y2) in face_boxes:
        y1c = max(0, y1 - padding)
        y2c = min(y2 + padding, img_bgr.shape[0] - 1)
        x1c = max(0, x1 - padding)
        x2c = min(x2 + padding, img_bgr.shape[1] - 1)
        face = img_bgr[y1c:y2c, x1c:x2c]
        
        if face.size == 0:
            continue
            
        # Use TTA
        gender, age = classify_face_tta(face)
        
        detections.append({
            'box': [x1, y1, x2, y2],
            'gender': gender,
            'age': age
        })

    return detections


def encode_image_to_dataurl(img_bgr):
    success, buf = cv2.imencode('.png', img_bgr)
    if not success:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f'data:image/png;base64,{b64}'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', '').strip()
    file = request.files.get('image')

    if not name:
        flash('Please enter a name.')
        return redirect(url_for('index'))

    saved_filename = None
    saved_path = None

    if file and file.filename:
        filename = secure_filename(file.filename)
        base, ext = os.path.splitext(filename)
        saved_filename = datetime.utcnow().strftime('%Y%m%d%H%M%S%f') + ext.lower()
        saved_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(saved_path)
    else:
        flash('Please upload an image or capture from camera.')
        return redirect(url_for('index'))

    try:
        # For the static upload result, we still want the drawn image
        result = run_inference(saved_path)
    except Exception as e:
        flash(f'Inference failed: {e}')
        return redirect(url_for('index'))

    processed_filename = 'processed_' + saved_filename
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    cv2.imwrite(processed_path, result['result_img'])

    # Log to DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'INSERT INTO detections (name, gender, age, image_filename, created_at) VALUES (?, ?, ?, ?, ?)',
        (
            name,
            result['gender'],
            result['age'],
            saved_filename,
            datetime.utcnow().isoformat()
        )
    )
    conn.commit()
    conn.close()

    return render_template(
        'index.html',
        name=name,
        gender=result['gender'],
        age=result['age'],
        processed_image=url_for('static', filename=f'processed/{processed_filename}')
    )


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.get_json(silent=True) or {}
    frame_data = data.get('frame')
    if not frame_data:
        return { 'error': 'No frame provided' }, 400
    try:
        header, encoded = frame_data.split(',', 1)
        binary = base64.b64decode(encoded)
        np_arr = np.frombuffer(binary, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return { 'error': 'Invalid image data' }, 400
            
        detections = run_inference_on_array(img_bgr)
        return { 'detections': detections }

    except Exception as e:
        print(e)
        return { 'error': str(e) }, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
