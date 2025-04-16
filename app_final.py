from flask import Flask, render_template, Response, request, redirect, url_for, flash, session, jsonify
from flask_socketio import SocketIO, emit, disconnect
from functools import wraps
import os
import cv2
import numpy as np
import datetime
import random
import pygame
import time
import pandas as pd
from threading import Thread, Lock
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
import csv

app = Flask(__name__)
app.secret_key = 'abc123412123'
app.config['SESSION_TYPE'] = 'filesystem'
socketio = SocketIO(app)

# Paths
LOG_FILE = 'attendance_log.csv'
AUDIO_SUCCESS = 'Audio/success-1-6297.mp3'
AUDIO_FAIL = 'Audio/wrong_5.mp3'
FACES_DIR = './faces'
DET_MODEL = './weights/det_10g.onnx'
REC_MODEL = './weights/w600k_r50.onnx'

# Locks
video_lock = Lock()
sound_lock = Lock()
log_lock = Lock()

# Check files
if not os.path.exists(DET_MODEL):
    raise FileNotFoundError(f"Detection model not found at {DET_MODEL}")
if not os.path.exists(REC_MODEL):
    raise FileNotFoundError(f"Recognition model not found at {REC_MODEL}")
if not os.path.exists(AUDIO_SUCCESS):
    raise FileNotFoundError(f"Success sound file not found at {AUDIO_SUCCESS}")
if not os.path.exists(AUDIO_FAIL):
    raise FileNotFoundError(f"Fail sound file not found at {AUDIO_FAIL}")
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

# Init log file
def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Thời gian", "Tên", "Check-in", "Check-out"])
    else:
        # Kiểm tra và sửa file nếu cần
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines or lines[0].strip() != "Thời gian,Tên,Check-in,Check-out":
            with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Thời gian", "Tên", "Check-in", "Check-out"])
                # Chỉ giữ các dòng có đúng 4 cột (đã bỏ cột Status)
                for line in lines[1:]:
                    cols = line.strip().split(',')
                    if len(cols) >= 4:  # Nếu có ít nhất 4 cột
                        writer.writerow(cols[:4])  # Chỉ lấy 4 cột đầu

init_log_file()

# Init sound
try:
    pygame.mixer.init()
except pygame.error as e:
    raise RuntimeError(f"Error initializing pygame mixer: {e}")

try:
    success_sound = pygame.mixer.Sound(AUDIO_SUCCESS)
    fail_sound = pygame.mixer.Sound(AUDIO_FAIL)
except pygame.error as e:
    raise RuntimeError(f"Error loading sound files: {e}")

# Play sound async
def play_sound_async(sound):
    def _play():
        with sound_lock:
            pygame.mixer.stop()
            sound.play()
            while pygame.mixer.get_busy():
                pygame.time.wait(100)
    Thread(target=_play).start()

# Load models
detector = SCRFD(DET_MODEL, input_size=(640, 640), conf_thres=0.5)
recognizer = ArcFace(REC_MODEL)

# Build targets
def build_targets():
    targets = []
    for filename in os.listdir(FACES_DIR):
        name = os.path.splitext(filename)[0]
        path = os.path.join(FACES_DIR, filename)
        image = cv2.imread(path)
        if image is None:
            continue
        bboxes, kpss = detector.detect(image, max_num=1)
        if len(kpss) == 0:
            continue
        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))
    return targets

targets = build_targets()
colors = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _, name in targets}

last_log_time = 0
COOLDOWN_SECONDS = 5

# SỬA HÀM LOG_ATTENDANCE
def log_attendance(name, status):
    global last_log_time
    current_time = time.time()
    
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    check_in = time_now if status == "checkin" else ""
    check_out = time_now if status == "checkout" else ""
    
    if name and name != "Unknown" and current_time - last_log_time >= COOLDOWN_SECONDS:
        with log_lock:
            # Thoát dấu phẩy trong name
            safe_name = name.replace(',', '_')
            with open(LOG_FILE, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time_now, safe_name, check_in, check_out])  # Đã bỏ status
            last_log_time = current_time
            try:
                df = pd.read_csv(LOG_FILE, on_bad_lines='skip').tail(5)[::-1]
                socketio.emit('attendance_update', df.to_dict(orient='records'), namespace='/admin')
            except Exception as e:
                print(f"Error reading CSV: {e}")
        play_sound_async(success_sound)
        return True
    else:
        play_sound_async(fail_sound)
        return False

# SỬA HÀM GENERATE_FRAMES
def generate_frames(action, user_agent):
    if not video_lock.acquire(blocking=False):
        print("Another video session is active")
        error_frame = create_error_frame("Video đang được sử dụng")
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, "Video đang được sử dụng")
        return

    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            error_frame = create_error_frame("Không thể mở camera")
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, "Không thể mở camera")
            return

        # Thiết lập kích thước đầu vào đúng cho model
        detector.input_size = (640, 640)
        
        name_detected = None
        success = False
        face_detected = False
        start_time = time.time()
        duration = 5  # Duy trì camera trong 5 giây
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                error_frame = create_error_frame("Không thể đọc frame")
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, "Không thể đọc frame")
                time.sleep(0.1)
                continue
                
            frame = cv2.flip(frame, 1)
            # Resize frame để tối ưu băng thông
            frame = cv2.resize(frame, (640, 480))  # Giảm độ phân giải xuống 640x480
            original_size = frame.shape[:2]
            
            # Resize frame cho detector
            frame_for_detection = cv2.resize(frame, detector.input_size)
            
            try:
                bboxes, kpss = detector.detect(frame_for_detection, max_num=1)
                scale_y = original_size[0] / detector.input_size[1]
                scale_x = original_size[1] / detector.input_size[0]
                
                if len(bboxes) > 0:
                    face_detected = True
                    for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
                        bbox_scaled = bbox.copy()
                        bbox_scaled[0] *= scale_x
                        bbox_scaled[1] *= scale_y
                        bbox_scaled[2] *= scale_x
                        bbox_scaled[3] *= scale_y
                        
                        kps_scaled = kps.copy()
                        for j in range(kps.shape[0]):
                            kps_scaled[j][0] *= scale_x
                            kps_scaled[j][1] *= scale_y
                        
                        embedding = recognizer(frame, kps_scaled)
                        best_match = "Unknown"
                        max_sim = 0
                        
                        for emb_target, name in targets:
                            sim = compute_similarity(embedding, emb_target)
                            if sim > max_sim and sim > 0.6:
                                max_sim = sim
                                best_match = name
                                
                        if best_match != "Unknown":
                            color = colors.get(best_match, (0, 255, 0))
                            draw_bbox_info(frame, bbox_scaled.astype(int), max_sim, best_match, color)
                            name_detected = best_match
                        else:
                            draw_bbox(frame, bbox_scaled.astype(int), (0, 0, 255))
                            name_detected = "Unknown"
            
                if face_detected and name_detected and name_detected != "Unknown":
                    success = log_attendance(name_detected, action)
                
            except Exception as e:
                print(f"Error in face detection/recognition: {e}")
                error_frame = create_error_frame(f"Lỗi nhận diện: {str(e)}")
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, f"Lỗi nhận diện: {str(e)}")
                time.sleep(0.1)
                continue
            
            # Chuyển frame thành JPEG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'), (success, name_detected)
            
            time.sleep(0.03)  # ~30 FPS
        
        # Kết quả cuối cùng
        if not face_detected:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'), (False, "Không phát hiện khuôn mặt")
        elif name_detected == "Unknown":
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'), (False, "Không nhận diện được khuôn mặt")
                
    except Exception as e:
        print(f"Error in generate_frames: {e}")
        error_frame = create_error_frame(f"Lỗi camera: {str(e)}")
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, f"Lỗi camera: {str(e)}")
    finally:
        if cap and cap.isOpened():
            cap.release()
        video_lock.release()
        print("Camera released")

# Hàm tạo frame lỗi
def create_error_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()
# Hard-coded users
users = {
    "user1": {"password": "pass123", "role": "user"},
    "admin1": {"password": "admin123", "role": "admin"}
}

# Login required decorator
def login_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' not in session or session['role'] != role:
                flash("Vui lòng đăng nhập!", "error")
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        if username in users and users[username]["password"] == password and users[username]["role"] == role:
            session['username'] = username
            session['role'] = role
            if role == 'user':
                return redirect(url_for('index'))
            elif role == 'admin':
                return redirect(url_for('admin'))
        else:
            flash("Sai tên đăng nhập hoặc mật khẩu!", "error")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    flash("Đã đăng xuất!", "success")
    return redirect(url_for('login'))

@app.route('/index')
@login_required('user')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
@login_required('admin')
def admin():
    global targets, colors

    if request.method == 'POST':
        if 'file' not in request.files or 'name' not in request.form:
            flash("Vui lòng chọn file và nhập tên!", "error")
        else:
            file = request.files['file']
            name = request.form['name'].strip()
            if file.filename == '' or not name:
                flash("File hoặc tên không hợp lệ!", "error")
            elif not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                flash("Chỉ hỗ trợ file .jpg, .jpeg, .png!", "error")
            else:
                filename = f"{name}.jpg"
                filepath = os.path.join(FACES_DIR, filename)
                if os.path.exists(filepath):
                    flash("Tên nhân viên đã tồn tại!", "error")
                else:
                    file.save(filepath)
                    targets = build_targets()
                    colors = {n: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _, n in targets}
                    flash(f"Đã thêm nhân viên {name} thành công!", "success")

    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip').tail(5)[::-1]
        data = df.to_dict(orient='records')
    except Exception as e:
        print(f"Error reading admin CSV: {e}")
        flash("Không tìm thấy file log!", "error")
        data = []
    return render_template('admin.html', data=data)

@app.route('/get_attendance')
@login_required('admin')
def get_attendance():
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip').tail(5)[::-1]
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        print(f"Error reading attendance CSV: {e}")
        return jsonify([])

@app.route('/video_feed')
@login_required('user')
def video_feed():
    action = request.args.get('action', 'checkin')
    user_agent = request.headers.get('User-Agent', '').lower()
    
    def stream():
        for frame, _ in generate_frames(action, user_agent):
            yield frame
    
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/checkin', methods=['POST'])
@login_required('user')
def checkin():
    action = 'checkin'
    user_agent = request.headers.get('User-Agent', '').lower()
    
    # Gọi generate_frames để nhận diện
    for _, result in generate_frames(action, user_agent):
        success, name_detected = result
        break
    else:
        success, name_detected = None, None
    
    if success is None:
        flash(f"Ghi vào thất bại: {name_detected or 'Lỗi không xác định'}", "error")
    elif success:
        flash(f"Ghi vào thành công cho {name_detected}", "success")
    else:
        if name_detected is None:
            reason = "Không phát hiện khuôn mặt (có thể camera bị che)"
        elif name_detected == "Unknown":
            reason = "Không nhận diện được khuôn mặt"
        else:
            reason = "Điểm danh quá gần lần trước"
        flash(f"Ghi vào thất bại: {reason}", "error")
    
    return redirect(url_for('index'))

@app.route('/checkout', methods=['POST'])
@login_required('user')
def checkout():
    action = 'checkout'
    user_agent = request.headers.get('User-Agent', '').lower()
    
    # Gọi generate_frames để nhận diện
    for _, result in generate_frames(action, user_agent):
        success, name_detected = result
        break
    else:
        success, name_detected = None, None
    
    if success is None:
        flash(f"Ghi ra thất bại: {name_detected or 'Lỗi không xác định'}", "error")
    elif success:
        flash(f"Ghi ra thành công cho {name_detected}", "success")
    else:
        if name_detected is None:
            reason = "Không phát hiện khuôn mặt (có thể camera bị che)"
        elif name_detected == "Unknown":
            reason = "Không nhận diện được khuôn mặt"
        else:
            reason = "Điểm danh quá gần lần trước"
        flash(f"Ghi ra thất bại: {reason}", "error")
    
    return redirect(url_for('index'))

@socketio.on('connect', namespace='/admin')
def handle_connect():
    if 'username' not in session or session['role'] != 'admin':
        disconnect()
    else:
        print('Admin connected to /admin namespace')

if __name__ == '__main__':
    socketio.run(app, debug=True)