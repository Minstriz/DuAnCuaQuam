from flask import Flask, render_template, Response, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import datetime
import random
import pygame
import time
from threading import Thread

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

app = Flask(__name__)
app.secret_key = 'abc123412123'

# Paths
LOG_FILE = 'attendance_log.csv'
AUDIO_SUCCESS = 'Audio/success-1-6297.mp3'
AUDIO_FAIL = 'Audio/wrong_5.mp3'
FACES_DIR = './faces'
DET_MODEL = './weights/det_10g.onnx'
REC_MODEL = './weights/w600k_r50.onnx'

# Check files
if not os.path.exists(DET_MODEL):
    raise FileNotFoundError(f"Detection model not found at {DET_MODEL}")
if not os.path.exists(REC_MODEL):
    raise FileNotFoundError(f"Recognition model not found at {REC_MODEL}")
if not os.path.exists(AUDIO_SUCCESS):
    raise FileNotFoundError(f"Success sound file not found at {AUDIO_SUCCESS}")
if not os.path.exists(AUDIO_FAIL):
    raise FileNotFoundError(f"Fail sound file not found at {AUDIO_FAIL}")

# Init sound
pygame.mixer.init()
try:
    success_sound = pygame.mixer.Sound(AUDIO_SUCCESS)
    fail_sound = pygame.mixer.Sound(AUDIO_FAIL)
except pygame.error as e:
    raise RuntimeError(f"Error loading sound files: {e}")

# Play sound async
def play_sound_async(sound):
    def _play():
        pygame.mixer.stop()
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
    Thread(target=_play).start()

# Load models
detector = SCRFD(DET_MODEL, input_size=(640, 640), conf_thres=0.3)
recognizer = ArcFace(REC_MODEL)

# Build targets
def build_targets():
    targets = []
    for filename in os.listdir(FACES_DIR):
        name = os.path.splitext(filename)[0]
        path = os.path.join(FACES_DIR, filename)
        image = cv2.imread(path)
        bboxes, kpss = detector.detect(image, max_num=1)
        if len(kpss) == 0:
            continue
        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))
    return targets

targets = build_targets()
colors = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _, name in targets}

last_log_time = 0

# Init CSV
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("Thời gian,Tên,Check-in,Check-out\n")

def log_attendance(name, status):
    global last_log_time
    current_time = time.time()

    if name != "Unknown" and current_time - last_log_time >= 3:
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_in = time_now if status == "checkin" else ""
        check_out = time_now if status == "checkout" else ""
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{time_now},{name},{check_in},{check_out}\n")
        play_sound_async(success_sound)
        last_log_time = current_time
        return True
    return False

def generate_frames(action):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    start_time = time.time()
    name_detected = "Unknown"
    logged = False
    success = False

    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        bboxes, kpss = detector.detect(frame, max_num=1)

        if len(bboxes) > 0:
            for bbox, kps in zip(bboxes, kpss):
                embedding = recognizer(frame, kps)
                best_match = "Unknown"
                max_sim = 0

                for emb_target, name in targets:
                    sim = compute_similarity(embedding, emb_target)
                    if sim > max_sim and sim > 0.4:
                        max_sim = sim
                        best_match = name

                if best_match != "Unknown":
                    color = colors[best_match]
                    draw_bbox_info(frame, bbox.astype(int), max_sim, best_match, color)
                    name_detected = best_match
                else:
                    draw_bbox(frame, bbox.astype(int), (0, 0, 255))

        if not logged and name_detected != "Unknown":
            success = log_attendance(name_detected, action)
            logged = True

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    # ✅ Phát âm thanh & thông báo sau 3 giây
    if success:
        print("✅ Nhận diện thành công, phát âm thanh OK")
        flash(f"{'Ghi vào' if action == 'checkin' else 'Ghi ra'} thành công cho {name_detected}", "success")
        success_sound.play()
    else:
        print("❌ Không nhận diện được, phát âm thanh thất bại")
        flash(f"{'Ghi vào' if action == 'checkin' else 'Ghi ra'} thất bại: Không nhận diện được khuôn mặt", "error")
        fail_sound.play()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    action = request.args.get('action', 'checkin')
    return Response(generate_frames(action), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/checkin', methods=['POST'])
def checkin():
    return redirect(url_for('video_feed', action='checkin'))

@app.route('/checkout', methods=['POST'])
def checkout():
    return redirect(url_for('video_feed', action='checkout'))

if __name__ == '__main__':
    app.run(debug=True)
