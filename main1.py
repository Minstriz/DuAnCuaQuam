import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np

import onnxruntime
from typing import Union, List, Tuple
from datetime import datetime

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition")
    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_r50.onnx")
    parser.add_argument("--similarity-thresh", type=float, default=0.4)
    parser.add_argument("--confidence-thresh", type=float, default=0.5)
    parser.add_argument("--faces-dir", type=str, default="./faces")
    parser.add_argument("--source", type=str, default="0")  # default webcam
    parser.add_argument("--max-num", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def log_attendance(name: str, status: str):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/attendance.txt", "a") as f:
        f.write(f"{timestamp} | {name} | {status}\n")
    print(f"📝 {status} - {name} at {timestamp}")


def build_targets(detector, recognizer, params: argparse.Namespace) -> List[Tuple[np.ndarray, str]]:
    targets = []
    for filename in os.listdir(params.faces_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        name = filename.rsplit(".", 1)[0]
        image_path = os.path.join(params.faces_dir, filename)
        image = cv2.imread(image_path)
        bboxes, kpss = detector.detect(image, max_num=1)
        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue
        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))
    return targets


def frame_processor(
    frame: np.ndarray,
    detector: SCRFD,
    recognizer: ArcFace,
    targets: List[Tuple[np.ndarray, str]],
    colors: dict,
    params: argparse.Namespace
) -> Tuple[np.ndarray, str]:
    bboxes, kpss = detector.detect(frame, params.max_num)
    best_match_name = "Unknown"
    max_similarity = 0

    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        if best_match_name != "Unknown":
            color = colors[best_match_name]
            draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=color)
        else:
            draw_bbox(frame, bbox, (255, 0, 0))

    return frame, best_match_name


def main(params):
    setup_logging(params.log_level)

    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)

    targets = build_targets(detector, recognizer, params)
    if len(targets) == 0:
        raise Exception("No valid faces found in faces folder.")

    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    if params.source.isdigit():
        params.source = int(params.source)

    cap = cv2.VideoCapture(params.source)
    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    print("🎥 Press A = Check-in | B = Check-out | Q = Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, best_match_name = frame_processor(frame, detector, recognizer, targets, colors, params)

        cv2.imshow("Face Attendance", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a") and best_match_name != "Unknown":
            log_attendance(best_match_name, "Check-in")
        elif key == ord("b") and best_match_name != "Unknown":
            log_attendance(best_match_name, "Check-out")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
