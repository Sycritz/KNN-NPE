"""
Face Enrollment — captures face crops from webcam and stores them per identity.
Usage: python enroll_faces.py --name "John"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

DATA_DIR = Path(__file__).parent / "face_data"
CAPTURE_SIZE = (640, 480)
CROP_SIZE = 160
NUM_FRAMES = 30
MIN_FACE_CONFIDENCE = 0.97


def build_detector(device: torch.device) -> MTCNN:
    return MTCNN(
        image_size=CROP_SIZE,
        margin=20,
        keep_all=False,
        min_face_size=60,
        thresholds=[0.7, 0.8, 0.9],
        post_process=False,
        device=device,
    )


def enroll(name: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = build_detector(device)

    save_dir = DATA_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(save_dir.glob("*.png")))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])

    collected = 0
    print(f"Enrolling '{name}' — collecting {NUM_FRAMES} face crops. Press 'q' to abort.")

    while collected < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box, conf = detector.detect(rgb)

        status = f"Captured: {collected}/{NUM_FRAMES}"

        if box is not None and conf[0] is not None and conf[0] >= MIN_FACE_CONFIDENCE:
            x1, y1, x2, y2 = box[0].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)

            face_rgb = rgb[y1:y2, x1:x2]
            if face_rgb.size > 0:
                face_rgb = cv2.resize(face_rgb, (CROP_SIZE, CROP_SIZE))
                idx = existing + collected
                cv2.imwrite(str(save_dir / f"{idx:04d}.png"),
                            cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
                collected += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf[0]:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Face Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected == NUM_FRAMES:
        print(f"Done — {collected} crops saved to {save_dir}")
    else:
        print(f"Aborted — {collected} crops saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a face identity from webcam.")
    parser.add_argument("--name", type=str, required=True, help="Identity label.")
    parser.add_argument("--frames", type=int, default=NUM_FRAMES,
                        help="Number of crops to capture.")
    args = parser.parse_args()
    NUM_FRAMES = args.frames
    enroll(args.name)
