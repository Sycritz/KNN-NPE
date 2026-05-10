"""
Live Face Recognition — real-time webcam inference using a serialised k-NN
model with an "Unknown" distance threshold to reject false positives.
Usage: python recognize_live.py
"""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

MODEL_PATH = Path(__file__).parent / "knn_face_model.pkl"
CAPTURE_SIZE = (640, 480)
DETECT_SCALE = 0.5
CROP_SIZE = 160
UNKNOWN_THRESHOLD = 0.95
BOX_COLORS = {
    "known": (0, 200, 80),
    "unknown": (0, 80, 220),
}


def load_model() -> dict:
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run train_knn.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_pipeline(device: torch.device) -> tuple[MTCNN, InceptionResnetV1]:
    detector = MTCNN(
        image_size=CROP_SIZE,
        margin=20,
        keep_all=True,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.8],
        post_process=False,
        device=device,
    )
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return detector, embedder


def preprocess_crop(face_tensor: torch.Tensor) -> torch.Tensor:
    return ((face_tensor.float() - 127.5) / 128.0).unsqueeze(0)


def recognise() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_data = load_model()
    clf = model_data["classifier"]
    le = model_data["label_encoder"]

    detector, embedder = build_pipeline(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])

    print("Live recognition running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))

        boxes, confs = detector.detect(small)

        if boxes is not None:
            boxes = boxes / DETECT_SCALE
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                if conf is None or conf < 0.85:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_rgb = rgb[y1:y2, x1:x2]
                if face_rgb.size == 0:
                    continue

                face_rgb = cv2.resize(face_rgb, (CROP_SIZE, CROP_SIZE))
                face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
                face_tensor = (face_tensor - 127.5) / 128.0

                with torch.no_grad():
                    embedding = embedder(face_tensor.unsqueeze(0).to(device))
                emb_np = embedding.cpu().numpy()

                distances, indices = clf.kneighbors(emb_np, n_neighbors=1)
                dist = distances[0][0]
                pred_idx = clf.predict(emb_np)[0]
                name = le.inverse_transform([pred_idx])[0]

                if dist > UNKNOWN_THRESHOLD:
                    label = "Unknown"
                    confidence = 0.0
                    color = BOX_COLORS["unknown"]
                else:
                    confidence = max(0.0, 1.0 - dist / UNKNOWN_THRESHOLD)
                    label = f"{name} ({confidence:.0%})"
                    color = BOX_COLORS["known"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        n_faces = len(boxes) if boxes is not None else 0
        cv2.putText(frame, f"Faces: {n_faces}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("k-NN Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognise()
