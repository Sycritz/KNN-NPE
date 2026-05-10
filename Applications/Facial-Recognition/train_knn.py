"""
k-NN Training Pipeline — extracts 512-d FaceNet embeddings from enrolled
face crops and trains a distance-weighted KNeighborsClassifier.
Usage: python train_knn.py
"""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent / "face_data"
MODEL_PATH = Path(__file__).parent / "knn_face_model.pkl"
CROP_SIZE = 160
K = 5


def load_embedder(device: torch.device) -> InceptionResnetV1:
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    tensor = (tensor - 127.5) / 128.0
    return tensor


def extract_embeddings(device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    embedder = load_embedder(device)
    identities = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    if not identities:
        sys.exit(f"No identity folders found in {DATA_DIR}")

    all_embeddings = []
    all_labels = []

    for identity_dir in identities:
        images = sorted(identity_dir.glob("*.png"))
        if not images:
            print(f"Skipping empty folder: {identity_dir.name}")
            continue

        print(f"Processing '{identity_dir.name}' — {len(images)} images")
        batch = torch.stack([preprocess(cv2.imread(str(p))) for p in images])
        batch = batch.to(device)

        with torch.no_grad():
            embeddings = embedder(batch).cpu().numpy()

        all_embeddings.append(embeddings)
        all_labels.extend([identity_dir.name] * len(embeddings))

    X = np.vstack(all_embeddings)
    y = np.array(all_labels)
    return X, y


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y = extract_embeddings(device)
    print(f"Embeddings: {X.shape}, Classes: {np.unique(y)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    k = min(K, len(X) - 1)
    clf = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric="euclidean",
        algorithm="ball_tree",
    )
    clf.fit(X, y_enc)

    payload = {
        "classifier": clf,
        "label_encoder": le,
        "embedding_dim": X.shape[1],
        "k": k,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Model saved → {MODEL_PATH}  (k={k}, {len(le.classes_)} classes)")


if __name__ == "__main__":
    train()
