"""
k-NN Face Recognition — Kaggle Notebook Script
Cells are separated by # %% markers for easy copy-paste.
"""

# %% [1] Dependencies
# !pip install facenet-pytorch scikit-learn opencv-python-headless pillow -q

# %% [2] Imports

import os
import pickle
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from IPython.display import display

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("face_data")
MODEL_PATH = Path("knn_face_model.pkl")
CROP_SIZE = 160
K = 5

print(f"Device: {DEVICE}")

# %% [3] Download sample face images (LFW subset)

LFW_BASE = "https://vis-www.cs.umass.edu/lfw/images"

IDENTITIES = {
    "George_W_Bush": [
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0001.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0002.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0003.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0004.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0005.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0006.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0007.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0008.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0009.jpg",
        f"{LFW_BASE}/George_W_Bush/George_W_Bush_0010.jpg",
    ],
    "Colin_Powell": [
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0001.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0002.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0003.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0004.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0005.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0006.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0007.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0008.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0009.jpg",
        f"{LFW_BASE}/Colin_Powell/Colin_Powell_0010.jpg",
    ],
    "Tony_Blair": [
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0001.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0002.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0003.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0004.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0005.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0006.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0007.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0008.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0009.jpg",
        f"{LFW_BASE}/Tony_Blair/Tony_Blair_0010.jpg",
    ],
}

TEST_IMAGES = {
    "George_W_Bush": f"{LFW_BASE}/George_W_Bush/George_W_Bush_0011.jpg",
    "Colin_Powell": f"{LFW_BASE}/Colin_Powell/Colin_Powell_0011.jpg",
    "Tony_Blair": f"{LFW_BASE}/Tony_Blair/Tony_Blair_0011.jpg",
}

for name, urls in IDENTITIES.items():
    save_dir = DATA_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(urls):
        dest = save_dir / f"{i:04d}.jpg"
        if not dest.exists():
            urlretrieve(url, str(dest))
    print(f"Downloaded {len(urls)} images for '{name}'")

test_dir = Path("test_images")
test_dir.mkdir(exist_ok=True)
for name, url in TEST_IMAGES.items():
    dest = test_dir / f"{name}.jpg"
    if not dest.exists():
        urlretrieve(url, str(dest))
print(f"Downloaded {len(TEST_IMAGES)} test images")

# %% [4] Face Detection and Cropping (Enrollment)

detector = MTCNN(
    image_size=CROP_SIZE,
    margin=20,
    keep_all=False,
    min_face_size=40,
    post_process=False,
    device=DEVICE,
)

crop_dir = Path("face_crops")

for name in IDENTITIES:
    src_dir = DATA_DIR / name
    dst_dir = crop_dir / name
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(src_dir.glob("*.jpg"))
    cropped = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, confs = detector.detect(rgb)
        if boxes is None or confs[0] is None or confs[0] < 0.9:
            continue

        x1, y1, x2, y2 = boxes[0].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)

        face = rgb[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.resize(face, (CROP_SIZE, CROP_SIZE))
        cv2.imwrite(str(dst_dir / img_path.name),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        cropped += 1

    print(f"Cropped {cropped}/{len(images)} faces for '{name}'")

# %% [5] Feature Extraction (512-d FaceNet Embeddings)

embedder = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    tensor = (tensor - 127.5) / 128.0
    return tensor


all_embeddings = []
all_labels = []

for name in sorted(IDENTITIES.keys()):
    identity_dir = crop_dir / name
    images = sorted(identity_dir.glob("*.jpg"))
    if not images:
        print(f"Skipping '{name}' — no crops found")
        continue

    batch = torch.stack([preprocess(cv2.imread(str(p))) for p in images])
    batch = batch.to(DEVICE)

    with torch.no_grad():
        embeddings = embedder(batch).cpu().numpy()

    all_embeddings.append(embeddings)
    all_labels.extend([name] * len(embeddings))
    print(f"Extracted {len(embeddings)} embeddings for '{name}'")

X = np.vstack(all_embeddings)
y = np.array(all_labels)
print(f"\nTotal: {X.shape[0]} embeddings, {len(np.unique(y))} classes")
print(f"Embedding shape: {X.shape}")

# %% [6] Train k-NN Classifier

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

print(f"k-NN model trained (k={k}, {len(le.classes_)} classes)")
print(f"Classes: {list(le.classes_)}")
print(f"Saved to {MODEL_PATH}")

# %% [7] Inference on Test Images

UNKNOWN_THRESHOLD = 0.95

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

clf = model_data["classifier"]
le = model_data["label_encoder"]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(TEST_IMAGES), figsize=(5 * len(TEST_IMAGES), 5))

for ax, (true_name, _) in zip(axes, TEST_IMAGES.items()):
    img_path = test_dir / f"{true_name}.jpg"
    img = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, confs = detector.detect(rgb)

    if boxes is not None and confs[0] is not None and confs[0] >= 0.9:
        x1, y1, x2, y2 = boxes[0].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)

        face = rgb[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (CROP_SIZE, CROP_SIZE))

        tensor = preprocess(cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
        with torch.no_grad():
            emb = embedder(tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()

        distances, indices = clf.kneighbors(emb, n_neighbors=1)
        dist = distances[0][0]
        pred_idx = clf.predict(emb)[0]
        pred_name = le.inverse_transform([pred_idx])[0]

        if dist > UNKNOWN_THRESHOLD:
            label = "Unknown"
            confidence = 0.0
        else:
            confidence = max(0.0, 1.0 - dist / UNKNOWN_THRESHOLD)
            label = f"{pred_name} ({confidence:.0%})"

        color = "green" if pred_name == true_name else "red"
        cv2.rectangle(rgb, (x1, y1), (x2, y2),
                      (0, 255, 0) if color == "green" else (255, 0, 0), 3)
    else:
        label = "No face detected"
        color = "gray"

    ax.imshow(rgb)
    ax.set_title(f"True: {true_name}\nPred: {label}", fontsize=11,
                 color=color, fontweight="bold")
    ax.axis("off")

plt.suptitle("k-NN Face Recognition — Test Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("recognition_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Results saved to recognition_results.png")

# %%
