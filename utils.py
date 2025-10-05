import numpy as np
import os
import json

WRIST = 0
MIDDLE_MCP = 9

DATA_DIR = "dataset_json"   # directory to keep JSON files
os.makedirs(DATA_DIR, exist_ok=True)


def landmarks_to_np(landmarks):
    """landmarks: list of 21 mediapipe landmark objects with .x,.y,.z
       returns np.array shape (21,3)
    """
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    return arr

def hand_size(pts: np.array):
    # scale using distance wrist <-> middle_mcp (or max pairwise)
    return np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])


def make_feature_vector(landmarks):
    pts = landmarks_to_np(landmarks)  # (21,3)
    # relative coords to wrist
    rel = pts - pts[WRIST]
    s = hand_size(pts)
    rel /= s  # normalize scale
    rel_flat = rel.flatten()  # 21*3 = 63
    # pairwise distances between tips (index, middle, ring, pinky, thumb)
    tip_inds = [4, 8, 12, 16, 20]
    tips = pts[tip_inds] - pts[WRIST]
    tips /= s
    dists = []
    for i in range(len(tip_inds)):
        for j in range(i+1, len(tip_inds)):
            dists.append(np.linalg.norm(tips[i] - tips[j]))
    dists = np.array(dists, dtype=np.float32)  # 10 distances
    feat = np.concatenate([rel_flat, dists])
    return feat  # shape = 63 + 10 = 73


def save_sample(label, landmarks):
    """Save landmarks/features to a JSON file named <label>.json"""
    filepath = os.path.join(DATA_DIR, f"{label}.json")

    # convert landmarks into simple nested list (not numpy)
    feat = make_feature_vector(landmarks).tolist()

    # if file exists, load it, else start new list
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = []

    # append new feature vector
    data.append(feat)

    # save back to file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved sample for '{label}' -> {filepath} (total {len(data)} samples)")
