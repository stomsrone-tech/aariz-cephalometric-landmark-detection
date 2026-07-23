"""
ISBI2015Dataset — for use as a FROZEN external validation set only (no training on this data).

Calibrated to the Kaggle mirror schema (jiahongqian/cephalometric-landmarks):
- {root}/train_senior.csv, test1_senior.csv, test2_senior.csv
- Columns: image_path, 1_x, 1_y, 2_x, 2_y, ..., 19_x, 19_y  (numbered 1-19, no names)
- {root}/cepha400/cepha400/{image_path}.jpg

Landmark order 1-19 confirmed against three independent sources (converged, see chat):
Sella, Nasion, Orbitale, Porion, Subspinale(A-point), Supramentale(B-point), Pogonion,
Menton, Gnathion, Gonion, Lower Incisor Tip, Upper Incisor Tip, Upper Lip, Lower Lip,
Subnasale, Soft Tissue Pogonion, Posterior Nasal Spine, Anterior Nasal Spine, Articulare.

Pixel spacing: fixed 0.1 mm/pixel for all images (single device, Soredex CRANEX Excel,
1935x2400 resolution) — well-established constant across the ISBI2015 literature, unlike
the per-image variable pixel_size in Aariz's mapping CSV.

Only "senior" annotations are present in this particular mirror (no junior) — noted as a
minor limitation vs. the original dataset's dual annotations; standard practice in this
field treats a single expert annotator as acceptable ground truth for external validation.
"""

import os
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# Index 0 = ISBI landmark #1, ..., index 18 = ISBI landmark #19
ISBI_LANDMARK_ORDER = [
    "Sella", "Nasion", "Orbitale", "Porion", "A-point", "B-point", "Pogonion",
    "Menton", "Gnathion", "Gonion", "Lower Incisor Tip", "Upper Incisor Tip",
    "Labrale superius", "Labrale inferius", "Subnasale", "Soft Tissue Pogonion",
    "Posterior Nasal Spine", "Anterior Nasal Spine", "Articulare",
]
assert len(ISBI_LANDMARK_ORDER) == 19

ISBI_PIXEL_SIZE_MM = 0.1  # fixed for the whole dataset

# Import Aariz's canonical order so we can compute the overlap mask once, consistently.
try:
    from aariz_dataset import LANDMARK_ORDER as AARIZ_LANDMARK_ORDER
except ImportError:
    AARIZ_LANDMARK_ORDER = None  # allows standalone use/testing without aariz_dataset.py present


def get_aariz_overlap_indices():
    """Returns the list of indices into Aariz's 29-landmark order that correspond to
    ISBI's 19 landmarks, in ISBI order. Use this to slice a trained model's 29-channel
    output down to the 19 comparable landmarks for external validation.
    Requires aariz_dataset.py to be importable (for AARIZ_LANDMARK_ORDER)."""
    if AARIZ_LANDMARK_ORDER is None:
        raise ImportError("aariz_dataset.py not found — needed to compute the overlap mapping")
    indices = []
    for title in ISBI_LANDMARK_ORDER:
        if title not in AARIZ_LANDMARK_ORDER:
            raise ValueError(f"ISBI landmark '{title}' not found in Aariz's landmark list — mapping is broken")
        indices.append(AARIZ_LANDMARK_ORDER.index(title))
    return indices


class ISBI2015Dataset(Dataset):
    def __init__(self, root, split, image_size=512):
        """
        root: path containing train_senior.csv / test1_senior.csv / test2_senior.csv and cepha400/cepha400/
        split: 'train' | 'test1' | 'test2' | 'test' (test = test1+test2 combined, standard practice
               in this literature since both together form the official 250-image test set)
        """
        assert split in ("train", "test1", "test2", "test")
        self.root = root
        self.image_size = image_size
        self.image_dir = os.path.join(root, "cepha400", "cepha400")

        if split == "test":
            rows = self._load_csv(os.path.join(root, "test1_senior.csv"))
            rows += self._load_csv(os.path.join(root, "test2_senior.csv"))
        else:
            rows = self._load_csv(os.path.join(root, f"{split}_senior.csv"))
        self.rows = rows

    @staticmethod
    def _load_csv(path):
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.image_dir, row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        landmarks_px = np.zeros((19, 2), dtype=np.float32)
        for i in range(19):
            landmarks_px[i, 0] = float(row[f"{i+1}_x"])
            landmarks_px[i, 1] = float(row[f"{i+1}_y"])

        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        landmarks_resized = landmarks_px.copy()
        landmarks_resized[:, 0] *= scale_x
        landmarks_resized[:, 1] *= scale_y

        return {
            "image_path": row["image_path"],
            "image": torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0,
            "landmarks_px_resized": torch.from_numpy(landmarks_resized).float(),
            "orig_size": (orig_h, orig_w),
            "pixel_size_mm": ISBI_PIXEL_SIZE_MM,
        }


def evaluate_on_isbi(model_predictions_29, isbi_ground_truth_19, orig_size, image_size, pixel_size_mm=ISBI_PIXEL_SIZE_MM):
    """
    model_predictions_29: (29, 2) array — a trained Aariz model's decoded predictions,
        in Aariz's canonical 29-landmark order, at image_size resolution.
    isbi_ground_truth_19: (19, 2) array — ISBI ground truth, in ISBI_LANDMARK_ORDER,
        at image_size resolution (i.e. sample["landmarks_px_resized"] from ISBI2015Dataset).
    Returns: (19,) array of per-landmark radial error in mm, in ISBI_LANDMARK_ORDER —
        this is the actual external-validation number for Phase 6.
    """
    from aariz_dataset import pixel_error_to_mm
    overlap_indices = get_aariz_overlap_indices()
    pred_overlap = model_predictions_29[overlap_indices]  # (19, 2), reordered into ISBI order
    return pixel_error_to_mm(pred_overlap, isbi_ground_truth_19, orig_size, image_size, pixel_size_mm)


if __name__ == "__main__":
    # Quick self-test when run directly: python3 isbi2015_dataset.py
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "mock"
    ds = ISBI2015Dataset(root=root, split="train", image_size=512)
    print(f"Loaded {len(ds)} training rows")
    sample = ds[0]
    print("image shape:", sample["image"].shape)
    print("landmarks shape:", sample["landmarks_px_resized"].shape)
    print("pixel_size_mm:", sample["pixel_size_mm"])

    ds_test = ISBI2015Dataset(root=root, split="test", image_size=512)
    print(f"Combined test1+test2: {len(ds_test)} rows")

    if AARIZ_LANDMARK_ORDER is not None:
        overlap = get_aariz_overlap_indices()
        print("Aariz channel indices matching ISBI's 19 landmarks:", overlap)
