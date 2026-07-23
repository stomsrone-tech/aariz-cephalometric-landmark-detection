"""
AarizDataset — calibrated to the confirmed Aariz schema (Phase 3).

Confirmed from Phase 0 exploration:
- root/{train,valid,test}/Cephalograms/{ceph_id}.{png,jpg,jpeg,bmp}
- root/{train,valid,test}/Annotations/Cephalometric Landmarks/{Junior Orthodontists,Senior Orthodontists}/{ceph_id}.json
- JSON schema: {"ceph_id": ..., "landmarks": [{"title","symbol","value":{"x","y"}}, ...]}  (29 landmarks)
- root/cephalogram_machine_mappings.csv: cephalogram_id, machine, pixel_size (mm/px), image_format, mode (Train/Valid/Test)
"""

import os
import json
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# Canonical landmark order — fixed once, used everywhere so heatmap channel i
# always corresponds to the same anatomical landmark across the whole pipeline.
LANDMARK_ORDER = [
    "A-point", "Anterior Nasal Spine", "B-point", "Menton", "Nasion", "Orbitale",
    "Pogonion", "Posterior Nasal Spine", "Pronasale", "Ramus", "Sella", "Articulare",
    "Condylion", "Gnathion", "Gonion", "Porion", "Lower 2nd PM Cusp Tip",
    "Lower Incisor Tip", "Lower Molar Cusp Tip", "Upper 2nd PM Cusp Tip",
    "Upper Incisor Apex", "Upper Incisor Tip", "Upper Molar Cusp Tip",
    "Lower Incisor Apex", "Labrale inferius", "Labrale superius",
    "Soft Tissue Nasion", "Soft Tissue Pogonion", "Subnasale",
]
assert len(LANDMARK_ORDER) == 29
TITLE_TO_IDX = {t: i for i, t in enumerate(LANDMARK_ORDER)}

# Subset used for fair comparison against the published Aariz leaderboard (Table 3).
# CephRes-MHNet's paper (and the other leaderboard entries it reports, which were all
# evaluated by CephRes-MHNet's own authors under the same protocol) restrict evaluation
# to 19 of Aariz's 29 landmarks, not all 29 -- reporting a full-29-landmark MRE against
# that leaderboard is not apples-to-apples. Of their 19 named landmarks, 16 map confidently
# onto Aariz's canonical names below; the remaining 3 (Distal U6, U1 Tip, L6 Occlusal) use
# dental notation with no unambiguous correspondence to Aariz's specific dental landmark
# names and are deliberately excluded rather than guessed at.
CEPHRES_MHNET_COMPARABLE_LANDMARKS = LANDMARK_ORDER[:16]  # A-point through Porion in canonical order
assert CEPHRES_MHNET_COMPARABLE_LANDMARKS == [
    "A-point", "Anterior Nasal Spine", "B-point", "Menton", "Nasion", "Orbitale",
    "Pogonion", "Posterior Nasal Spine", "Pronasale", "Ramus", "Sella", "Articulare",
    "Condylion", "Gnathion", "Gonion", "Porion",
]


def load_machine_mapping(root):
    """cephalogram_id -> dict(machine, pixel_size, image_format, mode)"""
    csv_path = os.path.join(root, "cephalogram_machine_mappings.csv")
    mapping = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["cephalogram_id"]] = {
                "machine": row["machine"],
                "pixel_size": float(row["pixel_size"]),
                "image_format": row["image_format"],
                "mode": row["mode"],
            }
    return mapping


def parse_landmark_json(path):
    """Returns a (29, 2) float32 array in canonical LANDMARK_ORDER, pixel coordinates.
    Raises if a landmark title is missing or unrecognized, rather than silently
    misaligning channels — this is exactly the kind of bug that would silently
    corrupt every downstream metric.
    """
    with open(path) as f:
        data = json.load(f)
    coords = np.full((29, 2), np.nan, dtype=np.float32)
    seen = set()
    for lm in data["landmarks"]:
        title = lm["title"]
        if title not in TITLE_TO_IDX:
            raise ValueError(f"Unrecognized landmark title '{title}' in {path}")
        idx = TITLE_TO_IDX[title]
        coords[idx] = [lm["value"]["x"], lm["value"]["y"]]
        seen.add(title)
    missing = set(LANDMARK_ORDER) - seen
    if missing:
        raise ValueError(f"Missing landmarks {missing} in {path}")
    return coords


def find_image_path(cephalograms_dir, ceph_id, image_format_hint=None):
    """Image extension isn't always the CSV's image_format in practice-safe code —
    check the hinted extension first, then fall back to scanning."""
    if image_format_hint:
        candidate = os.path.join(cephalograms_dir, f"{ceph_id}.{image_format_hint}")
        if os.path.exists(candidate):
            return candidate
    for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
        candidate = os.path.join(cephalograms_dir, f"{ceph_id}.{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No image found for {ceph_id} in {cephalograms_dir}")


class AarizDataset(Dataset):
    COMBINED_LABEL = "Combined (Junior+Senior mean)"

    def __init__(self, root, split, annotator="Junior Orthodontists",
                 image_size=512, heatmap_size=128, sigma=2.0, transform=None):
        """
        root: path to the 'Aariz' folder (containing train/valid/test + the mapping CSV)
        split: 'train' | 'valid' | 'test'
        annotator: 'Junior Orthodontists' | 'Senior Orthodontists' | AarizDataset.COMBINED_LABEL
            The Aariz dataset's own documentation states the mean of the Junior and Senior
            annotation sets is the intended ground truth (each of those sets is itself
            already an averaged/reviewed composite of two raters) -- COMBINED_LABEL
            reproduces that recommended convention rather than using Junior alone.
        image_size: model input size (square resize)
        heatmap_size: output heatmap resolution (square)
        sigma: Gaussian std for heatmap generation, in heatmap-pixel units
        """
        assert split in ("train", "valid", "test")
        self.root = root
        self.split = split
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transform
        self.annotator = annotator
        self.use_combined = (annotator == self.COMBINED_LABEL)

        self.ceph_dir = os.path.join(root, split, "Cephalograms")
        base_ann_dir = os.path.join(root, split, "Annotations", "Cephalometric Landmarks")
        self.junior_dir = os.path.join(base_ann_dir, "Junior Orthodontists")
        self.senior_dir = os.path.join(base_ann_dir, "Senior Orthodontists")
        self.ann_dir = self.junior_dir if self.use_combined else os.path.join(base_ann_dir, annotator)
        self.mapping = load_machine_mapping(root)

        # File listing always comes from the Junior folder when combined (both folders
        # should contain the same ceph_ids; Junior is the reference listing here).
        listing_dir = self.junior_dir if self.use_combined else self.ann_dir
        ann_files = sorted(f for f in os.listdir(listing_dir) if f.endswith(".json"))
        self.ceph_ids = [os.path.splitext(f)[0] for f in ann_files]

        # Cross-check against the mapping CSV's 'mode' column as an integrity check —
        # catches cases where a file ended up in the wrong split folder.
        for cid in self.ceph_ids:
            if cid not in self.mapping:
                raise ValueError(f"{cid} has an annotation file but no entry in the machine mapping CSV")
            expected_mode = self.mapping[cid]["mode"].lower()
            if expected_mode != split:
                raise ValueError(
                    f"{cid} is in the '{split}' folder but the mapping CSV says mode='{expected_mode}'"
                )

    def __len__(self):
        return len(self.ceph_ids)

    def __getitem__(self, idx):
        ceph_id = self.ceph_ids[idx]
        meta = self.mapping[ceph_id]
        pixel_size_mm = meta["pixel_size"]

        img_path = find_image_path(self.ceph_dir, ceph_id, meta["image_format"])
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        if self.use_combined:
            junior_coords = parse_landmark_json(os.path.join(self.junior_dir, f"{ceph_id}.json"))
            senior_coords = parse_landmark_json(os.path.join(self.senior_dir, f"{ceph_id}.json"))
            landmarks_px = (junior_coords + senior_coords) / 2.0  # (29, 2), original pixel space
        else:
            ann_path = os.path.join(self.ann_dir, f"{ceph_id}.json")
            landmarks_px = parse_landmark_json(ann_path)  # (29, 2) in ORIGINAL image pixel coords

        # Resize image to model input size; rescale landmark coords to match.
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        landmarks_resized = landmarks_px.copy()
        landmarks_resized[:, 0] *= scale_x
        landmarks_resized[:, 1] *= scale_y

        # Landmark coords in heatmap-resolution space (used to place Gaussians)
        hm_scale_x = self.heatmap_size / self.image_size
        hm_scale_y = self.heatmap_size / self.image_size
        landmarks_hm = landmarks_resized.copy()
        landmarks_hm[:, 0] *= hm_scale_x
        landmarks_hm[:, 1] *= hm_scale_y

        heatmaps = generate_heatmaps(landmarks_hm, self.heatmap_size, self.sigma)

        sample = {
            "ceph_id": ceph_id,
            "image": torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0,
            "heatmaps": torch.from_numpy(heatmaps).float(),
            "landmarks_px_resized": torch.from_numpy(landmarks_resized).float(),  # ground truth @ image_size, for eval
            "orig_size": (orig_h, orig_w),
            "pixel_size_mm": pixel_size_mm,  # mm per pixel AT ORIGINAL RESOLUTION — see note below
            "machine": meta["machine"],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def generate_heatmaps(landmarks_hm, heatmap_size, sigma):
    """landmarks_hm: (N, 2) array of (x, y) in heatmap-pixel coordinates.
    Returns (N, heatmap_size, heatmap_size) float32 Gaussian heatmaps.
    """
    n = landmarks_hm.shape[0]
    heatmaps = np.zeros((n, heatmap_size, heatmap_size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size), indexing="ij")
    for i in range(n):
        x0, y0 = landmarks_hm[i]
        if np.isnan(x0) or np.isnan(y0):
            continue
        heatmaps[i] = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return heatmaps


def decode_heatmaps(heatmaps, image_size, heatmap_size):
    """Inverse of generate_heatmaps + the resize in __getitem__.
    heatmaps: (N, heatmap_size, heatmap_size) numpy or torch array (model output or GT)
    Returns (N, 2) array of (x, y) in image_size-resolution pixel coordinates.
    """
    if hasattr(heatmaps, "numpy"):
        heatmaps = heatmaps.numpy()
    n = heatmaps.shape[0]
    coords = np.zeros((n, 2), dtype=np.float32)
    scale = image_size / heatmap_size
    for i in range(n):
        flat_idx = np.argmax(heatmaps[i])
        y, x = np.unravel_index(flat_idx, heatmaps[i].shape)
        coords[i] = [x * scale, y * scale]
    return coords


def inter_observer_comparison(root, split):
    """Compares Junior vs Senior Orthodontist annotations directly, in original pixel
    space -- no image loading or resizing needed, since we're comparing two humans'
    raw coordinate annotations against each other, not evaluating a model.
    Returns (errors_mm (N, 29), ceph_ids (list of N)) -- same shape convention as the
    model evaluation functions in train_eval.py, so summarize_errors() works unchanged
    on the output, giving directly comparable MRE/SDR/CI numbers.
    """
    mapping = load_machine_mapping(root)
    junior_dir = os.path.join(root, split, "Annotations", "Cephalometric Landmarks", "Junior Orthodontists")
    senior_dir = os.path.join(root, split, "Annotations", "Cephalometric Landmarks", "Senior Orthodontists")

    ceph_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(junior_dir) if f.endswith(".json"))

    errors_mm = []
    used_ids = []
    for ceph_id in ceph_ids:
        senior_path = os.path.join(senior_dir, f"{ceph_id}.json")
        if not os.path.exists(senior_path):
            continue  # skip if this image lacks a Senior annotation, rather than crash
        junior_path = os.path.join(junior_dir, f"{ceph_id}.json")

        junior_coords = parse_landmark_json(junior_path)  # (29, 2), original pixel space
        senior_coords = parse_landmark_json(senior_path)

        pixel_size_mm = mapping[ceph_id]["pixel_size"]
        diff_px = junior_coords - senior_coords
        err_mm = np.linalg.norm(diff_px, axis=1) * pixel_size_mm  # isotropic pixel size, no resize involved

        errors_mm.append(err_mm)
        used_ids.append(ceph_id)

    return np.stack(errors_mm, axis=0), used_ids


def pixel_error_to_mm(pred_px_at_image_size, gt_px_at_image_size, orig_size, image_size, pixel_size_mm):
    """Converts a per-landmark pixel error (measured at image_size resolution, i.e.
    after the resize in __getitem__) into millimetres, correctly undoing the resize.

    Aariz images have non-square aspect ratios (e.g. 1198x959, confirmed in Phase 0),
    but are resized to a square image_size, so x and y use different scale factors.
    Averaging those into one scalar before computing a Euclidean distance would distort
    the error for non-square images. Instead: convert the resized-pixel error back to
    ORIGINAL-pixel units per axis first, then apply pixel_size_mm (assumed isotropic —
    true for standard X-ray sensors), then take the Euclidean norm in mm.
    """
    orig_h, orig_w = orig_size
    scale_x = image_size / orig_w
    scale_y = image_size / orig_h

    diff_resized = pred_px_at_image_size - gt_px_at_image_size  # (N, 2), in resized-pixel space
    dx_orig = diff_resized[:, 0] / scale_x
    dy_orig = diff_resized[:, 1] / scale_y

    dx_mm = dx_orig * pixel_size_mm
    dy_mm = dy_orig * pixel_size_mm
    radial_error_mm = np.sqrt(dx_mm ** 2 + dy_mm ** 2)
    return radial_error_mm
