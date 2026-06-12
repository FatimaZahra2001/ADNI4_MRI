import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_nii(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"Could not find any of {names}")


def first_existing(root, names):
    root = Path(root)
    for n in names:
        p = root / n
        if p.is_file():
            return p
    return None


def robust_z(x, eps=1e-6):
    vals = x[np.isfinite(x)]
    med = np.median(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    return ((x - med) / (q3 - q1 + eps)).astype(np.float32)


def resize_3d_np(x, shape):
    t = torch.from_numpy(x).float()[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float32)


def crop_bbox(vol, mask, out_shape=(64, 64, 64), margin=6, include_mask=False):
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        raise ValueError("Empty ROI mask")

    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, np.array(vol.shape))

    crop_vol = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    crop_mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    crop_vol = resize_3d_np(crop_vol, out_shape)
    crop_mask = resize_3d_np(crop_mask.astype(np.float32), out_shape)
    crop_vol = crop_vol * (crop_mask > 0.1)

    if include_mask:
        return np.stack([crop_vol, crop_mask], axis=0).astype(np.float32)

    return crop_vol[None].astype(np.float32)


def compute_bbox(mask, margin=0):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        shape = np.array(mask.shape)
        return np.array([0, 0, 0], dtype=np.int32), (shape - 1).astype(np.int32)

    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin, np.array(mask.shape) - 1)
    return mins.astype(np.int32), maxs.astype(np.int32)


def centroid(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    return np.round(coords.mean(axis=0)).astype(np.int32)


def clamp_center(c, shape, patch_size):
    h = patch_size // 2
    c = np.asarray(c).astype(np.int32)
    return np.array([
        np.clip(c[0], h, shape[0] - h - 1),
        np.clip(c[1], h, shape[1] - h - 1),
        np.clip(c[2], h, shape[2] - h - 1),
    ], dtype=np.int32)


def crop(vol, c, patch_size):
    c = clamp_center(c, vol.shape, patch_size)
    h = patch_size // 2
    x, y, z = c
    return vol[x-h:x+h, y-h:y+h, z-h:z+h]


def extract_patch(volume, c, patch_size):
    half = patch_size // 2
    x, y, z = [int(v) for v in c]
    return volume[
        x - half:x - half + patch_size,
        y - half:y - half + patch_size,
        z - half:z - half + patch_size,
    ]


def zscore_brain(t1, brain):
    out = np.zeros_like(t1, dtype=np.float32)
    vals = t1[brain]
    out[brain] = (vals - vals.mean()) / (vals.std() + 1e-6)
    return np.nan_to_num(out)


def zscore_in_mask(volume, mask):
    mask = mask.astype(bool)
    out = np.zeros_like(volume, dtype=np.float32)

    if mask.sum() == 0:
        return out

    vals = volume[mask]
    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-6:
        std = 1.0

    out[mask] = (volume[mask] - mean) / std
    return out.astype(np.float32)


def zscore_patch(patch, mask):
    out = np.zeros_like(patch, dtype=np.float32)
    vals = patch[mask > 0]
    if vals.size > 10:
        out[mask > 0] = (vals - vals.mean()) / (vals.std() + 1e-6)
    return np.nan_to_num(out)


def brain_frac(brain, center, patch_size):
    return float(crop(brain.astype(np.float32), center, patch_size).mean())


def brain_fraction(mask, center, patch_size):
    p = extract_patch(mask.astype(np.uint8), center, patch_size)
    return float(p.mean())


def local_valid_center(
    desired,
    brain_mask,
    patch_size,
    min_brain_fraction,
    search_radius=12,
    step=4,
):
    desired = clamp_center(desired, brain_mask.shape, patch_size)

    best_c = desired.copy()
    best_frac = brain_fraction(brain_mask, best_c, patch_size)

    if best_frac >= min_brain_fraction:
        return best_c

    offsets = range(-search_radius, search_radius + 1, step)

    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                c = desired + np.array([dx, dy, dz], dtype=np.int32)
                c = clamp_center(c, brain_mask.shape, patch_size)
                frac = brain_fraction(brain_mask, c, patch_size)

                if frac > best_frac:
                    best_frac = frac
                    best_c = c.copy()

                if frac >= min_brain_fraction:
                    return c.astype(np.int32)

    return best_c.astype(np.int32)


def normalise_slice(x):
    x = np.nan_to_num(x)
    nz = x[x != 0]
    if nz.size > 0:
        lo, hi = np.percentile(nz, [1, 99])
    else:
        lo, hi = float(x.min()), float(x.max())
    x = np.clip(x, lo, hi)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(thr),
    }


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    })
    return out


def tune_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0, 1, 101)))
    best_t, best = 0.5, -1
    for t in qs:
        score = balanced_accuracy_score(y, (p >= t).astype(int))
        if score > best:
            best = score
            best_t = float(t)
    return best_t


def best_threshold(y, prob):
    best_t, best_b = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        pred = (prob >= t).astype(int)
        b = balanced_accuracy_score(y, pred)
        if b > best_b:
            best_b = b
            best_t = t
    return float(best_t), float(best_b)


def tune_threshold_by_metric(y_true, y_prob, metric="balanced_accuracy"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = float(score)
            best_t = float(t)

    return best_t, best_score


def get_mask(sdir, names, shape):
    sdir = Path(sdir)
    for n in names:
        p = sdir / n
        if p.exists():
            m = load_nii(p) > 0
            if m.shape == shape:
                return m
    return None


def filter_existing_subjects(df, roi_root):
    ptid_col = find_col(df, ["PTID", "ptid", "Subject", "subject"])
    t1_col = find_col(df, ["T1_MNI_path", "t1_mni_path", "t1_path", "T1_path", "t1", "T1"])

    required_masks = [
        "MTL_core_mask.nii.gz",
        "MTL_roi_mask.nii.gz",
        "TP_mask.nii.gz",
    ]

    keep, missing = [], []

    for _, row in df.iterrows():
        sid = str(row[ptid_col])
        ok = Path(str(row[t1_col])).exists()

        for mask_name in required_masks:
            if not (Path(roi_root) / sid / mask_name).exists():
                ok = False

        keep.append(ok)
        if not ok:
            missing.append(sid)

    out = df.loc[keep].reset_index(drop=True)

    print(f"[FILTER] kept={len(out)} removed={len(missing)}")
    if missing:
        print("[FILTER] first missing:", missing[:10])

    return out


def get_patch_centres(sdir, t1, brain, patch_size):
    shape = t1.shape

    hip_l = get_mask(sdir, ["hippocampus_left.nii.gz"], shape)
    hip_r = get_mask(sdir, ["hippocampus_right.nii.gz"], shape)
    mtl = get_mask(sdir, ["MTL_core_fixed_mask.nii.gz"], shape)

    if hip_l is None or hip_r is None or mtl is None:
        return None

    c_lh = centroid(hip_l & brain)
    c_rh = centroid(hip_r & brain)
    c_mtl = centroid(mtl & brain)

    if c_lh is None or c_rh is None or c_mtl is None:
        return None

    c_temp = c_mtl.copy()
    c_temp[1] = min(shape[1] - patch_size // 2 - 1, c_temp[1] + 10)
    c_temp[2] = min(shape[2] - patch_size // 2 - 1, c_temp[2] + 14)

    centres = {
        "left_hippocampus": c_lh,
        "right_hippocampus": c_rh,
        "mtl_core": c_mtl,
        "temporal_context": c_temp,
    }

    for k in list(centres.keys()):
        centres[k] = clamp_center(centres[k], shape, patch_size)

    return centres
