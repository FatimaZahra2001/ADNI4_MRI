from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import resnet34

from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

UTILS_DIR = Path(__file__).resolve().parents[1] / "metric_learning"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from utils import (
    brain_fraction,
    centroid,
    clamp_center,
    compute_binary_metrics,
    compute_bbox,
    extract_patch,
    first_existing,
    load_nii,
    local_valid_center,
    normalise_slice,
    seed_all,
    tune_threshold_by_metric as tune_threshold,
    zscore_in_mask,
)


PATCH_NAMES = [
    "MTL_all",
    "MTL_left",
    "MTL_right",
    "left_temporal",
    "right_temporal",
    "inferior_temporal",
]


# ============================================================
# Visualisation
# ============================================================

def save_attention_bar(patch_names, attn, out_path, title):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(attn)

    plt.figure(figsize=(8, 4))
    plt.barh([patch_names[i] for i in order], np.asarray(attn)[order])
    plt.xlabel("Attention weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc_curve(y_true, y_prob, out_path, title):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = auc(fpr, tpr)

    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"AUC={score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def save_confusion(y_true, y_prob, threshold, out_path, title):
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["HC", "MCI"])
    plt.yticks([0, 1], ["HC", "MCI"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def save_patch_visualisation(
    t1: np.ndarray,
    patches: np.ndarray,
    centers: np.ndarray,
    patch_names: List[str],
    attn: np.ndarray,
    prob: float,
    label: int,
    ptid: str,
    out_path: Path,
    patch_size: int,
    top_k: int = 6,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(attn)[::-1][:top_k]
    half = patch_size // 2

    fig = plt.figure(figsize=(16, 4 * len(order)))
    gs = fig.add_gridspec(len(order), 4, width_ratios=[1, 1, 1, 0.9])

    for r, idx in enumerate(order):
        c = centers[idx].astype(int)
        cx, cy, cz = [int(v) for v in c]

        axes = [fig.add_subplot(gs[r, i]) for i in range(4)]

        axes[0].imshow(normalise_slice(t1[cx, :, :].T), cmap="gray", origin="lower")
        axes[0].add_patch(plt.Rectangle((cz - half, cy - half), patch_size, patch_size, fill=False, edgecolor="red", linewidth=2))
        axes[0].set_title(f"{patch_names[idx]} sagittal")

        axes[1].imshow(normalise_slice(t1[:, cy, :].T), cmap="gray", origin="lower")
        axes[1].add_patch(plt.Rectangle((cz - half, cx - half), patch_size, patch_size, fill=False, edgecolor="red", linewidth=2))
        axes[1].set_title("coronal")

        axes[2].imshow(normalise_slice(t1[:, :, cz].T), cmap="gray", origin="lower")
        axes[2].add_patch(plt.Rectangle((cy - half, cx - half), patch_size, patch_size, fill=False, edgecolor="red", linewidth=2))
        axes[2].set_title("axial")

        p = patches[idx, 0]
        mid = patch_size // 2
        axes[3].imshow(normalise_slice(p[:, :, mid].T), cmap="gray", origin="lower")
        axes[3].set_title(f"patch | attn={attn[idx]:.3f}")

        for ax in axes:
            ax.axis("off")

    fig.suptitle(f"{ptid} | label={label} | prob_mci={prob:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# Manifest
# ============================================================

def load_manifest(args) -> pd.DataFrame:
    df = pd.read_csv(args.csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "PTID" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain PTID and label columns")

    df["PTID"] = df["PTID"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    df = df[df["label"].isin([0, 1])].copy()

    leakage_cols = [
        c for c in df.columns
        if c not in ["PTID", "label"]
        and any(k in c.lower() for k in ["cdglobal", "cdr", "mmse", "diagnosis", "dx"])
    ]

    if leakage_cols:
        print("[INFO] Dropping leakage-like manifest columns:", leakage_cols, flush=True)
        df = df.drop(columns=leakage_cols)

    root = Path(args.image_root)
    rows = []
    missing = []

    for _, r in df.iterrows():
        ptid = str(r["PTID"])
        sdir = root / ptid

        t1 = first_existing(sdir, args.t1_names)
        roi = first_existing(sdir, args.roi_names)

        if t1 is None or roi is None:
            missing.append((ptid, t1 is not None, roi is not None))
            continue

        row = r.to_dict()
        row["T1_path"] = str(t1)
        row["ROI_path"] = str(roi)
        rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)

    print(f"[MANIFEST] original_rows={len(df)} usable_rows={len(out)} missing={len(missing)}", flush=True)
    print("[MANIFEST] label counts:", out["label"].value_counts().to_dict(), flush=True)

    if missing:
        print("[MANIFEST] first missing:", missing[:10], flush=True)

    return out


def build_weighted_sampler(labels: np.ndarray):
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = weights[labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ============================================================
# Dataset
# ============================================================

class ROIT1MultiPatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        patch_size: int,
        min_brain_fraction: float,
        train_mode: bool,
        seed: int,
        debug_name: str,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.patch_size = int(patch_size)
        self.min_brain_fraction = float(min_brain_fraction)
        self.train_mode = bool(train_mode)
        self.seed = int(seed)

        print(
            f"[DATASET] {debug_name} n={len(self.df)} "
            f"labels={self.df['label'].value_counts().to_dict()} "
            f"patch_size={self.patch_size} min_brain_fraction={self.min_brain_fraction}",
            flush=True,
        )

    def __len__(self):
        return len(self.df)

    def _make_centers(self, t1_raw: np.ndarray, roi: np.ndarray, rng: np.random.Generator):
        brain_mask = t1_raw > 0
        roi = roi > 0
        mid_x = roi.shape[0] // 2

        left_roi = roi.copy()
        right_roi = roi.copy()
        left_roi[mid_x:, :, :] = False
        right_roi[:mid_x, :, :] = False

        all_c = centroid(roi)
        left_c = centroid(left_roi)
        right_c = centroid(right_roi)

        if all_c is None:
            mins, maxs = compute_bbox(brain_mask)
            all_c = np.round((mins + maxs) / 2).astype(np.int32)

        if left_c is None:
            left_c = all_c.copy()
            left_c[0] = max(left_c[0] - self.patch_size, self.patch_size // 2)

        if right_c is None:
            right_c = all_c.copy()
            right_c[0] = min(right_c[0] + self.patch_size, t1_raw.shape[0] - self.patch_size // 2 - 1)

        lateral_offset = max(12, self.patch_size // 2)
        inferior_offset = max(10, self.patch_size // 3)

        desired = {
            "MTL_all": all_c,
            "MTL_left": left_c,
            "MTL_right": right_c,
            "left_temporal": left_c + np.array([-lateral_offset, 0, 0], dtype=np.int32),
            "right_temporal": right_c + np.array([lateral_offset, 0, 0], dtype=np.int32),
            "inferior_temporal": all_c + np.array([0, 0, -inferior_offset], dtype=np.int32),
        }

        centers = []
        for name in PATCH_NAMES:
            c = desired[name].copy()

            if self.train_mode:
                c = c + rng.integers(-3, 4, size=3)

            c = local_valid_center(
                desired=c,
                brain_mask=brain_mask,
                patch_size=self.patch_size,
                min_brain_fraction=self.min_brain_fraction,
                search_radius=12,
                step=4,
            )
            centers.append(c)

        return np.stack(centers, axis=0).astype(np.int32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        ptid = str(row["PTID"])
        label = float(row["label"])
        t1_path = str(row["T1_path"])
        roi_path = str(row["ROI_path"])

        t1_raw = load_nii(t1_path)
        roi = load_nii(roi_path) > 0

        if t1_raw.shape != roi.shape:
            raise ValueError(f"Shape mismatch for {ptid}: T1={t1_raw.shape}, ROI={roi.shape}")

        brain_mask = t1_raw > 0
        t1 = zscore_in_mask(t1_raw, brain_mask)

        rng = np.random.default_rng(self.seed + idx if not self.train_mode else None)
        centers = self._make_centers(t1_raw, roi, rng)

        patches = []
        brain_fracs = []
        roi_fracs = []

        for c in centers:
            patch = extract_patch(t1, c, self.patch_size)[None, ...].astype(np.float32)
            b_patch = extract_patch(brain_mask.astype(np.uint8), c, self.patch_size)
            r_patch = extract_patch(roi.astype(np.uint8), c, self.patch_size)

            if self.train_mode:
                if random.random() < 0.5:
                    patch = np.flip(patch, axis=1).copy()
                if random.random() < 0.5:
                    patch = np.flip(patch, axis=2).copy()
                if random.random() < 0.4:
                    scale = random.uniform(0.97, 1.03)
                    shift = random.uniform(-0.03, 0.03)
                    patch = (patch * scale + shift).astype(np.float32)
                if random.random() < 0.25:
                    patch = patch + np.random.normal(0, 0.015, size=patch.shape).astype(np.float32)

            patches.append(patch)
            brain_fracs.append(float(b_patch.mean()))
            roi_fracs.append(float(r_patch.mean()))

        patches = np.stack(patches, axis=0).astype(np.float32)

        return {
            "patches": torch.from_numpy(patches),
            "label": torch.tensor(label, dtype=torch.float32),
            "ptid": ptid,
            "centers": torch.from_numpy(centers),
            "brain_fracs": torch.tensor(brain_fracs, dtype=torch.float32),
            "roi_fracs": torch.tensor(roi_fracs, dtype=torch.float32),
            "patch_names": PATCH_NAMES,
            "t1_path": t1_path,
            "roi_path": roi_path,
        }


# ============================================================
# Model
# ============================================================

class MedicalNetPatchEncoder(nn.Module):
    """
    Important fix:
    MedicalNet backbone is used as feature extractor.
    Projection layer remains trainable even when backbone is frozen.
    """

    def __init__(self, patch_size: int, feature_dim: int, dropout: float):
        super().__init__()

        self.backbone = resnet34(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=0,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, patch_size, patch_size, patch_size)
            f = self.backbone(dummy)
            if f.dim() == 5:
                f = F.adaptive_avg_pool3d(f, 1).flatten(1)
            backbone_dim = int(f.shape[1])

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() == 5:
            f = F.adaptive_avg_pool3d(f, 1).flatten(1)
        return self.proj(f)


def load_medicalnet_weights(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v

    model_state = model.state_dict()
    loadable = {}
    skipped = []

    for k, v in clean.items():
        candidates = [
            f"encoder.backbone.{k}",
            f"backbone.{k}",
            k,
        ]

        matched = None
        for ck_name in candidates:
            if ck_name in model_state:
                matched = ck_name
                break

        if matched is None:
            skipped.append((k, "missing"))
            continue

        if tuple(model_state[matched].shape) != tuple(v.shape):
            skipped.append((matched, tuple(v.shape), tuple(model_state[matched].shape)))
            continue

        loadable[matched] = v

    msg = model.load_state_dict(loadable, strict=False)
    print(f"[MedicalNet] loaded tensors={len(loadable)} skipped={len(skipped)}", flush=True)
    print(f"[MedicalNet] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
    if skipped:
        print("[MedicalNet] first skipped:", skipped[:10], flush=True)


class ROIResNet34AttentionMIL(nn.Module):
    def __init__(self, patch_size: int, feature_dim: int, attn_dim: int, dropout: float):
        super().__init__()

        self.encoder = MedicalNetPatchEncoder(
            patch_size=patch_size,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        self.attn_v = nn.Sequential(
            nn.Linear(feature_dim, attn_dim),
            nn.Tanh(),
        )
        self.attn_u = nn.Sequential(
            nn.Linear(feature_dim, attn_dim),
            nn.Sigmoid(),
        )
        self.attn_w = nn.Linear(attn_dim, 1)

        self.patch_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, patches):
        b, p, c, d, h, w = patches.shape

        x = patches.reshape(b * p, c, d, h, w)
        feats = self.encoder(x).reshape(b, p, -1)

        raw_attn = self.attn_w(self.attn_v(feats) * self.attn_u(feats)).squeeze(-1)
        attn = torch.softmax(raw_attn, dim=1)

        patch_logits = self.patch_classifier(feats).squeeze(-1)
        bag_logit = torch.sum(attn * patch_logits, dim=1)

        return bag_logit, attn, patch_logits


def set_backbone_trainable(model, trainable: bool):
    for p in model.encoder.backbone.parameters():
        p.requires_grad = trainable

    # Projection/head always trainable
    for p in model.encoder.proj.parameters():
        p.requires_grad = True


def make_optimizer(model, args):
    backbone_params = [p for p in model.encoder.backbone.parameters() if p.requires_grad]
    head_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("encoder.backbone.") and p.requires_grad
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_encoder})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


# ============================================================
# Training
# ============================================================

def get_patch_names_from_batch(raw_names, bs):
    return [list(PATCH_NAMES) for _ in range(bs)]


def run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp=False,
    patch_loss_weight=0.15,
    save_viz_dir: Path | None = None,
    max_viz: int = 0,
    patch_size: int = 48,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_samples = 0

    all_labels = []
    all_probs = []
    all_rows = []
    viz_saved = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            patches = batch["patches"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp):
                bag_logits, attn, patch_logits = model(patches)

                bag_loss = criterion(bag_logits, y)
                patch_targets = y[:, None].expand_as(patch_logits)
                patch_loss = F.binary_cross_entropy_with_logits(patch_logits, patch_targets)

                loss = bag_loss + patch_loss_weight * patch_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()

            probs = torch.sigmoid(bag_logits).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            attn_np = attn.detach().cpu().numpy()
            patch_prob_np = torch.sigmoid(patch_logits).detach().cpu().numpy()

            patches_np = patches.detach().cpu().numpy()
            centers_np = batch["centers"].cpu().numpy()
            brain_fracs_np = batch["brain_fracs"].cpu().numpy()
            roi_fracs_np = batch["roi_fracs"].cpu().numpy()

            bs = patches.shape[0]
            total_loss += float(loss.item()) * bs
            n_samples += bs

            per_sample_names = get_patch_names_from_batch(batch["patch_names"], bs)

            for i in range(bs):
                patch_names = per_sample_names[i]
                order = np.argsort(attn_np[i])[::-1]

                all_labels.append(int(labels[i]))
                all_probs.append(float(probs[i]))

                all_rows.append({
                    "ptid": batch["ptid"][i],
                    "label": int(labels[i]),
                    "prob_mci": float(probs[i]),
                    "patch_names": json.dumps(patch_names),
                    "patch_attention": json.dumps(attn_np[i].tolist()),
                    "patch_prob_mci": json.dumps(patch_prob_np[i].tolist()),
                    "top_patch_names": json.dumps([patch_names[j] for j in order]),
                    "top_patch_indices": json.dumps(order.tolist()),
                    "top_patch_centers": json.dumps(centers_np[i][order].tolist()),
                    "top_patch_attention": json.dumps(attn_np[i][order].tolist()),
                    "top_patch_prob_mci": json.dumps(patch_prob_np[i][order].tolist()),
                    "top_patch_brain_fracs": json.dumps(brain_fracs_np[i][order].tolist()),
                    "top_patch_roi_fracs": json.dumps(roi_fracs_np[i][order].tolist()),
                    "t1_path": batch["t1_path"][i],
                    "roi_path": batch["roi_path"][i],
                })

                if (not is_train) and save_viz_dir is not None and viz_saved < max_viz:
                    try:
                        t1_raw = load_nii(batch["t1_path"][i])
                        title = f"{batch['ptid'][i]} | label={int(labels[i])} prob_mci={float(probs[i]):.3f}"

                        save_patch_visualisation(
                            t1=t1_raw,
                            patches=patches_np[i],
                            centers=centers_np[i],
                            patch_names=patch_names,
                            attn=attn_np[i],
                            prob=float(probs[i]),
                            label=int(labels[i]),
                            ptid=batch["ptid"][i],
                            out_path=save_viz_dir / f"{batch['ptid'][i]}_patches.png",
                            patch_size=patch_size,
                            top_k=min(6, len(patch_names)),
                        )

                        save_attention_bar(
                            patch_names,
                            attn_np[i],
                            save_viz_dir / f"{batch['ptid'][i]}_attention_bar.png",
                            title,
                        )

                        viz_saved += 1
                    except Exception as e:
                        print(f"[VIZ WARN] {batch['ptid'][i]}: {e}", flush=True)

    return total_loss / max(1, n_samples), all_labels, all_probs, pd.DataFrame(all_rows)


def run_one_fold(fold_idx, train_df, val_df, args, device):
    fold_dir = Path(args.out_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "checkpoints").mkdir(exist_ok=True)
    (fold_dir / "predictions").mkdir(exist_ok=True)
    (fold_dir / "visualisations").mkdir(exist_ok=True)

    print(f"\n========== ROI RESNET34 MULTI-PATCH FOLD {fold_idx} ==========", flush=True)
    print(f"Train size: {len(train_df)} labels={train_df['label'].value_counts().to_dict()}", flush=True)
    print(f"Val size:   {len(val_df)} labels={val_df['label'].value_counts().to_dict()}", flush=True)

    train_ds = ROIT1MultiPatchDataset(
        train_df,
        args.patch_size,
        args.min_brain_fraction,
        True,
        args.seed + fold_idx,
        "train",
    )
    val_ds = ROIT1MultiPatchDataset(
        val_df,
        args.patch_size,
        args.min_brain_fraction,
        False,
        args.seed + 1000 + fold_idx,
        "val",
    )

    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        sampler = build_weighted_sampler(train_df["label"].astype(int).to_numpy())
        shuffle = False
        print("[INFO] Using WeightedRandomSampler", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = ROIResNet34AttentionMIL(
        patch_size=args.patch_size,
        feature_dim=args.feature_dim,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
    ).to(device)

    load_medicalnet_weights(model, args.medicalnet_ckpt)

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_metric = -math.inf
    best_epoch = -1
    best_threshold = 0.5
    best_ckpt = fold_dir / "checkpoints" / "best.pt"
    epochs_without_improve = 0
    history = []

    optimizer = None
    scheduler = None
    backbone_trainable = None

    for epoch in range(1, args.epochs + 1):
        should_train_backbone = epoch > args.freeze_backbone_epochs

        if should_train_backbone != backbone_trainable:
            backbone_trainable = should_train_backbone
            set_backbone_trainable(model, trainable=backbone_trainable)
            optimizer = make_optimizer(model, args)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - epoch + 1),
                eta_min=1e-7,
            )
            print(f"[FREEZE] epoch={epoch} backbone_trainable={backbone_trainable}", flush=True)

        t0 = time.time()

        train_loss, train_y, train_prob, train_pred = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            args.amp,
            args.patch_loss_weight,
            patch_size=args.patch_size,
        )

        save_viz = epoch == 1 or epoch % args.viz_every == 0

        val_loss, val_y, val_prob, val_pred = run_epoch(
            model,
            val_loader,
            device,
            criterion,
            None,
            None,
            args.amp,
            args.patch_loss_weight,
            save_viz_dir=(fold_dir / "visualisations" / f"epoch_{epoch:03d}") if save_viz else None,
            max_viz=args.max_viz_per_fold if save_viz else 0,
            patch_size=args.patch_size,
        )

        if scheduler is not None:
            scheduler.step()

        train_metrics = compute_binary_metrics(train_y, train_prob, threshold=0.5)
        val_metrics = compute_binary_metrics(val_y, val_prob, threshold=0.5)

        tuned_thr, _ = tune_threshold(val_y, val_prob, metric=args.threshold_metric)
        val_tuned = compute_binary_metrics(val_y, val_prob, threshold=tuned_thr)

        epoch_time = time.time() - t0

        row = {
            "fold": fold_idx,
            "epoch": epoch,
            "backbone_trainable": backbone_trainable,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_auc": train_metrics["auc"],
            "val_auc": val_metrics["auc"],
            "val_bal_acc_05": val_metrics["balanced_accuracy"],
            "val_bal_acc_tuned": val_tuned["balanced_accuracy"],
            "val_f1_tuned": val_tuned["f1"],
            "tuned_threshold": tuned_thr,
            "epoch_time_sec": epoch_time,
        }
        history.append(row)

        print(
            f"Fold {fold_idx} | Epoch {epoch:03d} | backbone_trainable={backbone_trainable} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} | "
            f"val_bACC@tuned={val_tuned['balanced_accuracy']:.4f} | "
            f"val_f1@tuned={val_tuned['f1']:.4f} | "
            f"thr={tuned_thr:.4f} | time={epoch_time:.1f}s",
            flush=True,
        )

        current_metric = val_metrics["auc"]
        if np.isnan(current_metric):
            current_metric = -math.inf

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_threshold = tuned_thr
            epochs_without_improve = 0

            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_auc": best_metric,
                    "best_threshold": best_threshold,
                    "args": vars(args),
                },
                best_ckpt,
            )

            train_pred.to_csv(fold_dir / "predictions" / "best_train_predictions_raw.csv", index=False)
            val_pred.to_csv(fold_dir / "predictions" / "best_val_predictions_raw.csv", index=False)

            print(f"[BEST] Fold {fold_idx}: epoch={epoch} val_auc={best_metric:.4f}", flush=True)
        else:
            epochs_without_improve += 1
            print(f"[EARLY STOP] fold {fold_idx}: {epochs_without_improve}/{args.early_stopping_patience}", flush=True)

        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

        if epochs_without_improve >= args.early_stopping_patience:
            break

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    best_threshold = float(ckpt.get("best_threshold", 0.5))

    val_loss, val_y, val_prob, val_pred = run_epoch(
        model,
        val_loader,
        device,
        criterion,
        None,
        None,
        args.amp,
        args.patch_loss_weight,
        save_viz_dir=fold_dir / "visualisations" / "best_model",
        max_viz=args.max_viz_per_fold,
        patch_size=args.patch_size,
    )

    metrics_05 = compute_binary_metrics(val_y, val_prob, threshold=0.5)
    metrics_tuned = compute_binary_metrics(val_y, val_prob, threshold=best_threshold)

    val_pred["fold"] = fold_idx
    val_pred["pred_label_tuned"] = (val_pred["prob_mci"].values >= best_threshold).astype(int)
    val_pred["threshold_used"] = best_threshold
    val_pred.to_csv(fold_dir / "predictions" / "val_predictions_best.csv", index=False)

    save_roc_curve(val_y, val_prob, fold_dir / "val_roc_best.png", f"Fold {fold_idx} ROC")
    save_confusion(val_y, val_prob, best_threshold, fold_dir / "val_confusion_best.png", f"Fold {fold_idx} confusion")

    summary = {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_auc": best_metric,
        "best_threshold": best_threshold,
        "val_loss": val_loss,
        **{f"val05_{k}": v for k, v in metrics_05.items()},
        **{f"valtuned_{k}": v for k, v in metrics_tuned.items()},
    }

    with open(fold_dir / "fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== FOLD {fold_idx} SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    torch.cuda.empty_cache()
    return summary, val_pred


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--medicalnet_ckpt", required=True)

    ap.add_argument("--t1_names", nargs="+", default=["T1_norm.nii.gz", "T1_MNI.nii.gz"])
    ap.add_argument("--roi_names", nargs="+", default=["MTL_core_mask.nii.gz", "MTL_roi_mask.nii.gz", "MTL_roi_mask_aligned.nii.gz"])

    ap.add_argument("--patch_size", type=int, default=48)
    ap.add_argument("--min_brain_fraction", type=float, default=0.70)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=15)

    ap.add_argument("--lr_encoder", type=float, default=5e-6)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--feature_dim", type=int, default=128)
    ap.add_argument("--attn_dim", type=int, default=64)
    ap.add_argument("--patch_loss_weight", type=float, default=0.15)

    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--test_size", type=float, default=0.0)
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--threshold_metric", choices=["balanced_accuracy", "f1"], default="balanced_accuracy")
    ap.add_argument("--early_stopping_patience", type=int, default=12)

    ap.add_argument("--viz_every", type=int, default=15)
    ap.add_argument("--max_viz_per_fold", type=int, default=8)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(args.out_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    df = load_manifest(args)
    y = df["label"].astype(int).to_numpy()

    if args.test_size > 0:
        dev_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=args.test_size,
            stratify=y,
            random_state=args.seed,
        )
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        df = df.iloc[dev_idx].copy().reset_index(drop=True)

        test_df.to_csv(Path(args.out_dir) / "heldout_test_manifest.csv", index=False)
        print("[HELDOUT TEST] saved manifest only; CV runs on dev set.", flush=True)
        print("[DEV LABELS]", df["label"].value_counts().to_dict(), flush=True)
        print("[TEST LABELS]", test_df["label"].value_counts().to_dict(), flush=True)

    y = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    summaries = []
    oof_preds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df.iloc[val_idx].copy().reset_index(drop=True)

        summary, pred = run_one_fold(fold_idx, train_df, val_df, args, device)
        summaries.append(summary)
        oof_preds.append(pred)

    out_dir = Path(args.out_dir)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "cv_fold_summary.csv", index=False)

    oof_df = pd.concat(oof_preds, ignore_index=True)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    oof_y = oof_df["label"].astype(int).to_numpy()
    oof_prob = oof_df["prob_mci"].astype(float).to_numpy()

    oof_thr, _ = tune_threshold(oof_y, oof_prob, metric=args.threshold_metric)
    oof_05 = compute_binary_metrics(oof_y, oof_prob, threshold=0.5)
    oof_tuned = compute_binary_metrics(oof_y, oof_prob, threshold=oof_thr)

    save_roc_curve(oof_y, oof_prob, out_dir / "oof_roc.png", "OOF ROC")
    save_confusion(oof_y, oof_prob, oof_thr, out_dir / "oof_confusion_tuned.png", "OOF confusion tuned")

    attn_rows = []
    for _, r in oof_df.iterrows():
        names = json.loads(r["patch_names"])
        attn = json.loads(r["patch_attention"])
        patch_prob = json.loads(r["patch_prob_mci"])

        for n, a, pp in zip(names, attn, patch_prob):
            attn_rows.append({
                "ptid": r["ptid"],
                "label": int(r["label"]),
                "patch": n,
                "attention": float(a),
                "patch_prob_mci": float(pp),
            })

    attn_df = pd.DataFrame(attn_rows)
    attn_df.to_csv(out_dir / "oof_patch_attention_long.csv", index=False)

    attn_summary = (
        attn_df
        .groupby(["patch", "label"])
        .agg(
            attention_mean=("attention", "mean"),
            attention_std=("attention", "std"),
            patch_prob_mci_mean=("patch_prob_mci", "mean"),
            patch_prob_mci_std=("patch_prob_mci", "std"),
            count=("attention", "count"),
        )
        .reset_index()
    )
    attn_summary.to_csv(out_dir / "oof_patch_attention_summary.csv", index=False)

    cv_summary = {
        "model": "roi_t1_resnet34_medicalnet_attention_mil_v2",
        "n_folds": args.n_folds,
        "mean_best_val_auc": float(summary_df["best_val_auc"].mean()),
        "std_best_val_auc": float(summary_df["best_val_auc"].std()),
        "mean_valtuned_balanced_accuracy": float(summary_df["valtuned_balanced_accuracy"].mean()),
        "std_valtuned_balanced_accuracy": float(summary_df["valtuned_balanced_accuracy"].std()),
        "mean_valtuned_f1": float(summary_df["valtuned_f1"].mean()),
        "std_valtuned_f1": float(summary_df["valtuned_f1"].std()),
        "oof_threshold": float(oof_thr),
        **{f"oof05_{k}": v for k, v in oof_05.items()},
        **{f"ooftuned_{k}": v for k, v in oof_tuned.items()},
    }

    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    print("\n===== ROI T1 RESNET34 MEDICALNET MULTI-PATCH CV SUMMARY =====", flush=True)
    print(json.dumps(cv_summary, indent=2), flush=True)

    print(f"Saved: {out_dir / 'cv_summary.json'}", flush=True)
    print(f"Saved: {out_dir / 'oof_predictions.csv'}", flush=True)
    print(f"Saved: {out_dir / 'oof_patch_attention_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
