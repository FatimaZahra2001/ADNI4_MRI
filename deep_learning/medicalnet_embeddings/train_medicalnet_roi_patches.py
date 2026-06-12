import argparse, json, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from monai.networks.nets import resnet

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, accuracy_score,
    f1_score, confusion_matrix, roc_curve
)

UTILS_DIR = Path(__file__).resolve().parents[1] / "metric_learning"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from utils import (
    best_threshold,
    crop_bbox,
    filter_existing_subjects,
    find_col,
    load_nii,
    robust_z,
    seed_all,
)

# =========================
# Utils
# =========================

def load_medicalnet_into_monai(model, ckpt_path):
    if ckpt_path is None or str(ckpt_path).lower() == "none":
        print("[MedicalNet] no checkpoint provided")
        return

    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        print(f"[MedicalNet] checkpoint not found: {ckpt_path}")
        return

    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    print(f"[MedicalNet] loaded from: {ckpt_path}")
    print(f"[MedicalNet] missing={len(missing)} unexpected={len(unexpected)}")


# =========================
# Dataset
# =========================

class MedicalNetROIDataset(Dataset):
    def __init__(
        self,
        df,
        roi_root,
        patch_shape=(64, 64, 64),
        margin=6,
        augment=False,
    ):
        self.df = df.reset_index(drop=True)
        self.roi_root = Path(roi_root)
        self.patch_shape = tuple(patch_shape)
        self.margin = margin
        self.augment = augment

        self.ptid_col = find_col(df, ["PTID", "ptid", "Subject", "subject"])
        self.t1_col = find_col(df, ["T1_MNI_path", "t1_mni_path", "t1_path", "T1_path", "t1", "T1"])
        self.label_col = find_col(df, ["label", "y", "diagnosis_bin"])

        self.mask_defs = [
            ("MTL_core", "MTL_core_mask.nii.gz"),
            ("MTL_roi", "MTL_roi_mask.nii.gz"),
            ("temporal", "TP_mask.nii.gz"),
        ]

    @property
    def roi_names(self):
        return [x[0] for x in self.mask_defs]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sid = str(row[self.ptid_col])
        t1 = robust_z(load_nii(row[self.t1_col]))

        patches = []

        for roi_name, mask_file in self.mask_defs:
            mask_path = self.roi_root / sid / mask_file
            mask = (load_nii(mask_path) > 0).astype(np.float32)

            patch = crop_bbox(
                t1,
                mask,
                out_shape=self.patch_shape,
                margin=self.margin,
            )

            patches.append(patch)

        x = np.stack(patches, axis=0)  # [R, 1, H, W, D]

        if self.augment:
            if random.random() < 0.5:
                x = x[..., ::-1].copy()
            if random.random() < 0.5:
                x = x[..., :, ::-1, :].copy()

        y = int(row[self.label_col])

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor(y).float(),
            "id": sid,
        }


# =========================
# Model
# =========================

class ResNet18ROIEncoder(nn.Module):
    def __init__(self, medicalnet_ckpt=None, emb_dim=256, freeze_until_layer4=False):
        super().__init__()

        self.backbone = resnet.resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=0,
        )

        load_medicalnet_into_monai(self.backbone, medicalnet_ckpt)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64, 64)
            feat = self.backbone(dummy)
            feat_dim = feat.shape[1] if feat.dim() == 2 else feat.flatten(1).shape[1]

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )

        if freeze_until_layer4:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = False
                if "layer4" in name:
                    p.requires_grad = True

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() > 2:
            f = f.flatten(1)
        return self.proj(f)


class MedicalNetROIAttentionModel(nn.Module):
    def __init__(self, n_rois, medicalnet_ckpt=None, emb_dim=256, freeze_until_layer4=False):
        super().__init__()

        self.encoder = ResNet18ROIEncoder(
            medicalnet_ckpt=medicalnet_ckpt,
            emb_dim=emb_dim,
            freeze_until_layer4=freeze_until_layer4,
        )

        self.attn = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.cls = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: [B, R, 1, H, W, D]
        b, r, c, h, w, d = x.shape
        x = x.view(b * r, c, h, w, d)

        e = self.encoder(x)
        e = e.view(b, r, -1)

        a = self.attn(e).squeeze(-1)
        a = torch.softmax(a, dim=1)

        z = (e * a.unsqueeze(-1)).sum(dim=1)
        logit = self.cls(z).squeeze(1)

        return logit, a


# =========================
# Training
# =========================

def run_epoch(model, loader, opt, device, pos_weight=None):
    is_train = opt is not None
    model.train(is_train)

    losses, ys, probs = [], [], []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        if is_train:
            opt.zero_grad(set_to_none=True)

        logit, _ = model(x)

        loss = F.binary_cross_entropy_with_logits(
            logit,
            y,
            pos_weight=pos_weight,
        )

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        prob = torch.sigmoid(logit).detach().cpu().numpy()

        losses.append(loss.item())
        ys.extend(y.detach().cpu().numpy().astype(int).tolist())
        probs.extend(prob.tolist())

    ys = np.array(ys)
    probs = np.array(probs)
    pred = (probs >= 0.5).astype(int)

    auc = roc_auc_score(ys, probs) if len(np.unique(ys)) == 2 else np.nan

    return {
        "loss": float(np.mean(losses)),
        "auc": float(auc),
        "bacc": float(balanced_accuracy_score(ys, pred)),
        "acc": float(accuracy_score(ys, pred)),
        "f1": float(f1_score(ys, pred, zero_division=0)),
        "y": ys,
        "prob": probs,
    }


def collect_predictions(model, loader, device, roi_names):
    model.eval()

    ys, probs = [], []
    attn_rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].cpu().numpy().astype(int)

            logit, attn = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()
            attn = attn.cpu().numpy()

            ys.extend(y.tolist())
            probs.extend(prob.tolist())

            for a in attn:
                attn_rows.append({roi_names[i]: float(a[i]) for i in range(len(roi_names))})

    return np.array(ys), np.array(probs), attn_rows


# =========================
# Plots
# =========================

def plot_history(history, outpath):
    df = pd.DataFrame(history)

    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_auc"], label="Train AUC")
    plt.plot(df["epoch"], df["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_roc(y, prob, outpath, title):
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_cm(y, prob, threshold, outpath, title):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred)

    plt.figure(figsize=(4.6, 4))
    plt.imshow(cm)
    plt.xticks([0, 1], ["HC", "MCI"])
    plt.yticks([0, 1], ["HC", "MCI"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_attention(attn_rows, roi_names, outpath):
    df = pd.DataFrame(attn_rows)
    means = df[roi_names].mean().sort_values()

    plt.figure(figsize=(7, 4))
    plt.barh(means.index, means.values)
    plt.xlabel("Mean attention weight")
    plt.title("Mean ROI attention")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--roi_root", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--medicalnet_ckpt", default="/rds/projects/j/jouaitim-mri-test/ADNI4/pretrained/medicalnet/resnet_18_23dataset.pth")

    parser.add_argument("--patch_shape", nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument("--margin", type=int, default=6)

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--freeze_until_layer4", action="store_true")

    args = parser.parse_args()

    seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = filter_existing_subjects(df, args.roi_root)

    label_col = find_col(df, ["label", "y", "diagnosis_bin"])
    df = df[df[label_col].isin([0, 1])].reset_index(drop=True)

    y = df[label_col].astype(int).values

    print("[DATA]", df.shape)
    print("[LABELS]", dict(pd.Series(y).value_counts().sort_index()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device)

    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    roi_names = ["MTL_core", "MTL_roi", "temporal"]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    all_y, all_prob = [], []
    fold_summaries = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), 1):
        print(f"\n========== FOLD {fold} ==========")

        fold_dir = outdir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)

        train_y = train_df[label_col].astype(int).values
        counts = np.bincount(train_y)

        sample_weights = 1.0 / counts[train_y]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_ds = MedicalNetROIDataset(
            train_df,
            roi_root=args.roi_root,
            patch_shape=tuple(args.patch_shape),
            margin=args.margin,
            augment=True,
        )

        val_ds = MedicalNetROIDataset(
            val_df,
            roi_root=args.roi_root,
            patch_shape=tuple(args.patch_shape),
            margin=args.margin,
            augment=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = MedicalNetROIAttentionModel(
            n_rois=len(roi_names),
            medicalnet_ckpt=args.medicalnet_ckpt,
            emb_dim=256,
            freeze_until_layer4=args.freeze_until_layer4,
        ).to(device)

        pos_weight = torch.tensor([counts[0] / max(counts[1], 1)], dtype=torch.float32, device=device)

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_auc = -1
        best_epoch = -1
        bad = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, opt, device, pos_weight=pos_weight)
            va = run_epoch(model, val_loader, None, device, pos_weight=pos_weight)

            history.append({
                "epoch": epoch,
                "train_loss": tr["loss"],
                "val_loss": va["loss"],
                "train_auc": tr["auc"],
                "val_auc": va["auc"],
                "train_bacc": tr["bacc"],
                "val_bacc": va["bacc"],
                "train_f1": tr["f1"],
                "val_f1": va["f1"],
            })

            print(
                f"Fold {fold} | Epoch {epoch:03d} | "
                f"loss={tr['loss']:.4f}/{va['loss']:.4f} | "
                f"AUC={tr['auc']:.4f}/{va['auc']:.4f} | "
                f"bACC={tr['bacc']:.4f}/{va['bacc']:.4f} | "
                f"F1={tr['f1']:.4f}/{va['f1']:.4f}"
            )

            if va["auc"] > best_auc:
                best_auc = va["auc"]
                best_epoch = epoch
                bad = 0
                torch.save(model.state_dict(), fold_dir / "best_model.pt")
                np.save(fold_dir / "best_val_y.npy", va["y"])
                np.save(fold_dir / "best_val_prob.npy", va["prob"])
            else:
                bad += 1

            if bad >= args.patience:
                print(f"[EARLY STOP] fold={fold} epoch={epoch} best_epoch={best_epoch} best_auc={best_auc:.4f}")
                break

        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        plot_history(history, fold_dir / "history_auc.png")

        model.load_state_dict(torch.load(fold_dir / "best_model.pt", map_location=device))
        ys, probs, attn_rows = collect_predictions(model, val_loader, device, roi_names)

        thr, tuned_bacc = best_threshold(ys, probs)
        pred = (probs >= thr).astype(int)

        fold_summary = {
            "fold": fold,
            "best_epoch": best_epoch,
            "auc": float(roc_auc_score(ys, probs)),
            "threshold": thr,
            "balanced_accuracy": float(balanced_accuracy_score(ys, pred)),
            "accuracy": float(accuracy_score(ys, pred)),
            "f1": float(f1_score(ys, pred, zero_division=0)),
        }

        fold_summaries.append(fold_summary)

        all_y.extend(ys.tolist())
        all_prob.extend(probs.tolist())

        pd.DataFrame(attn_rows).to_csv(fold_dir / "attention_weights.csv", index=False)

        plot_roc(ys, probs, fold_dir / "roc_curve.png", f"Fold {fold} ROC")
        plot_cm(ys, probs, thr, fold_dir / "confusion_matrix.png", f"Fold {fold} confusion")
        plot_attention(attn_rows, roi_names, fold_dir / "mean_attention.png")

    all_y = np.array(all_y)
    all_prob = np.array(all_prob)

    oof_thr, _ = best_threshold(all_y, all_prob)
    oof_pred = (all_prob >= oof_thr).astype(int)

    final = {
        "oof_auc": float(roc_auc_score(all_y, all_prob)),
        "oof_threshold": oof_thr,
        "oof_balanced_accuracy": float(balanced_accuracy_score(all_y, oof_pred)),
        "oof_accuracy": float(accuracy_score(all_y, oof_pred)),
        "oof_f1": float(f1_score(all_y, oof_pred, zero_division=0)),
        "mean_fold_auc": float(np.mean([x["auc"] for x in fold_summaries])),
        "std_fold_auc": float(np.std([x["auc"] for x in fold_summaries])),
        "folds": fold_summaries,
    }

    with open(outdir / "final_summary.json", "w") as f:
        json.dump(final, f, indent=2)

    plot_roc(all_y, all_prob, outdir / "OOF_roc_curve.png", "OOF ROC")
    plot_cm(all_y, all_prob, oof_thr, outdir / "OOF_confusion_matrix.png", "OOF confusion")

    print("\n===== FINAL SUMMARY =====")
    print(json.dumps(final, indent=2))
    print("[DONE]", outdir)


if __name__ == "__main__":
    main()
