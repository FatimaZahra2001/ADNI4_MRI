import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from monai.networks.nets import resnet34

from utils import (
    brain_frac,
    clamp_center,
    crop,
    get_patch_centres,
    load_nii,
    metrics,
    seed_all,
    tune_threshold,
    zscore_brain,
    zscore_patch,
)


# =========================
# CONFIG
# =========================

SEED = 42
N_FOLDS = 5

MANIFEST = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_t1_clinical_manifest.csv")
ROOT = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm")
OUT = Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_multipatch_metric_learning")

MEDICALNET_CKPT = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/pretrained/medicalnet/resnet_34_23dataset.pth")

T1_NAME = "T1_norm.nii.gz"

PATCH_SIZE = 64
MIN_BRAIN_FRAC = 0.55

BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 60
PATIENCE = 14

EMBED_DIM = 128
FEAT_DIM = 256

FREEZE_EPOCHS = 8
LR_HEAD = 3e-4
LR_BACKBONE = 5e-7
WEIGHT_DECAY = 1e-4

SUPCON_WEIGHT = 0.7
BCE_WEIGHT = 0.3
TEMPERATURE = 0.15

USE_CLINICAL_SEVERE = False 
SAVE_GRADCAM_N = 8


# =========================
# DATASET
# =========================

class MultiPatchDataset(Dataset):
    def __init__(self, df, train=False, qc_dir=None, save_qc=False):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.qc_dir = Path(qc_dir) if qc_dir else None
        self.save_qc = save_qc

        if self.qc_dir:
            self.qc_dir.mkdir(parents=True, exist_ok=True)

        self.patch_names = ["left_hippocampus", "right_hippocampus", "mtl_core", "temporal_context"]

    def __len__(self):
        return len(self.df)

    def _save_qc(self, ptid, patches):
        if not self.save_qc or self.qc_dir is None:
            return
        if len(list(self.qc_dir.glob("*.png"))) > 60:
            return

        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        for i, name in enumerate(self.patch_names):
            p = patches[i, 0]
            ax[i].imshow(p[:, :, PATCH_SIZE // 2].T, cmap="gray", origin="lower")
            ax[i].set_title(name, fontsize=8)
            ax[i].axis("off")
        fig.suptitle(ptid)
        plt.tight_layout()
        plt.savefig(self.qc_dir / f"{ptid}.png", dpi=150)
        plt.close()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ptid = str(row["PTID"]).strip()
        label = int(row["label"])

        sdir = ROOT / ptid
        t1 = load_nii(sdir / T1_NAME)
        brain = t1 > 0

        t1z = zscore_brain(t1, brain)

        centres = get_patch_centres(sdir, t1, brain, PATCH_SIZE)
        if centres is None:
            raise RuntimeError(f"missing centres: {ptid}")

        patches = []
        fracs = []
        used_centres = []

        for name in self.patch_names:
            c = centres[name].copy()

            if self.train:
                c = c + np.random.randint(-2, 3, size=3)
                c = clamp_center(c, t1.shape, PATCH_SIZE)

            bf = brain_frac(brain, c, PATCH_SIZE)

            # fallback: if context patch has low brain, fall back to MTL centre.
            if bf < MIN_BRAIN_FRAC and name == "temporal_context":
                c = centres["mtl_core"].copy()
                c[0] = c[0] + np.random.randint(-8, 9)
                c[1] = c[1] + np.random.randint(-8, 9)
                c = clamp_center(c, t1.shape, PATCH_SIZE)
                bf = brain_frac(brain, c, PATCH_SIZE)

            if bf < MIN_BRAIN_FRAC:
                c = centres["mtl_core"].copy()
                bf = brain_frac(brain, c, PATCH_SIZE)

            brain_patch = crop(brain.astype(np.float32), c, PATCH_SIZE)
            patch = crop(t1z, c, PATCH_SIZE)
            patch = zscore_patch(patch, brain_patch)
            patch = patch * brain_patch
            patches.append(patch[None, ...].astype(np.float32))
            fracs.append(bf)
            used_centres.append(c.tolist())

        patches = np.stack(patches, axis=0)

        self._save_qc(ptid, patches)

        return {
            "x": torch.tensor(patches, dtype=torch.float32),
            "y": torch.tensor(label, dtype=torch.long),
            "ptid": ptid,
            "centres": used_centres,
            "brain_fracs": fracs,
        }


def collate(batch):
    x = torch.stack([b["x"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    ptids = [b["ptid"] for b in batch]
    centres = [b["centres"] for b in batch]
    fracs = [b["brain_fracs"] for b in batch]
    return x, y, ptids, centres, fracs


# =========================
# MODEL
# =========================

class MultiPatchMetricNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = resnet34(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=FEAT_DIM,
        )

        self.patch_type = nn.Parameter(torch.randn(1, 4, FEAT_DIM) * 0.02)

        self.patch_attn = nn.Sequential(
            nn.Linear(FEAT_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.proj = nn.Sequential(
            nn.Linear(FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, EMBED_DIM),
        )

        self.classifier = nn.Linear(EMBED_DIM, 1)

    def set_trainable(self, train_layer4=False):
        for p in self.encoder.parameters():
            p.requires_grad = False

        if train_layer4:
            for name, p in self.encoder.named_parameters():
                if name.startswith("layer4") or name.startswith("fc"):
                    p.requires_grad = True

    def encode_patches(self, x):
        b, p, c, d, h, w = x.shape
        f = self.encoder(x.reshape(b * p, c, d, h, w)).reshape(b, p, -1)
        f = f + self.patch_type[:, :p, :]

        scores = self.patch_attn(f).squeeze(-1)
        attn = torch.softmax(scores, dim=1)

        bag = (attn.unsqueeze(-1) * f).sum(dim=1)
        emb = self.proj(bag)
        emb = F.normalize(emb, dim=1)

        logit = self.classifier(emb).squeeze(-1)
        return emb, logit, attn


def load_medicalnet(model):
    ckpt = torch.load(MEDICALNET_CKPT, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    own = model.state_dict()
    loadable = {}

    for k, v in state.items():
        k = k.replace("module.", "")
        mk = f"encoder.{k}"
        if mk in own and own[mk].shape == v.shape:
            loadable[mk] = v

    msg = model.load_state_dict(loadable, strict=False)
    print(f"[MedicalNet] loaded={len(loadable)} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


def supervised_contrastive_loss(features, labels, temperature=0.15):
    features = F.normalize(features, dim=1)
    labels = labels.view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=features.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    positives = mask.sum(dim=1)
    valid = positives > 0

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positives + 1e-8)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    return -mean_log_prob_pos[valid].mean()


def make_optimizer(model):
    head, backbone = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.layer4") or name.startswith("encoder.fc"):
            backbone.append(p)
        else:
            head.append(p)

    return torch.optim.AdamW(
        [
            {"params": head, "lr": LR_HEAD},
            {"params": backbone, "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


# =========================
# TRAIN/EVAL
# =========================

def make_sampler(labels):
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels)
    weights = 1.0 / np.maximum(counts, 1)
    sample_w = weights[labels]
    return WeightedRandomSampler(torch.DoubleTensor(sample_w), len(sample_w), replacement=True)


def run_epoch(model, loader, opt=None, pos_weight=None):
    train = opt is not None
    model.train(train)

    ys, ps, losses = [], [], []
    rows = []

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for x, y, ptids, centres, fracs in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            emb, logit, attn = model.encode_patches(x)

            bce = F.binary_cross_entropy_with_logits(
                logit,
                y.float(),
                pos_weight=pos_weight,
            )
            supcon = supervised_contrastive_loss(emb, y, TEMPERATURE)

            loss = BCE_WEIGHT * bce + SUPCON_WEIGHT * supcon

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            prob = torch.sigmoid(logit).detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            attn_np = attn.detach().cpu().numpy()

            ys.extend(y_np.tolist())
            ps.extend(prob.tolist())
            losses.append(float(loss.item()))

            for i, ptid in enumerate(ptids):
                rows.append({
                    "PTID": ptid,
                    "label": int(y_np[i]),
                    "prob_metric_head": float(prob[i]),
                    "attn_left_hippocampus": float(attn_np[i, 0]),
                    "attn_right_hippocampus": float(attn_np[i, 1]),
                    "attn_mtl_core": float(attn_np[i, 2]),
                    "attn_temporal_context": float(attn_np[i, 3]),
                    "centres": json.dumps(centres[i]),
                    "brain_fracs": json.dumps(fracs[i]),
                })

    return np.mean(losses), np.array(ys), np.array(ps), pd.DataFrame(rows)


def extract_embeddings(model, loader):
    model.eval()

    embs, ys, ptids = [], [], []
    rows = []

    with torch.no_grad():
        for x, y, p, centres, fracs in loader:
            x = x.cuda(non_blocking=True)
            emb, logit, attn = model.encode_patches(x)

            embs.append(emb.cpu().numpy())
            ys.extend(y.numpy().tolist())
            ptids.extend(p)

            prob = torch.sigmoid(logit).cpu().numpy()
            attn_np = attn.cpu().numpy()

            for i, ptid in enumerate(p):
                rows.append({
                    "PTID": ptid,
                    "label": int(y[i].item()),
                    "prob_metric_head": float(prob[i]),
                    "attn_left_hippocampus": float(attn_np[i, 0]),
                    "attn_right_hippocampus": float(attn_np[i, 1]),
                    "attn_mtl_core": float(attn_np[i, 2]),
                    "attn_temporal_context": float(attn_np[i, 3]),
                })

    return np.concatenate(embs, axis=0), np.array(ys), ptids, pd.DataFrame(rows)


def plot_embedding_2d(emb, y, title, out_path):
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
        z = reducer.fit_transform(emb)
        method = "UMAP"
    else:
        z = TSNE(n_components=2, perplexity=min(30, len(y)//4), random_state=SEED).fit_transform(emb)
        method = "t-SNE"

    plt.figure(figsize=(6, 5))
    plt.scatter(z[y == 0, 0], z[y == 0, 1], s=18, alpha=0.7, label="CN")
    plt.scatter(z[y == 1, 0], z[y == 1, 1], s=18, alpha=0.7, label="MCI")
    plt.title(f"{title} ({method})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def evaluate_svm(train_emb, train_y, val_emb, val_y):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_emb)
    Xva = scaler.transform(val_emb)

    base = LinearSVC(C=0.2, class_weight="balanced", random_state=SEED, max_iter=10000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xtr, train_y)

    p = clf.predict_proba(Xva)[:, 1]
    thr = tune_threshold(val_y, p)

    return p, metrics(val_y, p, 0.5), metrics(val_y, p, thr), thr


# =========================
# GRADCAM
# =========================

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activ = None
        self.grad = None

        layer = model.encoder.layer4

        def fwd_hook(module, inp, out):
            self.activ = out

        def bwd_hook(module, grad_in, grad_out):
            self.grad = grad_out[0]

        self.h1 = layer.register_forward_hook(fwd_hook)
        self.h2 = layer.register_full_backward_hook(bwd_hook)

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def cam_for_patch(self, patch):
        self.model.zero_grad(set_to_none=True)

        f = self.model.encoder(patch)
        score = f.norm(dim=1).sum()
        score.backward()

        grad = self.grad
        activ = self.activ

        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * activ).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_gradcams(model, dataset, out_dir, n=8):
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    gc = GradCAM(model)

    for i in range(min(n, len(dataset))):
        item = dataset[i]
        x = item["x"]
        y = int(item["y"].item())
        ptid = item["ptid"]

        fig, ax = plt.subplots(4, 2, figsize=(6, 12))

        for pidx, name in enumerate(dataset.patch_names):
            patch = x[pidx:pidx+1].cuda()
            cam = gc.cam_for_patch(patch)

            img = x[pidx, 0].numpy()
            sl = PATCH_SIZE // 2

            ax[pidx, 0].imshow(img[:, :, sl].T, cmap="gray", origin="lower")
            ax[pidx, 0].set_title(f"{name} T1")
            ax[pidx, 0].axis("off")

            ax[pidx, 1].imshow(img[:, :, sl].T, cmap="gray", origin="lower")
            ax[pidx, 1].imshow(cam[:, :, sl].T, cmap="hot", alpha=0.45, origin="lower")
            ax[pidx, 1].set_title(f"{name} GradCAM")
            ax[pidx, 1].axis("off")

        fig.suptitle(f"{ptid} label={y}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{ptid}_gradcam.png", dpi=150)
        plt.close()

    gc.remove()


# =========================
# MAIN TRAINING
# =========================

def train_fold(fold, train_df, val_df, out_dir):
    fold_dir = out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MultiPatchDataset(train_df, train=True, qc_dir=fold_dir / "patch_qc", save_qc=(fold == 1))
    val_ds = MultiPatchDataset(val_df, train=False, qc_dir=fold_dir / "patch_qc", save_qc=(fold == 1))

    sampler = make_sampler(train_df["label"].values)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True
    )

    model = MultiPatchMetricNet().cuda()
    load_medicalnet(model)

    ytr = train_df["label"].values.astype(int)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32).cuda()

    best_auc = -1
    best_state = None
    bad = 0
    hist = []

    current_trainable = None
    opt = None

    for epoch in range(1, EPOCHS + 1):
        train_layer4 = epoch > FREEZE_EPOCHS

        if current_trainable != train_layer4:
            model.set_trainable(train_layer4)
            opt = make_optimizer(model)
            current_trainable = train_layer4
            print(f"[FOLD {fold}] epoch={epoch} layer4_trainable={train_layer4}", flush=True)

        train_loss, train_y, train_p, _ = run_epoch(model, train_loader, opt, pos_weight)
        val_loss, val_y, val_p, val_rows = run_epoch(model, val_loader, None, pos_weight)

        val_auc = roc_auc_score(val_y, val_p)
        val_pr = average_precision_score(val_y, val_p)

        hist.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc_metric_head": val_auc,
            "val_pr_metric_head": val_pr,
        })
        pd.DataFrame(hist).to_csv(fold_dir / "history.csv", index=False)

        print(
            f"Fold {fold} | Epoch {epoch:03d} | "
            f"loss={train_loss:.4f}/{val_loss:.4f} "
            f"val_auc={val_auc:.4f} val_pr={val_pr:.4f}",
            flush=True
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            val_rows.to_csv(fold_dir / "best_metric_head_predictions.csv", index=False)
            bad = 0
        else:
            bad += 1

        if bad >= PATIENCE:
            print(f"[EARLY STOP] fold={fold} best_auc={best_auc:.4f}", flush=True)
            break

    model.load_state_dict(best_state)

    train_emb, train_y, train_ptids, train_rows = extract_embeddings(model, train_loader)
    val_emb, val_y, val_ptids, val_rows = extract_embeddings(model, val_loader)

    np.save(fold_dir / "train_embeddings.npy", train_emb)
    np.save(fold_dir / "val_embeddings.npy", val_emb)

    train_rows.to_csv(fold_dir / "train_embedding_rows.csv", index=False)
    val_rows.to_csv(fold_dir / "val_embedding_rows.csv", index=False)

    svm_p, svm_m05, svm_mtuned, svm_thr = evaluate_svm(train_emb, train_y, val_emb, val_y)

    out_val = pd.DataFrame({
        "PTID": val_ptids,
        "label": val_y,
        "svm_prob_mci": svm_p,
        "fold": fold,
    })
    out_val = out_val.merge(val_rows, on=["PTID", "label"], how="left")
    out_val.to_csv(fold_dir / "val_svm_predictions.csv", index=False)

    all_emb = np.concatenate([train_emb, val_emb], axis=0)
    all_y = np.concatenate([train_y, val_y], axis=0)
    plot_embedding_2d(all_emb, all_y, f"Fold {fold} embeddings", fold_dir / "embedding_2d.png")

    save_gradcams(model, val_ds, fold_dir / "gradcam", n=SAVE_GRADCAM_N)

    torch.save({"model": model.state_dict(), "best_auc": best_auc}, fold_dir / "best_metric_model.pt")

    summary = {
        "fold": fold,
        "metric_head_best_auc": float(best_auc),
        **{f"svm05_{k}": v for k, v in svm_m05.items()},
        **{f"svmtuned_{k}": v for k, v in svm_mtuned.items()},
        "svm_tuned_threshold": float(svm_thr),
    }

    return summary, out_val


def filter_usable(df):
    rows = []
    missing = []

    for _, row in df.iterrows():
        ptid = str(row["PTID"]).strip()
        sdir = ROOT / ptid

        ok = True
        for f in [T1_NAME, "hippocampus_left.nii.gz", "hippocampus_right.nii.gz", "MTL_core_fixed_mask.nii.gz"]:
            if not (sdir / f).exists():
                ok = False

        if ok:
            rows.append(row)
        else:
            missing.append(ptid)

    out = pd.DataFrame(rows).reset_index(drop=True)
    pd.DataFrame({"PTID": missing}).to_csv(OUT / "missing_metric_subjects.csv", index=False)

    print("[USABLE]", len(out), "missing", len(missing), flush=True)
    print(out["label"].value_counts().to_dict(), flush=True)
    return out


def main():
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)
    df["PTID"] = df["PTID"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    if USE_CLINICAL_SEVERE and "clinical_severe_mci" in df.columns:
        df["clinical_severe_mci"] = df["clinical_severe_mci"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[df["clinical_severe_mci"]].copy()

    df = filter_usable(df)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    summaries = []
    oof_rows = []

    for fold, (tr, va) in enumerate(skf.split(df, df["label"]), start=1):
        print(f"\n========== FOLD {fold} ==========", flush=True)
        train_df = df.iloc[tr].reset_index(drop=True)
        val_df = df.iloc[va].reset_index(drop=True)

        print("train:", train_df["label"].value_counts().to_dict(), flush=True)
        print("val:", val_df["label"].value_counts().to_dict(), flush=True)

        s, pred = train_fold(fold, train_df, val_df, OUT)
        summaries.append(s)
        oof_rows.append(pred)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "cv_fold_summary.csv", index=False)

    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(OUT / "oof_svm_predictions.csv", index=False)

    y = oof["label"].values.astype(int)
    p = oof["svm_prob_mci"].values.astype(float)
    thr = tune_threshold(y, p)

    final = {
        **{f"oof05_{k}": v for k, v in metrics(y, p, 0.5).items()},
        **{f"ooftuned_{k}": v for k, v in metrics(y, p, thr).items()},
        "tuned_threshold": float(thr),
    }

    with open(OUT / "cv_final_summary.json", "w") as f:
        json.dump(final, f, indent=2)

    print("\n===== FINAL OOF SVM SUMMARY =====")
    print(json.dumps(final, indent=2))
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
