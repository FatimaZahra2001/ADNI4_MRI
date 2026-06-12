import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import resnet

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
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


def load_medicalnet(model, ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[MedicalNet] missing checkpoint: {ckpt_path}")
        return

    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    clean = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[MedicalNet] loaded: {ckpt_path}")
    print(f"[MedicalNet] missing={len(missing)} unexpected={len(unexpected)}")


class ROIPatchDataset(Dataset):
    def __init__(self, df, roi_root, patch_shape=(64,64,64), margin=8):
        self.df = df.reset_index(drop=True)
        self.roi_root = Path(roi_root)
        self.patch_shape = tuple(patch_shape)
        self.margin = margin

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
            mask = (load_nii(self.roi_root / sid / mask_file) > 0).astype(np.float32)
            p = crop_bbox(t1, mask, self.patch_shape, self.margin)
            patches.append(p)

        x = np.stack(patches, axis=0)  # [R, 1, H, W, D]
        y = int(row[self.label_col])

        return {
            "x": torch.from_numpy(x).float(),
            "y": y,
            "id": sid,
        }


class MedicalNetFeatureExtractor(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.backbone = resnet.resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=512,
        )
        load_medicalnet(self.backbone, ckpt_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() > 2:
            f = f.flatten(1)
        return f


def extract_embeddings(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = filter_existing_subjects(df, args.roi_root)

    label_col = find_col(df, ["label", "y", "diagnosis_bin"])
    ptid_col = find_col(df, ["PTID", "ptid", "Subject", "subject"])

    df = df[df[label_col].isin([0,1])].reset_index(drop=True)

    ds = ROIPatchDataset(
        df,
        roi_root=args.roi_root,
        patch_shape=tuple(args.patch_shape),
        margin=args.margin,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = MedicalNetFeatureExtractor(args.medicalnet_ckpt).to(device)
    extractor.eval()

    all_rows = []
    roi_names = ds.roi_names

    print("[EXTRACT] device:", device)
    print("[EXTRACT] subjects:", len(ds))
    print("[EXTRACT] ROIs:", roi_names)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)  # [B, R, 1, H, W, D]
            b, r, c, h, w, d = x.shape
            x_flat = x.view(b * r, c, h, w, d)

            emb = extractor(x_flat).cpu().numpy()
            emb = emb.reshape(b, r, -1)

            for i in range(b):
                row = {
                    "PTID": batch["id"][i],
                    "label": int(batch["y"][i]),
                }

                parts = []
                for ri, roi in enumerate(roi_names):
                    e = emb[i, ri]
                    parts.append(e)
                    for j, val in enumerate(e):
                        row[f"{roi}_emb_{j:04d}"] = float(val)

                concat = np.concatenate(parts)
                for j, val in enumerate(concat):
                    row[f"concat_emb_{j:04d}"] = float(val)

                all_rows.append(row)

    emb_df = pd.DataFrame(all_rows)
    emb_path = outdir / "medicalnet_roi_embeddings.csv"
    emb_df.to_csv(emb_path, index=False)

    print("[SAVED]", emb_path)
    print("[EMBEDDINGS]", emb_df.shape)

    with torch.no_grad():
        dummy = torch.zeros(1, 1, *args.patch_shape).to(device)
        test_emb = extractor(dummy)
        print("[CHECK] embedding shape:", tuple(test_emb.shape))

    return emb_path


def metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp + 1e-8),
        "f1": f1_score(y, pred, zero_division=0),
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_models():
    return {
        "logreg_l2": (
            LogisticRegression(class_weight="balanced", max_iter=5000, solver="liblinear"),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        ),
        "logreg_elasticnet": (
            LogisticRegression(class_weight="balanced", max_iter=10000, solver="saga", penalty="elasticnet"),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3], "clf__l1_ratio": [0.1,0.3,0.5,0.7,0.9]},
        ),
        "svm_linear": (
            SVC(kernel="linear", probability=True, class_weight="balanced"),
            {"clf__C": [0.01,0.03,0.1,0.3,1,3]},
        ),
        "svm_rbf": (
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
            {"clf__C": [0.1,0.3,1,3,10], "clf__gamma": ["scale",0.001,0.003,0.01,0.03]},
        ),
        "extra_trees": (
            ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
            {"clf__max_depth": [2,3,4,None], "clf__min_samples_leaf": [2,4,8], "clf__max_features": ["sqrt",0.5,None]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            {"clf__max_depth": [2,3,4,None], "clf__min_samples_leaf": [2,4,8], "clf__max_features": ["sqrt",0.5,None]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {"clf__n_estimators": [100,200], "clf__learning_rate": [0.01,0.03,0.1], "clf__max_depth": [1,2,3]},
        ),
    }


def plot_bar(df, metric, outpath, top_n=20):
    d = df.sort_values(metric, ascending=False).head(top_n)
    plt.figure(figsize=(11, max(5, 0.38 * len(d))))
    plt.barh(d["run_name"][::-1], d[metric][::-1])
    plt.xlabel(metric.replace("_", " ").upper())
    plt.title(f"Top models by {metric.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_roc(y, prob, outpath, title):
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def run_ml(args, emb_path):
    outdir = Path(args.outdir)
    df = pd.read_csv(emb_path)

    y = df["label"].astype(int).values
    feature_cols = [c for c in df.columns if c not in ["PTID", "label"]]

    X = df[feature_cols].copy()

    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)

    models = make_models()
    rows = []
    pred_store = {}

    k_values = [min(k, len(feature_cols)) for k in args.k_values]
    k_values = sorted(set([k for k in k_values if k >= 2]))

    for model_name, (base, grid) in models.items():
        for k in k_values:
            run_name = f"{model_name}_k{k}"
            print("\n---", run_name, "---")

            oof_prob = np.zeros(len(y))
            thresholds = []

            for fold, (tr, va) in enumerate(outer.split(X, y), 1):
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("select", SelectKBest(f_classif, k=k)),
                    ("clf", base),
                ])

                search = GridSearchCV(
                    pipe,
                    grid,
                    scoring=args.scoring,
                    cv=inner,
                    n_jobs=args.n_jobs,
                    refit=True,
                )

                search.fit(X.iloc[tr], y[tr])
                best = search.best_estimator_

                train_prob = cross_val_predict(
                    best,
                    X.iloc[tr],
                    y[tr],
                    cv=inner,
                    method="predict_proba",
                    n_jobs=args.n_jobs,
                )[:, 1]

                thr, _ = best_threshold(y[tr], train_prob)
                thresholds.append(thr)

                best.fit(X.iloc[tr], y[tr])
                prob = best.predict_proba(X.iloc[va])[:, 1]
                oof_prob[va] = prob

                m = metrics(y[va], prob, thr)
                print(f"Fold {fold}: AUC={m['roc_auc']:.3f} bACC={m['balanced_accuracy']:.3f} recall={m['recall']:.3f}")

            final_thr = float(np.median(thresholds))
            final = metrics(y, oof_prob, final_thr)

            row = {
                "run_name": run_name,
                "model": model_name,
                "k": k,
                "n_subjects": len(y),
                "n_hc": int((y == 0).sum()),
                "n_mci": int((y == 1).sum()),
                "n_features_total": len(feature_cols),
                **final,
            }

            rows.append(row)
            pred_store[run_name] = (y.copy(), oof_prob.copy(), final_thr)

    res = pd.DataFrame(rows).sort_values(["balanced_accuracy", "roc_auc", "pr_auc"], ascending=False)
    res.to_csv(outdir / "embedding_ml_leaderboard.csv", index=False)

    print("\n===== TOP RESULTS =====")
    print(res.head(20).to_string(index=False))

    plot_bar(res, "balanced_accuracy", outdir / "top_balanced_accuracy.png")
    plot_bar(res, "roc_auc", outdir / "top_roc_auc.png")
    plot_bar(res, "pr_auc", outdir / "top_pr_auc.png")

    best_name = res.iloc[0]["run_name"]
    yb, pb, tb = pred_store[best_name]
    plot_roc(yb, pb, outdir / "best_roc_curve.png", f"Best embedding model: {best_name}")

    with open(outdir / "summary.json", "w") as f:
        json.dump({"best_run": best_name, "best_metrics": res.iloc[0].to_dict()}, f, indent=2)

    print("[DONE]", outdir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--roi_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--medicalnet_ckpt", required=True)

    parser.add_argument("--patch_shape", nargs=3, type=int, default=[64,64,64])
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--scoring", default="roc_auc", choices=["roc_auc", "balanced_accuracy", "average_precision"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[8,16,32,64,128,256,512])

    parser.add_argument("--reuse_embeddings", action="store_true")

    args = parser.parse_args()
    seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    emb_path = outdir / "medicalnet_roi_embeddings.csv"

    if args.reuse_embeddings and emb_path.exists():
        print("[REUSE]", emb_path)
    else:
        emb_path = extract_embeddings(args)

    run_ml(args, emb_path)


if __name__ == "__main__":
    main()
