import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

SEED = 42
N_FOLDS = 5

MANIFEST_CSV = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_dual_mri_manifest_with_demog.csv")
IMAGE_ROOT = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm")
OUT_DIR = Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_compact_pr_auc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

T1_NAME = "T1_norm.nii.gz"

ROI_MASKS = {
    "MTL_core_fixed": "MTL_core_fixed_mask.nii.gz",
    "MTL_roi": "MTL_roi_mask.nii.gz",
    "hippocampus_left": "hippocampus_left.nii.gz",
    "hippocampus_right": "hippocampus_right.nii.gz",
    "amygdala_left": "amygdala_left.nii.gz",
    "amygdala_right": "amygdala_right.nii.gz",
}

TAB_COLS = ["Age_Baseline", "Gender"]


def load_nii(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= threshold).astype(int)

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
        "threshold": float(threshold),
    }


def tune_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0, 1, 101)))
    best_t, best_score = 0.5, -1

    for t in qs:
        score = balanced_accuracy_score(y, (p >= t).astype(int))
        if score > best_score:
            best_t, best_score = float(t), float(score)

    return best_t


def roi_stats(imgz, brain, mask):
    mask = (mask > 0) & brain

    if mask.sum() < 5:
        return {
            "vox": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p90": np.nan,
        }

    vals = imgz[mask]
    return {
        "vox": float(mask.sum()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p90": float(np.percentile(vals, 90)),
    }


def asym(l, r):
    return (l - r) / (l + r + 1e-8)


def extract_subject(row):
    ptid = str(row["PTID"]).strip()
    sdir = IMAGE_ROOT / ptid
    t1_path = sdir / T1_NAME

    if not t1_path.exists():
        return None, "missing_t1"

    t1 = load_nii(t1_path)
    brain = t1 > 0

    if brain.sum() < 1000:
        return None, "bad_brain"

    vals = t1[brain]
    t1z = np.zeros_like(t1, dtype=np.float32)
    t1z[brain] = (vals - vals.mean()) / (vals.std() + 1e-6)

    feats = {
        "PTID": ptid,
        "label": int(row["label"]),
        "Age_Baseline": row.get("Age_Baseline", np.nan),
        "Gender": row.get("Gender", np.nan),
        "brain_vox": float(brain.sum()),
    }

    for roi, fname in ROI_MASKS.items():
        p = sdir / fname

        if not p.exists():
            for k in ["vox", "mean", "std", "p90"]:
                feats[f"{roi}_{k}"] = np.nan
            continue

        mask = load_nii(p)

        if mask.shape != t1.shape:
            for k in ["vox", "mean", "std", "p90"]:
                feats[f"{roi}_{k}"] = np.nan
            continue

        stats = roi_stats(t1z, brain, mask)

        for k, v in stats.items():
            feats[f"{roi}_{k}"] = v

    # Core asymmetry features.
    for stat in ["vox", "mean", "std", "p90"]:
        hl = feats.get(f"hippocampus_left_{stat}", np.nan)
        hr = feats.get(f"hippocampus_right_{stat}", np.nan)
        al = feats.get(f"amygdala_left_{stat}", np.nan)
        ar = feats.get(f"amygdala_right_{stat}", np.nan)

        feats[f"hippocampus_asym_{stat}"] = asym(hl, hr)
        feats[f"amygdala_asym_{stat}"] = asym(al, ar)

        feats[f"hippocampus_sum_{stat}"] = hl + hr
        feats[f"amygdala_sum_{stat}"] = al + ar

    l_mtl_vox = feats.get("hippocampus_left_vox", np.nan) + feats.get("amygdala_left_vox", np.nan)
    r_mtl_vox = feats.get("hippocampus_right_vox", np.nan) + feats.get("amygdala_right_vox", np.nan)

    feats["hippo_amyg_left_vox"] = l_mtl_vox
    feats["hippo_amyg_right_vox"] = r_mtl_vox
    feats["hippo_amyg_asym_vox"] = asym(l_mtl_vox, r_mtl_vox)
    feats["hippo_amyg_sum_vox"] = l_mtl_vox + r_mtl_vox

    return feats, "ok"


def build_features(df):
    rows, missing = [], []

    for _, row in df.iterrows():
        feats, reason = extract_subject(row)
        if feats is None:
            missing.append({"PTID": row["PTID"], "reason": reason})
        else:
            rows.append(feats)

    feat = pd.DataFrame(rows)
    feat.to_csv(OUT_DIR / "compact_t1_features.csv", index=False)
    pd.DataFrame(missing).to_csv(OUT_DIR / "missing_subjects.csv", index=False)

    print(f"[FEATURES] usable={len(feat)} missing={len(missing)}")
    print("[FEATURES] labels:", feat["label"].value_counts().to_dict())

    return feat


def make_models(k):
    base_svm = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("clf", SVC(kernel="rbf", C=0.7, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
    ])

    base_linear_svm = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("clf", SVC(kernel="linear", C=0.3, class_weight="balanced", probability=True, random_state=SEED)),
    ])

    return {
        f"elasticnet_k{k}_C0.05": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=k)),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,
                C=0.05,
                solver="saga",
                class_weight="balanced",
                max_iter=10000,
                random_state=SEED,
            )),
        ]),
        f"elasticnet_k{k}_C0.1": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=k)),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,
                C=0.1,
                solver="saga",
                class_weight="balanced",
                max_iter=10000,
                random_state=SEED,
            )),
        ]),
        f"logreg_l1_k{k}": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=k)),
            ("clf", LogisticRegression(
                penalty="l1",
                C=0.15,
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            )),
        ]),
        f"logreg_l2_k{k}": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=k)),
            ("clf", LogisticRegression(
                penalty="l2",
                C=0.25,
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            )),
        ]),
        f"svm_rbf_k{k}": base_svm,
        f"svm_linear_k{k}": base_linear_svm,
        f"rf_k{k}": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", RandomForestClassifier(
                n_estimators=800,
                max_depth=3,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),
        f"extra_trees_k{k}": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", ExtraTreesClassifier(
                n_estimators=800,
                max_depth=3,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),
    }


def selected_features_from_pipeline(model, feature_cols):
    try:
        selector = model.named_steps["select"]
        mask = selector.get_support()
        return [c for c, keep in zip(feature_cols, mask) if keep]
    except Exception:
        return []


def run_cv(feat, feature_cols):
    y = feat["label"].values.astype(int)
    X = feat[feature_cols].copy()
    ptids = feat["PTID"].values

    ks = [8, 12, 16, 20, min(25, len(feature_cols))]
    models = {}
    for k in ks:
        models.update(make_models(k))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    summary_rows = []
    oof = pd.DataFrame({"PTID": ptids, "label": y})

    feature_freq = []

    for model_name, model in models.items():
        probs = np.zeros(len(feat), dtype=float)
        fold_rows = []

        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
            probs[va] = p

            selected = selected_features_from_pipeline(model, feature_cols)
            for s in selected:
                feature_freq.append({"model": model_name, "fold": fold, "feature": s})

            thr = tune_threshold(y[va], p)

            fold_rows.append({
                "model": model_name,
                "fold": fold,
                **{f"val05_{k}": v for k, v in metrics(y[va], p, 0.5).items()},
                **{f"valtuned_{k}": v for k, v in metrics(y[va], p, thr).items()},
            })

        oof[f"prob_{model_name}"] = probs
        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(OUT_DIR / f"{model_name}_fold_metrics.csv", index=False)

        oof_thr = tune_threshold(y, probs)

        summary_rows.append({
            "model": model_name,
            "n_features_total": len(feature_cols),
            **{f"oof05_{k}": v for k, v in metrics(y, probs, 0.5).items()},
            **{f"ooftuned_{k}": v for k, v in metrics(y, probs, oof_thr).items()},
            "fold_roc_auc_mean": float(fold_df["val05_roc_auc"].mean()),
            "fold_pr_auc_mean": float(fold_df["val05_pr_auc"].mean()),
            "fold_bacc05_mean": float(fold_df["val05_balanced_accuracy"].mean()),
            "fold_bacc_tuned_mean": float(fold_df["valtuned_balanced_accuracy"].mean()),
        })

    summary = pd.DataFrame(summary_rows).sort_values("oof05_pr_auc", ascending=False)
    summary.to_csv(OUT_DIR / "compact_t1_model_summary.csv", index=False)
    oof.to_csv(OUT_DIR / "compact_t1_oof_predictions.csv", index=False)

    ff = pd.DataFrame(feature_freq)
    if len(ff):
        freq = (
            ff.groupby("feature")
            .size()
            .reset_index(name="selected_count")
            .sort_values("selected_count", ascending=False)
        )
        freq.to_csv(OUT_DIR / "feature_selection_frequency.csv", index=False)

    print("\n===== BEST BY PR-AUC =====")
    print(summary[[
        "model",
        "oof05_roc_auc",
        "oof05_pr_auc",
        "oof05_balanced_accuracy",
        "ooftuned_balanced_accuracy",
        "ooftuned_recall",
        "ooftuned_precision",
    ]].head(20).to_string(index=False))

    return summary


def main():
    df = pd.read_csv(MANIFEST_CSV)
    df["PTID"] = df["PTID"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    for c in TAB_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    feat = build_features(df)

    ignore = {"PTID", "label"}
    feature_cols = [
        c for c in feat.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(feat[c])
    ]

    print("[FEATURES USED]", len(feature_cols))
    print(feature_cols)

    run_cv(feat, feature_cols)

    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
