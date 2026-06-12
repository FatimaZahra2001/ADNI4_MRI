import json
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# =========================================================
# PATHS
# =========================================================

MANIFEST_CSV = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_dual_mri_manifest_with_demog.csv")
IMAGE_ROOT = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm")

# Optional CNN OOF files. Add/remove paths here.
CNN_OOF_PATHS = [
    Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/adni_anatomy_patch_selector_demog_v3_selective_topk/oof_predictions.csv"),
    Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/adni_anatomy_patch_selector_demog_v2_topk/oof_predictions.csv"),
    Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/adni_mtl_context64_attention/oof_predictions.csv"),
]

OUT_DIR = Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/adni_roi_tabular_baseline_ensemble")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5

TAB_COLS = ["Age_Baseline", "Gender"]

ROI_MASKS = {
    "MTL_core_fixed": "MTL_core_fixed_mask.nii.gz",
    "MTL_roi": "MTL_roi_mask.nii.gz",
    "hippocampus_left": "hippocampus_left.nii.gz",
    "hippocampus_right": "hippocampus_right.nii.gz",
    "amygdala_left": "amygdala_left.nii.gz",
    "amygdala_right": "amygdala_right.nii.gz",
}

T1_NAME = "T1_norm.nii.gz"


# =========================================================
# BASIC METRICS
# =========================================================

def metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= threshold).astype(int)

    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "auc": float(auc),
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


def load_nii(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def safe_roi_stats(t1, brain, mask):
    mask = (mask > 0) & brain

    if mask.sum() < 5:
        return {
            "vox": np.nan,
            "frac_brain": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
        }

    vals = t1[mask]
    return {
        "vox": float(mask.sum()),
        "frac_brain": float(mask.sum() / max(brain.sum(), 1)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


def add_asymmetry(feats, left_name, right_name):
    for suffix in ["vox", "frac_brain", "mean", "std", "median", "p10", "p90"]:
        l = feats.get(f"{left_name}_{suffix}", np.nan)
        r = feats.get(f"{right_name}_{suffix}", np.nan)

        feats[f"{left_name}_minus_{right_name}_{suffix}"] = l - r
        feats[f"{left_name}_plus_{right_name}_{suffix}"] = l + r
        feats[f"{left_name}_asym_{suffix}"] = (l - r) / (l + r + 1e-8)


def extract_subject_features(row):
    ptid = str(row["PTID"]).strip()
    sdir = IMAGE_ROOT / ptid

    t1_path = sdir / T1_NAME
    if not t1_path.exists():
        return None, "missing_t1"

    t1 = load_nii(t1_path)
    brain = t1 > 0

    if brain.sum() < 1000:
        return None, "bad_brain_mask"

    vals = t1[brain]
    t1z = np.zeros_like(t1, dtype=np.float32)
    t1z[brain] = (vals - vals.mean()) / (vals.std() + 1e-6)

    feats = {
        "PTID": ptid,
        "label": int(row["label"]),
        "brain_vox": float(brain.sum()),
        "brain_mean_raw": float(vals.mean()),
        "brain_std_raw": float(vals.std()),
    }

    for c in TAB_COLS:
        feats[c] = row.get(c, np.nan)

    missing_masks = []

    for roi_name, fname in ROI_MASKS.items():
        p = sdir / fname

        if not p.exists():
            missing_masks.append(roi_name)
            for suffix in ["vox", "frac_brain", "mean", "std", "median", "p10", "p90"]:
                feats[f"{roi_name}_{suffix}"] = np.nan
            continue

        m = load_nii(p)

        if m.shape != t1.shape:
            missing_masks.append(f"{roi_name}_shape_mismatch")
            for suffix in ["vox", "frac_brain", "mean", "std", "median", "p10", "p90"]:
                feats[f"{roi_name}_{suffix}"] = np.nan
            continue

        stats = safe_roi_stats(t1z, brain, m)

        for k, v in stats.items():
            feats[f"{roi_name}_{k}"] = v

    add_asymmetry(feats, "hippocampus_left", "hippocampus_right")
    add_asymmetry(feats, "amygdala_left", "amygdala_right")

    # Combined MTL asymmetry proxy from hippocampus + amygdala.
    l_mtl = feats.get("hippocampus_left_vox", np.nan) + feats.get("amygdala_left_vox", np.nan)
    r_mtl = feats.get("hippocampus_right_vox", np.nan) + feats.get("amygdala_right_vox", np.nan)
    feats["hippo_amyg_left_vox"] = l_mtl
    feats["hippo_amyg_right_vox"] = r_mtl
    feats["hippo_amyg_asym_vox"] = (l_mtl - r_mtl) / (l_mtl + r_mtl + 1e-8)

    feats["missing_masks"] = ";".join(missing_masks)

    return feats, "ok"


def build_feature_table(df):
    rows, missing = [], []

    for _, row in df.iterrows():
        feats, reason = extract_subject_features(row)
        if feats is None:
            missing.append((str(row["PTID"]), reason))
        else:
            rows.append(feats)

    feat_df = pd.DataFrame(rows)

    print(f"[FEATURES] usable={len(feat_df)} missing={len(missing)}")
    print("[FEATURES] labels:", feat_df["label"].value_counts().to_dict())
    if missing:
        print("[FEATURES] first missing:", missing[:20])

    feat_df.to_csv(OUT_DIR / "roi_tabular_features.csv", index=False)
    pd.DataFrame(missing, columns=["PTID", "reason"]).to_csv(OUT_DIR / "missing_subjects.csv", index=False)

    return feat_df


# =========================================================
# MODELS
# =========================================================

def get_models():
    models = {}

    models["logreg_l2"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k="all")),
        ("clf", LogisticRegression(
            penalty="l2",
            C=0.5,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )),
    ])

    models["logreg_l1"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k="all")),
        ("clf", LogisticRegression(
            penalty="l1",
            C=0.25,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )),
    ])

    ridge = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RidgeClassifier(class_weight="balanced", alpha=2.0)),
    ])
    models["ridge_calibrated"] = CalibratedClassifierCV(ridge, method="sigmoid", cv=3)

    models["svm_linear"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", SVC(
            kernel="linear",
            C=0.25,
            probability=True,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])

    models["svm_rbf"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=0.7,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])

    models["rf"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=700,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    models["extra_trees"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=700,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    models["gb"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.8,
            random_state=SEED,
        )),
    ])

    if HAS_XGB:
        models["xgb"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=250,
                max_depth=2,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                reg_alpha=0.5,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=4,
            )),
        ])

    if HAS_LGBM:
        models["lgbm"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(
                n_estimators=250,
                max_depth=2,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                reg_alpha=0.5,
                class_weight="balanced",
                random_state=SEED,
                verbose=-1,
            )),
        ])

    return models


def get_prob(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))

    pred = model.predict(X)
    return pred.astype(float)


def evaluate_models_cv(feat_df, feature_cols, subset_name):
    y = feat_df["label"].values.astype(int)
    X = feat_df[feature_cols].copy()
    ptids = feat_df["PTID"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    models = get_models()

    summary_rows = []
    oof_prob_cols = {}

    for model_name, model in models.items():
        oof = np.zeros(len(feat_df), dtype=float)
        fold_rows = []

        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]

            model.fit(Xtr, ytr)
            p = get_prob(model, Xva)
            oof[va] = p

            thr = tune_threshold(yva, p)

            row = {
                "subset": subset_name,
                "model": model_name,
                "fold": fold,
                **{f"val05_{k}": v for k, v in metrics(yva, p, 0.5).items()},
                **{f"valtuned_{k}": v for k, v in metrics(yva, p, thr).items()},
            }
            fold_rows.append(row)

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(OUT_DIR / f"{subset_name}_{model_name}_fold_metrics.csv", index=False)

        oof_prob_cols[f"prob_{model_name}"] = oof

        oof_thr = tune_threshold(y, oof)

        summary = {
            "subset": subset_name,
            "model": model_name,
            "n_features": len(feature_cols),
            **{f"oof05_{k}": v for k, v in metrics(y, oof, 0.5).items()},
            **{f"ooftuned_{k}": v for k, v in metrics(y, oof, oof_thr).items()},
            "fold_auc_mean": float(fold_df["val05_auc"].mean()),
            "fold_auc_std": float(fold_df["val05_auc"].std()),
            "fold_bacc05_mean": float(fold_df["val05_balanced_accuracy"].mean()),
            "fold_bacc_tuned_mean": float(fold_df["valtuned_balanced_accuracy"].mean()),
        }
        summary_rows.append(summary)

    oof_df = pd.DataFrame({
        "PTID": ptids,
        "label": y,
    })

    for k, v in oof_prob_cols.items():
        oof_df[k] = v

    summary_df = pd.DataFrame(summary_rows).sort_values("oof05_auc", ascending=False)

    oof_df.to_csv(OUT_DIR / f"{subset_name}_roi_model_oof_predictions.csv", index=False)
    summary_df.to_csv(OUT_DIR / f"{subset_name}_roi_model_summary.csv", index=False)

    return summary_df, oof_df


# =========================================================
# SUBSETS AND LABEL NOISE
# =========================================================

def add_clear_mci_proxy(feat_df):
    """
    No clinical severity column required. This defines 'clear-MCI' structurally:
    MCI subjects with low MTL/hippocampal volume proxies.

    This is NOT final clinical truth; it is a signal-proving subset.
    """
    df = feat_df.copy()

    cn = df[df["label"] == 0]

    for col in [
        "MTL_core_fixed_vox",
        "MTL_roi_vox",
        "hippocampus_left_vox",
        "hippocampus_right_vox",
        "hippo_amyg_left_vox",
        "hippo_amyg_right_vox",
    ]:
        if col in df.columns:
            cutoff = cn[col].quantile(0.35)
            df[f"{col}_low_vs_CN35"] = df[col] <= cutoff

    low_cols = [c for c in df.columns if c.endswith("_low_vs_CN35")]
    if low_cols:
        df["clear_mci_structural_score"] = df[low_cols].sum(axis=1)
    else:
        df["clear_mci_structural_score"] = 0

    df["subset_all"] = True
    df["subset_clear_mci"] = (df["label"] == 0) | ((df["label"] == 1) & (df["clear_mci_structural_score"] >= 2))
    df["subset_very_clear_mci"] = (df["label"] == 0) | ((df["label"] == 1) & (df["clear_mci_structural_score"] >= 3))

    return df


def label_noise_report(feat_df):
    rows = []

    cn = feat_df[feat_df["label"] == 0]
    mci = feat_df[feat_df["label"] == 1]

    for col in [c for c in feat_df.columns if c.endswith("_vox") or c.endswith("_asym_vox") or c.endswith("_mean")]:
        if col in ["brain_vox"]:
            continue

        cn_vals = pd.to_numeric(cn[col], errors="coerce")
        mci_vals = pd.to_numeric(mci[col], errors="coerce")

        rows.append({
            "feature": col,
            "CN_mean": cn_vals.mean(),
            "MCI_mean": mci_vals.mean(),
            "MCI_minus_CN": mci_vals.mean() - cn_vals.mean(),
            "CN_std": cn_vals.std(),
            "MCI_std": mci_vals.std(),
            "overlap_note": "large overlap likely" if abs(mci_vals.mean() - cn_vals.mean()) < 0.25 * (cn_vals.std() + 1e-8) else "",
        })

    report = pd.DataFrame(rows)
    report["abs_effect_proxy"] = report["MCI_minus_CN"].abs() / (report["CN_std"] + 1e-8)
    report = report.sort_values("abs_effect_proxy", ascending=False)
    report.to_csv(OUT_DIR / "label_noise_structural_overlap_report.csv", index=False)

    return report


# =========================================================
# HYBRID ENSEMBLE
# =========================================================

def load_cnn_oof():
    frames = []

    for i, p in enumerate(CNN_OOF_PATHS, start=1):
        if not p.exists():
            continue

        df = pd.read_csv(p)

        ptid_col = "ptid" if "ptid" in df.columns else "PTID"
        prob_col = "prob_mci"

        if ptid_col not in df.columns or prob_col not in df.columns:
            continue

        tmp = df[[ptid_col, prob_col]].copy()
        tmp.columns = ["PTID", f"cnn_prob_{i}"]
        tmp["PTID"] = tmp["PTID"].astype(str).str.strip()
        frames.append(tmp)

    if not frames:
        return None

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="PTID", how="outer")

    return out


def ensemble_oof(base_oof, roi_oof, subset_name):
    cnn = load_cnn_oof()

    ens = roi_oof.copy()
    ens["PTID"] = ens["PTID"].astype(str).str.strip()

    if cnn is not None:
        ens = ens.merge(cnn, on="PTID", how="left")

    prob_cols = [c for c in ens.columns if c.startswith("prob_") or c.startswith("cnn_prob_")]
    prob_cols = [c for c in prob_cols if ens[c].notna().sum() > 10]

    rows = []

    # Single average ensemble.
    ens["ensemble_mean"] = ens[prob_cols].mean(axis=1)

    y = ens["label"].values.astype(int)
    p = ens["ensemble_mean"].values.astype(float)
    thr = tune_threshold(y, p)

    rows.append({
        "subset": subset_name,
        "ensemble": "mean_all_available",
        "prob_cols": json.dumps(prob_cols),
        **{f"oof05_{k}": v for k, v in metrics(y, p, 0.5).items()},
        **{f"ooftuned_{k}": v for k, v in metrics(y, p, thr).items()},
    })

    # Best pair/triple simple search.
    from itertools import combinations

    for r in [2, 3]:
        for cols in combinations(prob_cols, r):
            pp = ens[list(cols)].mean(axis=1).values.astype(float)
            thr = tune_threshold(y, pp)
            rows.append({
                "subset": subset_name,
                "ensemble": f"mean_{r}",
                "prob_cols": json.dumps(list(cols)),
                **{f"oof05_{k}": v for k, v in metrics(y, pp, 0.5).items()},
                **{f"ooftuned_{k}": v for k, v in metrics(y, pp, thr).items()},
            })

    ens.to_csv(OUT_DIR / f"{subset_name}_hybrid_ensemble_oof_predictions.csv", index=False)

    ens_summary = pd.DataFrame(rows).sort_values("ooftuned_balanced_accuracy", ascending=False)
    ens_summary.to_csv(OUT_DIR / f"{subset_name}_hybrid_ensemble_summary.csv", index=False)

    return ens_summary


# =========================================================
# MAIN
# =========================================================

def main():
    df = pd.read_csv(MANIFEST_CSV)
    df["PTID"] = df["PTID"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    for c in TAB_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    feat_df = build_feature_table(df)
    feat_df = add_clear_mci_proxy(feat_df)

    noise_report = label_noise_report(feat_df)

    ignore_cols = {
        "PTID", "label", "missing_masks",
        "subset_all", "subset_clear_mci", "subset_very_clear_mci",
    }

    feature_cols = [
        c for c in feat_df.columns
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(feat_df[c])
    ]

    print(f"[FEATURES] n_features={len(feature_cols)}")
    print("[MODELS] xgboost:", HAS_XGB, "lightgbm:", HAS_LGBM)

    all_summaries = []
    all_ens = []

    subsets = {
        "all_CN_vs_all_MCI": feat_df["subset_all"],
        "CN_vs_clear_MCI": feat_df["subset_clear_mci"],
        "CN_vs_very_clear_MCI": feat_df["subset_very_clear_mci"],
    }

    for subset_name, mask in subsets.items():
        sub = feat_df[mask].copy().reset_index(drop=True)

        counts = sub["label"].value_counts().to_dict()
        print(f"\n===== {subset_name} =====")
        print("counts:", counts)

        if min(counts.values()) < 10:
            print("Skipping: too few minority samples")
            continue

        summary, oof = evaluate_models_cv(sub, feature_cols, subset_name)
        all_summaries.append(summary)

        ens_summary = ensemble_oof(feat_df, oof, subset_name)
        if ens_summary is not None:
            all_ens.append(ens_summary)

        print(summary[[
            "model", "oof05_auc", "oof05_balanced_accuracy",
            "ooftuned_balanced_accuracy", "ooftuned_recall", "ooftuned_precision"
        ]].head(10).to_string(index=False))

    final_model_summary = pd.concat(all_summaries, ignore_index=True)
    final_model_summary.to_csv(OUT_DIR / "ALL_roi_model_summary.csv", index=False)

    if all_ens:
        final_ens = pd.concat(all_ens, ignore_index=True)
        final_ens.to_csv(OUT_DIR / "ALL_hybrid_ensemble_summary.csv", index=False)

    with open(OUT_DIR / "README.txt", "w") as f:
        f.write("ROI/tabular baseline + CNN hybrid ensemble\n")
        f.write("==========================================\n\n")
        f.write(f"Manifest: {MANIFEST_CSV}\n")
        f.write(f"Image root: {IMAGE_ROOT}\n")
        f.write(f"Subjects with ROI features: {len(feat_df)}\n\n")
        f.write("Feature groups:\n")
        f.write("- ROI voxel volumes / brain fractions\n")
        f.write("- ROI intensity mean/std/median/p10/p90\n")
        f.write("- left-right asymmetry for hippocampus and amygdala\n")
        f.write("- MTL/hippocampus-amygdala combined proxies\n")
        f.write("- age and sex\n\n")
        f.write("Subset logic:\n")
        f.write("- all_CN_vs_all_MCI: full CN/MCI classification\n")
        f.write("- CN_vs_clear_MCI: MCI with structural MTL abnormality proxy score >= 2\n")
        f.write("- CN_vs_very_clear_MCI: MCI with structural MTL abnormality proxy score >= 3\n\n")
        f.write("Important: clear-MCI subsets are only for proving structural signal.\n")
        f.write("They are not final clinical labels.\n\n")
        f.write("Existing pymba/poset analysis can later be ensembled too if its OOF outputs are saved.\n")
        f.write("This ROI script is intentionally faster and more direct than the CNN patch selector.\n")

    print("\nSaved to:", OUT_DIR)
    print("\nBest overall ROI models:")
    print(final_model_summary.sort_values("ooftuned_balanced_accuracy", ascending=False).head(15).to_string(index=False))

    if all_ens:
        print("\nBest hybrid ensembles:")
        print(final_ens.sort_values("ooftuned_balanced_accuracy", ascending=False).head(15).to_string(index=False))

    print("\nTop structural overlap/effect proxies:")
    print(noise_report.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
