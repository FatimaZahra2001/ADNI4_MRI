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
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier


SEED = 42
N_FOLDS = 5

MANIFEST_CSV = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_dual_mri_manifest_with_demog.csv")
IMAGE_ROOT = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm")

CDR_CSV = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/csvs/CDR_07Nov2025.csv")
MMSE_CSV = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/csvs/MMSE_07Nov2025.csv")

OUT_DIR = Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_flair_roi_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAB_COLS = ["Age_Baseline", "Gender"]

ROI_MASKS = {
    "MTL_core_fixed": "MTL_core_fixed_mask.nii.gz",
    "MTL_roi": "MTL_roi_mask.nii.gz",
    "hippocampus_left": "hippocampus_left.nii.gz",
    "hippocampus_right": "hippocampus_right.nii.gz",
    "amygdala_left": "amygdala_left.nii.gz",
    "amygdala_right": "amygdala_right.nii.gz",
}

MODALITIES = {
    "T1": "T1_norm.nii.gz",
    "FLAIR": "FLAIR_norm.nii.gz",
}


def load_nii(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def find_col(df, candidates):
    for cand in candidates:
        for col in df.columns:
            if col.lower() == cand.lower():
                return col
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in col.lower():
                return col
    return None


def merge_clinical(df):
    df = df.copy()
    df["PTID"] = df["PTID"].astype(str).str.strip()

    if MMSE_CSV.exists():
        mmse = pd.read_csv(MMSE_CSV)
        mmse["PTID"] = mmse["PTID"].astype(str).str.strip()
        keep = ["PTID"] + [c for c in mmse.columns if "mmse" in c.lower() or "mmscore" in c.lower()]
        mmse = mmse[list(dict.fromkeys(keep))].drop_duplicates("PTID", keep="first")
        df = df.merge(mmse, on="PTID", how="left")
        print(f"[CLINICAL] merged MMSE: {MMSE_CSV}")

    if CDR_CSV.exists():
        cdr = pd.read_csv(CDR_CSV)
        cdr["PTID"] = cdr["PTID"].astype(str).str.strip()
        keep = ["PTID"] + [c for c in cdr.columns if "cdr" in c.lower()]
        cdr = cdr[list(dict.fromkeys(keep))].drop_duplicates("PTID", keep="first")
        df = df.merge(cdr, on="PTID", how="left")
        print(f"[CLINICAL] merged CDR: {CDR_CSV}")

    mmse_col = find_col(df, ["MMSE", "MMSCORE", "MMSEASON"])
    cdrsb_col = find_col(df, ["CDRSB", "SUMBOX", "CDR"])

    print("[CLINICAL COLS]", {"mmse": mmse_col, "cdrsb": cdrsb_col})

    if mmse_col is not None:
        df[mmse_col] = pd.to_numeric(df[mmse_col], errors="coerce")
        df["MMSE_USE"] = df[mmse_col]
    else:
        df["MMSE_USE"] = np.nan

    if cdrsb_col is not None:
        df[cdrsb_col] = pd.to_numeric(df[cdrsb_col], errors="coerce")
        df["CDRSB_USE"] = df[cdrsb_col]
    else:
        df["CDRSB_USE"] = np.nan

    return df


def metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= threshold).astype(int)

    try:
        roc = roc_auc_score(y, p)
    except Exception:
        roc = np.nan

    try:
        pr = average_precision_score(y, p)
    except Exception:
        pr = np.nan

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
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


def roi_stats(img, brain, mask):
    mask = (mask > 0) & brain

    if mask.sum() < 5:
        return {k: np.nan for k in ["vox", "frac_brain", "mean", "std", "median", "p10", "p90"]}

    vals = img[mask]

    return {
        "vox": float(mask.sum()),
        "frac_brain": float(mask.sum() / max(brain.sum(), 1)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


def add_asymmetry(feats, prefix, left, right):
    for suffix in ["vox", "frac_brain", "mean", "std", "median", "p10", "p90"]:
        l = feats.get(f"{prefix}_{left}_{suffix}", np.nan)
        r = feats.get(f"{prefix}_{right}_{suffix}", np.nan)

        feats[f"{prefix}_{left}_minus_{right}_{suffix}"] = l - r
        feats[f"{prefix}_{left}_plus_{right}_{suffix}"] = l + r
        feats[f"{prefix}_{left}_asym_{suffix}"] = (l - r) / (l + r + 1e-8)


def extract_subject_features(row):
    ptid = str(row["PTID"]).strip()
    sdir = IMAGE_ROOT / ptid

    t1_path = sdir / "T1_norm.nii.gz"
    flair_path = sdir / "FLAIR_norm.nii.gz"

    if not t1_path.exists():
        return None, "missing_t1"

    t1 = load_nii(t1_path)
    brain = t1 > 0

    if brain.sum() < 1000:
        return None, "bad_brain"

    feats = {
        "PTID": ptid,
        "label": int(row["label"]),
        "Age_Baseline": row.get("Age_Baseline", np.nan),
        "Gender": row.get("Gender", np.nan),
        "MMSE_USE": row.get("MMSE_USE", np.nan),
        "CDRSB_USE": row.get("CDRSB_USE", np.nan),
        "brain_vox": float(brain.sum()),
    }

    for mod, fname in MODALITIES.items():
        img_path = sdir / fname

        if not img_path.exists():
            continue

        img = load_nii(img_path)

        if img.shape != t1.shape:
            print(f"[WARN] shape mismatch {ptid} {mod}: {img.shape} vs {t1.shape}")
            continue

        vals = img[brain]
        imgz = np.zeros_like(img, dtype=np.float32)
        imgz[brain] = (vals - vals.mean()) / (vals.std() + 1e-6)

        feats[f"{mod}_brain_mean_raw"] = float(vals.mean())
        feats[f"{mod}_brain_std_raw"] = float(vals.std())

        for roi_name, mask_name in ROI_MASKS.items():
            mask_path = sdir / mask_name

            if not mask_path.exists():
                for suffix in ["vox", "frac_brain", "mean", "std", "median", "p10", "p90"]:
                    feats[f"{mod}_{roi_name}_{suffix}"] = np.nan
                continue

            mask = load_nii(mask_path)

            if mask.shape != t1.shape:
                for suffix in ["vox", "frac_brain", "mean", "std", "median", "p10", "p90"]:
                    feats[f"{mod}_{roi_name}_{suffix}"] = np.nan
                continue

            stats = roi_stats(imgz, brain, mask)

            for k, v in stats.items():
                feats[f"{mod}_{roi_name}_{k}"] = v

        add_asymmetry(feats, mod, "hippocampus_left", "hippocampus_right")
        add_asymmetry(feats, mod, "amygdala_left", "amygdala_right")

        l_mtl = feats.get(f"{mod}_hippocampus_left_vox", np.nan) + feats.get(f"{mod}_amygdala_left_vox", np.nan)
        r_mtl = feats.get(f"{mod}_hippocampus_right_vox", np.nan) + feats.get(f"{mod}_amygdala_right_vox", np.nan)

        feats[f"{mod}_hippo_amyg_left_vox"] = l_mtl
        feats[f"{mod}_hippo_amyg_right_vox"] = r_mtl
        feats[f"{mod}_hippo_amyg_asym_vox"] = (l_mtl - r_mtl) / (l_mtl + r_mtl + 1e-8)

    return feats, "ok"


def build_feature_table(df):
    rows, missing = [], []

    for _, row in df.iterrows():
        feats, reason = extract_subject_features(row)
        if feats is None:
            missing.append({"PTID": row["PTID"], "reason": reason})
        else:
            rows.append(feats)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "t1_flair_roi_features.csv", index=False)
    pd.DataFrame(missing).to_csv(OUT_DIR / "missing_subjects.csv", index=False)

    print(f"[FEATURES] usable={len(out)} missing={len(missing)}")
    print("[FEATURES] labels:", out["label"].value_counts().to_dict())

    return out


def get_models():
    return {
        "logreg_l1": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", C=0.25, class_weight="balanced",
                solver="liblinear", max_iter=5000, random_state=SEED
            )),
        ]),
        "logreg_l2": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", C=0.5, class_weight="balanced",
                solver="liblinear", max_iter=5000, random_state=SEED
            )),
        ]),
        "svm_rbf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=0.7, gamma="scale",
                probability=True, class_weight="balanced", random_state=SEED
            )),
        ]),
        "rf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=700, max_depth=4, min_samples_leaf=8,
                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
            )),
        ]),
        "extra_trees": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=700, max_depth=4, min_samples_leaf=8,
                class_weight="balanced", random_state=SEED, n_jobs=-1
            )),
        ]),
        "gb": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.03,
                max_depth=2, subsample=0.8, random_state=SEED
            )),
        ]),
    }


def run_cv(df, feature_cols, experiment_name):
    y = df["label"].values.astype(int)
    X = df[feature_cols].copy()
    ptids = df["PTID"].values

    models = get_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    summary_rows = []
    oof_df = pd.DataFrame({"PTID": ptids, "label": y})

    for model_name, model in models.items():
        oof = np.zeros(len(df), dtype=float)
        fold_rows = []

        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
            oof[va] = p

            thr = tune_threshold(y[va], p)

            fold_rows.append({
                "experiment": experiment_name,
                "model": model_name,
                "fold": fold,
                **{f"val05_{k}": v for k, v in metrics(y[va], p, 0.5).items()},
                **{f"valtuned_{k}": v for k, v in metrics(y[va], p, thr).items()},
            })

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(OUT_DIR / f"{experiment_name}_{model_name}_fold_metrics.csv", index=False)

        oof_df[f"prob_{model_name}"] = oof
        oof_thr = tune_threshold(y, oof)

        summary_rows.append({
            "experiment": experiment_name,
            "model": model_name,
            "n_subjects": len(df),
            "n_cn": int((y == 0).sum()),
            "n_mci": int((y == 1).sum()),
            "n_features": len(feature_cols),
            **{f"oof05_{k}": v for k, v in metrics(y, oof, 0.5).items()},
            **{f"ooftuned_{k}": v for k, v in metrics(y, oof, oof_thr).items()},
            "fold_roc_auc_mean": float(fold_df["val05_roc_auc"].mean()),
            "fold_pr_auc_mean": float(fold_df["val05_pr_auc"].mean()),
            "fold_bacc05_mean": float(fold_df["val05_balanced_accuracy"].mean()),
            "fold_bacc_tuned_mean": float(fold_df["valtuned_balanced_accuracy"].mean()),
        })

    summary = pd.DataFrame(summary_rows).sort_values("oof05_pr_auc", ascending=False)
    summary.to_csv(OUT_DIR / f"{experiment_name}_summary.csv", index=False)
    oof_df.to_csv(OUT_DIR / f"{experiment_name}_oof_predictions.csv", index=False)

    print(f"\n===== {experiment_name} =====")
    print(summary[[
        "model", "oof05_roc_auc", "oof05_pr_auc",
        "oof05_balanced_accuracy", "ooftuned_balanced_accuracy",
        "ooftuned_recall", "ooftuned_precision"
    ]].to_string(index=False))

    return summary


def main():
    df = pd.read_csv(MANIFEST_CSV)
    df["PTID"] = df["PTID"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    for c in TAB_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df = merge_clinical(df)
    feat = build_feature_table(df)

    feat["clinical_clear"] = (
        (feat["label"] == 0)
        | (
            (feat["label"] == 1)
            & (
                (feat["MMSE_USE"] <= 26)
                | (feat["CDRSB_USE"] >= 1.0)
            )
        )
    )

    print("[SUBSETS]")
    print("all:", feat["label"].value_counts().to_dict())
    print("clinical_clear:", feat[feat["clinical_clear"]]["label"].value_counts().to_dict())
    print("MMSE missing:", feat["MMSE_USE"].isna().sum())
    print("CDRSB missing:", feat["CDRSB_USE"].isna().sum())

    ignore = {"PTID", "label", "MMSE_USE", "CDRSB_USE", "clinical_clear"}

    all_numeric = [
        c for c in feat.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(feat[c])
    ]

    t1_cols = [
        c for c in all_numeric
        if c.startswith("T1_") or c in TAB_COLS or c == "brain_vox"
    ]

    t1_flair_cols = [
        c for c in all_numeric
        if c.startswith("T1_") or c.startswith("FLAIR_") or c in TAB_COLS or c == "brain_vox"
    ]

    experiments = [
        ("A_all_CN_vs_all_MCI_T1_only", feat.copy(), t1_cols),
        ("B_all_CN_vs_all_MCI_T1_FLAIR", feat.copy(), t1_flair_cols),
        ("C_CN_vs_clinical_clear_MCI_T1_only", feat[feat["clinical_clear"]].copy(), t1_cols),
        ("D_CN_vs_clinical_clear_MCI_T1_FLAIR", feat[feat["clinical_clear"]].copy(), t1_flair_cols),
    ]

    all_summaries = []

    for name, sub, cols in experiments:
        sub = sub.reset_index(drop=True)
        counts = sub["label"].value_counts().to_dict()

        if len(counts) < 2 or min(counts.values()) < 10:
            print(f"[SKIP] {name}, counts={counts}")
            continue

        s = run_cv(sub, cols, name)
        all_summaries.append(s)

    final = pd.concat(all_summaries, ignore_index=True)
    final.to_csv(OUT_DIR / "ALL_t1_flair_roi_ablation_summary.csv", index=False)

    print("\n===== BEST OVERALL BY PR-AUC =====")
    print(final.sort_values("oof05_pr_auc", ascending=False).head(20).to_string(index=False))

    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
