import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from scipy.ndimage import binary_erosion

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    IsolationForest
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_FOLDS = 5
TEST_SIZE = 0.20

MANIFEST = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_t1_clinical_manifest.csv")
ROOT = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm")

OUT = Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_radiomics_normative_clean_eval")
OUT.mkdir(parents=True, exist_ok=True)

T1_NAME = "T1_norm.nii.gz"

ROI_MASKS = {
    "MTL_core_fixed": "MTL_core_fixed_mask.nii.gz",
    "hippocampus_left": "hippocampus_left.nii.gz",
    "hippocampus_right": "hippocampus_right.nii.gz",
    "amygdala_left": "amygdala_left.nii.gz",
    "amygdala_right": "amygdala_right.nii.gz",
}

LEAKAGE_COL_KEYWORDS = [
    "cdglobal", "cdr", "cdrsb", "mmse", "diagnosis", "dx",
    "clinical_severe_mci", "split", "viscode", "visdate"
]


# ============================================================
# BASIC HELPERS
# ============================================================

def load_nii(p):
    return nib.load(str(p)).get_fdata(dtype=np.float32)


def asym(l, r):
    return (l - r) / (l + r + 1e-8)


def safe_roc_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def safe_pr_auc(y, p):
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return np.nan


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": safe_roc_auc(y, p),
        "pr_auc": safe_pr_auc(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(thr),
    }


def tune_threshold(y, p):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -1

    for t in thresholds:
        score = balanced_accuracy_score(y, (p >= t).astype(int))
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


def get_prob(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))

    return model.predict(X).astype(float)


# ============================================================
# MRI FEATURE EXTRACTION
# ============================================================

def first_order_features(vals):
    vals = vals[np.isfinite(vals)]

    if vals.size < 10:
        return {}

    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        "skew": float(skew(vals)),
        "kurtosis": float(kurtosis(vals)),
    }


def shape_features(mask):
    mask = mask.astype(bool)
    vox = int(mask.sum())

    if vox == 0:
        return {
            "vox": 0,
            "surface_proxy": 0,
            "surface_to_volume": 0,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_z": 0,
            "bbox_ratio_xy": 0,
            "bbox_ratio_xz": 0,
            "bbox_ratio_yz": 0,
            "compactness_proxy": 0,
        }

    eroded = binary_erosion(mask)
    surface = int(mask.sum() - eroded.sum())

    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox = maxs - mins + 1

    bx, by, bz = bbox.astype(float)

    return {
        "vox": vox,
        "surface_proxy": surface,
        "surface_to_volume": float(surface / (vox + 1e-8)),
        "bbox_x": float(bx),
        "bbox_y": float(by),
        "bbox_z": float(bz),
        "bbox_ratio_xy": float(bx / (by + 1e-8)),
        "bbox_ratio_xz": float(bx / (bz + 1e-8)),
        "bbox_ratio_yz": float(by / (bz + 1e-8)),
        "compactness_proxy": float(vox / (bx * by * bz + 1e-8)),
    }


def extract_roi_features(img, mask, prefix):
    out = {}

    m = mask > 0
    vals = img[m]

    for k, v in shape_features(m).items():
        out[f"{prefix}_{k}"] = v

    for k, v in first_order_features(vals).items():
        out[f"{prefix}_{k}"] = v

    return out


def extract_subject(row):
    ptid = str(row["PTID"]).strip()
    subj_dir = ROOT / ptid

    t1_path = subj_dir / T1_NAME
    if not t1_path.exists():
        return None

    try:
        img = load_nii(t1_path)
    except Exception:
        return None

    feats = {
        "PTID": ptid,
        "label": int(row["label"]),
        "Age_Baseline": float(row.get("Age_Baseline", np.nan)),
        "Gender": float(row.get("Gender", np.nan)),
        "brain_vox": float(np.isfinite(img).sum()),
    }

    roi_vox = {}

    for roi_name, mask_name in ROI_MASKS.items():
        mask_path = subj_dir / mask_name
        if not mask_path.exists():
            continue

        try:
            mask = load_nii(mask_path)
        except Exception:
            continue

        roi_feats = extract_roi_features(img, mask, roi_name)
        feats.update(roi_feats)

        vox_key = f"{roi_name}_vox"
        roi_vox[roi_name] = feats.get(vox_key, np.nan)

    # biologically useful ratios/asymmetry
    hl = roi_vox.get("hippocampus_left", np.nan)
    hr = roi_vox.get("hippocampus_right", np.nan)
    al = roi_vox.get("amygdala_left", np.nan)
    ar = roi_vox.get("amygdala_right", np.nan)
    mtl = roi_vox.get("MTL_core_fixed", np.nan)

    feats["hipp_asym_vox"] = asym(hl, hr)
    feats["amyg_asym_vox"] = asym(al, ar)

    hsum = hl + hr
    asum = al + ar

    feats["hipp_total_vox"] = hsum
    feats["amyg_total_vox"] = asum
    feats["hipp_to_mtl"] = hsum / (mtl + 1e-8)
    feats["amyg_to_mtl"] = asum / (mtl + 1e-8)
    feats["mtl_to_brain"] = mtl / (feats["brain_vox"] + 1e-8)

    return feats


def build_features(df, out_dir):
    rows = []
    missing = []

    for _, row in df.iterrows():
        r = extract_subject(row)
        if r is None:
            missing.append(str(row["PTID"]))
        else:
            rows.append(r)

    feat = pd.DataFrame(rows)

    feat.to_csv(out_dir / "radiomics_raw_features.csv", index=False)
    pd.DataFrame({"PTID": missing}).to_csv(out_dir / "missing_subjects.csv", index=False)

    print("[FEATURES]", feat.shape)
    print("[LABELS]", feat["label"].value_counts().to_dict())

    return feat


# ============================================================
# TRANSFORMERS
# ============================================================

class NormativeDeviationTransformer(BaseEstimator, TransformerMixin):
    """
    Fits CN-only age/gender/brain adjusted normative models
    on TRAINING DATA ONLY inside each fold.
    """

    def __init__(self, covars=("Age_Baseline", "Gender", "brain_vox"), min_cn=30, alpha=1.0):
        self.covars = covars
        self.min_cn = min_cn
        self.alpha = alpha

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y)

        self.feature_names_in_ = list(X.columns)
        self.models_ = {}
        self.resid_std_ = {}

        cn_mask = y == 0
        cn = X.loc[cn_mask].copy()

        usable_covars = [c for c in list(self.covars) if c in X.columns]
        self.usable_covars_ = usable_covars

        if len(usable_covars) == 0:
            return self

        cov_imputer = SimpleImputer(strategy="median")
        Xcn_cov = cov_imputer.fit_transform(cn[usable_covars])
        self.cov_imputer_ = cov_imputer

        for col in self.feature_names_in_:
            if col in usable_covars:
                continue

            if not pd.api.types.is_numeric_dtype(X[col]):
                continue

            vals = cn[col].values
            valid = np.isfinite(vals)

            if valid.sum() < self.min_cn:
                continue

            reg = Ridge(alpha=self.alpha)
            reg.fit(Xcn_cov[valid], vals[valid])

            pred = reg.predict(Xcn_cov[valid])
            resid_std = np.std(vals[valid] - pred) + 1e-6

            self.models_[col] = reg
            self.resid_std_[col] = resid_std

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = X.copy()

        if not hasattr(self, "models_") or len(self.models_) == 0:
            return out

        Xcov = self.cov_imputer_.transform(X[self.usable_covars_])

        for col, reg in self.models_.items():
            pred = reg.predict(Xcov)
            out[f"{col}_normdev"] = (X[col].values - pred) / self.resid_std_[col]

        return out


class CNAnomalyTransformer(BaseEstimator, TransformerMixin):
    """
    Fits CN-only anomaly models on TRAINING DATA ONLY.
    Adds anomaly score columns to train/val/test safely.
    """

    def __init__(self, contamination=0.25, nu=0.25, random_state=42):
        self.contamination = contamination
        self.nu = nu
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y)

        self.feature_names_in_ = list(X.columns)

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()

        X_imp = self.imputer_.fit_transform(X)
        X_scaled = self.scaler_.fit_transform(X_imp)

        cn_mask = y == 0
        Xcn = X_scaled[cn_mask]

        self.iso_ = IsolationForest(
            n_estimators=400,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.iso_.fit(Xcn)

        self.ocsvm_ = OneClassSVM(gamma="scale", nu=self.nu)
        self.ocsvm_.fit(Xcn)

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        X_imp = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imp)

        out = X.copy()
        out["iso_cn_anomaly"] = -self.iso_.score_samples(X_scaled)
        out["ocsvm_cn_anomaly"] = -self.ocsvm_.score_samples(X_scaled)

        return out


# ============================================================
# MODELS
# ============================================================

def make_models(k):
    models = {}

    for C in [0.05, 0.1, 0.5, 1.0, 2.0]:
        models[f"elasticnet_C{C}_k{k}"] = Pipeline([
            ("normdev", NormativeDeviationTransformer()),
            ("cnanom", CNAnomalyTransformer(random_state=SEED)),
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=k)),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,
                C=C,
                solver="saga",
                class_weight="balanced",
                max_iter=10000,
                random_state=SEED,
            )),
        ])

    for C in [0.5, 1.0, 2.0, 5.0, 10.0]:
        models[f"svm_rbf_C{C}_k{k}"] = Pipeline([
            ("normdev", NormativeDeviationTransformer()),
            ("cnanom", CNAnomalyTransformer(random_state=SEED)),
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=k)),
            ("clf", SVC(
                kernel="rbf",
                C=C,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=SEED,
            )),
        ])

    models[f"extra_trees_k{k}"] = Pipeline([
        ("normdev", NormativeDeviationTransformer()),
        ("cnanom", CNAnomalyTransformer(random_state=SEED)),
        ("imp", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("clf", ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    models[f"random_forest_k{k}"] = Pipeline([
        ("normdev", NormativeDeviationTransformer()),
        ("cnanom", CNAnomalyTransformer(random_state=SEED)),
        ("imp", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("clf", RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    models[f"gradient_boosting_k{k}"] = Pipeline([
        ("normdev", NormativeDeviationTransformer()),
        ("cnanom", CNAnomalyTransformer(random_state=SEED)),
        ("imp", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=SEED,
        )),
    ])

    return models


def selected_features(model, feature_cols):
    try:
        names = list(model.named_steps["normdev"].transform(pd.DataFrame(columns=feature_cols)).columns)
    except Exception:
        names = feature_cols

    try:
        mask = model.named_steps["select"].get_support()
        return [f for f, keep in zip(names, mask) if keep]
    except Exception:
        return []


# ============================================================
# PLOTS
# ============================================================

def plot_leaderboard(res, out_dir, metric="val_balanced_accuracy", top_n=20):
    d = res.sort_values(metric, ascending=False).head(top_n)

    plt.figure(figsize=(10, max(4, 0.35 * len(d))))
    plt.barh(d["model"][::-1], d[metric][::-1])
    plt.xlabel(metric.replace("_", " ").upper())
    plt.title(f"Top models by {metric.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(out_dir / f"top_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc(y, p, out_path, title):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)

    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr(y, p, out_path, title):
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    plt.figure(figsize=(5.5, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion(y, p, thr, out_path, title):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

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
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# EVALUATION
# ============================================================

def run_cv_and_test(feat, feature_cols, out_dir):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    X = feat[feature_cols].copy()
    y = feat["label"].values.astype(int)
    ptids = feat["PTID"].values

    dev_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEED,
    )

    X_dev = X.iloc[dev_idx].reset_index(drop=True)
    y_dev = y[dev_idx]
    ptid_dev = ptids[dev_idx]

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]
    ptid_test = ptids[test_idx]

    split_df = pd.DataFrame({
        "PTID": np.concatenate([ptid_dev, ptid_test]),
        "label": np.concatenate([y_dev, y_test]),
        "split": ["dev"] * len(y_dev) + ["test"] * len(y_test),
    })
    split_df.to_csv(out_dir / "trainval_test_split.csv", index=False)

    print("[DEV COUNTS]", dict(pd.Series(y_dev).value_counts().sort_index()))
    print("[TEST COUNTS]", dict(pd.Series(y_test).value_counts().sort_index()))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    k_values = [6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 75]
    k_values = sorted(set([min(k, len(feature_cols)) for k in k_values if min(k, len(feature_cols)) >= 2]))

    models = {}
    for k in k_values:
        models.update(make_models(k))

    rows = []
    oof_df = pd.DataFrame({"PTID": ptid_dev, "label": y_dev})
    feature_rows = []

    best_model_name = None
    best_score = -1
    best_oof = None
    best_thr = 0.5

    for name, model in models.items():
        print(f"\n========== {name} ==========")

        oof = np.zeros(len(y_dev), dtype=float)
        fold_rows = []

        for fold, (tr, va) in enumerate(skf.split(X_dev, y_dev), 1):
            m = clone(model)
            m.fit(X_dev.iloc[tr], y_dev[tr])

            p = get_prob(m, X_dev.iloc[va])
            oof[va] = p

            thr_fold = tune_threshold(y_dev[tr], get_prob(m, X_dev.iloc[tr]))
            fold_metric = metrics(y_dev[va], p, thr_fold)
            fold_metric["fold"] = fold
            fold_rows.append(fold_metric)

            print(
                f"Fold {fold}: "
                f"AUC={fold_metric['roc_auc']:.3f} "
                f"bACC={fold_metric['balanced_accuracy']:.3f} "
                f"recall={fold_metric['recall']:.3f} "
                f"thr={thr_fold:.3f}"
            )

            for f in selected_features(m, feature_cols):
                feature_rows.append({"model": name, "fold": fold, "feature": f})

        thr = tune_threshold(y_dev, oof)

        val05 = metrics(y_dev, oof, 0.5)
        valtuned = metrics(y_dev, oof, thr)

        oof_df[f"prob_{name}"] = oof

        row = {
            "model": name,
            "n_dev": len(y_dev),
            "n_test": len(y_test),
            "n_features_total": len(feature_cols),
            "val_threshold": thr,
            **{f"val05_{k}": v for k, v in val05.items()},
            **{f"valtuned_{k}": v for k, v in valtuned.items()},
            "mean_fold_auc": float(np.nanmean([r["roc_auc"] for r in fold_rows])),
            "std_fold_auc": float(np.nanstd([r["roc_auc"] for r in fold_rows])),
            "mean_fold_bacc": float(np.nanmean([r["balanced_accuracy"] for r in fold_rows])),
            "std_fold_bacc": float(np.nanstd([r["balanced_accuracy"] for r in fold_rows])),
        }
        rows.append(row)

        score = valtuned["balanced_accuracy"]

        if score > best_score:
            best_score = score
            best_model_name = name
            best_oof = oof.copy()
            best_thr = thr

    res = pd.DataFrame(rows).sort_values(
        ["valtuned_balanced_accuracy", "valtuned_roc_auc", "valtuned_pr_auc"],
        ascending=False,
    )

    res.to_csv(out_dir / "cv_model_results.csv", index=False)
    oof_df.to_csv(out_dir / "dev_oof_predictions.csv", index=False)

    if feature_rows:
        freq = (
            pd.DataFrame(feature_rows)
            .groupby("feature")
            .size()
            .reset_index(name="selected_count")
            .sort_values("selected_count", ascending=False)
        )
        freq.to_csv(out_dir / "feature_selection_frequency.csv", index=False)

    print("\n===== TOP 20 BY DEV BALANCED ACCURACY =====")
    print(res[[
        "model",
        "valtuned_roc_auc",
        "valtuned_pr_auc",
        "valtuned_balanced_accuracy",
        "valtuned_accuracy",
        "valtuned_precision",
        "valtuned_recall",
        "valtuned_specificity",
        "valtuned_f1",
        "val_threshold",
    ]].head(20).to_string(index=False))

    plot_leaderboard(res.rename(columns={
        "valtuned_balanced_accuracy": "val_balanced_accuracy",
        "valtuned_roc_auc": "val_roc_auc",
        "valtuned_pr_auc": "val_pr_auc",
    }), plot_dir, metric="val_balanced_accuracy")

    plot_leaderboard(res.rename(columns={
        "valtuned_balanced_accuracy": "val_balanced_accuracy",
        "valtuned_roc_auc": "val_roc_auc",
        "valtuned_pr_auc": "val_pr_auc",
    }), plot_dir, metric="val_roc_auc")

    # refit best on all development data, evaluate ONCE on untouched test
    print("\n[BEST DEV MODEL]", best_model_name)

    best_model = clone(models[best_model_name])
    best_model.fit(X_dev, y_dev)

    test_prob = get_prob(best_model, X_test)
    test_metrics_05 = metrics(y_test, test_prob, 0.5)
    test_metrics_tuned = metrics(y_test, test_prob, best_thr)

    pd.DataFrame({
        "PTID": ptid_test,
        "label": y_test,
        "prob": test_prob,
        "pred05": (test_prob >= 0.5).astype(int),
        "pred_tuned": (test_prob >= best_thr).astype(int),
    }).to_csv(out_dir / "test_predictions_best_model.csv", index=False)

    test_summary = {
        "best_model": best_model_name,
        "dev_selected_threshold": best_thr,
        **{f"test05_{k}": v for k, v in test_metrics_05.items()},
        **{f"testtuned_{k}": v for k, v in test_metrics_tuned.items()},
    }

    pd.DataFrame([test_summary]).to_csv(out_dir / "test_metrics_best_model.csv", index=False)

    print("\n===== FINAL TEST PERFORMANCE: BEST DEV MODEL =====")
    for k, v in test_summary.items():
        print(f"{k}: {v}")

    plot_roc(y_dev, best_oof, plot_dir / "best_dev_oof_roc.png", f"Dev OOF ROC: {best_model_name}")
    plot_pr(y_dev, best_oof, plot_dir / "best_dev_oof_pr.png", f"Dev OOF PR: {best_model_name}")
    plot_confusion(y_dev, best_oof, best_thr, plot_dir / "best_dev_oof_confusion.png", f"Dev OOF Confusion: {best_model_name}")

    plot_roc(y_test, test_prob, plot_dir / "best_test_roc.png", f"Test ROC: {best_model_name}")
    plot_pr(y_test, test_prob, plot_dir / "best_test_pr.png", f"Test PR: {best_model_name}")
    plot_confusion(y_test, test_prob, best_thr, plot_dir / "best_test_confusion.png", f"Test Confusion: {best_model_name}")

    return res


# ============================================================
# EXPERIMENT WRAPPER
# ============================================================

def remove_leakage_columns(df):
    drop = []

    for c in df.columns:
        cl = c.lower()
        if c in ["PTID", "label"]:
            continue
        if any(k in cl for k in LEAKAGE_COL_KEYWORDS):
            drop.append(c)

    if drop:
        print("[DROPPING LEAKAGE-LIKE COLUMNS]", drop)
        df = df.drop(columns=drop)

    return df


def run_experiment(name, df):
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================")
    print("EXPERIMENT:", name)
    print("COUNTS:", df["label"].value_counts().to_dict())
    print("============================")

    feat = build_features(df, out_dir)
    feat = remove_leakage_columns(feat)

    feature_cols = [
        c for c in feat.columns
        if c not in ["PTID", "label"]
        and pd.api.types.is_numeric_dtype(feat[c])
    ]

    feat.to_csv(out_dir / "radiomics_raw_feature_table_no_leakage.csv", index=False)

    print("[FINAL BASE FEATURES BEFORE PIPELINE TRANSFORMS]", len(feature_cols))

    run_cv_and_test(feat, feature_cols, out_dir)

    print("Saved:", out_dir)


def main():
    df = pd.read_csv(MANIFEST)

    df["PTID"] = df["PTID"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # Keep only binary HC/MCI labels
    df = df[df["label"].isin([0, 1])].copy()

    for c in ["Age_Baseline", "Gender"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    if "clinical_severe_mci" in df.columns:
        if df["clinical_severe_mci"].dtype == object:
            df["clinical_severe_mci"] = (
                df["clinical_severe_mci"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
        else:
            df["clinical_severe_mci"] = df["clinical_severe_mci"].astype(bool)

    run_experiment("all_mci_clean_holdout", df.copy())

    if "clinical_severe_mci" in df.columns:
        severe = df[df["clinical_severe_mci"]].copy()
        if severe["label"].value_counts().min() >= 20:
            run_experiment("clinical_severe_mci_clean_holdout", severe)


if __name__ == "__main__":
    main()
