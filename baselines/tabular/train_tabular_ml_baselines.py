import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")


LABEL_CANDIDATES = ["label", "y", "diagnosis_bin", "CDGLOBAL_STR"]
ID_CANDIDATES = ["PTID", "ptid", "Subject", "subject", "RID", "subject_id"]


CSV_FILES = {
    "normdev_all_mci": "/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_normative_deviation_all_mci/raw_roi_features.csv",
    "normdev_clinical_severe_mci": "/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_normative_deviation_clinical_severe_mci/raw_roi_features.csv",
    "roi_tabular_baseline": "/rds/projects/j/jouaitim-mri-test/fatima/outputs/adni_roi_tabular_baseline_ensemble/roi_tabular_features.csv",
    "clinical_roi_features": "/rds/projects/j/jouaitim-mri-test/fatima/outputs/adni_roi_clinical_severity_explainable/roi_tabular_clinical_features.csv",
    "compact_t1_features": "/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_compact_pr_auc/compact_t1_features.csv",
    "radiomics_all_mci": "/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_radiomics_normative/all_mci/radiomics_raw_features.csv",
    "radiomics_clinical_severe_mci": "/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_radiomics_normative/clinical_severe_mci/radiomics_raw_features.csv",
}


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def clean_label(y):
    if y.dtype == object:
        mapping = {
            "HC": 0, "CN": 0, "CONTROL": 0, "NORMAL": 0,
            "MCI": 1, "EMCI": 1, "LMCI": 1,
            "0": 0, "1": 1,
        }
        return y.astype(str).str.upper().map(mapping).astype(int).values
    return y.astype(int).values


def get_feature_cols(df, label_col, id_col):
    exclude = {label_col}
    if id_col:
        exclude.add(id_col)

    leakage_keywords = [
        "diagnosis", "label", "target", "split", "fold",
        "cdglobal_str", "class", "group"
    ]

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(k in c.lower() for k in leakage_keywords):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    return cols


def make_models():
    return {
        "logreg_l2": (
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
            {
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            },
        ),

        "logreg_elasticnet": (
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                max_iter=10000,
                random_state=42,
            ),
            {
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3],
                "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        ),

        "svm_linear": (
            SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            {
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3],
            },
        ),

        "svm_rbf": (
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            {
                "clf__C": [0.1, 0.3, 1, 3, 10],
                "clf__gamma": ["scale", 0.001, 0.003, 0.01, 0.03],
            },
        ),

        "random_forest": (
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "clf__max_depth": [2, 3, 4, None],
                "clf__min_samples_leaf": [2, 4, 8],
                "clf__max_features": ["sqrt", 0.5, None],
            },
        ),

        "extra_trees": (
            ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "clf__max_depth": [2, 3, 4, None],
                "clf__min_samples_leaf": [2, 4, 8],
                "clf__max_features": ["sqrt", 0.5, None],
            },
        ),

        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.01, 0.03, 0.1],
                "clf__max_depth": [1, 2, 3],
            },
        ),
    }


def build_pipe(model, k):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", model),
    ])


def prob_from_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def best_threshold(y_true, prob):
    best_t = 0.5
    best_bacc = -1

    for t in np.linspace(0.05, 0.95, 181):
        pred = (prob >= t).astype(int)
        bacc = balanced_accuracy_score(y_true, pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_t = t

    return float(best_t), float(best_bacc)


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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_bar(df, metric, outpath, top_n=20):
    d = df.sort_values(metric, ascending=False).head(top_n)
    plt.figure(figsize=(11, max(5, 0.38 * len(d))))
    plt.barh(d["run_name"][::-1], d[metric][::-1])
    plt.xlabel(metric.replace("_", " ").upper())
    plt.title(f"Top models by {metric.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc(y, prob, outpath, title):
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)

    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr(y, prob, outpath, title):
    precision, recall, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)

    plt.figure(figsize=(5.5, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cm(y, prob, threshold, outpath, title):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred)

    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["HC", "MCI"])
    plt.yticks([0, 1], ["HC", "MCI"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=15)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(pipe, feature_cols, outpath, title, top_n=25):
    selector = pipe.named_steps["select"]
    clf = pipe.named_steps["clf"]
    selected = np.array(feature_cols)[selector.get_support()]

    values = None
    if hasattr(clf, "coef_"):
        values = np.abs(clf.coef_).ravel()
    elif hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_

    if values is None:
        return

    order = np.argsort(values)[::-1][:top_n]

    plt.figure(figsize=(8, max(5, 0.35 * len(order))))
    plt.barh(selected[order][::-1], values[order][::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def run_one_csv(name, csv_path, outdir, args):
    print(f"\n\n{'='*100}")
    print(f"DATASET: {name}")
    print(f"CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    label_col = find_col(df, LABEL_CANDIDATES)
    id_col = find_col(df, ID_CANDIDATES, required=False)

    y = clean_label(df[label_col])
    feature_cols = get_feature_cols(df, label_col, id_col)

    print(f"Shape: {df.shape}")
    print(f"Label col: {label_col}")
    print(f"ID col: {id_col}")
    print(f"Labels: {dict(pd.Series(y).value_counts().sort_index())}")
    print(f"Numeric usable features: {len(feature_cols)}")

    X = df[feature_cols].copy()

    dataset_dir = outdir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "features_used.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    models = make_models()
    outer_cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    inner_cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)

    rows = []
    prediction_store = {}

    for model_name, (base_model, grid_params) in models.items():
        for k_raw in args.k_values:
            k = min(k_raw, len(feature_cols))
            if k < 2:
                continue

            run_name = f"{name}__{model_name}__k{k}"
            print(f"\n--- {run_name} ---")

            oof_prob = np.zeros(len(y))
            thresholds = []
            fold_rows = []

            for fold, (tr, va) in enumerate(outer_cv.split(X, y), 1):
                X_tr, X_va = X.iloc[tr], X.iloc[va]
                y_tr, y_va = y[tr], y[va]

                pipe = build_pipe(clone(base_model), k)

                search = GridSearchCV(
                    pipe,
                    grid_params,
                    scoring=args.scoring,
                    cv=inner_cv,
                    n_jobs=args.n_jobs,
                    refit=True,
                )

                search.fit(X_tr, y_tr)
                best_model = search.best_estimator_

                try:
                    tr_prob_cv = cross_val_predict(
                        clone(best_model), X_tr, y_tr,
                        cv=inner_cv,
                        method="predict_proba",
                        n_jobs=args.n_jobs,
                    )[:, 1]
                except Exception:
                    tr_score_cv = cross_val_predict(
                        clone(best_model), X_tr, y_tr,
                        cv=inner_cv,
                        method="decision_function",
                        n_jobs=args.n_jobs,
                    )
                    tr_prob_cv = 1 / (1 + np.exp(-tr_score_cv))

                threshold, train_bacc = best_threshold(y_tr, tr_prob_cv)

                best_model.fit(X_tr, y_tr)
                va_prob = prob_from_model(best_model, X_va)

                oof_prob[va] = va_prob
                thresholds.append(threshold)

                m = metrics(y_va, va_prob, threshold)
                m["fold"] = fold
                m["train_threshold_bacc"] = train_bacc
                fold_rows.append(m)

                print(
                    f"Fold {fold}: AUC={m['roc_auc']:.3f} "
                    f"bACC={m['balanced_accuracy']:.3f} "
                    f"recall={m['recall']:.3f} "
                    f"spec={m['specificity']:.3f} "
                    f"thr={threshold:.3f}"
                )

            final_thr = float(np.median(thresholds))
            final = metrics(y, oof_prob, final_thr)

            row = {
                "dataset": name,
                "run_name": run_name,
                "model": model_name,
                "k": k,
                "n_subjects": len(y),
                "n_hc": int((y == 0).sum()),
                "n_mci": int((y == 1).sum()),
                "n_features_total": len(feature_cols),
                **final,
                "mean_fold_auc": float(np.mean([r["roc_auc"] for r in fold_rows])),
                "std_fold_auc": float(np.std([r["roc_auc"] for r in fold_rows])),
                "mean_fold_bacc": float(np.mean([r["balanced_accuracy"] for r in fold_rows])),
                "std_fold_bacc": float(np.std([r["balanced_accuracy"] for r in fold_rows])),
            }

            rows.append(row)
            prediction_store[run_name] = (y.copy(), oof_prob.copy(), final_thr, feature_cols, clone(base_model), grid_params, k)

            pd.DataFrame(fold_rows).to_csv(dataset_dir / f"{run_name}_folds.csv", index=False)

    result_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "roc_auc", "pr_auc"],
        ascending=False,
    )

    result_df.to_csv(dataset_dir / "leaderboard.csv", index=False)

    print(f"\nTOP RESULTS FOR {name}")
    print(result_df.head(15)[[
        "run_name", "roc_auc", "pr_auc", "balanced_accuracy",
        "accuracy", "precision", "recall", "specificity", "f1", "threshold"
    ]].to_string(index=False))

    plot_bar(result_df, "balanced_accuracy", dataset_dir / "top_balanced_accuracy.png")
    plot_bar(result_df, "roc_auc", dataset_dir / "top_roc_auc.png")
    plot_bar(result_df, "pr_auc", dataset_dir / "top_pr_auc.png")

    best_name = result_df.iloc[0]["run_name"]
    y_best, prob_best, thr_best, cols_best, model_best, grid_best, k_best = prediction_store[best_name]

    plot_roc(y_best, prob_best, dataset_dir / "best_roc_curve.png", f"{name}: Best ROC")
    plot_pr(y_best, prob_best, dataset_dir / "best_pr_curve.png", f"{name}: Best PR")
    plot_cm(y_best, prob_best, thr_best, dataset_dir / "best_confusion_matrix.png", f"{name}: Best confusion matrix")

    # refit best on all data for feature importance
    pipe = build_pipe(model_best, k_best)
    search = GridSearchCV(
        pipe,
        grid_best,
        scoring=args.scoring,
        cv=inner_cv,
        n_jobs=args.n_jobs,
        refit=True,
    )
    search.fit(X, y)

    plot_feature_importance(
        search.best_estimator_,
        cols_best,
        dataset_dir / "best_feature_importance.png",
        f"{name}: top selected features",
    )

    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="/rds/projects/j/jouaitim-mri-test/fatima/outputs/phase1_ml_baselines")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--scoring", default="roc_auc", choices=["roc_auc", "balanced_accuracy", "average_precision"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[4, 6, 8, 10, 12, 16, 20, 30, 40, 50, 80, 120])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for name, path in CSV_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"[SKIP missing] {name}: {path}")
            continue

        try:
            res = run_one_csv(name, path, outdir, args)
            all_results.append(res)
        except Exception as e:
            print(f"[FAILED] {name}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values(
            ["balanced_accuracy", "roc_auc", "pr_auc"],
            ascending=False,
        )
        combined.to_csv(outdir / "ALL_DATASETS_LEADERBOARD.csv", index=False)

        print("\n\n===== OVERALL TOP 30 =====")
        print(combined.head(30)[[
            "dataset", "run_name", "roc_auc", "pr_auc",
            "balanced_accuracy", "accuracy", "precision",
            "recall", "specificity", "f1", "threshold"
        ]].to_string(index=False))

        plot_bar(combined, "balanced_accuracy", outdir / "OVERALL_top_balanced_accuracy.png", top_n=30)
        plot_bar(combined, "roc_auc", outdir / "OVERALL_top_roc_auc.png", top_n=30)
        plot_bar(combined, "pr_auc", outdir / "OVERALL_top_pr_auc.png", top_n=30)

    print(f"\n[DONE] Saved to: {outdir}")


if __name__ == "__main__":
    main()
