#!/bin/bash
#SBATCH --job-name=make_clean_csvs
#SBATCH --qos=bbdefault
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/make_clean_csvs_%j.out
#SBATCH --error=logs/make_clean_csvs_%j.err

set -euo pipefail

echo "===== START CLEAN MRI CSV CREATION ====="
date
hostname

module purge
module load bear-apps/2024a
module load Miniforge3/25.3.0-3

source activate /rds/projects/j/jouaitim-mri-test/envs/adni4_gpu

mkdir -p logs

python << 'EOF'
import pandas as pd
from pathlib import Path

# ============================================================
# INPUTS
# ============================================================

FEATURE_CSV = Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/t1_radiomics_normative/all_mci/radiomics_raw_features.csv")

# Change this if your original clinical CSV is elsewhere
CLINICAL_CSV = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/CDR_07Nov2025.csv")

OUTDIR = Path("/rds/projects/j/jouaitim-mri-test/fatima/outputs/clean_mri_classification_csvs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

print(f"[LOAD FEATURES] {FEATURE_CSV}")
feat = pd.read_csv(FEATURE_CSV)

print(f"[FEATURES SHAPE] {feat.shape}")
print(f"[FEATURE COLUMNS FIRST 20] {feat.columns[:20].tolist()}")

if not CLINICAL_CSV.exists():
    raise FileNotFoundError(f"Clinical CSV not found: {CLINICAL_CSV}")

print(f"[LOAD CLINICAL] {CLINICAL_CSV}")
clin = pd.read_csv(CLINICAL_CSV)

print(f"[CLINICAL SHAPE] {clin.shape}")
print(f"[CLINICAL COLUMNS FIRST 30] {clin.columns[:30].tolist()}")

# ============================================================
# FIND ID + CDR COLUMN
# ============================================================

id_candidates = ["PTID", "ptid", "Subject", "subject", "RID"]
cdr_candidates = ["CDGLOBAL", "CDGLOBAL_STR", "CDR_GLOBAL", "CDR", "CDR_GLOBAL_SCORE"]

feat_id = next((c for c in id_candidates if c in feat.columns), None)
clin_id = next((c for c in id_candidates if c in clin.columns), None)
cdr_col = next((c for c in cdr_candidates if c in clin.columns), None)

if feat_id is None:
    raise ValueError(f"No subject ID column found in feature CSV. Tried: {id_candidates}")

if clin_id is None:
    raise ValueError(f"No subject ID column found in clinical CSV. Tried: {id_candidates}")

if cdr_col is None:
    raise ValueError(f"No CDR/global column found in clinical CSV. Tried: {cdr_candidates}")

print(f"[FEATURE ID COL] {feat_id}")
print(f"[CLINICAL ID COL] {clin_id}")
print(f"[CDR COL] {cdr_col}")

# ============================================================
# CLEAN CLINICAL LABELS
# ============================================================

clin_small = clin[[clin_id, cdr_col]].copy()
clin_small = clin_small.rename(columns={clin_id: "PTID", cdr_col: "CDGLOBAL"})

clin_small["CDGLOBAL"] = pd.to_numeric(clin_small["CDGLOBAL"], errors="coerce")

clin_small = clin_small.dropna(subset=["PTID", "CDGLOBAL"])
clin_small["PTID"] = clin_small["PTID"].astype(str)

# If multiple rows per PTID, keep first available.
# If you have visit-date alignment later, replace this with scan-date matched label.
clin_small = clin_small.sort_values(["PTID"]).drop_duplicates("PTID", keep="first")

print("[CDGLOBAL COUNTS IN CLINICAL]")
print(clin_small["CDGLOBAL"].value_counts(dropna=False).sort_index())

# ============================================================
# MERGE FEATURES + CLEAN LABEL
# ============================================================

feat = feat.copy()
feat["PTID"] = feat[feat_id].astype(str)

# Drop old label if present so we rebuild labels only from CDGLOBAL
old_label_cols = [
    "label", "y", "diagnosis_bin", "DX", "diagnosis", "CDGLOBAL", "CDGLOBAL_STR",
    "MMSE", "MMSE_USE", "CDRSB", "CDRSB_USE"
]

to_drop = [c for c in old_label_cols if c in feat.columns and c != "PTID"]
print(f"[DROP POSSIBLE LEAKAGE COLS FROM FEATURES] {to_drop}")
feat = feat.drop(columns=to_drop, errors="ignore")

df = feat.merge(clin_small, on="PTID", how="inner")

print(f"[MERGED SHAPE] {df.shape}")
print("[MERGED CDGLOBAL COUNTS]")
print(df["CDGLOBAL"].value_counts(dropna=False).sort_index())

# ============================================================
# SAVE CLEAN TASK CSVS
# ============================================================

def save_task(df_in, keep_values, label_map, name):
    sub = df_in[df_in["CDGLOBAL"].isin(keep_values)].copy()
    sub["label"] = sub["CDGLOBAL"].map(label_map).astype(int)

    # CDGLOBAL used to create label only; remove before modelling
    sub = sub.drop(columns=["CDGLOBAL"])

    # Put PTID + label first
    cols = ["PTID", "label"] + [c for c in sub.columns if c not in ["PTID", "label"]]
    sub = sub[cols]

    out = OUTDIR / f"{name}.csv"
    sub.to_csv(out, index=False)

    print("\n" + "="*80)
    print(f"[SAVED] {out}")
    print(f"[SHAPE] {sub.shape}")
    print("[LABEL COUNTS]")
    print(sub["label"].value_counts().sort_index().to_dict())
    print("[FIRST 15 COLS]")
    print(sub.columns[:15].tolist())

    return out

# Main thesis task: HC vs MCI
save_task(
    df,
    keep_values=[0, 0.5],
    label_map={0: 0, 0.5: 1},
    name="clean_hc_vs_mci"
)

# Sanity/pipeline validation: HC vs AD/dementia-level impairment
save_task(
    df,
    keep_values=[0, 1],
    label_map={0: 0, 1: 1},
    name="clean_hc_vs_ad"
)

# Three-class exploratory task: HC vs MCI vs AD
sub = df[df["CDGLOBAL"].isin([0, 0.5, 1])].copy()
sub["label"] = sub["CDGLOBAL"].map({0: 0, 0.5: 1, 1: 2}).astype(int)
sub = sub.drop(columns=["CDGLOBAL"])
cols = ["PTID", "label"] + [c for c in sub.columns if c not in ["PTID", "label"]]
sub = sub[cols]

out = OUTDIR / "clean_hc_vs_mci_vs_ad.csv"
sub.to_csv(out, index=False)

print("\n" + "="*80)
print(f"[SAVED] {out}")
print(f"[SHAPE] {sub.shape}")
print("[LABEL COUNTS]")
print(sub["label"].value_counts().sort_index().to_dict())
print("[FIRST 15 COLS]")
print(sub.columns[:15].tolist())

# ============================================================
# SAFETY CHECK
# ============================================================

for f in OUTDIR.glob("clean_*.csv"):
    check = pd.read_csv(f)
    forbidden = [c for c in check.columns if any(x in c.lower() for x in [
        "cdglobal", "cdrsb", "mmse", "diagnosis", "dx"
    ])]

    print("\n" + "-"*80)
    print(f"[CHECK] {f.name}")
    print("shape:", check.shape)
    print("label counts:", check["label"].value_counts().sort_index().to_dict())
    print("forbidden/leakage-like columns:", forbidden)

print("\n[DONE]")
print(f"Clean CSVs saved to: {OUTDIR}")

EOF

echo "===== END CLEAN MRI CSV CREATION ====="
date
