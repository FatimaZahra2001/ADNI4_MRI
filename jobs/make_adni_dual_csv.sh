#!/bin/bash
set -euo pipefail

module purge
module load bear-apps/2024a
module load Miniforge3/25.3.0-3
source activate /rds/projects/j/jouaitim-mri-test/envs/adni4_gpu

SRC_CSV="/rds/projects/j/jouaitim-mri-test/fatima/code/DINO/preprocessing/adni4_subject_splits_mni.csv"
MRI_ROOT="/rds/projects/j/jouaitim-mri-test/UNIFIED_PREPROC_MNI/adni"
OUT_CSV="/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_dual_mri_manifest.csv"

python - <<'PY'
import pandas as pd
from pathlib import Path

src_csv = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/DINO/preprocessing/adni4_subject_splits_mni.csv")
mri_root = Path("/rds/projects/j/jouaitim-mri-test/UNIFIED_PREPROC_MNI/adni")
out_csv = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_dual_mri_manifest.csv")

def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

if src_csv.exists():
    print(f"[INFO] Using source CSV: {src_csv}")
    df = pd.read_csv(src_csv)
    df.columns = [str(c).strip() for c in df.columns]
    print("[INFO] Source columns:", df.columns.tolist())

    ptid_col = find_col(df.columns, ["PTID", "ptid"])
    split_col = find_col(df.columns, ["split", "split20", "Split", "fold_split"])
    label_col = find_col(df.columns, ["label", "diagnosis", "dx", "target"])

    if ptid_col is None:
        raise ValueError("Could not find PTID column in source CSV")

    if split_col is None:
        print("[WARN] No split column found. Will create 70/15/15 split.")
    if label_col is None:
        print("[WARN] No label column found. Setting all labels to 0 for now.")

    out = pd.DataFrame()
    out["PTID"] = df[ptid_col].astype(str).str.strip()

    if label_col is not None:
        out["label"] = df[label_col]
    else:
        out["label"] = 0

    out["T1_MNI_path"] = out["PTID"].apply(lambda x: str(mri_root / x / "T1_MNI.nii.gz"))
    out["FLAIR_MNI_path"] = out["PTID"].apply(lambda x: str(mri_root / x / "FLAIR_MNI.nii.gz"))

    out["t1_exists"] = out["T1_MNI_path"].apply(lambda p: Path(p).is_file())
    out["flair_exists"] = out["FLAIR_MNI_path"].apply(lambda p: Path(p).is_file())
    out = out[out["t1_exists"] & out["flair_exists"]].copy()

    if split_col is not None:
        out["split"] = df.loc[out.index, split_col].astype(str).str.strip().str.lower()
        split_map = {
            "train": "train",
            "training": "train",
            "tr": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "dev": "val",
            "test": "test",
            "te": "test",
        }
        out["split"] = out["split"].map(lambda x: split_map.get(x, x))
        valid = out["split"].isin(["train", "val", "test"])
        if not valid.all():
            print("[WARN] Some split values were not train/val/test. Rebuilding split for those rows.")
            bad = out.loc[~valid].copy()
            good = out.loc[valid].copy()

            bad = bad.sample(frac=1, random_state=42).reset_index(drop=True)
            n = len(bad)
            train_end = int(0.70 * n)
            val_end = int(0.85 * n)
            bad["split"] = "train"
            bad.loc[train_end:val_end-1, "split"] = "val"
            bad.loc[val_end:, "split"] = "test"

            out = pd.concat([good, bad], ignore_index=True)
    else:
        out = out.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(out)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        out["split"] = "train"
        out.loc[train_end:val_end-1, "split"] = "val"
        out.loc[val_end:, "split"] = "test"

else:
    print(f"[WARN] Source CSV not found: {src_csv}")
    print("[WARN] Building manifest directly from MRI folders and setting all labels to 0.")
    rows = []
    for ptid_dir in sorted(mri_root.iterdir()):
        if not ptid_dir.is_dir():
            continue
        t1 = ptid_dir / "T1_MNI.nii.gz"
        flair = ptid_dir / "FLAIR_MNI.nii.gz"
        if t1.is_file() and flair.is_file():
            rows.append({
                "PTID": ptid_dir.name,
                "label": 0,
                "T1_MNI_path": str(t1),
                "FLAIR_MNI_path": str(flair),
            })

    out = pd.DataFrame(rows)
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(out)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    out["split"] = "train"
    out.loc[train_end:val_end-1, "split"] = "val"
    out.loc[val_end:, "split"] = "test"

# keep only required columns in correct order
out = out[["PTID", "split", "label", "T1_MNI_path", "FLAIR_MNI_path"]].copy()

out_csv.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_csv, index=False)

print(f"[INFO] Saved CSV to: {out_csv}")
print("[INFO] Rows:", len(out))
print("[INFO] Split counts:")
print(out["split"].value_counts(dropna=False).to_string())

print("[INFO] First 5 rows:")
print(out.head().to_string(index=False))
PY

echo
echo "Done."
echo "CSV created at:"
echo "$OUT_CSV"
