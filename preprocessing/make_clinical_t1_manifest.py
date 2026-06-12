from pathlib import Path
import pandas as pd
import numpy as np

CODE = Path("/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI")
ROOT = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm")

BASE_MANIFEST = CODE / "adni_dual_mri_manifest_with_demog.csv"
MMSE_CSV = CODE / "MMSE_07Nov2025.csv"
CDR_CSV = CODE / "CDR_07Nov2025.csv"
DEMO_CSV = CODE / "RMT_PTDEMOG_07Nov2025.csv"

OUT = CODE / "adni_t1_clinical_manifest.csv"


def pick_baseline(df, score_cols):
    df = df.copy()
    df["PTID"] = df["PTID"].astype(str).str.strip()

    visit_col = None
    for c in ["VISCODE2", "VISCODE", "VISCODE_x"]:
        if c in df.columns:
            visit_col = c
            break

    if visit_col:
        df["_visit_rank"] = df[visit_col].astype(str).str.lower().map({
            "bl": 0, "sc": 1, "m00": 2
        }).fillna(99)
        df = df.sort_values(["PTID", "_visit_rank"])
    else:
        df["_visit_rank"] = 0

    keep = ["PTID"] + [c for c in score_cols if c in df.columns]
    return df[keep + ["_visit_rank"]].drop_duplicates("PTID", keep="first").drop(columns=["_visit_rank"])


def main():
    base = pd.read_csv(BASE_MANIFEST)
    base["PTID"] = base["PTID"].astype(str).str.strip()
    base["label"] = base["label"].astype(int)

    # Keep only subjects with T1.
    base["t1_path"] = base["PTID"].apply(lambda x: str(ROOT / x / "T1_norm.nii.gz"))
    base["has_t1"] = base["PTID"].apply(lambda x: (ROOT / x / "T1_norm.nii.gz").exists())
    base = base[base["has_t1"]].copy()

    mmse = pd.read_csv(MMSE_CSV)
    cdr = pd.read_csv(CDR_CSV)

    mmse_col = "MMSCORE" if "MMSCORE" in mmse.columns else "MMSEASON"
    cdr_cols = [c for c in ["CDGLOBAL", "CDRSB", "CDMEMORY"] if c in cdr.columns]

    mmse_b = pick_baseline(mmse, [mmse_col]).rename(columns={mmse_col: "MMSCORE"})
    cdr_b = pick_baseline(cdr, cdr_cols)

    df = base.merge(mmse_b, on="PTID", how="left")
    df = df.merge(cdr_b, on="PTID", how="left")

    for c in ["MMSCORE", "CDGLOBAL", "CDRSB", "CDMEMORY"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clinical subgroups, using only clinical variables, not MRI features.
    is_cn = df["label"] == 0
    is_mci = df["label"] == 1

    df["target_all_mci"] = df["label"]

    df["clinical_severe_mci"] = is_cn | (
        is_mci & (
            (df["MMSCORE"] <= 26) |
            (df["CDRSB"] >= 1.5) |
            (df.get("CDGLOBAL", np.nan) >= 0.5)
        )
    )

    df["clinical_strict_mci"] = is_cn | (
        is_mci & (
            (df["MMSCORE"] <= 26) &
            (df["CDRSB"] >= 1.0)
        )
    )

    df["clinical_very_strict_mci"] = is_cn | (
        is_mci & (
            (df["MMSCORE"] <= 25) &
            (df["CDRSB"] >= 1.5)
        )
    )

    df["mci_severity_group"] = "CN"
    df.loc[is_mci & (df["MMSCORE"] > 26) & (df["CDRSB"] < 1.5), "mci_severity_group"] = "mild_or_unclear_MCI"
    df.loc[is_mci & df["clinical_severe_mci"], "mci_severity_group"] = "clinical_severe_MCI"
    df.loc[is_mci & df["clinical_strict_mci"], "mci_severity_group"] = "clinical_strict_MCI"
    df.loc[is_mci & df["clinical_very_strict_mci"], "mci_severity_group"] = "clinical_very_strict_MCI"

    print("Saved:", OUT)
    print("Total usable T1:", len(df))
    print("Labels:", df["label"].value_counts().to_dict())
    print("MMSE missing:", df["MMSCORE"].isna().sum())
    print("CDRSB missing:", df["CDRSB"].isna().sum())
    print("Severe subset counts:", df[df["clinical_severe_mci"]]["label"].value_counts().to_dict())
    print("Strict subset counts:", df[df["clinical_strict_mci"]]["label"].value_counts().to_dict())
    print("Very strict subset counts:", df[df["clinical_very_strict_mci"]]["label"].value_counts().to_dict())

    df.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
