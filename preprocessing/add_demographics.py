import pandas as pd

# =========================
# PATHS
# =========================
MANIFEST = "/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_dual_mri_manifest.csv"
DEMOG = "/rds/projects/j/jouaitim-mri-test/ADNI4/csvs/RMT_PTDEMOG_07Nov2025.csv"
OUT = "/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/adni_dual_mri_manifest_with_demog.csv"

# =========================
# LOAD
# =========================
print("Loading files...")
mri = pd.read_csv(MANIFEST)
demo = pd.read_csv(DEMOG)

# =========================
# CLEAN KEYS
# =========================
mri["PTID"] = mri["PTID"].astype(str).str.strip()
demo["PTID"] = demo["PTID"].astype(str).str.strip()

# remove bad PTIDs
demo = demo[demo["PTID"].notna()]
demo = demo[demo["PTID"] != "nan"]

# =========================
# KEEP BASELINE ONLY
# =========================
# If multiple rows per subject → keep earliest
if "RMT_Timepoint" in demo.columns:
    demo = demo.sort_values(["PTID", "RMT_Timepoint"])
else:
    demo = demo.sort_values(["PTID"])

demo_base = demo.drop_duplicates("PTID", keep="first")

# =========================
# SELECT FEATURES
# =========================
keep_cols = ["PTID", "Age_Baseline", "Gender"]

# optional
if "RMT_Education" in demo.columns:
    keep_cols.append("RMT_Education")

demo_base = demo_base[keep_cols]

# =========================
# MERGE
# =========================
merged = mri.merge(demo_base, on="PTID", how="left")

# =========================
# DEBUG INFO
# =========================
print("\n===== DEBUG =====")
print("MRI rows:", len(mri))
print("Merged rows:", len(merged))

print("\nMissing values:")
print("Age missing:", merged["Age_Baseline"].isna().sum())
print("Gender missing:", merged["Gender"].isna().sum())

print("\nGender distribution:")
print(merged["Gender"].value_counts(dropna=False))

print("\nExample rows:")
print(merged.head())

# =========================
# SAVE
# =========================
merged.to_csv(OUT, index=False)

print("\nSaved to:", OUT)
print("===== DONE =====")
