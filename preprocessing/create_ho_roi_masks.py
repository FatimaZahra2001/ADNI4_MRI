from pathlib import Path
import os
import re
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np


ROOT = Path("/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm")
FSLDIR = Path(os.environ.get("FSLDIR", "/rds/bear-apps/2022b/EL8-haswell/software/FSL/6.0.7.19/fsl"))


ROI_KEYWORDS = {
    "hippocampus_left": ["Left Hippocampus"],
    "hippocampus_right": ["Right Hippocampus"],
    "amygdala_left": ["Left Amygdala"],
    "amygdala_right": ["Right Amygdala"],

    "parahippocampal_left": ["Left Parahippocampal"],
    "parahippocampal_right": ["Right Parahippocampal"],

    "temporal_left": ["Left Temporal"],
    "temporal_right": ["Right Temporal"],
}


def find_xml(name_contains):
    atlas_dir = FSLDIR / "data" / "atlases"
    matches = list(atlas_dir.glob(f"*{name_contains}*.xml"))
    if not matches:
        raise FileNotFoundError(f"Could not find XML containing {name_contains} in {atlas_dir}")
    return matches[0]


def parse_labels(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []
    for lab in root.iter("label"):
        idx = int(lab.attrib["index"])
        text = (lab.text or "").strip()
        labels.append((idx, text))

    return labels


def matching_indices(labels, keywords):
    hits = []
    for idx, text in labels:
        for kw in keywords:
            if kw.lower() in text.lower():
                hits.append((idx, text))
    return hits


def make_mask(data, indices):
    mask = np.zeros(data.shape, dtype=bool)

    for idx, _name in indices:
        m1 = data == idx
        m2 = data == (idx + 1)

        if m2.sum() > m1.sum():
            mask |= m2
        else:
            mask |= m1

    return mask.astype(np.uint8)


def main():
    sub_xml = find_xml("Subcortical")
    cort_xml = find_xml("Cortical")

    print(f"Using subcortical XML: {sub_xml}")
    print(f"Using cortical XML:    {cort_xml}")

    sub_labels = parse_labels(sub_xml)
    cort_labels = parse_labels(cort_xml)

    roi_sources = {}

    for roi_name, kws in ROI_KEYWORDS.items():
        if roi_name.startswith(("hippocampus", "amygdala")):
            hits = matching_indices(sub_labels, kws)
            roi_sources[roi_name] = ("sub", hits)
        else:
            hits = matching_indices(cort_labels, kws)
            roi_sources[roi_name] = ("cort", hits)

        print(f"\n{roi_name}:")
        for idx, name in roi_sources[roi_name][1]:
            print(f"  index={idx} name={name}")

    subjects = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    print(f"\nSubjects: {len(subjects)}")

    made = 0
    skipped = 0

    for sdir in subjects:
        sub_path = sdir / "HO_sub_in_T1norm.nii.gz"
        cort_path = sdir / "HO_cort_in_T1norm.nii.gz"

        if not sub_path.exists() or not cort_path.exists():
            skipped += 1
            continue

        sub_img = nib.load(str(sub_path))
        cort_img = nib.load(str(cort_path))

        sub_data = sub_img.get_fdata()
        cort_data = cort_img.get_fdata()

        for roi_name, (source, hits) in roi_sources.items():
            if not hits:
                continue

            if source == "sub":
                mask = make_mask(sub_data, hits)
                ref_img = sub_img
            else:
                mask = make_mask(cort_data, hits)
                ref_img = cort_img

            out_path = sdir / f"{roi_name}.nii.gz"

            nib.save(
                nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header),
                str(out_path),
            )

            made += 1

    print(f"\nDone. Made masks: {made}. Skipped subjects: {skipped}")


if __name__ == "__main__":
    main()
