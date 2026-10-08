import tarfile
import gzip
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current file:
# Neuroverse/Preprocessing/ProcessingScripts/analyze_dataset.py
#
# parents[0] = ProcessingScripts
# parents[1] = Preprocessing
# parents[2] = Neuroverse

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TAR_FILE = PROJECT_ROOT / "BraTS2021_Training_Data.tar"

INDEX_FILE = (
    PROJECT_ROOT
    / "Preprocessing"
    / "ProcessedData"
    / "brats_patient_index.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Preprocessing"
    / "ProcessedData"
    / "brats_subject_analysis.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

if not TAR_FILE.exists():
    raise FileNotFoundError(
        f"Training TAR file not found:\n{TAR_FILE}"
    )

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f"Patient index not found:\n{INDEX_FILE}"
    )


# ============================================================
# LOAD PATIENT INDEX
# ============================================================

patients_df = pd.read_csv(INDEX_FILE)

patient_ids = patients_df["patient_id"].tolist()

print("=" * 70)
print("BraTS 2021 DATASET ANALYSIS")
print("=" * 70)

print(f"\nTotal patients in index: {len(patient_ids)}")
print(f"TAR file: {TAR_FILE}")
print(f"Output: {OUTPUT_FILE}")


# ============================================================
# FIND SEGMENTATION FILES
# ============================================================

print("\n" + "-" * 70)
print("Scanning TAR archive for segmentation files...")
print("-" * 70)

segmentation_files = {}

with tarfile.open(TAR_FILE, "r") as tar:

    for member in tar:

        name = member.name

        if not name.endswith("_seg.nii.gz"):
            continue

        # Example:
        # ./BraTS2021_00000/BraTS2021_00000_seg.nii.gz

        match = re.search(
            r"(BraTS2021_\d+)/.*_seg\.nii\.gz$",
            name
        )

        if match:
            patient_id = match.group(1)
            segmentation_files[patient_id] = name


print(
    f"Segmentation files found: "
    f"{len(segmentation_files)}"
)


# ============================================================
# CHECK FOR MISSING SEGMENTATIONS
# ============================================================

missing_seg = [
    patient_id
    for patient_id in patient_ids
    if patient_id not in segmentation_files
]

if missing_seg:

    print("\nWARNING!")
    print(
        f"Missing segmentation files: "
        f"{len(missing_seg)}"
    )

    print("First few missing patients:")

    for patient_id in missing_seg[:10]:
        print(patient_id)

else:

    print("All patients have segmentation files.")


# ============================================================
# ANALYZE EACH PATIENT
# ============================================================

results = []

print("\n" + "-" * 70)
print("Analyzing segmentation volumes...")
print("-" * 70)

with tarfile.open(TAR_FILE, "r") as tar:

    total = len(patient_ids)

    for i, patient_id in enumerate(patient_ids, start=1):

        # Print progress every 50 patients
        if i == 1 or i % 50 == 0 or i == total:

            print(
                f"Processing "
                f"{i}/{total} "
                f"({i / total * 100:.1f}%)"
            )

        seg_path = segmentation_files.get(patient_id)

        if seg_path is None:
            continue

        try:

            # ------------------------------------------------
            # Extract segmentation file from TAR to memory
            # ------------------------------------------------

            member = tar.getmember(seg_path)

            file_object = tar.extractfile(member)

            if file_object is None:
                print(
                    f"Could not read segmentation: "
                    f"{patient_id}"
                )
                continue

            compressed_data = file_object.read()

            # ------------------------------------------------
            # Decompress .nii.gz
            # ------------------------------------------------

            nii_data = gzip.decompress(
                compressed_data
            )

            # ------------------------------------------------
            # Load NIfTI from memory
            # ------------------------------------------------

            image = nib.Nifti1Image.from_bytes(
                nii_data
            )

            data = np.asarray(
                image.dataobj
            )

            header = image.header

            # ------------------------------------------------
            # IMAGE SHAPE
            # ------------------------------------------------

            shape = data.shape

            # ------------------------------------------------
            # VOXEL SPACING
            # ------------------------------------------------

            spacing = header.get_zooms()[:3]

            spacing_x = float(spacing[0])
            spacing_y = float(spacing[1])
            spacing_z = float(spacing[2])

            voxel_volume_mm3 = (
                spacing_x
                * spacing_y
                * spacing_z
            )

            # ------------------------------------------------
            # TOTAL VOXELS
            # ------------------------------------------------

            total_voxels = data.size

            # ------------------------------------------------
            # SEGMENTATION LABEL COUNTS
            # ------------------------------------------------

            label_0 = int(
                np.count_nonzero(data == 0)
            )

            label_1 = int(
                np.count_nonzero(data == 1)
            )

            label_2 = int(
                np.count_nonzero(data == 2)
            )

            label_4 = int(
                np.count_nonzero(data == 4)
            )

            # ------------------------------------------------
            # UNIQUE LABELS
            # ------------------------------------------------

            unique_labels = np.unique(data)

            unique_labels = [
                int(x)
                for x in unique_labels
            ]

            # ------------------------------------------------
            # UNEXPECTED LABELS
            # ------------------------------------------------

            expected_labels = {
                0, 1, 2, 4
            }

            unexpected_labels = [
                x
                for x in unique_labels
                if x not in expected_labels
            ]

            # ------------------------------------------------
            # WHOLE TUMOR
            #
            # WT = 1 + 2 + 4
            # ------------------------------------------------

            wt_voxels = (
                label_1
                + label_2
                + label_4
            )

            wt_volume_mm3 = (
                wt_voxels
                * voxel_volume_mm3
            )

            wt_volume_cm3 = (
                wt_volume_mm3 / 1000.0
            )

            # ------------------------------------------------
            # TUMOR CORE
            #
            # TC = 1 + 4
            # ------------------------------------------------

            tc_voxels = (
                label_1
                + label_4
            )

            tc_volume_mm3 = (
                tc_voxels
                * voxel_volume_mm3
            )

            tc_volume_cm3 = (
                tc_volume_mm3 / 1000.0
            )

            # ------------------------------------------------
            # ENHANCING TUMOR
            #
            # ET = 4
            # ------------------------------------------------

            et_voxels = label_4

            et_volume_mm3 = (
                et_voxels
                * voxel_volume_mm3
            )

            et_volume_cm3 = (
                et_volume_mm3 / 1000.0
            )

            # ------------------------------------------------
            # TUMOR PERCENTAGE
            # ------------------------------------------------

            tumor_percentage = (
                wt_voxels
                / total_voxels
                * 100
            )

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            results.append({

                "patient_id": patient_id,

                # Image information
                "shape_x": shape[0],
                "shape_y": shape[1],
                "shape_z": shape[2],

                "spacing_x_mm": spacing_x,
                "spacing_y_mm": spacing_y,
                "spacing_z_mm": spacing_z,

                "voxel_volume_mm3":
                    voxel_volume_mm3,

                "total_voxels":
                    total_voxels,

                # Raw labels
                "label_0_voxels":
                    label_0,

                "label_1_voxels":
                    label_1,

                "label_2_voxels":
                    label_2,

                "label_4_voxels":
                    label_4,

                # Tumor regions
                "WT_voxels":
                    wt_voxels,

                "TC_voxels":
                    tc_voxels,

                "ET_voxels":
                    et_voxels,

                # Tumor volumes
                "WT_volume_cm3":
                    wt_volume_cm3,

                "TC_volume_cm3":
                    tc_volume_cm3,

                "ET_volume_cm3":
                    et_volume_cm3,

                # Tumor percentage
                "tumor_percentage":
                    tumor_percentage,

                # Labels
                "unique_labels":
                    str(unique_labels),

                "unexpected_labels":
                    str(unexpected_labels)
            })

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{patient_id}: {e}"
            )


# ============================================================
# CREATE DATAFRAME
# ============================================================

analysis_df = pd.DataFrame(results)


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)

print(
    f"\nPatients successfully analyzed: "
    f"{len(analysis_df)}"
)

print(
    f"Patients missing from analysis: "
    f"{len(patient_ids) - len(analysis_df)}"
)


# ============================================================
# IMAGE SHAPE
# ============================================================

print("\n")
print("-" * 70)
print("IMAGE SHAPE DISTRIBUTION")
print("-" * 70)

shape_distribution = (
    analysis_df[
        [
            "shape_x",
            "shape_y",
            "shape_z"
        ]
    ]
    .value_counts()
)

print(shape_distribution)


# ============================================================
# VOXEL SPACING
# ============================================================

print("\n")
print("-" * 70)
print("VOXEL SPACING DISTRIBUTION")
print("-" * 70)

spacing_distribution = (
    analysis_df[
        [
            "spacing_x_mm",
            "spacing_y_mm",
            "spacing_z_mm"
        ]
    ]
    .round(4)
    .value_counts()
)

print(spacing_distribution)


# ============================================================
# WHOLE TUMOR STATISTICS
# ============================================================

print("\n")
print("-" * 70)
print("WHOLE TUMOR (WT) VOLUME — cm³")
print("-" * 70)

print(
    analysis_df[
        "WT_volume_cm3"
    ].describe()
)


# ============================================================
# TUMOR CORE STATISTICS
# ============================================================

print("\n")
print("-" * 70)
print("TUMOR CORE (TC) VOLUME — cm³")
print("-" * 70)

print(
    analysis_df[
        "TC_volume_cm3"
    ].describe()
)


# ============================================================
# ENHANCING TUMOR STATISTICS
# ============================================================

print("\n")
print("-" * 70)
print("ENHANCING TUMOR (ET) VOLUME — cm³")
print("-" * 70)

print(
    analysis_df[
        "ET_volume_cm3"
    ].describe()
)


# ============================================================
# TUMOR PERCENTAGE
# ============================================================

print("\n")
print("-" * 70)
print("TUMOR VOXEL PERCENTAGE")
print("-" * 70)

print(
    analysis_df[
        "tumor_percentage"
    ].describe()
)


# ============================================================
# TOTAL LABEL COUNTS
# ============================================================

print("\n")
print("-" * 70)
print("TOTAL SEGMENTATION LABEL COUNTS")
print("-" * 70)

print(
    "Label 1:",
    analysis_df[
        "label_1_voxels"
    ].sum()
)

print(
    "Label 2:",
    analysis_df[
        "label_2_voxels"
    ].sum()
)

print(
    "Label 4:",
    analysis_df[
        "label_4_voxels"
    ].sum()
)


# ============================================================
# CHECK UNEXPECTED LABELS
# ============================================================

print("\n")
print("-" * 70)
print("UNEXPECTED SEGMENTATION LABELS")
print("-" * 70)

unexpected = analysis_df[
    analysis_df[
        "unexpected_labels"
    ] != "[]"
]

if len(unexpected) == 0:

    print("None found.")

else:

    print(
        unexpected[
            [
                "patient_id",
                "unexpected_labels"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

analysis_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("DATASET ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nAnalysis file saved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nRows: {len(analysis_df)}"
)

print(
    f"Columns: {len(analysis_df.columns)}"
)
