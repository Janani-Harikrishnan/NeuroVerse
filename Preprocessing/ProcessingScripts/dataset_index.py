import tarfile
import re
import pandas as pd
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

TAR_FILE = "BraTS2021_Training_Data.tar"
OUTPUT_FILE = "brats_patient_index.csv"


# ============================================================
# FIND ALL PATIENTS AND THEIR FILES
# ============================================================

patients = defaultdict(dict)

print("=" * 60)
print("BraTS 2021 Dataset Indexing")
print("=" * 60)

print(f"\nReading: {TAR_FILE}")
print("This may take a little time...\n")


with tarfile.open(TAR_FILE, "r") as tar:

    for member in tar:

        name = member.name

        # Example:
        # ./BraTS2021_00000/BraTS2021_00000_flair.nii.gz

        match = re.search(r"BraTS2021_(\d+)/", name)

        if not match:
            continue

        patient_id = f"BraTS2021_{match.group(1)}"

        # Initialize patient
        if patient_id not in patients:
            patients[patient_id] = {
                "patient_id": patient_id,
                "t1": None,
                "t1ce": None,
                "t2": None,
                "flair": None,
                "seg": None
            }

        # Store file path inside TAR
        if name.endswith("_t1.nii.gz"):
            patients[patient_id]["t1"] = name

        elif name.endswith("_t1ce.nii.gz"):
            patients[patient_id]["t1ce"] = name

        elif name.endswith("_t2.nii.gz"):
            patients[patient_id]["t2"] = name

        elif name.endswith("_flair.nii.gz"):
            patients[patient_id]["flair"] = name

        elif name.endswith("_seg.nii.gz"):
            patients[patient_id]["seg"] = name


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(list(patients.values()))

# Sort by patient ID
df = df.sort_values("patient_id").reset_index(drop=True)


# ============================================================
# BASIC DATASET ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"\nTotal patients found: {len(df)}")


# ------------------------------------------------------------
# Check missing files
# ------------------------------------------------------------

print("\nMissing files:")
print("-" * 40)

missing = df[
    ["t1", "t1ce", "t2", "flair", "seg"]
].isna().sum()

print(missing)


# ------------------------------------------------------------
# Complete patients
# ------------------------------------------------------------

required_columns = ["t1", "t1ce", "t2", "flair", "seg"]

complete_mask = df[required_columns].notna().all(axis=1)

complete_patients = df[complete_mask]

incomplete_patients = df[~complete_mask]

print("\nComplete patients:", len(complete_patients))
print("Incomplete patients:", len(incomplete_patients))


# ------------------------------------------------------------
# Show incomplete patients
# ------------------------------------------------------------

if len(incomplete_patients) > 0:

    print("\nIncomplete patient details:")
    print("-" * 40)

    print(
        incomplete_patients[
            ["patient_id", "t1", "t1ce", "t2", "flair", "seg"]
        ].to_string(index=False)
    )


# ============================================================
# DISPLAY FIRST FEW PATIENTS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 PATIENTS")
print("=" * 60)

print(
    df.head(5).to_string(index=False)
)


# ============================================================
# DISPLAY LAST FEW PATIENTS
# ============================================================

print("\n" + "=" * 60)
print("LAST 5 PATIENTS")
print("=" * 60)

print(
    df.tail(5).to_string(index=False)
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("INDEX CREATED")
print("=" * 60)

print(f"\nSaved to: {OUTPUT_FILE}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
