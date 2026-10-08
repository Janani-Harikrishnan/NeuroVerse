from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

# Current file:
# Neuroverse/Preprocessing/ProcessingScripts/create_train_test_split.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Preprocessing"
    / "ProcessedData"
    / "brats_subject_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Preprocessing"
    / "ProcessedData"
)

TRAINING_POOL_FILE = (
    OUTPUT_DIR
    / "training_pool.csv"
)

GLOBAL_TEST_FILE = (
    OUTPUT_DIR
    / "global_test.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.15

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GLOBAL TRAIN / TEST SUBJECT SPLIT")
print("=" * 70)

print(f"\nInput file:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal subjects: {len(df)}")


# ============================================================
# CHECK FOR DUPLICATES
# ============================================================

duplicates = df["patient_id"].duplicated().sum()

if duplicates > 0:

    raise ValueError(
        f"Found {duplicates} duplicate patient IDs."
    )

print("Duplicate patient IDs: 0")


# ============================================================
# SPLIT SUBJECTS
# ============================================================

training_pool, global_test = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True
)


# ============================================================
# SORT PATIENTS
# ============================================================

training_pool = (
    training_pool
    .sort_values("patient_id")
    .reset_index(drop=True)
)

global_test = (
    global_test
    .sort_values("patient_id")
    .reset_index(drop=True)
)


# ============================================================
# CHECK FOR OVERLAP
# ============================================================

training_ids = set(
    training_pool["patient_id"]
)

test_ids = set(
    global_test["patient_id"]
)

overlap = training_ids.intersection(test_ids)

if len(overlap) > 0:

    raise ValueError(
        f"DATA LEAKAGE DETECTED: "
        f"{len(overlap)} patients appear in both sets."
    )

print("\nPatient overlap: 0")


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

training_pool.to_csv(
    TRAINING_POOL_FILE,
    index=False
)

global_test.to_csv(
    GLOBAL_TEST_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(
    f"\nTotal subjects:       {len(df)}"
)

print(
    f"Training pool:        {len(training_pool)} "
    f"({len(training_pool) / len(df) * 100:.2f}%)"
)

print(
    f"Global test:          {len(global_test)} "
    f"({len(global_test) / len(df) * 100:.2f}%)"
)

print("\nRandom seed:", RANDOM_SEED)

print("\nFiles created:")

print(
    f"Training pool:\n{TRAINING_POOL_FILE}"
)

print(
    f"\nGlobal test:\n{GLOBAL_TEST_FILE}"
)


# ============================================================
# VERIFY PATIENT COUNTS
# ============================================================

print("\n" + "-" * 70)
print("VERIFICATION")
print("-" * 70)

print(
    "Training + Test =",
    len(training_pool) + len(global_test)
)

print(
    "Original dataset =",
    len(df)
)

if (
    len(training_pool) + len(global_test)
    == len(df)
):
    print("\n✓ All subjects accounted for.")

else:
    print("\n✗ ERROR: Subject count mismatch!")


if len(overlap) == 0:
    print("✓ No patient appears in both sets.")

print("\n" + "=" * 70)
