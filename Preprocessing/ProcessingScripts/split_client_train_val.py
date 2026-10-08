from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

# Script:
# Neuroverse/Preprocessing/ProcessingScripts/
# split_client_train_val.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLIENTS_DIR = PROJECT_ROOT / "IID_Clients"


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 3

VALIDATION_SIZE = 0.20

RANDOM_SEED = 42


# ============================================================
# START
# ============================================================

print("=" * 70)
print("LOCAL CLIENT TRAIN / VALIDATION SPLIT")
print("=" * 70)

print(f"\nValidation ratio: {VALIDATION_SIZE * 100:.0f}%")
print(f"Training ratio: {(1 - VALIDATION_SIZE) * 100:.0f}%")
print(f"Random seed: {RANDOM_SEED}")


# ============================================================
# PROCESS EACH CLIENT
# ============================================================

all_train_ids = set()
all_val_ids = set()

for client_number in range(1, NUM_CLIENTS + 1):

    client_name = f"C{client_number}"

    client_dir = CLIENTS_DIR / client_name

    client_file = (
        client_dir
        / f"client_{client_number}.csv"
    )

    train_file = client_dir / "train.csv"
    val_file = client_dir / "val.csv"


    # --------------------------------------------------------
    # Check client file
    # --------------------------------------------------------

    if not client_file.exists():

        raise FileNotFoundError(
            f"Client file not found:\n{client_file}"
        )


    # --------------------------------------------------------
    # Load client
    # --------------------------------------------------------

    df = pd.read_csv(client_file)

    print("\n" + "-" * 70)
    print(f"{client_name}")
    print("-" * 70)

    print(
        f"Total patients: {len(df)}"
    )


    # --------------------------------------------------------
    # Check duplicates
    # --------------------------------------------------------

    duplicate_count = (
        df["patient_id"].duplicated().sum()
    )

    if duplicate_count > 0:

        raise ValueError(
            f"{client_name} contains "
            f"{duplicate_count} duplicate patient IDs."
        )

    print("Duplicate patients: 0")


    # --------------------------------------------------------
    # Split client
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )


    # --------------------------------------------------------
    # Sort for reproducibility/readability
    # --------------------------------------------------------

    train_df = (
        train_df
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Check overlap
    # --------------------------------------------------------

    train_ids = set(
        train_df["patient_id"]
    )

    val_ids = set(
        val_df["patient_id"]
    )

    overlap = train_ids.intersection(val_ids)

    if overlap:

        raise ValueError(
            f"DATA LEAKAGE in {client_name}: "
            f"{len(overlap)} patients appear "
            f"in both train and validation."
        )


    # --------------------------------------------------------
    # Check patient accounting
    # --------------------------------------------------------

    original_ids = set(
        df["patient_id"]
    )

    if train_ids.union(val_ids) != original_ids:

        raise ValueError(
            f"Patient accounting error in {client_name}."
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_df.to_csv(
        train_file,
        index=False
    )

    val_df.to_csv(
        val_file,
        index=False
    )


    # --------------------------------------------------------
    # Store IDs for final global verification
    # --------------------------------------------------------

    all_train_ids.update(train_ids)
    all_val_ids.update(val_ids)


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Train patients: {len(train_df)} "
        f"({len(train_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Validation patients: {len(val_df)} "
        f"({len(val_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Train + Validation: "
        f"{len(train_df) + len(val_df)}"
    )

    print(
        f"Saved train:\n{train_file}"
    )

    print(
        f"Saved validation:\n{val_file}"
    )

    print("✓ No train/validation overlap")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CLIENT SUMMARY")
print("=" * 70)

total_train = 0
total_val = 0

for client_number in range(1, NUM_CLIENTS + 1):

    client_dir = (
        CLIENTS_DIR / f"C{client_number}"
    )

    train_df = pd.read_csv(
        client_dir / "train.csv"
    )

    val_df = pd.read_csv(
        client_dir / "val.csv"
    )

    total_train += len(train_df)
    total_val += len(val_df)

    print(
        f"\nC{client_number}:"
    )

    print(
        f"  Train: {len(train_df)}"
    )

    print(
        f"  Validation: {len(val_df)}"
    )

    print(
        f"  Total: "
        f"{len(train_df) + len(val_df)}"
    )


# ============================================================
# CHECK CROSS-CLIENT TRAINING OVERLAP
# ============================================================

print("\n" + "-" * 70)
print("CROSS-CLIENT VERIFICATION")
print("-" * 70)

client_train_sets = []
client_val_sets = []

for client_number in range(1, NUM_CLIENTS + 1):

    client_dir = (
        CLIENTS_DIR / f"C{client_number}"
    )

    train_ids = set(
        pd.read_csv(
            client_dir / "train.csv"
        )["patient_id"]
    )

    val_ids = set(
        pd.read_csv(
            client_dir / "val.csv"
        )["patient_id"]
    )

    client_train_sets.append(train_ids)
    client_val_sets.append(val_ids)


# ------------------------------------------------------------
# Check train-train overlap
# ------------------------------------------------------------

train_overlap = set()

for i in range(NUM_CLIENTS):

    for j in range(i + 1, NUM_CLIENTS):

        overlap = (
            client_train_sets[i]
            .intersection(client_train_sets[j])
        )

        train_overlap.update(overlap)


if train_overlap:

    raise ValueError(
        "Patients are shared between client "
        "training sets."
    )

print(
    "✓ No patient overlap between client "
    "training sets."
)


# ------------------------------------------------------------
# Check validation-validation overlap
# ------------------------------------------------------------

val_overlap = set()

for i in range(NUM_CLIENTS):

    for j in range(i + 1, NUM_CLIENTS):

        overlap = (
            client_val_sets[i]
            .intersection(client_val_sets[j])
        )

        val_overlap.update(overlap)


if val_overlap:

    raise ValueError(
        "Patients are shared between client "
        "validation sets."
    )

print(
    "✓ No patient overlap between client "
    "validation sets."
)


# ------------------------------------------------------------
# Check train-validation cross-client overlap
# ------------------------------------------------------------

cross_overlap = set()

for train_set in client_train_sets:

    for val_set in client_val_sets:

        cross_overlap.update(
            train_set.intersection(val_set)
        )


if cross_overlap:

    raise ValueError(
        "DATA LEAKAGE: A patient appears in "
        "training of one client and validation "
        "of another client."
    )

print(
    "✓ No cross-client train/validation overlap."
)


# ============================================================
# FINAL COUNTS
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(
    f"\nTotal local training patients: "
    f"{total_train}"
)

print(
    f"Total local validation patients: "
    f"{total_val}"
)

print(
    f"Total patients across clients: "
    f"{total_train + total_val}"
)

print("\nOutput structure:")

print(
    "IID_Clients/"
)

print(
    "├── C1/"
)

print(
    "│   ├── client_1.csv"
)

print(
    "│   ├── train.csv"
)

print(
    "│   └── val.csv"
)

print(
    "├── C2/"
)

print(
    "│   ├── client_2.csv"
)

print(
    "│   ├── train.csv"
)

print(
    "│   └── val.csv"
)

print(
    "└── C3/"
)

print(
    "    ├── client_3.csv"
)

print(
    "    ├── train.csv"
)

print(
    "    └── val.csv"
)

print("\n✓ All patient-level leakage checks passed.")
