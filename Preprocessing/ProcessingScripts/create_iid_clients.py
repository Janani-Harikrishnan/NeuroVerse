from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

# Script location:
# Neuroverse/Preprocessing/ProcessingScripts/create_iid_clients.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_POOL_FILE = (
    PROJECT_ROOT
    / "Global_Data"
    / "training_pool.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "IID_Clients"


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 3
RANDOM_SEED = 42


# ============================================================
# LOAD TRAINING POOL
# ============================================================

print("=" * 70)
print("IID FEDERATED CLIENT SPLIT")
print("=" * 70)

print(f"\nTraining pool:")
print(TRAINING_POOL_FILE)

df = pd.read_csv(TRAINING_POOL_FILE)

print(f"\nTotal training-pool patients: {len(df)}")


# ============================================================
# CHECK PATIENT IDS
# ============================================================

if df["patient_id"].duplicated().any():

    raise ValueError(
        "Duplicate patient IDs found in training pool."
    )

print("Duplicate patient IDs: 0")


# ============================================================
# VERIFY REQUIRED DATA
# ============================================================

if len(df) < NUM_CLIENTS:

    raise ValueError(
        "Number of patients is smaller than "
        "the number of clients."
    )


# ============================================================
# SHUFFLE PATIENTS
# ============================================================

df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SPLIT INTO 3 IID CLIENTS
# ============================================================

client_dfs = []

base_size = len(df) // NUM_CLIENTS
remainder = len(df) % NUM_CLIENTS

start = 0

for client_number in range(1, NUM_CLIENTS + 1):

    # Distribute remainder across first clients
    client_size = base_size

    if client_number <= remainder:
        client_size += 1

    end = start + client_size

    client_df = (
        df.iloc[start:end]
        .copy()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    client_dfs.append(client_df)

    start = end


# ============================================================
# CREATE CLIENT DIRECTORIES
# ============================================================

for client_number in range(1, NUM_CLIENTS + 1):

    client_dir = (
        OUTPUT_DIR
        / f"C{client_number}"
    )

    client_dir.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# SAVE CLIENT FILES
# ============================================================

print("\n" + "-" * 70)
print("CLIENT DISTRIBUTION")
print("-" * 70)

for client_number, client_df in enumerate(
    client_dfs,
    start=1
):

    output_file = (
        OUTPUT_DIR
        / f"C{client_number}"
        / f"client_{client_number}.csv"
    )

    client_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nClient {client_number}: "
        f"{len(client_df)} patients"
    )

    print(
        f"Saved to:\n{output_file}"
    )


# ============================================================
# VERIFY ALL PATIENTS ARE INCLUDED
# ============================================================

all_client_ids = []

for client_df in client_dfs:

    all_client_ids.extend(
        client_df["patient_id"].tolist()
    )

all_client_ids = set(all_client_ids)

original_ids = set(
    df["patient_id"]
)

print("\n" + "-" * 70)
print("VERIFICATION")
print("-" * 70)

print(
    f"Original training pool: "
    f"{len(original_ids)}"
)

print(
    f"Patients across clients: "
    f"{len(all_client_ids)}"
)

if len(all_client_ids) != len(original_ids):

    raise ValueError(
        "Patient count mismatch."
    )

if all_client_ids != original_ids:

    raise ValueError(
        "Some patients are missing or extra."
    )

print("✓ All training-pool patients accounted for.")


# ============================================================
# CHECK FOR CLIENT OVERLAP
# ============================================================

overlap_found = False

for i in range(NUM_CLIENTS):

    for j in range(i + 1, NUM_CLIENTS):

        ids_i = set(
            client_dfs[i]["patient_id"]
        )

        ids_j = set(
            client_dfs[j]["patient_id"]
        )

        overlap = ids_i.intersection(ids_j)

        if overlap:

            overlap_found = True

            print(
                f"ERROR: Client {i+1} and "
                f"Client {j+1} have "
                f"{len(overlap)} overlapping patients."
            )


if overlap_found:

    raise ValueError(
        "Patient overlap detected between clients."
    )

else:

    print("✓ No patient overlap between clients.")


# ============================================================
# VERIFY CLIENT SIZES
# ============================================================

print("\n" + "-" * 70)
print("CLIENT SIZES")
print("-" * 70)

for client_number, client_df in enumerate(
    client_dfs,
    start=1
):

    print(
        f"Client {client_number}: "
        f"{len(client_df)} patients"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("IID CLIENT SPLIT COMPLETE")
print("=" * 70)

print("\nDirectory structure:")

print(
    f"{OUTPUT_DIR}/C1/client_1.csv"
)

print(
    f"{OUTPUT_DIR}/C2/client_2.csv"
)

print(
    f"{OUTPUT_DIR}/C3/client_3.csv"
)
