from pathlib import Path

import pandas as pd
from scipy.stats import kruskal


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLIENT_DIR = PROJECT_ROOT / "IID_Clients"

CLIENT_FILES = {
    "C1": CLIENT_DIR / "C1" / "client_1.csv",
    "C2": CLIENT_DIR / "C2" / "client_2.csv",
    "C3": CLIENT_DIR / "C3" / "client_3.csv",
}


# ============================================================
# FEATURES TO COMPARE
# ============================================================

FEATURES = [
    "WT_volume_cm3",
    "TC_volume_cm3",
    "ET_volume_cm3",
    "tumor_percentage",
]


# ============================================================
# LOAD CLIENT DATA
# ============================================================

print("=" * 75)
print("IID CLIENT DISTRIBUTION VERIFICATION")
print("=" * 75)

clients = {}

for client_name, file_path in CLIENT_FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Client file not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    clients[client_name] = df

    print(
        f"\n{client_name}: "
        f"{len(df)} patients"
    )


# ============================================================
# CHECK PATIENT OVERLAP
# ============================================================

print("\n" + "-" * 75)
print("PATIENT OVERLAP CHECK")
print("-" * 75)

sets = {
    name: set(df["patient_id"])
    for name, df in clients.items()
}

for c1, c2 in [
    ("C1", "C2"),
    ("C1", "C3"),
    ("C2", "C3"),
]:

    overlap = sets[c1].intersection(sets[c2])

    print(
        f"{c1} ∩ {c2}: "
        f"{len(overlap)} patients"
    )

    if overlap:
        raise ValueError(
            f"Patient overlap detected between "
            f"{c1} and {c2}."
        )


print("\n✓ No patient overlap.")


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("DESCRIPTIVE STATISTICS")
print("=" * 75)

for feature in FEATURES:

    print("\n" + "-" * 75)
    print(feature)
    print("-" * 75)

    rows = []

    for client_name, df in clients.items():

        values = df[feature]

        rows.append({
            "Client": client_name,
            "N": len(values),
            "Mean": values.mean(),
            "Std": values.std(),
            "Median": values.median(),
            "Min": values.min(),
            "Q1": values.quantile(0.25),
            "Q3": values.quantile(0.75),
            "Max": values.max(),
        })

    stats_df = pd.DataFrame(rows)

    print(
        stats_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# KRUSKAL-WALLIS TEST
# ============================================================

print("\n" + "=" * 75)
print("KRUSKAL-WALLIS DISTRIBUTION TEST")
print("=" * 75)

print(
    "\nH0: The three clients come from the same distribution."
)

print(
    "H1: At least one client has a different distribution."
)

print(
    "\nInterpretation:"
)

print(
    "p >= 0.05 → no statistically significant difference detected"
)

print(
    "p < 0.05  → statistically significant difference detected"
)


test_results = []


for feature in FEATURES:

    c1_values = clients["C1"][feature]
    c2_values = clients["C2"][feature]
    c3_values = clients["C3"][feature]

    statistic, p_value = kruskal(
        c1_values,
        c2_values,
        c3_values
    )

    if p_value >= 0.05:
        interpretation = "Similar distribution"
    else:
        interpretation = "Significant difference"

    test_results.append({
        "Feature": feature,
        "H_statistic": statistic,
        "p_value": p_value,
        "Interpretation": interpretation,
    })


results_df = pd.DataFrame(test_results)

print(
    "\n" +
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("OVERALL IID ASSESSMENT")
print("=" * 75)

similar_features = sum(
    results_df["p_value"] >= 0.05
)

different_features = sum(
    results_df["p_value"] < 0.05
)

print(
    f"\nFeatures with no significant difference: "
    f"{similar_features}/{len(FEATURES)}"
)

print(
    f"Features with significant difference: "
    f"{different_features}/{len(FEATURES)}"
)


if different_features == 0:

    print(
        "\n✓ The three clients are statistically "
        "consistent with an IID distribution "
        "for the tested features."
    )

elif similar_features >= 3:

    print(
        "\n✓ The clients are largely similar, "
        "although some differences exist."
    )

else:

    print(
        "\n⚠ The clients show meaningful "
        "distribution differences."
    )


print("\n" + "=" * 75)
