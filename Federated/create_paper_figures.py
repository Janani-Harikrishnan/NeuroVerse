import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Neuroverse/Federated/Full_results")

COMBINED = os.path.join(ROOT, "Combined_Comparison")
TABLES = os.path.join(COMBINED, "tables")
FINAL_DIR = os.path.join(COMBINED, "final_comparison")
CONV_DIR = os.path.join(COMBINED, "convergence")

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(CONV_DIR, exist_ok=True)

# ============================================================
# LOAD FINAL RESULTS
# ============================================================

csv_path = os.path.join(
    TABLES,
    "FINAL_FOUR_EXPERIMENT_COMPARISON.csv"
)

df = pd.read_csv(csv_path)

# Short names for figures
names = [
    "E1 - 3D U-Net + FedAvg",
    "E2 - Swin UNETR + FedAvg",
    "E3 - Swin UNETR + Adaptive FL",
    "E4 - Swin UNETR + Quantum Kernel",
]

short_names = [
    "E1\n3D U-Net + FedAvg",
    "E2\nSwin + FedAvg",
    "E3\nSwin + Adaptive FL",
    "E4\nSwin + Quantum Kernel",
]

# ============================================================
# 1. SINGLE IMAGE: ALL IMPORTANT FINAL METRICS
# ============================================================

metrics = [
    ("Mean Dice", "Mean Dice", 0, 1),
    ("Mean IoU", "Mean IoU", 0, 1),
    ("Precision", "Mean Precision", 0, 1),
    ("Sensitivity", "Mean Sensitivity", 0, 1),
    ("MCC", "MCC", 0, 1),
    ("ROC-AUC", "ROC-AUC", 0, 1),
    ("PR-AUC", "PR-AUC", 0, 1),
    ("HD95", "HD95", 0, None),
]

fig, axes = plt.subplots(
    2,
    4,
    figsize=(20, 10)
)

fig.suptitle(
    "Final Performance Comparison of Four Federated Learning Experiments",
    fontsize=18,
    fontweight="bold"
)

for ax, (title, column, ymin, ymax) in zip(
    axes.flat,
    metrics
):

    values = df[column].values

    x = np.arange(len(values))

    bars = ax.bar(
        x,
        values
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["E1", "E2", "E3", "E4"]
    )

    ax.set_ylabel(title)

    if ymax is not None:
        ax.set_ylim(ymin, ymax)

    for bar, value in zip(bars, values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.grid(
        axis="y",
        alpha=0.25
    )

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

output = os.path.join(
    FINAL_DIR,
    "FINAL_ALL_METRICS_2x4.png"
)

plt.savefig(
    output,
    dpi=400,
    bbox_inches="tight"
)

plt.close()

print("Created:")
print(output)


# ============================================================
# 2. SINGLE IMAGE: MOST IMPORTANT METRICS ONLY
# ============================================================

important = [
    ("Mean Dice", "Mean Dice"),
    ("Mean IoU", "Mean IoU"),
    ("Sensitivity", "Mean Sensitivity"),
    ("MCC", "MCC"),
    ("ROC-AUC", "ROC-AUC"),
    ("PR-AUC", "PR-AUC"),
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 9)
)

fig.suptitle(
    "Key Segmentation Performance Metrics",
    fontsize=18,
    fontweight="bold"
)

for ax, (title, column) in zip(
    axes.flat,
    important
):

    values = df[column].values

    x = np.arange(4)

    ax.bar(
        x,
        values
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["E1", "E2", "E3", "E4"]
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel(title)

    for i, value in enumerate(values):

        ax.text(
            i,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.grid(
        axis="y",
        alpha=0.25
    )

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

output = os.path.join(
    FINAL_DIR,
    "KEY_METRICS_2x3.png"
)

plt.savefig(
    output,
    dpi=400,
    bbox_inches="tight"
)

plt.close()

print("Created:")
print(output)


# ============================================================
# 3. REGION-WISE DICE IN ONE FIGURE
# ============================================================

region_csv = os.path.join(
    TABLES,
    "REGION_WISE_DICE.csv"
)

region_df = pd.read_csv(region_csv)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

x = np.arange(4)
width = 0.25

for i, region in enumerate(
    ["WT", "TC", "ET"]
):

    values = region_df[region].values

    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=region
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.set_xticks(x)

ax.set_xticklabels(
    ["E1", "E2", "E3", "E4"]
)

ax.set_ylabel(
    "Dice Score"
)

ax.set_ylim(
    0.75,
    0.95
)

ax.set_title(
    "Region-wise Dice Performance",
    fontsize=16,
    fontweight="bold"
)

ax.legend(
    title="Tumor Region"
)

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

output = os.path.join(
    FINAL_DIR,
    "REGION_WISE_DICE_SINGLE.png"
)

plt.savefig(
    output,
    dpi=400,
    bbox_inches="tight"
)

plt.close()

print("Created:")
print(output)


# ============================================================
# 4. E4 QUANTUM DIVERSITY CURVE
# ============================================================

E4 = os.path.join(
    ROOT,
    "E4_Swin_QuantumKernel"
)

qk_file = os.path.join(
    E4,
    "raw_results",
    "quantum_kernel_history.csv"
)

if os.path.exists(qk_file):

    qdf = pd.read_csv(qk_file)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for client in sorted(
        qdf["client"].unique()
    ):

        cdf = qdf[
            qdf["client"] == client
        ].sort_values("round")

        ax.plot(
            cdf["round"],
            cdf["quantum_diversity"],
            marker="o",
            linewidth=2,
            label=client
        )

    ax.set_xlabel(
        "Federated Round"
    )

    ax.set_ylabel(
        "Quantum Diversity"
    )

    ax.set_title(
        "Quantum Kernel Diversity Across Federated Rounds",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xticks(
        sorted(qdf["round"].unique())
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    output = os.path.join(
        CONV_DIR,
        "E4_quantum_diversity_curve.png"
    )

    plt.savefig(
        output,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    print("Created:")
    print(output)


# ============================================================
# 5. E4 CLIENT ADAPTIVE WEIGHTS CURVE
# ============================================================

adaptive_file = os.path.join(
    E4,
    "raw_results",
    "adaptive_weights.csv"
)

if os.path.exists(adaptive_file):

    adf = pd.read_csv(
        adaptive_file
    )

    print()
    print("Adaptive weights columns:")
    print(list(adf.columns))

    # Find columns automatically
    round_col = next(
        (
            c for c in adf.columns
            if c.lower() == "round"
        ),
        None
    )

    client_col = next(
        (
            c for c in adf.columns
            if c.lower() in
            ["client", "client_id", "node"]
        ),
        None
    )

    weight_col = next(
        (
            c for c in adf.columns
            if "weight" in c.lower()
            and "normalized" not in c.lower()
        ),
        None
    )

    if (
        round_col
        and client_col
        and weight_col
    ):

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for client in sorted(
            adf[client_col].unique()
        ):

            cdf = adf[
                adf[client_col] == client
            ].sort_values(
                round_col
            )

            ax.plot(
                cdf[round_col],
                cdf[weight_col],
                marker="o",
                linewidth=2,
                label=str(client)
            )

        ax.set_xlabel(
            "Federated Round"
        )

        ax.set_ylabel(
            "Aggregation Weight"
        )

        ax.set_title(
            "Adaptive Client Aggregation Weights",
            fontsize=16,
            fontweight="bold"
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

        plt.tight_layout()

        output = os.path.join(
            CONV_DIR,
            "E4_adaptive_weights_curve.png"
        )

        plt.savefig(
            output,
            dpi=400,
            bbox_inches="tight"
        )

        plt.close()

        print("Created:")
        print(output)


print()
print("=" * 70)
print("PAPER FIGURES CREATED")
print("=" * 70)
