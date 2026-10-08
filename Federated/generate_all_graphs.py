import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

ROOT = os.path.expanduser("~/Neuroverse/Federated/Full_results")

E1 = os.path.join(ROOT, "E1_3D_UNet_FedAvg")
E2 = os.path.join(ROOT, "E2_Swin_FedAvg")
E3 = os.path.join(ROOT, "E3_Swin_Adaptive")
E4 = os.path.join(ROOT, "E4_Swin_QuantumKernel")

COMBINED = os.path.join(ROOT, "Combined_Comparison")

for path in [
    os.path.join(E1, "graphs"),
    os.path.join(E2, "graphs"),
    os.path.join(E3, "graphs"),
    os.path.join(E4, "graphs"),
    os.path.join(COMBINED, "convergence"),
    os.path.join(COMBINED, "final_comparison"),
    os.path.join(COMBINED, "region_wise"),
    os.path.join(COMBINED, "radar"),
]:
    os.makedirs(path, exist_ok=True)


# ============================================================
# HELPER
# ============================================================

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 1. LOAD FINAL METRICS
# ============================================================

final_csv = os.path.join(
    COMBINED,
    "tables",
    "FINAL_FOUR_EXPERIMENT_COMPARISON.csv"
)

df = pd.read_csv(final_csv)

short_names = [
    "E1\n3D U-Net + FedAvg",
    "E2\nSwin + FedAvg",
    "E3\nSwin + Adaptive FL",
    "E4\nSwin + Quantum Kernel"
]


# ============================================================
# 2. INDIVIDUAL EXPERIMENT FINAL-METRIC GRAPHS
# ============================================================

metric_info = [
    ("Mean Dice", "mean_dice", "Mean Dice"),
    ("Mean IoU", "mean_iou", "Mean IoU"),
    ("Mean Precision", "precision", "Mean Precision"),
    ("Mean Sensitivity", "sensitivity", "Mean Sensitivity"),
    ("MCC", "mcc", "Mean MCC"),
    ("ROC-AUC", "roc_auc", "Mean ROC-AUC"),
    ("PR-AUC", "pr_auc", "Mean PR-AUC"),
    ("HD95", "hd95", "Mean HD95"),
]

for exp_path, exp_name in [
    (E1, "E1"),
    (E2, "E2"),
    (E3, "E3"),
    (E4, "E4"),
]:
    row = df[df["Experiment"].str.startswith(exp_name)].iloc[0]

    graph_dir = os.path.join(exp_path, "graphs")

    for column, filename, title in metric_info:

        value = (
            row[column]
            if column in df.columns
            else row[
                {
                    "Mean Dice": "Mean Dice",
                    "Mean IoU": "Mean IoU",
                    "Mean Precision": "Mean Precision",
                    "Mean Sensitivity": "Mean Sensitivity",
                    "MCC": "MCC",
                    "ROC-AUC": "ROC-AUC",
                    "PR-AUC": "PR-AUC",
                    "HD95": "HD95",
                }[column]
            ]
        )

        plt.figure(figsize=(6, 5))
        plt.bar([exp_name], [value])
        plt.ylabel(title)
        plt.title(f"{exp_name} — Final {title}")
        plt.ylim(0, max(1.0, value * 1.15))

        if column == "HD95":
            plt.ylim(0, value * 1.25)

        plt.text(
            0,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

        savefig(os.path.join(graph_dir, f"{filename}.png"))


# ============================================================
# 3. FOUR-EXPERIMENT FINAL COMPARISONS
# ============================================================

comparison_metrics = [
    ("Mean Dice", "final_mean_dice.png"),
    ("Mean IoU", "final_mean_iou.png"),
    ("Mean Precision", "final_precision.png"),
    ("Mean Sensitivity", "final_sensitivity.png"),
    ("MCC", "final_mcc.png"),
    ("ROC-AUC", "final_roc_auc.png"),
    ("PR-AUC", "final_pr_auc.png"),
    ("HD95", "final_hd95.png"),
]

for column, filename in comparison_metrics:

    plt.figure(figsize=(10, 6))

    values = df[column].values

    bars = plt.bar(short_names, values)

    plt.ylabel(column)
    plt.title(f"Final {column} Comparison")

    if column != "HD95":
        plt.ylim(0, 1.05)
    else:
        plt.ylim(0, max(values) * 1.15)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    savefig(
        os.path.join(
            COMBINED,
            "final_comparison",
            filename,
        )
    )


# ============================================================
# 4. 2x2 PAPER COMPARISON FIGURE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

four_metrics = [
    ("Mean Dice", axes[0, 0]),
    ("Mean IoU", axes[0, 1]),
    ("ROC-AUC", axes[1, 0]),
    ("HD95", axes[1, 1]),
]

for metric, ax in four_metrics:

    values = df[metric].values

    bars = ax.bar(short_names, values)

    ax.set_title(metric)
    ax.set_ylabel(metric)

    if metric != "HD95":
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(0, max(values) * 1.15)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

fig.suptitle(
    "Final Performance Comparison of Four Federated Learning Experiments",
    fontsize=16,
)

savefig(
    os.path.join(
        COMBINED,
        "final_comparison",
        "09_FINAL_2x2_COMPARISON.png",
    )
)


# ============================================================
# 5. REGION-WISE DICE
# ============================================================

region_csv = os.path.join(
    COMBINED,
    "tables",
    "REGION_WISE_DICE.csv"
)

region_df = pd.read_csv(region_csv)

plt.figure(figsize=(11, 6))

x = np.arange(len(region_df))
width = 0.25

for i, region in enumerate(["WT", "TC", "ET"]):
    plt.bar(
        x + (i - 1) * width,
        region_df[region],
        width,
        label=region,
    )

plt.xticks(x, short_names)
plt.ylabel("Dice Score")
plt.title("Region-wise Dice Comparison")
plt.ylim(0, 1.0)
plt.legend()

savefig(
    os.path.join(
        COMBINED,
        "region_wise",
        "REGION_WISE_DICE_COMPARISON.png",
    )
)


# ============================================================
# 6. RADAR CHART
# ============================================================

radar_metrics = [
    "Mean Dice",
    "Mean IoU",
    "Mean Sensitivity",
    "MCC",
    "ROC-AUC",
    "PR-AUC",
]

labels = radar_metrics
angles = np.linspace(
    0,
    2 * np.pi,
    len(labels),
    endpoint=False,
).tolist()

angles += angles[:1]

fig = plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)

for _, row in df.iterrows():

    values = [
        row[m]
        for m in radar_metrics
    ]

    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=2,
        label=row["Experiment"],
    )

    ax.fill(
        angles,
        values,
        alpha=0.08,
    )

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.set_ylim(0, 1)

plt.title(
    "Multi-Metric Performance Comparison",
    pad=25,
)

plt.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.15),
)

savefig(
    os.path.join(
        COMBINED,
        "radar",
        "MULTI_METRIC_RADAR.png",
    )
)


# ============================================================
# 7. E4 QUANTUM DIVERSITY VS ROUND
# ============================================================

qk_history = os.path.join(
    E4,
    "raw_results",
    "quantum_kernel_history.csv",
)

if os.path.exists(qk_history):

    qdf = pd.read_csv(qk_history)

    plt.figure(figsize=(10, 6))

    for client in sorted(qdf["client"].unique()):

        cdf = qdf[qdf["client"] == client]

        plt.plot(
            cdf["round"],
            cdf["quantum_diversity"],
            marker="o",
            label=client,
        )

    plt.xlabel("Federated Round")
    plt.ylabel("Quantum Diversity")
    plt.title(
        "Quantum Kernel Diversity Across Federated Rounds"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    savefig(
        os.path.join(
            E4,
            "graphs",
            "01_quantum_diversity_vs_round.png",
        )
    )


# ============================================================
# 8. E4 QUANTUM KERNEL HEATMAPS
# ============================================================

for round_no in [2, 5, 10]:

    kernel_file = os.path.join(
        E4,
        "raw_results",
        f"quantum_kernel_round_{round_no}.npy",
    )

    if not os.path.exists(kernel_file):
        continue

    kernel = np.load(kernel_file)

    plt.figure(figsize=(6, 5))

    plt.imshow(
        kernel,
        interpolation="nearest",
    )

    plt.colorbar(
        label="Kernel Similarity"
    )

    plt.xticks(
        range(kernel.shape[0]),
        [f"C{i+1}" for i in range(kernel.shape[0])],
    )

    plt.yticks(
        range(kernel.shape[0]),
        [f"C{i+1}" for i in range(kernel.shape[0])],
    )

    plt.xlabel("Client")
    plt.ylabel("Client")

    plt.title(
        f"Quantum Kernel Similarity — Round {round_no}"
    )

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            plt.text(
                j,
                i,
                f"{kernel[i, j]:.3f}",
                ha="center",
                va="center",
            )

    savefig(
        os.path.join(
            E4,
            "graphs",
            f"02_quantum_kernel_heatmap_round_{round_no}.png",
        )
    )


# ============================================================
# 9. E4 ADAPTIVE WEIGHTS
# ============================================================

adaptive_file = os.path.join(
    E4,
    "raw_results",
    "adaptive_weights.csv",
)

if os.path.exists(adaptive_file):

    adf = pd.read_csv(adaptive_file)

    # Automatically find likely round/client/weight columns
    round_col = next(
        (c for c in adf.columns if c.lower() == "round"),
        None,
    )

    client_col = next(
        (
            c
            for c in adf.columns
            if c.lower() in ["client", "client_id", "node"]
        ),
        None,
    )

    weight_col = next(
        (
            c
            for c in adf.columns
            if "weight" in c.lower()
            and "normalized" not in c.lower()
        ),
        None,
    )

    if round_col and client_col and weight_col:

        plt.figure(figsize=(10, 6))

        for client in sorted(
            adf[client_col].unique()
        ):

            cdf = adf[
                adf[client_col] == client
            ]

            plt.plot(
                cdf[round_col],
                cdf[weight_col],
                marker="o",
                label=str(client),
            )

        plt.xlabel("Federated Round")
        plt.ylabel("Aggregation Weight")
        plt.title(
            "Adaptive Client Aggregation Weights"
        )

        plt.grid(True, alpha=0.3)
        plt.legend()

        savefig(
            os.path.join(
                E4,
                "graphs",
                "06_adaptive_weights_vs_round.png",
            )
        )


# ============================================================
# 10. COPY / ORGANIZE EXISTING E4 GRAPH
# ============================================================

old_graph = os.path.join(
    E4,
    "graphs",
    "quantum_quantum_diversity.png",
)

new_graph = os.path.join(
    E4,
    "graphs",
    "quantum_diversity.png",
)

if os.path.exists(old_graph):
    os.replace(old_graph, new_graph)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("GRAPH GENERATION COMPLETE")
print("=" * 70)
print()

for name, path in [
    ("E1", os.path.join(E1, "graphs")),
    ("E2", os.path.join(E2, "graphs")),
    ("E3", os.path.join(E3, "graphs")),
    ("E4", os.path.join(E4, "graphs")),
    (
        "Combined",
        COMBINED,
    ),
]:
    files = []

    for root, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(".png"):
                files.append(
                    os.path.relpath(
                        os.path.join(root, f),
                        path,
                    )
                )

    print(f"{name}: {len(files)} PNG files")

print()
print("No fabricated E4 Dice/IoU convergence curve was created.")
print("E4 convergence-specific plots are based on actual quantum data.")
