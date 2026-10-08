from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE = Path.home() / "Neuroverse" / "Federated"
FULL = BASE / "Full_results"
OUT = FULL / "Combined_Comparison"

TABLES = OUT / "tables"
GRAPHS = OUT / "final_comparison"
RADAR = OUT / "radar"
REGION = OUT / "region_wise"

for d in [TABLES, GRAPHS, RADAR, REGION]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# FOUR ACTUAL EXPERIMENTS
# ============================================================

experiments = {
    "E1 - 3D U-Net + FedAvg":
        FULL / "E1_3D_UNet_FedAvg/raw_results/GlobalTest/global_test_summary.json",

    "E2 - Swin UNETR + FedAvg":
        FULL / "E2_Swin_FedAvg/raw_results/GlobalTest/global_test_summary.json",

    "E3 - Swin UNETR + Adaptive FL":
        FULL / "E3_Swin_Adaptive/raw_results/GlobalTest/global_test_summary.json",

    "E4 - Swin UNETR + Quantum Kernel":
        FULL / "E4_Swin_QuantumKernel/raw_results/GlobalTest/global_test_summary.json",
}


# ============================================================
# METRICS
# ============================================================

metrics = {
    "mean_accuracy": "Mean Accuracy",
    "mean_precision": "Mean Precision",
    "mean_sensitivity": "Mean Sensitivity",
    "mean_specificity": "Mean Specificity",
    "mean_f1": "Mean F1",
    "mean_dice": "Mean Dice",
    "mean_iou": "Mean IoU",
    "mean_balanced_accuracy": "Balanced Accuracy",
    "mean_mcc": "MCC",
    "mean_hd95": "HD95",
    "mean_roc_auc": "ROC-AUC",
    "mean_pr_auc": "PR-AUC",
}


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

rows = []

for experiment, path in experiments.items():

    print(f"Loading: {experiment}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file:\n{path}"
        )

    with open(path, "r") as f:
        data = json.load(f)

    row = {
        "Experiment": experiment
    }

    for key, label in metrics.items():
        row[label] = data.get(key, np.nan)

    row["Patients"] = data.get("num_patients", np.nan)
    row["Inference Time (s)"] = data.get(
        "inference_time_seconds", np.nan
    )
    row["Total Evaluation Time (s)"] = data.get(
        "total_evaluation_time_seconds", np.nan
    )

    rows.append(row)


df = pd.DataFrame(rows)


# ============================================================
# SAVE MASTER CSV
# ============================================================

master_csv = TABLES / "FINAL_FOUR_EXPERIMENT_COMPARISON.csv"

df.to_csv(
    master_csv,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 100)
print("FINAL FOUR-EXPERIMENT COMPARISON")
print("=" * 100)
print()

print(
    df[
        [
            "Experiment",
            "Mean Dice",
            "Mean IoU",
            "Mean Precision",
            "Mean Sensitivity",
            "MCC",
            "ROC-AUC",
            "PR-AUC",
            "HD95",
        ]
    ].to_string(index=False)
)

print()
print(f"Saved: {master_csv}")


# ============================================================
# SAVE PAPER-FRIENDLY TABLE
# ============================================================

paper_columns = [
    "Experiment",
    "Mean Dice",
    "Mean IoU",
    "Mean Precision",
    "Mean Sensitivity",
    "Mean Specificity",
    "Balanced Accuracy",
    "MCC",
    "HD95",
    "ROC-AUC",
    "PR-AUC",
]

paper_df = df[paper_columns].copy()

paper_df.to_csv(
    TABLES / "PAPER_FINAL_METRICS_TABLE.csv",
    index=False
)


# ============================================================
# FINAL BAR CHARTS
# ============================================================

bar_metrics = [
    ("Mean Dice", "01_final_mean_dice.png"),
    ("Mean IoU", "02_final_mean_iou.png"),
    ("Mean Precision", "03_final_precision.png"),
    ("Mean Sensitivity", "04_final_sensitivity.png"),
    ("MCC", "05_final_mcc.png"),
    ("ROC-AUC", "06_final_roc_auc.png"),
    ("PR-AUC", "07_final_pr_auc.png"),
    ("HD95", "08_final_hd95.png"),
]


for metric, filename in bar_metrics:

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        df["Experiment"],
        df[metric]
    )

    plt.ylabel(metric)
    plt.title(f"Final {metric} Comparison")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(
        bars,
        df[metric]
    ):

        if pd.notna(value):

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()

    plt.savefig(
        GRAPHS / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 2 x 2 IMPORTANT COMPARISON
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 11)
)

plots = [
    ("Mean Dice", axes[0, 0]),
    ("Mean IoU", axes[0, 1]),
    ("ROC-AUC", axes[1, 0]),
    ("HD95", axes[1, 1]),
]

for metric, ax in plots:

    bars = ax.bar(
        df["Experiment"],
        df[metric]
    )

    ax.set_title(metric)
    ax.set_ylabel(metric)

    ax.tick_params(
        axis="x",
        rotation=25
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(
        bars,
        df[metric]
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

fig.suptitle(
    "Neuroverse - Final Model Comparison",
    fontsize=17
)

fig.tight_layout()

fig.savefig(
    GRAPHS / "09_FINAL_2x2_COMPARISON.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# REGION-WISE DICE
# ============================================================

region_rows = []

for experiment, path in experiments.items():

    with open(path, "r") as f:
        data = json.load(f)

    region_rows.append({
        "Experiment": experiment,
        "WT": data.get("WT_dice", np.nan),
        "TC": data.get("TC_dice", np.nan),
        "ET": data.get("ET_dice", np.nan),
    })


region_df = pd.DataFrame(region_rows)

region_df.to_csv(
    TABLES / "REGION_WISE_DICE.csv",
    index=False
)


x = np.arange(len(region_df))
width = 0.25

plt.figure(figsize=(13, 7))

plt.bar(
    x - width,
    region_df["WT"],
    width,
    label="WT"
)

plt.bar(
    x,
    region_df["TC"],
    width,
    label="TC"
)

plt.bar(
    x + width,
    region_df["ET"],
    width,
    label="ET"
)

plt.xticks(
    x,
    region_df["Experiment"],
    rotation=20,
    ha="right"
)

plt.ylabel("Dice")
plt.title("Region-wise Dice Comparison")

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    REGION / "REGION_WISE_DICE_COMPARISON.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RADAR CHART
# ============================================================

radar_metrics = [
    "Mean Dice",
    "Mean IoU",
    "Mean Sensitivity",
    "MCC",
    "ROC-AUC",
    "PR-AUC",
]

N = len(radar_metrics)

angles = np.linspace(
    0,
    2 * np.pi,
    N,
    endpoint=False
)

angles = np.concatenate(
    [angles, [angles[0]]]
)

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(
    111,
    polar=True
)

for _, row in df.iterrows():

    values = [
        row[m]
        for m in radar_metrics
    ]

    values.append(values[0])

    ax.plot(
        angles,
        values,
        linewidth=2,
        label=row["Experiment"]
    )

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    radar_metrics
)

ax.set_title(
    "Neuroverse - Multi-Metric Comparison",
    pad=25
)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.15)
)

plt.tight_layout()

plt.savefig(
    RADAR / "MULTI_METRIC_RADAR.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("=" * 100)
print("GRAPH GENERATION COMPLETE")
print("=" * 100)
