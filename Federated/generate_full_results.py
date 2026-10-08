from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Neuroverse" / "Federated"

FULL = BASE / "Full_results"

QK_RESULTS = BASE / "Quantum_Kernel_E3" / "results"

FL_GRAPHS = BASE / "FL_Graphs"


# ============================================================
# EXPERIMENT NAMES
# ============================================================

EXPERIMENTS = {
    "E1": "E1_3D_UNet",
    "E2": "E2_3D_UNet_Fed",
    "E3": "E3_Swin_FedAvg",
    "E4": "E4_Swin_Adaptive",
    "E5": "E5_Swin_QuantumKernel",
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for folder in EXPERIMENTS.values():
    (FULL / folder / "raw_results").mkdir(parents=True, exist_ok=True)
    (FULL / folder / "graphs").mkdir(parents=True, exist_ok=True)
    (FULL / folder / "final_metrics").mkdir(parents=True, exist_ok=True)

(FULL / "Combined_Comparison" / "convergence").mkdir(
    parents=True, exist_ok=True
)

(FULL / "Combined_Comparison" / "final_comparison").mkdir(
    parents=True, exist_ok=True
)

(FULL / "Combined_Comparison" / "radar").mkdir(
    parents=True, exist_ok=True
)

(FULL / "Combined_Comparison" / "region_wise").mkdir(
    parents=True, exist_ok=True
)

(FULL / "Combined_Comparison" / "tables").mkdir(
    parents=True, exist_ok=True
)


# ============================================================
# COPY QUANTUM KERNEL RESULTS
# ============================================================

if QK_RESULTS.exists():

    destination = FULL / "E5_Swin_QuantumKernel" / "raw_results"

    for item in QK_RESULTS.iterdir():

        target = destination / item.name

        if item.is_dir():
            shutil.copytree(
                item,
                target,
                dirs_exist_ok=True
            )
        else:
            shutil.copy2(item, target)

    print("Quantum Kernel results copied.")


# ============================================================
# LOAD QUANTUM FINAL RESULTS
# ============================================================

QK_SUMMARY = (
    QK_RESULTS
    / "GlobalTest"
    / "global_test_summary.json"
)

if QK_SUMMARY.exists():

    with open(QK_SUMMARY, "r") as f:
        qk = json.load(f)

    qk_df = pd.DataFrame([qk])

    qk_df.to_csv(
        FULL
        / "E5_Swin_QuantumKernel"
        / "final_metrics"
        / "final_metrics.csv",
        index=False
    )

    print("Quantum final metrics saved.")


# ============================================================
# QK QUANTUM KERNEL HISTORY
# ============================================================

QK_HISTORY = QK_RESULTS / "quantum_kernel_history.csv"

if QK_HISTORY.exists():

    df = pd.read_csv(QK_HISTORY)

    df.to_csv(
        FULL
        / "E5_Swin_QuantumKernel"
        / "final_metrics"
        / "quantum_kernel_history.csv",
        index=False
    )

    print("Quantum kernel history copied.")


# ============================================================
# QK ADAPTIVE WEIGHTS
# ============================================================

QK_WEIGHTS = QK_RESULTS / "adaptive_weights.csv"

if QK_WEIGHTS.exists():

    df = pd.read_csv(QK_WEIGHTS)

    df.to_csv(
        FULL
        / "E5_Swin_QuantumKernel"
        / "final_metrics"
        / "adaptive_weights.csv",
        index=False
    )


# ============================================================
# EXISTING ROUND-WISE E1/E2/E3 DATA
# ============================================================

ROUNDWISE = FL_GRAPHS / "round_wise_metrics_E1_E2_E3.csv"

if ROUNDWISE.exists():

    df = pd.read_csv(ROUNDWISE)

    df.to_csv(
        FULL
        / "Combined_Comparison"
        / "tables"
        / "round_wise_metrics_E1_E2_E3.csv",
        index=False
    )

    print(
        "Existing E1/E2/E3 round-wise metrics copied."
    )


# ============================================================
# UTILITY: FIND COLUMN
# ============================================================

def find_column(df, possible_names):

    lower_map = {
        c.lower().strip(): c
        for c in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


# ============================================================
# GENERIC CONVERGENCE PLOT
# ============================================================

def convergence_plot(
    df,
    metric,
    output,
    title
):

    round_col = find_column(
        df,
        [
            "round",
            "round_number",
            "server_round"
        ]
    )

    if round_col is None:
        print(
            f"Cannot find round column for {metric}"
        )
        return

    plt.figure(figsize=(10, 6))

    found = False

    for experiment in [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5"
    ]:

        candidates = [
            f"{experiment}_{metric}",
            f"{experiment}_{metric.lower()}",
            f"{experiment.lower()}_{metric.lower()}",
        ]

        column = find_column(
            df,
            candidates
        )

        if column:

            plt.plot(
                df[round_col],
                df[column],
                marker="o",
                linewidth=2,
                label=EXPERIMENTS[experiment]
            )

            found = True

    if not found:
        print(
            f"No columns found for {metric}"
        )
        plt.close()
        return

    plt.xlabel("Federated Round")
    plt.ylabel(metric.replace("_", " ").title())

    plt.title(title)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# GENERATE E1/E2/E3 CONVERGENCE GRAPHS
# ============================================================

if ROUNDWISE.exists():

    df = pd.read_csv(ROUNDWISE)

    metrics = [
        "mean_dice",
        "mean_iou",
        "mean_sensitivity",
        "mean_mcc",
        "mean_roc_auc",
        "mean_pr_auc",
        "mean_hd95"
    ]

    for metric in metrics:

        convergence_plot(
            df,
            metric,
            FULL
            / "Combined_Comparison"
            / "convergence"
            / f"{metric}_convergence.png",
            f"{metric.replace('_', ' ').title()} Convergence"
        )


# ============================================================
# QUANTUM KERNEL DIVERSITY CURVE
# ============================================================

if QK_HISTORY.exists():

    qdf = pd.read_csv(QK_HISTORY)

    print(
        "Quantum kernel history columns:",
        list(qdf.columns)
    )

    # Plot numerical columns that appear to represent
    # quantum diversity/similarity.

    for column in qdf.columns:

        if column.lower() in [
            "round",
            "server_round"
        ]:
            continue

        if pd.api.types.is_numeric_dtype(
            qdf[column]
        ):

            plt.figure(figsize=(9, 6))

            plt.plot(
                qdf.index + 2,
                qdf[column],
                marker="o",
                linewidth=2
            )

            plt.xlabel("Federated Round")
            plt.ylabel(column)

            plt.title(
                f"Quantum Kernel - {column}"
            )

            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            safe_name = (
                column
                .replace(" ", "_")
                .replace("/", "_")
            )

            plt.savefig(
                FULL
                / "E5_Swin_QuantumKernel"
                / "graphs"
                / f"quantum_{safe_name}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


print()
print("=" * 60)
print("GRAPH GENERATION FINISHED")
print("=" * 60)
print()
print(f"Results stored in:")
print(FULL)
