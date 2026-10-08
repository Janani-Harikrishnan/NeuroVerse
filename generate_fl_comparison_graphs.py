from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "Neuroverse"

# ------------------------------------------------------------
# Experiment result directories
# ------------------------------------------------------------

EXPERIMENTS = {
    "E1: 3D U-Net + FedAvg":
        ROOT / "Federated" / "FedAvg_BraTS" / "results",

    "E2: Swin UNETR + FedAvg":
        ROOT / "Federated" / "Swin_FedAvg" / "results",

    "E3: Swin UNETR + Adaptive FL":
        ROOT / "Federated" / "Swin_Adaptive_FL" / "results",
}

# ------------------------------------------------------------
# IMPORTANT:
# This fixes the comparison order everywhere.
# ------------------------------------------------------------

MODEL_ORDER = [
    "E1: 3D U-Net + FedAvg",
    "E2: Swin UNETR + FedAvg",
    "E3: Swin UNETR + Adaptive FL",
]

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "Visualization_Results"
    / "FL_Comparison_Graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SEARCH FOR SUMMARY FILE
# ============================================================

def find_summary_file(result_dir):

    possible_files = [

        result_dir
        / "GlobalTest"
        / "global_test_summary.json",

        result_dir
        / "GlobalTest"
        / "global_test_summary.csv",

        result_dir
        / "global_test_summary.json",

        result_dir
        / "global_test_summary.csv",
    ]

    for path in possible_files:

        if path.exists():

            return path

    # Recursive fallback
    json_files = list(
        result_dir.rglob(
            "*summary*.json"
        )
    )

    if json_files:

        return json_files[0]

    csv_files = list(
        result_dir.rglob(
            "*summary*.csv"
        )
    )

    if csv_files:

        return csv_files[0]

    return None


# ============================================================
# LOAD JSON SUMMARY
# ============================================================

def load_json(path):

    print(
        f"Reading: {path}"
    )

    with open(
        path,
        "r"
    ) as f:

        data = json.load(f)

    return data


# ============================================================
# LOAD CSV SUMMARY
# ============================================================

def load_csv(path):

    print(
        f"Reading: {path}"
    )

    data = {}

    with open(
        path,
        "r",
        newline=""
    ) as f:

        reader = csv.reader(f)

        rows = list(reader)

    # --------------------------------------------------------
    # Case 1:
    # key,value
    # --------------------------------------------------------

    for row in rows:

        if len(row) >= 2:

            key = row[0].strip()

            value = row[1].strip()

            try:

                value = float(value)

            except ValueError:

                pass

            data[key] = value

    return data


# ============================================================
# LOAD EXPERIMENT METRICS
# ============================================================

def load_experiment_metrics(
    model_name,
    result_dir
):

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    summary_file = find_summary_file(
        result_dir
    )

    if summary_file is None:

        raise FileNotFoundError(
            f"\nCould not find GlobalTest summary "
            f"inside:\n{result_dir}\n\n"
            "Expected something like:\n"
            "results/GlobalTest/global_test_summary.json"
        )

    print(
        f"Summary file: {summary_file}"
    )

    if summary_file.suffix.lower() == ".json":

        metrics = load_json(
            summary_file
        )

    else:

        metrics = load_csv(
            summary_file
        )

    print(
        f"Loaded {len(metrics)} metrics."
    )

    return metrics


# ============================================================
# LOAD ALL THREE MODELS
# ============================================================

def load_all_metrics():

    all_metrics = {}

    for model_name in MODEL_ORDER:

        result_dir = EXPERIMENTS[
            model_name
        ]

        all_metrics[
            model_name
        ] = load_experiment_metrics(
            model_name,
            result_dir
        )

    return all_metrics


# ============================================================
# GET METRIC
# ============================================================

def get_metric(
    metrics,
    key
):

    if key not in metrics:

        raise KeyError(
            f"Metric '{key}' not found.\n"
            f"Available examples:\n"
            f"{list(metrics.keys())[:30]}"
        )

    return float(
        metrics[key]
    )


# ============================================================
# COMMON GRAPH SETTINGS
# ============================================================

def prepare_axes(
    ax,
    title,
    ylabel
):

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    ax.set_ylabel(
        ylabel,
        fontsize=11
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.set_axisbelow(
        True
    )

    # Remove top/right borders
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


# ============================================================
# GRAPH 1
# MEAN DICE
# ============================================================

def graph_mean_dice(
    all_metrics
):

    values = [

        get_metric(
            all_metrics[model],
            "mean_dice"
        )

        for model in MODEL_ORDER
    ]

    labels = [
        "E1\n3D U-Net + FedAvg",
        "E2\nSwin UNETR + FedAvg",
        "E3\nSwin UNETR + Adaptive FL",
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(labels)
    )

    bars = ax.bar(
        x,
        values,
        width=0.60
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        fontsize=10
    )

    prepare_axes(
        ax,
        "Mean Dice Comparison",
        "Mean Dice"
    )

    # Dice naturally lies between 0 and 1
    ax.set_ylim(
        0.80,
        0.90
    )

    # --------------------------------------------------------
    # Values on bars
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "01_mean_dice_comparison.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {output}"
    )


# ============================================================
# GRAPH 2
# WT / TC / ET DICE
# ============================================================

def graph_region_dice(
    all_metrics
):

    regions = [
        "WT",
        "TC",
        "ET"
    ]

    metric_keys = [
        "WT_dice",
        "TC_dice",
        "ET_dice"
    ]

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    values = np.array([

        [
            get_metric(
                all_metrics[model],
                metric
            )

            for model in MODEL_ORDER
        ]

        for metric in metric_keys

    ])

    # Shape:
    # 3 regions × 3 models

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    x = np.arange(
        len(regions)
    )

    width = 0.24

    # --------------------------------------------------------
    # E1
    # --------------------------------------------------------

    bars1 = ax.bar(
        x - width,
        values[:, 0],
        width,
        label="E1: 3D U-Net + FedAvg"
    )

    # --------------------------------------------------------
    # E2
    # --------------------------------------------------------

    bars2 = ax.bar(
        x,
        values[:, 1],
        width,
        label="E2: Swin UNETR + FedAvg"
    )

    # --------------------------------------------------------
    # E3
    # --------------------------------------------------------

    bars3 = ax.bar(
        x + width,
        values[:, 2],
        width,
        label="E3: Swin UNETR + Adaptive FL"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Whole Tumor (WT)",
            "Tumor Core (TC)",
            "Enhancing Tumor (ET)"
        ]
    )

    prepare_axes(
        ax,
        "Region-wise Dice Comparison",
        "Dice"
    )

    ax.set_ylim(
        0.75,
        0.95
    )

    ax.legend(
        fontsize=9,
        loc="lower left"
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    for bars in [
        bars1,
        bars2,
        bars3
    ]:

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.004,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "02_region_wise_dice_comparison.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {output}"
    )


# ============================================================
# GRAPH 3
# MULTI-METRIC COMPARISON
# ============================================================

def graph_main_metrics(
    all_metrics
):

    metrics = [
        ("mean_dice", "Dice"),
        ("mean_iou", "IoU"),
        ("mean_sensitivity", "Sensitivity"),
        ("mean_mcc", "MCC"),
        ("mean_roc_auc", "ROC-AUC"),
        ("mean_pr_auc", "PR-AUC"),
    ]

    # --------------------------------------------------------
    # Create one subplot per metric
    #
    # The three models remain side-by-side
    # in EVERY subplot.
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9)
    )

    axes = axes.flatten()

    x = np.arange(
        len(MODEL_ORDER)
    )

    short_labels = [
        "E1",
        "E2",
        "E3"
    ]

    for ax, (
        metric_key,
        metric_name
    ) in zip(
        axes,
        metrics
    ):

        values = [

            get_metric(
                all_metrics[model],
                metric_key
            )

            for model in MODEL_ORDER

        ]

        bars = ax.bar(
            x,
            values,
            width=0.55
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            short_labels
        )

        ax.set_title(
            metric_name,
            fontsize=13,
            fontweight="bold"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.set_axisbelow(
            True
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        # All these metrics are in [0, 1]
        ax.set_ylim(
            0,
            1.05
        )

        # ----------------------------------------------------
        # Value labels
        # ----------------------------------------------------

        for bar, value in zip(
            bars,
            values
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_ylabel(
            metric_name,
            fontsize=9
        )

    fig.suptitle(
        "Performance Comparison Across Federated Models",
        fontsize=17,
        fontweight="bold",
        y=0.99
    )

    fig.text(
        0.5,
        0.01,
        (
            "E1 = 3D U-Net + FedAvg    |    "
            "E2 = Swin UNETR + FedAvg    |    "
            "E3 = Swin UNETR + Adaptive FL"
        ),
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            0.96
        ]
    )

    output = (
        OUTPUT_DIR
        / "03_main_metrics_comparison.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {output}"
    )


# ============================================================
# CREATE NUMERICAL SUMMARY
# ============================================================

def save_comparison_csv(
    all_metrics
):

    output = (
        OUTPUT_DIR
        / "FL_model_comparison.csv"
    )

    metrics = [
        "mean_dice",
        "mean_iou",
        "mean_sensitivity",
        "mean_specificity",
        "mean_mcc",
        "mean_roc_auc",
        "mean_pr_auc",
        "mean_hd95",
        "WT_dice",
        "TC_dice",
        "ET_dice",
    ]

    with open(
        output,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "Metric",
                "E1: 3D U-Net + FedAvg",
                "E2: Swin UNETR + FedAvg",
                "E3: Swin UNETR + Adaptive FL",
            ]
        )

        for metric in metrics:

            row = [
                metric
            ]

            for model in MODEL_ORDER:

                row.append(
                    get_metric(
                        all_metrics[
                            model
                        ],
                        metric
                    )
                )

            writer.writerow(
                row
            )

    print(
        f"✓ Saved: {output}"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    all_metrics
):

    print()
    print("=" * 90)
    print("FINAL FL MODEL COMPARISON")
    print("=" * 90)

    print(
        f"{'Metric':<25}"
        f"{'E1':>15}"
        f"{'E2':>15}"
        f"{'E3':>15}"
    )

    print(
        "-" * 90
    )

    metrics = [
        ("Mean Dice", "mean_dice"),
        ("Mean IoU", "mean_iou"),
        ("Sensitivity", "mean_sensitivity"),
        ("Specificity", "mean_specificity"),
        ("MCC", "mean_mcc"),
        ("ROC-AUC", "mean_roc_auc"),
        ("PR-AUC", "mean_pr_auc"),
        ("HD95", "mean_hd95"),
        ("WT Dice", "WT_dice"),
        ("TC Dice", "TC_dice"),
        ("ET Dice", "ET_dice"),
    ]

    for name, key in metrics:

        values = [

            get_metric(
                all_metrics[model],
                key
            )

            for model in MODEL_ORDER
        ]

        print(
            f"{name:<25}"
            f"{values[0]:>15.4f}"
            f"{values[1]:>15.4f}"
            f"{values[2]:>15.4f}"
        )

    print(
        "=" * 90
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("FEDERATED MODEL GRAPH GENERATION")
    print("=" * 90)

    print()
    print(
        "Models will always be displayed in this order:"
    )

    for model in MODEL_ORDER:

        print(
            f"  {model}"
        )

    print()
    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    # ========================================================
    # LOAD SAVED METRICS
    # ========================================================

    all_metrics = load_all_metrics()

    # ========================================================
    # PRINT VALUES
    # ========================================================

    print_summary(
        all_metrics
    )

    # ========================================================
    # GENERATE GRAPH 1
    # ========================================================

    graph_mean_dice(
        all_metrics
    )

    # ========================================================
    # GENERATE GRAPH 2
    # ========================================================

    graph_region_dice(
        all_metrics
    )

    # ========================================================
    # GENERATE GRAPH 3
    # ========================================================

    graph_main_metrics(
        all_metrics
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_comparison_csv(
        all_metrics
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 90)
    print("ALL GRAPHS GENERATED")
    print("=" * 90)

    print()
    print(
        f"Location:\n{OUTPUT_DIR}"
    )

    print()
    print(
        "Files:"
    )

    for file in sorted(
        OUTPUT_DIR.iterdir()
    ):

        print(
            f"  {file.name}"
        )

    print()


if __name__ == "__main__":

    main()
