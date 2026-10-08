import json
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Neuroverse" / "Federated"

PROJECTS = {
    "E1 – 3D U-Net + FedAvg":
        BASE / "FedAvg_BraTS" / "results",

    "E2 – Swin UNETR + FedAvg":
        BASE / "Swin_FedAvg" / "results",

    "E3 – Swin UNETR + Adaptive FL":
        BASE / "Swin_Adaptive_FL" / "results",
}

OUTPUT = BASE / "FL_Graphs"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def find_round_number(path):
    match = re.search(r"round_(\d+)", str(path))

    if match:
        return int(match.group(1))

    return None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_metric(data, names):

    for name in names:
        if name in data:
            return data[name]

    lower_map = {
        str(k).lower(): v
        for k, v in data.items()
    }

    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


# ============================================================
# LOAD ROUND-WISE VALIDATION DATA
# ============================================================

def load_validation_history(results_dir):

    records = []

    files = sorted(
        results_dir.glob(
            "client_metrics/*/round_*/validation_summary.json"
        )
    )

    for file in files:

        client = file.parent.parent.name
        round_number = find_round_number(file)

        if round_number is None:
            continue

        try:
            data = load_json(file)
        except Exception as e:
            print(f"Could not read {file}: {e}")
            continue

        record = {
            "client": client,
            "round": round_number,

            "dice": find_metric(
                data,
                ["mean_dice", "Mean Dice", "dice", "mean_f1"]
            ),

            "iou": find_metric(
                data,
                ["mean_iou", "Mean IoU", "iou"]
            ),

            "sensitivity": find_metric(
                data,
                [
                    "mean_sensitivity",
                    "Mean Sensitivity",
                    "sensitivity",
                ]
            ),

            "specificity": find_metric(
                data,
                [
                    "mean_specificity",
                    "Mean Specificity",
                    "specificity",
                ]
            ),

            "mcc": find_metric(
                data,
                ["mean_mcc", "Mean MCC", "mcc"]
            ),

            "roc_auc": find_metric(
                data,
                [
                    "mean_roc_auc",
                    "Mean ROC-AUC",
                    "roc_auc",
                    "mean_roc",
                ]
            ),

            "pr_auc": find_metric(
                data,
                [
                    "mean_pr_auc",
                    "Mean PR-AUC",
                    "pr_auc",
                ]
            ),

            "hd95": find_metric(
                data,
                ["mean_hd95", "Mean HD95", "hd95"]
            ),
        }

        records.append(record)

    return records


# ============================================================
# AVERAGE CLIENT METRICS PER ROUND
# ============================================================

def aggregate_rounds(records):

    metrics = [
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "mcc",
        "roc_auc",
        "pr_auc",
        "hd95",
    ]

    rounds = sorted(
        set(r["round"] for r in records)
    )

    output = []

    for rnd in rounds:

        round_records = [
            r for r in records
            if r["round"] == rnd
        ]

        row = {
            "round": rnd
        }

        for metric in metrics:

            values = [
                r[metric]
                for r in round_records
                if r[metric] is not None
            ]

            if values:
                row[metric] = sum(values) / len(values)
            else:
                row[metric] = None

        output.append(row)

    return output


# ============================================================
# LOAD FINAL GLOBALTEST SUMMARY
# ============================================================

def load_final_globaltest(results_dir):

    json_file = (
        results_dir /
        "GlobalTest" /
        "global_test_summary.json"
    )

    if json_file.exists():

        print(f"Loading final GlobalTest: {json_file}")

        data = load_json(json_file)

        return {
            "dice": find_metric(
                data,
                ["mean_dice", "Mean Dice"]
            ),

            "iou": find_metric(
                data,
                ["mean_iou", "Mean IoU"]
            ),

            "sensitivity": find_metric(
                data,
                ["mean_sensitivity", "Mean Sensitivity"]
            ),

            "specificity": find_metric(
                data,
                ["mean_specificity", "Mean Specificity"]
            ),

            "mcc": find_metric(
                data,
                ["mean_mcc", "Mean MCC"]
            ),

            "roc_auc": find_metric(
                data,
                ["mean_roc_auc", "Mean ROC-AUC"]
            ),

            "pr_auc": find_metric(
                data,
                ["mean_pr_auc", "Mean PR-AUC"]
            ),

            "hd95": find_metric(
                data,
                ["mean_hd95", "Mean HD95"]
            ),

            "WT_dice": find_metric(
                data,
                ["WT_dice", "WT Dice"]
            ),

            "TC_dice": find_metric(
                data,
                ["TC_dice", "TC Dice"]
            ),

            "ET_dice": find_metric(
                data,
                ["ET_dice", "ET Dice"]
            ),
        }

    print(
        f"WARNING: Final GlobalTest JSON not found in "
        f"{results_dir}"
    )

    return {}


# ============================================================
# LOAD E1 + E2 ROUND-WISE DATA
# ============================================================

histories = {}

for name in [
    "E1 – 3D U-Net + FedAvg",
    "E2 – Swin UNETR + FedAvg",
]:

    records = load_validation_history(
        PROJECTS[name]
    )

    histories[name] = aggregate_rounds(records)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for row in histories[name]:

        print(
            f"Round {row['round']:02d}: "
            f"Dice={row['dice']}, "
            f"ROC-AUC={row['roc_auc']}"
        )


# ============================================================
# LOAD FINAL RESULTS FOR ALL THREE
# ============================================================

final_results = {}

for name, path in PROJECTS.items():

    final_results[name] = load_final_globaltest(path)

    print("\nFINAL:", name)

    for key, value in final_results[name].items():
        print(f"  {key}: {value}")


# ============================================================
# ADD E3 FINAL GLOBALTEST AS ROUND 10
# ============================================================

e3_name = "E3 – Swin UNETR + Adaptive FL"

e3_final = final_results[e3_name]

if e3_final:

    histories[e3_name] = [
        {
            "round": 10,
            "dice": e3_final.get("dice"),
            "iou": e3_final.get("iou"),
            "sensitivity": e3_final.get("sensitivity"),
            "specificity": e3_final.get("specificity"),
            "mcc": e3_final.get("mcc"),
            "roc_auc": e3_final.get("roc_auc"),
            "pr_auc": e3_final.get("pr_auc"),
            "hd95": e3_final.get("hd95"),
        }
    ]

    print("\nE3 final point added at Round 10.")
    print("IMPORTANT: This is Final GlobalTest,")
    print("not intermediate validation.")


# ============================================================
# SAVE ROUND-WISE CSV
# ============================================================

csv_path = OUTPUT / "round_wise_metrics_E1_E2_E3.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Round",
        "Mean Dice",
        "Mean IoU",
        "Sensitivity",
        "Specificity",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
        "HD95",
        "Evaluation Type",
    ])

    for model, rows in histories.items():

        for row in rows:

            evaluation_type = (
                "Final GlobalTest"
                if model == e3_name
                else "Round-wise validation"
            )

            writer.writerow([
                model,
                row["round"],
                row["dice"],
                row["iou"],
                row["sensitivity"],
                row["specificity"],
                row["mcc"],
                row["roc_auc"],
                row["pr_auc"],
                row["hd95"],
                evaluation_type,
            ])

print(f"\nSaved: {csv_path}")


# ============================================================
# GENERIC CONVERGENCE GRAPH FUNCTION
# ============================================================

def make_convergence_graph(
    metric,
    ylabel,
    title,
    filename,
):

    plt.figure(figsize=(10, 6))

    # E1 and E2
    for model in [
        "E1 – 3D U-Net + FedAvg",
        "E2 – Swin UNETR + FedAvg",
    ]:

        rows = histories.get(model, [])

        x = [
            r["round"]
            for r in rows
            if r.get(metric) is not None
        ]

        y = [
            r[metric]
            for r in rows
            if r.get(metric) is not None
        ]

        if x:
            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=model,
            )

    # E3 FINAL POINT ONLY
    e3_rows = histories.get(e3_name, [])

    x3 = [
        r["round"]
        for r in e3_rows
        if r.get(metric) is not None
    ]

    y3 = [
        r[metric]
        for r in e3_rows
        if r.get(metric) is not None
    ]

    if x3:

        plt.scatter(
            x3,
            y3,
            s=100,
            marker="*",
            label="E3 – Adaptive FL (Final GlobalTest)",
            zorder=5,
        )

    plt.xlabel("Federated Round")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(
        range(1, 11)
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = OUTPUT / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# CONVERGENCE GRAPHS — ALL THREE
# ============================================================

make_convergence_graph(
    "dice",
    "Mean Dice",
    "Mean Dice Convergence",
    "01_mean_dice_convergence_E1_E2_E3.png",
)

make_convergence_graph(
    "roc_auc",
    "ROC-AUC",
    "ROC-AUC Convergence",
    "02_roc_auc_convergence_E1_E2_E3.png",
)

make_convergence_graph(
    "iou",
    "Mean IoU",
    "Mean IoU Convergence",
    "03_iou_convergence_E1_E2_E3.png",
)

make_convergence_graph(
    "sensitivity",
    "Sensitivity",
    "Sensitivity Convergence",
    "04_sensitivity_convergence_E1_E2_E3.png",
)

make_convergence_graph(
    "mcc",
    "MCC",
    "MCC Convergence",
    "05_mcc_convergence_E1_E2_E3.png",
)


# ============================================================
# FINAL MODEL COMPARISON — ALL THREE
# ============================================================

model_names = [
    "E1 – 3D U-Net + FedAvg",
    "E2 – Swin UNETR + FedAvg",
    "E3 – Swin UNETR + Adaptive FL",
]

short_names = [
    "E1\n3D U-Net + FedAvg",
    "E2\nSwin UNETR + FedAvg",
    "E3\nSwin UNETR + Adaptive FL",
]


def final_bar_chart(
    metric,
    ylabel,
    title,
    filename,
):

    values = []

    for model in model_names:

        value = final_results[model].get(metric)

        if value is None:
            value = 0

        values.append(value)

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        short_names,
        values,
    )

    plt.ylabel(ylabel)
    plt.title(title)

    # Display values on bars
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    path = OUTPUT / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# FINAL MAIN METRICS
# ============================================================

final_bar_chart(
    "dice",
    "Mean Dice",
    "Final Mean Dice Comparison",
    "07_final_mean_dice_E1_E2_E3.png",
)

final_bar_chart(
    "iou",
    "Mean IoU",
    "Final Mean IoU Comparison",
    "08_final_mean_iou_E1_E2_E3.png",
)

final_bar_chart(
    "sensitivity",
    "Sensitivity",
    "Final Sensitivity Comparison",
    "09_final_sensitivity_E1_E2_E3.png",
)

final_bar_chart(
    "mcc",
    "MCC",
    "Final MCC Comparison",
    "10_final_mcc_E1_E2_E3.png",
)

final_bar_chart(
    "roc_auc",
    "ROC-AUC",
    "Final ROC-AUC Comparison",
    "11_final_roc_auc_E1_E2_E3.png",
)

final_bar_chart(
    "pr_auc",
    "PR-AUC",
    "Final PR-AUC Comparison",
    "12_final_pr_auc_E1_E2_E3.png",
)

final_bar_chart(
    "hd95",
    "HD95",
    "Final HD95 Comparison",
    "13_final_hd95_E1_E2_E3.png",
)


# ============================================================
# REGION-WISE DICE
# ============================================================

region_names = [
    "WT",
    "TC",
    "ET",
]

region_keys = [
    "WT_dice",
    "TC_dice",
    "ET_dice",
]

x_positions = list(range(3))
width = 0.25

plt.figure(figsize=(10, 6))

for i, model in enumerate(model_names):

    values = [
        final_results[model].get(key, 0)
        for key in region_keys
    ]

    positions = [
        x + (i - 1) * width
        for x in x_positions
    ]

    bars = plt.bar(
        positions,
        values,
        width=width,
        label=model,
    )

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.xticks(
    x_positions,
    region_names,
)

plt.ylabel("Dice")
plt.xlabel("Tumor Region")
plt.title("Final Region-wise Dice Comparison")

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

path = OUTPUT / "14_region_wise_dice_E1_E2_E3.png"

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# ============================================================
# E3 ADAPTIVE WEIGHTS
# ============================================================

e3_weights = (
    PROJECTS[e3_name] /
    "adaptive_weights.csv"
)

if e3_weights.exists():

    c1 = []
    c2 = []
    c3 = []

    with open(e3_weights, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:

                rnd = int(
                    row.get("round")
                    or row.get("Round")
                )

                client = (
                    row.get("client")
                    or row.get("Client")
                    or row.get("client_id")
                )

                weight = float(
                    row.get("weight")
                    or row.get("Weight")
                    or row.get("aggregation_weight")
                )

            except Exception:
                continue

            if client == "C1":
                c1.append((rnd, weight))

            elif client == "C2":
                c2.append((rnd, weight))

            elif client == "C3":
                c3.append((rnd, weight))

    plt.figure(figsize=(10, 6))

    for data, label in [
        (c1, "C1"),
        (c2, "C2"),
        (c3, "C3"),
    ]:

        if data:

            data.sort()

            plt.plot(
                [x[0] for x in data],
                [x[1] for x in data],
                marker="o",
                linewidth=2,
                label=label,
            )

    plt.xlabel("Federated Round")
    plt.ylabel("Aggregation Weight")
    plt.title("E3 Adaptive Client Aggregation Weights")

    plt.xticks(range(1, 11))

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = OUTPUT / "15_E3_adaptive_weights.png"

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAPH GENERATION COMPLETE")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT)

print("\nGenerated PNG files:")

for file in sorted(OUTPUT.glob("*.png")):
    print(" ", file.name)

print("\nCSV:")
print(csv_path)

print("\nIMPORTANT:")
print(
    "E3 is represented at Round 10 using its FINAL GlobalTest "
    "result. It does not imply intermediate E3 validation."
)