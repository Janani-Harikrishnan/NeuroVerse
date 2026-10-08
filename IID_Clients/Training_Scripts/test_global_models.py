import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from scipy.ndimage import distance_transform_edt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path.home() / "Neuroverse"

GLOBAL_TEST_DIR = (
    PROJECT_ROOT
    / "IID_Clients"
    / "GlobalTest"
    / "test"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "IID_Clients"
    / "Training_Results_Final"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "IID_Clients"
    / "Global_Test_Results"
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (96, 96, 96)

IN_CHANNELS = 4
NUM_CLASSES = 4
BASE_CHANNELS = 16

EXPECTED_PATIENTS = 188

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Number of voxels per patient used for ROC/PR curves.
# This prevents huge memory usage.
ROC_PR_SAMPLES_PER_PATIENT = 5000

RANDOM_SEED = 42


# ============================================================
# EXACT MODEL USED DURING TRAINING
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class UNet3D(nn.Module):

    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        base=16
    ):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            base
        )

        self.enc2 = DoubleConv(
            base,
            base * 2
        )

        self.enc3 = DoubleConv(
            base * 2,
            base * 4
        )

        self.enc4 = DoubleConv(
            base * 4,
            base * 8
        )

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(
            base * 8,
            base * 16
        )

        # Decoder
        self.up4 = nn.ConvTranspose3d(
            base * 16,
            base * 8,
            2,
            stride=2
        )

        self.dec4 = DoubleConv(
            base * 16,
            base * 8
        )

        self.up3 = nn.ConvTranspose3d(
            base * 8,
            base * 4,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            base * 8,
            base * 4
        )

        self.up2 = nn.ConvTranspose3d(
            base * 4,
            base * 2,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            base * 4,
            base * 2
        )

        self.up1 = nn.ConvTranspose3d(
            base * 2,
            base,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            base * 2,
            base
        )

        self.out = nn.Conv3d(
            base,
            num_classes,
            kernel_size=1
        )


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        output = self.out(d1)

        return output


# ============================================================
# LOAD TRAINED CLIENT MODEL
# ============================================================

def load_client_model(client):

    model_path = (
        MODEL_ROOT
        / client
        / "best_model.pth"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{model_path}"
        )

    print()
    print("Loading:")
    print(model_path)

    model = UNet3D(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base=BASE_CHANNELS
    )

    checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=False
    )   

    # Support either:
    # 1. raw state_dict
    # 2. {"model_state_dict": ...}
    # 3. {"state_dict": ...}

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint["state_dict"]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # Remove DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value


    # IMPORTANT:
    # strict=True catches architecture mismatch.
    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print()
    print("✓ MODEL LOADED SUCCESSFULLY")

    return model


# ============================================================
# PATCH UTILITIES
# ============================================================

def get_patch_starts(
    length,
    patch
):

    if length <= patch:

        return [0]

    starts = list(
        range(
            0,
            length - patch + 1,
            patch
        )
    )

    if starts[-1] != length - patch:

        starts.append(
            length - patch
        )

    return starts


def pad_volume(image):

    """
    image shape:

        (4, X, Y, Z)
    """

    _, x, y, z = image.shape

    px = max(
        0,
        PATCH_SIZE[0] - x
    )

    py = max(
        0,
        PATCH_SIZE[1] - y
    )

    pz = max(
        0,
        PATCH_SIZE[2] - z
    )

    padded = np.pad(
        image,
        (
            (0, 0),
            (0, px),
            (0, py),
            (0, pz)
        ),
        mode="constant",
        constant_values=0
    )

    return padded


# ============================================================
# FULL VOLUME INFERENCE
# ============================================================

@torch.inference_mode()
def predict_volume(
    model,
    image
):

    original_shape = (
        image.shape[1:]
    )

    padded = pad_volume(
        image
    )

    _, X, Y, Z = padded.shape

    probabilities = np.zeros(
        (
            NUM_CLASSES,
            X,
            Y,
            Z
        ),
        dtype=np.float32
    )

    counts = np.zeros(
        (
            X,
            Y,
            Z
        ),
        dtype=np.float32
    )

    xs = get_patch_starts(
        X,
        PATCH_SIZE[0]
    )

    ys = get_patch_starts(
        Y,
        PATCH_SIZE[1]
    )

    zs = get_patch_starts(
        Z,
        PATCH_SIZE[2]
    )

    total_patches = (
        len(xs)
        * len(ys)
        * len(zs)
    )

    patch_number = 0

    for x in xs:

        for y in ys:

            for z in zs:

                patch_number += 1

                patch = padded[
                    :,
                    x:x + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    z:z + PATCH_SIZE[2]
                ]

                patch_tensor = (
                    torch.from_numpy(
                        patch
                    )
                    .float()
                    .unsqueeze(0)
                    .to(
                        DEVICE,
                        non_blocking=True
                    )
                )

                logits = model(
                    patch_tensor
                )

                probs = torch.softmax(
                    logits,
                    dim=1
                )[0].cpu().numpy()

                probabilities[
                    :,
                    x:x + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    z:z + PATCH_SIZE[2]
                ] += probs

                counts[
                    x:x + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    z:z + PATCH_SIZE[2]
                ] += 1.0

    probabilities /= np.maximum(
        counts,
        1e-8
    )[None, ...]

    probabilities = probabilities[
        :,
        :original_shape[0],
        :original_shape[1],
        :original_shape[2]
    ]

    prediction = np.argmax(
        probabilities,
        axis=0
    ).astype(
        np.uint8
    )

    return (
        prediction,
        probabilities
    )


# ============================================================
# REGION MASKS
# ============================================================

def region_masks(mask):

    """
    BraTS:

    0 = background
    1 = NCR
    2 = edema
    3 = enhancing tumor

    WT = 1 + 2 + 3
    TC = 1 + 3
    ET = 3
    """

    WT = mask > 0

    TC = (
        (mask == 1)
        |
        (mask == 3)
    )

    ET = (
        mask == 3
    )

    return (
        WT,
        TC,
        ET
    )


# ============================================================
# DICE
# ============================================================

def dice_score(
    pred,
    target
):

    pred = pred.astype(
        bool
    )

    target = target.astype(
        bool
    )

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    denominator = (
        pred.sum()
        +
        target.sum()
    )

    if denominator == 0:

        return 1.0

    return float(
        2.0
        * intersection
        / denominator
    )


# ============================================================
# IOU
# ============================================================

def iou_score(
    pred,
    target
):

    pred = pred.astype(
        bool
    )

    target = target.astype(
        bool
    )

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    union = np.logical_or(
        pred,
        target
    ).sum()

    if union == 0:

        return 1.0

    return float(
        intersection / union
    )


# ============================================================
# HD95
# ============================================================

def hd95(
    pred,
    target
):

    pred = pred.astype(
        bool
    )

    target = target.astype(
        bool
    )

    if pred.sum() == 0 and target.sum() == 0:

        return 0.0

    if pred.sum() == 0 or target.sum() == 0:

        return np.nan

    pred_distance = (
        distance_transform_edt(
            ~pred
        )
    )

    target_distance = (
        distance_transform_edt(
            ~target
        )
    )

    pred_surface = (
        pred
        &
        (
            distance_transform_edt(
                pred
            ) <= 1
        )
    )

    target_surface = (
        target
        &
        (
            distance_transform_edt(
                target
            ) <= 1
        )
    )

    distances_1 = (
        target_distance[
            pred_surface
        ]
    )

    distances_2 = (
        pred_distance[
            target_surface
        ]
    )

    distances = np.concatenate(
        [
            distances_1,
            distances_2
        ]
    )

    if len(distances) == 0:

        return np.nan

    return float(
        np.percentile(
            distances,
            95
        )
    )


# ============================================================
# STANDARD CLASSIFICATION METRICS
# ============================================================

def binary_metrics(
    pred,
    target
):

    p = (
        pred
        .astype(np.uint8)
        .reshape(-1)
    )

    t = (
        target
        .astype(np.uint8)
        .reshape(-1)
    )

    accuracy = accuracy_score(
        t,
        p
    )

    precision = precision_score(
        t,
        p,
        zero_division=0
    )

    sensitivity = recall_score(
        t,
        p,
        zero_division=0
    )

    f1 = f1_score(
        t,
        p,
        zero_division=0
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            t,
            p
        )
    )

    try:

        mcc = matthews_corrcoef(
            t,
            p
        )

    except Exception:

        mcc = 0.0


    tn = np.sum(
        (p == 0)
        &
        (t == 0)
    )

    fp = np.sum(
        (p == 1)
        &
        (t == 0)
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )


    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "f1":
            float(f1),

        "balanced_accuracy":
            float(balanced_accuracy),

        "mcc":
            float(mcc)
    }


# ============================================================
# ROC-AUC / PR-AUC
# ============================================================

def auc_metrics(
    probability,
    target
):

    p = probability.reshape(
        -1
    )

    t = (
        target
        .astype(np.uint8)
        .reshape(-1)
    )

    if len(
        np.unique(t)
    ) < 2:

        return (
            np.nan,
            np.nan
        )

    try:

        roc_auc = (
            roc_auc_score(
                t,
                p
            )
        )

    except Exception:

        roc_auc = np.nan


    try:

        pr_auc = (
            average_precision_score(
                t,
                p
            )
        )

    except Exception:

        pr_auc = np.nan


    return (
        float(roc_auc)
        if not np.isnan(roc_auc)
        else np.nan,

        float(pr_auc)
        if not np.isnan(pr_auc)
        else np.nan
    )


# ============================================================
# ONE PATIENT
# ============================================================

def evaluate_patient(
    model,
    image,
    mask,
    rng
):

    start = time.time()

    prediction, probabilities = (
        predict_volume(
            model,
            image
        )
    )

    WT_true, TC_true, ET_true = (
        region_masks(
            mask
        )
    )

    WT_pred, TC_pred, ET_pred = (
        region_masks(
            prediction
        )
    )


    regions = {

        "WT": (
            WT_pred,
            WT_true,
            probabilities[
                1:
            ].sum(axis=0)
        ),

        "TC": (
            TC_pred,
            TC_true,
            probabilities[
                [1, 3]
            ].sum(axis=0)
        ),

        "ET": (
            ET_pred,
            ET_true,
            probabilities[
                3
            ]
        )
    }


    results = {}

    roc_data = {}


    for region_name, (
        pred,
        target,
        probability
    ) in regions.items():

        # Dice
        results[
            f"{region_name}_dice"
        ] = dice_score(
            pred,
            target
        )

        # IoU
        results[
            f"{region_name}_iou"
        ] = iou_score(
            pred,
            target
        )

        # Accuracy / precision / recall /
        # specificity / F1 / balanced accuracy / MCC
        basic = binary_metrics(
            pred,
            target
        )

        for metric, value in basic.items():

            results[
                f"{region_name}_{metric}"
            ] = value


        # HD95
        results[
            f"{region_name}_hd95"
        ] = hd95(
            pred,
            target
        )


        # ROC-AUC / PR-AUC
        roc_auc, pr_auc = (
            auc_metrics(
                probability,
                target
            )
        )

        results[
            f"{region_name}_roc_auc"
        ] = roc_auc

        results[
            f"{region_name}_pr_auc"
        ] = pr_auc


        # ----------------------------------------------------
        # Sample voxels for ROC/PR graph
        # ----------------------------------------------------

        p = probability.reshape(
            -1
        )

        t = (
            target
            .astype(np.uint8)
            .reshape(-1)
        )

        sample_count = min(
            ROC_PR_SAMPLES_PER_PATIENT,
            len(t)
        )

        if sample_count < len(t):

            indices = rng.choice(
                len(t),
                size=sample_count,
                replace=False
            )

            p = p[indices]
            t = t[indices]

        roc_data[
            region_name
        ] = (
            p.astype(
                np.float32
            ),
            t
        )


    # ========================================================
    # MEAN WT / TC / ET
    # ========================================================

    metrics = [

        "dice",
        "iou",
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "mcc",
        "hd95",
        "roc_auc",
        "pr_auc"
    ]


    for metric in metrics:

        values = [

            results[
                f"WT_{metric}"
            ],

            results[
                f"TC_{metric}"
            ],

            results[
                f"ET_{metric}"
            ]
        ]

        results[
            f"mean_{metric}"
        ] = float(
            np.nanmean(
                values
            )
        )


    results[
        "inference_time_seconds"
    ] = (
        time.time()
        -
        start
    )


    return (
        results,
        roc_data
    )


# ============================================================
# PATIENT METRIC GRAPHS
# ============================================================

def save_patient_graphs(
    df,
    output_dir
):

    graph_dir = (
        output_dir
        / "graphs"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    metrics = [

        (
            "mean_dice",
            "Mean Dice"
        ),

        (
            "mean_iou",
            "Mean IoU"
        ),

        (
            "mean_accuracy",
            "Mean Accuracy"
        ),

        (
            "mean_precision",
            "Mean Precision"
        ),

        (
            "mean_sensitivity",
            "Mean Sensitivity"
        ),

        (
            "mean_specificity",
            "Mean Specificity"
        ),

        (
            "mean_f1",
            "Mean F1"
        ),

        (
            "mean_balanced_accuracy",
            "Balanced Accuracy"
        ),

        (
            "mean_mcc",
            "MCC"
        ),

        (
            "mean_roc_auc",
            "ROC-AUC"
        ),

        (
            "mean_pr_auc",
            "PR-AUC"
        ),

        (
            "mean_hd95",
            "HD95"
        )
    ]


    for column, title in metrics:

        if column not in df.columns:

            continue

        values = (
            df[column]
            .astype(float)
        )

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            values.values,
            linewidth=1
        )

        mean_value = (
            values.mean()
        )

        plt.axhline(
            mean_value,
            linestyle="--",
            label=(
                f"Mean = "
                f"{mean_value:.4f}"
            )
        )

        plt.title(
            f"{title} Across "
            f"{len(df)} Global Test Patients"
        )

        plt.xlabel(
            "Patient index"
        )

        plt.ylabel(
            title
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            graph_dir
            / f"{column}.png",
            dpi=200
        )

        plt.close()


# ============================================================
# REGION COMPARISON GRAPHS
# ============================================================

def save_region_graphs(
    region_df,
    output_dir
):

    graph_dir = (
        output_dir
        / "graphs"
    )

    metrics = [

        ("dice", "Dice"),

        ("iou", "IoU"),

        ("accuracy", "Accuracy"),

        ("precision", "Precision"),

        ("sensitivity", "Sensitivity"),

        ("specificity", "Specificity"),

        ("f1", "F1"),

        (
            "balanced_accuracy",
            "Balanced Accuracy"
        ),

        ("mcc", "MCC"),

        ("roc_auc", "ROC-AUC"),

        ("pr_auc", "PR-AUC")
    ]


    for column, title in metrics:

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            region_df["region"],
            region_df[column]
        )

        plt.title(
            f"{title}: WT vs TC vs ET"
        )

        plt.xlabel(
            "Region"
        )

        plt.ylabel(
            title
        )

        plt.tight_layout()

        plt.savefig(
            graph_dir
            / f"region_{column}.png",
            dpi=200
        )

        plt.close()


# ============================================================
# ROC + PR CURVES
# ============================================================

def save_roc_pr_curves(
    roc_store,
    output_dir
):

    graph_dir = (
        output_dir
        / "graphs"
    )


    # ========================================================
    # ROC
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    for region in [
        "WT",
        "TC",
        "ET"
    ]:

        if region not in roc_store:

            continue

        probabilities = np.concatenate(
            roc_store[
                region
            ]["probabilities"]
        )

        targets = np.concatenate(
            roc_store[
                region
            ]["targets"]
        )


        if len(
            np.unique(targets)
        ) < 2:

            continue


        fpr, tpr, _ = (
            roc_curve(
                targets,
                probabilities
            )
        )

        auc_value = (
            roc_auc_score(
                targets,
                probabilities
            )
        )


        plt.plot(
            fpr,
            tpr,
            label=(
                f"{region} "
                f"AUC={auc_value:.4f}"
            )
        )


        pd.DataFrame(
            {
                "fpr": fpr,
                "tpr": tpr
            }
        ).to_csv(
            output_dir
            / f"{region}_roc_curve.csv",
            index=False
        )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Global Test ROC Curves"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        graph_dir
        / "ROC_curves.png",
        dpi=200
    )

    plt.close()


    # ========================================================
    # PRECISION-RECALL
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )


    for region in [
        "WT",
        "TC",
        "ET"
    ]:

        if region not in roc_store:

            continue

        probabilities = np.concatenate(
            roc_store[
                region
            ]["probabilities"]
        )

        targets = np.concatenate(
            roc_store[
                region
            ]["targets"]
        )


        if len(
            np.unique(targets)
        ) < 2:

            continue


        precision, recall, _ = (
            precision_recall_curve(
                targets,
                probabilities
            )
        )

        pr_auc = (
            average_precision_score(
                targets,
                probabilities
            )
        )


        plt.plot(
            recall,
            precision,
            label=(
                f"{region} "
                f"AP={pr_auc:.4f}"
            )
        )


        pd.DataFrame(
            {
                "recall": recall,
                "precision": precision
            }
        ).to_csv(
            output_dir
            / f"{region}_pr_curve.csv",
            index=False
        )


    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Global Test Precision-Recall Curves"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        graph_dir
        / "PR_curves.png",
        dpi=200
    )

    plt.close()


# ============================================================
# EVALUATE CLIENT
# ============================================================

def evaluate_client(
    client
):

    print()
    print("=" * 80)
    print(
        f"TESTING CLIENT MODEL: {client}"
    )
    print("=" * 80)


    patient_files = sorted(
        GLOBAL_TEST_DIR.glob(
            "*.npz"
        )
    )


    print()
    print(
        f"Global test patients found: "
        f"{len(patient_files)}"
    )


    if len(patient_files) != EXPECTED_PATIENTS:

        print(
            f"WARNING: expected "
            f"{EXPECTED_PATIENTS}, "
            f"found {len(patient_files)}"
        )


    output_dir = (
        RESULT_ROOT
        / client
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD CLIENT MODEL
    # ========================================================

    model = load_client_model(
        client
    )


    # ========================================================
    # ROC/PR STORAGE
    # ========================================================

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    roc_store = {

        "WT": {
            "probabilities": [],
            "targets": []
        },

        "TC": {
            "probabilities": [],
            "targets": []
        },

        "ET": {
            "probabilities": [],
            "targets": []
        }
    }


    patient_results = []

    total_start = time.time()


    # ========================================================
    # TEST ALL 188 PATIENTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FULL GLOBAL TEST"
    )
    print("=" * 80)


    for index, patient_file in enumerate(
        patient_files,
        start=1
    ):

        data = np.load(
            patient_file
        )

        image = data[
            "image"
        ]

        mask = data[
            "mask"
        ]


        results, roc_data = (
            evaluate_patient(
                model,
                image,
                mask,
                rng
            )
        )


        results[
            "patient_id"
        ] = patient_file.stem


        patient_results.append(
            results
        )


        # Store ROC/PR samples
        for region in [
            "WT",
            "TC",
            "ET"
        ]:

            probabilities, targets = (
                roc_data[region]
            )

            roc_store[
                region
            ][
                "probabilities"
            ].append(
                probabilities
            )

            roc_store[
                region
            ][
                "targets"
            ].append(
                targets
            )


        print(
            f"[{index:3d}/"
            f"{len(patient_files)}] "
            f"{patient_file.stem} | "
            f"WT="
            f"{results['WT_dice']:.4f} "
            f"TC="
            f"{results['TC_dice']:.4f} "
            f"ET="
            f"{results['ET_dice']:.4f} | "
            f"{results['inference_time_seconds']:.2f}s"
        )


    total_time = (
        time.time()
        -
        total_start
    )


    df = pd.DataFrame(
        patient_results
    )


    # patient_id first
    df = df[
        ["patient_id"]
        +
        [
            c
            for c in df.columns
            if c != "patient_id"
        ]
    ]


    # ========================================================
    # SAVE PER-PATIENT METRICS
    # ========================================================

    patient_metrics_file = (
        output_dir
        / "patient_metrics.csv"
    )

    df.to_csv(
        patient_metrics_file,
        index=False
    )


    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary_metrics = [

        "mean_accuracy",

        "mean_precision",

        "mean_sensitivity",

        "mean_specificity",

        "mean_f1",

        "mean_dice",

        "mean_iou",

        "mean_balanced_accuracy",

        "mean_mcc",

        "mean_hd95",

        "mean_roc_auc",

        "mean_pr_auc"
    ]


    summary = {}


    for metric in summary_metrics:

        summary[
            metric
        ] = float(
            np.nanmean(
                df[metric].values
            )
        )


    summary[
        "num_patients"
    ] = len(df)


    summary[
        "total_inference_time_seconds"
    ] = float(
        total_time
    )


    summary[
        "mean_inference_time_seconds"
    ] = float(
        df[
            "inference_time_seconds"
        ].mean()
    )


    summary_file = (
        output_dir
        / "summary.csv"
    )


    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_file,
        index=False
    )


    # ========================================================
    # REGION SUMMARY
    # ========================================================

    region_rows = []


    for region in [
        "WT",
        "TC",
        "ET"
    ]:

        row = {
            "region":
                region
        }


        for metric in [

            "dice",
            "iou",
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "balanced_accuracy",
            "mcc",
            "hd95",
            "roc_auc",
            "pr_auc"
        ]:

            column = (
                f"{region}_{metric}"
            )

            row[
                metric
            ] = float(
                np.nanmean(
                    df[
                        column
                    ].values
                )
            )


        region_rows.append(
            row
        )


    region_df = pd.DataFrame(
        region_rows
    )


    region_file = (
        output_dir
        / "region_summary.csv"
    )


    region_df.to_csv(
        region_file,
        index=False
    )


    # ========================================================
    # GRAPHS
    # ========================================================

    print()
    print(
        "Saving graphs..."
    )


    save_patient_graphs(
        df,
        output_dir
    )


    save_region_graphs(
        region_df,
        output_dir
    )


    save_roc_pr_curves(
        roc_store,
        output_dir
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "GLOBAL TEST RESULTS"
    )
    print("=" * 80)

    print(
        f"Client Model       : "
        f"{client}"
    )

    print(
        f"Patients tested    : "
        f"{len(df)}"
    )

    print(
        f"Mean Accuracy      : "
        f"{summary['mean_accuracy']:.4f}"
    )

    print(
        f"Mean Precision     : "
        f"{summary['mean_precision']:.4f}"
    )

    print(
        f"Mean Sensitivity   : "
        f"{summary['mean_sensitivity']:.4f}"
    )

    print(
        f"Mean Specificity   : "
        f"{summary['mean_specificity']:.4f}"
    )

    print(
        f"Mean F1            : "
        f"{summary['mean_f1']:.4f}"
    )

    print(
        f"Mean Dice          : "
        f"{summary['mean_dice']:.4f}"
    )

    print(
        f"Mean IoU           : "
        f"{summary['mean_iou']:.4f}"
    )

    print(
        f"Mean Balanced Acc. : "
        f"{summary['mean_balanced_accuracy']:.4f}"
    )

    print(
        f"Mean MCC           : "
        f"{summary['mean_mcc']:.4f}"
    )

    print(
        f"Mean HD95          : "
        f"{summary['mean_hd95']:.4f}"
    )

    print(
        f"Mean ROC-AUC       : "
        f"{summary['mean_roc_auc']:.4f}"
    )

    print(
        f"Mean PR-AUC        : "
        f"{summary['mean_pr_auc']:.4f}"
    )

    print(
        f"Total inference    : "
        f"{summary['total_inference_time_seconds']:.2f}s"
    )


    print()
    print("=" * 80)
    print(
        "REGION RESULTS"
    )
    print("=" * 80)

    print(
        region_df.to_string(
            index=False
        )
    )


    print()
    print("=" * 80)
    print(
        "RESULTS SAVED"
    )
    print("=" * 80)

    print(
        f"Directory:\n"
        f"{output_dir}"
    )

    print(
        f"\nPatient metrics:\n"
        f"{patient_metrics_file}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_file}"
    )

    print(
        f"\nRegion summary:\n"
        f"{region_file}"
    )

    print(
        f"\nGraphs:\n"
        f"{output_dir / 'graphs'}"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client",
        required=True,
        choices=[
            "C1",
            "C2",
            "C3"
        ]
    )

    args = parser.parse_args()


    print("=" * 80)
    print(
        "NEUROVERSE — GLOBAL TEST EVALUATION"
    )
    print("=" * 80)


    print()
    print(
        f"Device: {DEVICE}"
    )


    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    else:

        print(
            "WARNING: CUDA NOT AVAILABLE"
        )


    print()
    print(
        f"Global test directory:\n"
        f"{GLOBAL_TEST_DIR}"
    )


    print()
    print(
        f"Expected patients: "
        f"{EXPECTED_PATIENTS}"
    )


    evaluate_client(
        args.client
    )


if __name__ == "__main__":

    main()